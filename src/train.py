"""
src/train.py
-------------
Fase 3 — Loss, metriche e training loop.

Ci sono ORA due metriche diverse, non confonderle:

1. "accuracy per mossa" (`run_epoch`, usata durante il training/validation
   ad ogni epoca): su ciascun ESEMPIO del dataset espanso (una singola
   posizione lungo una linea di matto) guarda se la mossa con score più
   alto è quella giusta. È la metrica naturale per il training a batch
   perché è veloce e differenziabile.

2. "accuracy sequenza completa" (`evaluate_full_sequence` /
   `solve_puzzle_with_gnn`, usata a fine training su val/test e nel
   confronto con l'LLM): a livello di PUZZLE INTERO. Si parte dalla
   posizione iniziale, si fa predire al modello la mossa, si applica la
   risposta forzata dell'avversario (presa dai dati reali, il modello non
   la deve indovinare visto che non tocca a lui), si ripete finché non
   finiscono le mosse della linea. Il puzzle conta come risolto solo se
   TUTTE le mosse del solver sono state predette correttamente. Questa è
   la metrica comparabile con quella della baseline LLM
   (`full_sequence_correct` in src/llm_baseline.py).

Uso da riga di comando (addestra sia il modello con timing che quello
senza, per l'ablation study):
    python -m scripts.run_training
"""

import json
import os
import time

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as pyg_softmax, scatter as pyg_scatter

import config
from src.dataset import ChessPuzzleGraphDataset
from src.model import TimedGNN
from src.chess_encoding import (NODE_FEAT_DIM, board_to_graph_data, solver_moves_sequence,
                                compute_think_time_sec)
from src.stats import mcnemar_from_flags
from src.data_pipeline import print_n_distribution, check_no_leakage_from_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _atomic_torch_save(obj, path: str):
    """Salva su file temporaneo nella stessa cartella e poi sostituisce il target."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_training_complete_marker(ckpt_path: str, include_timing: bool,
                                    final_epoch: int, best_val_acc: float):
    marker = ckpt_path + ".complete.json"
    payload = {
        "complete": True,
        "include_timing": bool(include_timing),
        "final_epoch": int(final_epoch),
        "best_val_acc": float(best_val_acc),
        "subset_size": config.SUBSET_SIZE,
        "num_epochs": config.NUM_EPOCHS,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": config.EARLY_STOPPING_MIN_DELTA,
        "learning_rate": config.LEARNING_RATE,
        "random_seed": config.RANDOM_SEED,
        "graph_cache_max_items": config.GRAPH_CACHE_MAX_ITEMS,
        "hidden_dim": config.HIDDEN_DIM,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
        "batch_size": config.BATCH_SIZE,
        "data_pipeline_schema_version": config.DATA_PIPELINE_SCHEMA_VERSION,
    }
    tmp = marker + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, marker)


# ----------------------------------------------------------------------
# 1. Training/validation a batch: accuracy PER MOSSA (per-ply)
# ----------------------------------------------------------------------

def compute_targets_and_offsets(batch):
    num_cand = batch.num_candidate_moves
    offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=num_cand.device),
        torch.cumsum(num_cand, dim=0)[:-1],
    ])
    correct_global_idx = offsets + batch.y.squeeze(-1)
    return correct_global_idx


def loss_and_metrics(scores, edge_batch, correct_global_idx, num_graphs, eps=1e-12):
    probs = pyg_softmax(scores, edge_batch)
    loss = -torch.log(probs[correct_global_idx] + eps).mean()

    # Calcola gli argmax raggruppati senza chiamare .item() per ogni grafo.
    # Il vecchio ciclo provocava centinaia di migliaia di sincronizzazioni CPU/GPU
    # per epoca sui subset grandi, rallentando drasticamente il training.
    max_scores = pyg_scatter(scores, edge_batch, dim=0,
                             dim_size=num_graphs, reduce="max")
    candidate_idx = torch.arange(scores.numel(), device=scores.device,
                                 dtype=torch.long)
    sentinel = torch.full_like(candidate_idx, scores.numel())
    winners = torch.where(scores == max_scores[edge_batch], candidate_idx, sentinel)
    pred_global_idx = pyg_scatter(winners, edge_batch, dim=0,
                                  dim_size=num_graphs, reduce="min")
    correct_flags = pred_global_idx.eq(correct_global_idx)

    acc = correct_flags.float().mean().item()
    return loss, acc, correct_flags


def run_epoch(model, loader, optimizer=None, progress_bar=None, progress_phase=None,
              progress_every=500):
    """Un'epoca di training o valutazione, a livello di ESEMPIO (posizione
    singola lungo una linea di matto), non di puzzle intero. Vedi
    docstring del modulo."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_examples = 0.0, 0, 0
    per_n_correct, per_n_total = {}, {}
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            scores, edge_batch = model(batch)
            correct_global_idx = compute_targets_and_offsets(batch)
            loss, acc, correct_flags = loss_and_metrics(
                scores, edge_batch, correct_global_idx, batch.num_graphs
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = batch.num_graphs
            total_loss += loss.item() * bs
            total_correct += acc * bs
            total_examples += bs

            mate_n_vals = batch.mate_n.squeeze(-1).cpu().tolist()
            for n_val, is_correct in zip(mate_n_vals, correct_flags.cpu().tolist()):
                per_n_total[n_val] = per_n_total.get(n_val, 0) + 1
                per_n_correct[n_val] = per_n_correct.get(n_val, 0) + int(is_correct)

            batches_done = batch_idx + 1
            # La barra viene abilitata soltanto durante la prima epoca. È una
            # singola barra condivisa tra training e validation; gli aggiornamenti
            # del postfix avvengono ogni N batch senza generare nuove righe.
            if progress_bar is not None:
                progress_bar.update(1)
                if (batches_done % progress_every == 0
                        or batches_done == len(loader)):
                    graph_cache = getattr(loader.dataset, "_data_cache", None)
                    cache_info = (
                        f"cache grafi {len(graph_cache):,}"
                        if graph_cache is not None else "cache grafi n/d"
                    )
                    progress_bar.set_postfix_str(
                        f"{progress_phase} | esempi {total_examples:,}/"
                        f"{len(loader.dataset):,} | {cache_info}",
                        refresh=False,
                    )

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    per_n_acc = {n: per_n_correct[n] / per_n_total[n] for n in sorted(per_n_total)}
    return avg_loss, avg_acc, per_n_acc


def maybe_subset(df: pd.DataFrame, n, seed=config.RANDOM_SEED):
    """Sottocampiona fino a n righe, bilanciando per mate_n e ridistribuendo
    la quota non utilizzata dai gruppi piccoli fra i gruppi con righe disponibili.

    Il subset raggiunge quindi n elementi quando il dataset ne contiene almeno n;
    N4/N5 vengono inclusi interamente se sono più piccoli della quota e il residuo
    viene distribuito in modo round-robin agli altri valori di n. Questo rende
    SUBSET_SIZE=300000 un target effettivo, non solo la somma delle quote iniziali.
    """
    if n is None or n >= len(df):
        return df
    target = max(0, min(int(n), len(df)))
    if target == 0:
        return df.iloc[0:0].copy().reset_index(drop=True)
    if "mate_n" not in df.columns:
        return df.sample(target, random_state=seed).reset_index(drop=True)

    groups = {key: group for key, group in df.groupby("mate_n", sort=True)}
    keys = list(groups)
    if not keys:
        return df.iloc[0:0].copy().reset_index(drop=True)

    base = target // len(keys)
    quota = {key: min(len(groups[key]), base) for key in keys}
    remaining = target - sum(quota.values())
    while remaining > 0:
        progressed = False
        for key in keys:
            if remaining == 0:
                break
            if quota[key] < len(groups[key]):
                quota[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    frames = []
    for key in keys:
        count = quota[key]
        if count:
            # Seed distinto ma deterministico per gruppo.
            group_seed = int(seed) + int(key)
            frames.append(groups[key].sample(count, random_state=group_seed))
    result = pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0].copy()
    # L'ordine è stabile e indipendente dall'ordine originario delle righe.
    return result.reset_index(drop=True)


def train_one_variant(df_train, df_val, include_timing: bool, ckpt_path: str, tag: str):
    """Addestra una variante del modello (con o senza timing) e salva il
    checkpoint migliore su validation accuracy PER MOSSA.

    Salva anche un checkpoint "resumable" (modello + optimizer + epoca +
    history + stato dell'early stopping) DOPO OGNI EPOCA in
    <ckpt_path>.resume.pt: se la sessione Colab si disconnette a metà,
    rilanciando lo stesso comando la funzione riparte dall'epoca
    successiva invece che da zero.

    Early stopping: se val_acc non migliora per config.EARLY_STOPPING_PATIENCE
    epoche consecutive, il training si interrompe prima di arrivare a
    NUM_EPOCHS. Il checkpoint salvato (ckpt_path) resta comunque quello
    della MIGLIOR epoca vista, non dell'ultima.
    """
    resume_path = ckpt_path + ".resume.pt"

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    train_ds = ChessPuzzleGraphDataset(df_train, include_timing=include_timing)
    val_ds = ChessPuzzleGraphDataset(df_val, include_timing=include_timing)
    print(f"[{tag}] esempi (puzzle, mossa) di training: {len(train_ds):,} "
          f"(da {len(df_train):,} puzzle) | validation: {len(val_ds):,} (da {len(df_val):,} puzzle)")
    ex_tr, ex_va = train_ds.examples_per_n(), val_ds.examples_per_n()
    print(f"[{tag}] esempi (posizione, mossa) per n:")
    print("        n   " + "  ".join(f"N{n:<7d}" for n in sorted(set(ex_tr) | set(ex_va))))
    print("   train    " + "  ".join(f"{ex_tr.get(n, 0):<8,d}" for n in sorted(set(ex_tr) | set(ex_va))))
    print("   val      " + "  ".join(f"{ex_va.get(n, 0):<8,d}" for n in sorted(set(ex_tr) | set(ex_va))))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = TimedGNN(NODE_FEAT_DIM, hidden_dim=config.HIDDEN_DIM,
                      num_layers=config.NUM_LAYERS, dropout=config.DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    last_completed_epoch = 0
    best_model_state = None

    if os.path.exists(resume_path):
        # I checkpoint staged sono locali e fidati; weights_only=False è necessario
        # per consentire il caricamento dello stato completo dell'optimizer nelle
        # versioni recenti di PyTorch.
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:  # compatibilità con versioni precedenti di PyTorch
            checkpoint = torch.load(resume_path, map_location=device)
        saved_cfg = checkpoint.get("run_config", {})
        current_cfg = {
            "include_timing": bool(include_timing),
            "subset_size": config.SUBSET_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": config.EARLY_STOPPING_MIN_DELTA,
            "learning_rate": config.LEARNING_RATE,
            "random_seed": config.RANDOM_SEED,
            "graph_cache_max_items": config.GRAPH_CACHE_MAX_ITEMS,
            "hidden_dim": config.HIDDEN_DIM,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "batch_size": config.BATCH_SIZE,
            "data_pipeline_schema_version": config.DATA_PIPELINE_SCHEMA_VERSION,
        }
        if saved_cfg and saved_cfg != current_cfg:
            raise RuntimeError(
                f"[{tag}] Il checkpoint di ripresa usa una configurazione diversa. "
                f"Salvata={saved_cfg}; attuale={current_cfg}. Non proseguo per evitare "
                "di mescolare esperimenti. Ripristina la configurazione originaria "
                "oppure rimuovi esplicitamente il checkpoint .resume.pt di questa fase."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint["history"]
        best_val_acc = checkpoint["best_val_acc"]
        best_model_state = checkpoint.get("best_model_state")
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        last_completed_epoch = int(checkpoint["epoch"])
        start_epoch = last_completed_epoch + 1
        # Ripristina RNG DOPO aver creato modello/optimizer: la prossima epoca
        # prosegue con lo stato casuale salvato alla fine dell'ultima epoca.
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states"):
            try:
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
            except Exception as exc:
                print(f"[{tag}] Nota: impossibile ripristinare interamente lo stato RNG CUDA ({exc}).")
        # Il resume contiene anche la copia dei pesi BEST. In questo modo il
        # checkpoint migliore resta riparabile se Colab si interrompe nel breve
        # intervallo fra i due salvataggi atomici.
        if best_model_state is not None:
            _atomic_torch_save(best_model_state, ckpt_path)
        print(f"[{tag}] Ripresa da checkpoint: riparto dall'epoca {start_epoch} "
              f"(best_val_acc finora: {best_val_acc:.3f}, "
              f"{epochs_without_improvement} epoche senza miglioramento)")

    if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE and last_completed_epoch > 0:
        print(f"[{tag}] Il checkpoint indica che l'early stopping era già scattato all'epoca "
              f"{last_completed_epoch}; finalizzo la fase senza eseguire un'altra epoca.")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            _atomic_torch_save(best_model_state, ckpt_path)
        elif os.path.exists(ckpt_path):
            try:
                best_state = torch.load(ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                best_state = torch.load(ckpt_path, map_location=device)
            if isinstance(best_state, dict) and "model_state" in best_state:
                best_state = best_state["model_state"]
            model.load_state_dict(best_state)
        _write_training_complete_marker(ckpt_path, include_timing, last_completed_epoch, best_val_acc)
        if os.path.exists(resume_path):
            os.remove(resume_path)
        return model, history, best_val_acc

    if start_epoch > config.NUM_EPOCHS:
        print(f"[{tag}] Già completate tutte le {config.NUM_EPOCHS} epoche in un run precedente.")
        # Caso raro: Colab è terminato dopo l'ultimo checkpoint d'epoca ma prima
        # della pulizia finale. Ripristina i pesi BEST e completa la fase in modo
        # idempotente.
        if os.path.exists(ckpt_path):
            try:
                best_state = torch.load(ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                best_state = torch.load(ckpt_path, map_location=device)
            if isinstance(best_state, dict) and "model_state" in best_state:
                best_state = best_state["model_state"]
            model.load_state_dict(best_state)
        _write_training_complete_marker(ckpt_path, include_timing, config.NUM_EPOCHS, best_val_acc)
        if os.path.exists(resume_path):
            os.remove(resume_path)
        return model, history, best_val_acc

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        # La barra viene mostrata soltanto per la prima epoca eseguita in questo
        # run (anche se si riprende da un checkpoint). Le successive mostrano
        # soltanto la riga riepilogativa a fine epoca.
        progress_bar = None
        is_first_epoch = epoch == start_epoch
        if is_first_epoch:
            print(f"[{tag}] avvio epoca {epoch:02d}/{config.NUM_EPOCHS} "
                  f"({len(train_ds):,} esempi train; {len(val_ds):,} val)", flush=True)
            progress_bar = tqdm(
                total=len(train_loader) + len(val_loader),
                desc=f"[{tag}] epoca {epoch:02d}/{config.NUM_EPOCHS}",
                unit="batch",
                dynamic_ncols=True,
                leave=True,
            )
            progress_bar.set_postfix_str(
                f"train | esempi 0/{len(train_ds):,} | cache grafi 0",
                refresh=True,
            )

        t0 = time.time()
        try:
            train_loss, train_acc, _ = run_epoch(
                model, train_loader, optimizer,
                progress_bar=progress_bar,
                progress_phase="train",
            )
            if progress_bar is not None:
                progress_bar.set_postfix_str(
                    f"validation | esempi 0/{len(val_ds):,} | cache grafi 0",
                    refresh=True,
                )
            val_loss, val_acc, _ = run_epoch(
                model, val_loader, optimizer=None,
                progress_bar=progress_bar,
                progress_phase="validation",
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc + config.EARLY_STOPPING_MIN_DELTA
        if improved:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            _atomic_torch_save(best_model_state, ckpt_path)
        else:
            epochs_without_improvement += 1

        # Checkpoint di ripresa: sovrascritto ogni epoca, non solo quando
        # migliora — serve a riprendere il training, non a scegliere il
        # modello finale (quello resta ckpt_path, solo sui best).
        run_config = {
            "include_timing": bool(include_timing),
            "subset_size": config.SUBSET_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": config.EARLY_STOPPING_MIN_DELTA,
            "learning_rate": config.LEARNING_RATE,
            "random_seed": config.RANDOM_SEED,
            "graph_cache_max_items": config.GRAPH_CACHE_MAX_ITEMS,
            "hidden_dim": config.HIDDEN_DIM,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "batch_size": config.BATCH_SIZE,
            "data_pipeline_schema_version": config.DATA_PIPELINE_SCHEMA_VERSION,
        }
        resume_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_val_acc": best_val_acc,
            "best_model_state": best_model_state,
            "epochs_without_improvement": epochs_without_improvement,
            "run_config": run_config,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        _atomic_torch_save(resume_payload, resume_path)

        marker = "" if improved else f" (nessun miglioramento da {epochs_without_improvement} epoche)"
        print(f"[{tag}] epoch {epoch:02d}/{config.NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc_per_mossa={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc_per_mossa={val_acc:.3f} | {dt:.1f}s{marker}")

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n[{tag}] Early stopping: nessun miglioramento della val accuracy "
                  f"per {config.EARLY_STOPPING_PATIENCE} epoche consecutive. "
                  f"Mi fermo all'epoca {epoch}/{config.NUM_EPOCHS}.")
            break

    print(f"\nMiglior val accuracy per mossa ({tag}): {best_val_acc:.3f} -> salvato in {ckpt_path}")
    # Il marker viene scritto PRIMA di rimuovere il resume: se Colab si interrompe
    # in questo intervallo, al riavvio train_one_variant vedrà l'epoca finale nel
    # resume e completerà la pulizia senza ripetere l'allenamento.
    final_epoch = history.get("epoch_last", config.NUM_EPOCHS)
    # history non dipende dal numero di epoche eseguite; ricava l'ultima epoca dal
    # checkpoint resume, che contiene il contatore effettivo in caso di early stop.
    if os.path.exists(resume_path):
        try:
            last_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            last_ckpt = torch.load(resume_path, map_location="cpu")
        final_epoch = int(last_ckpt.get("epoch", final_epoch))
    _write_training_complete_marker(ckpt_path, include_timing, final_epoch, best_val_acc)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    return model, history, best_val_acc


def evaluate_on_test(model, df_test, include_timing: bool):
    """Accuracy PER MOSSA sul test set (veloce, a batch). Per l'accuracy a
    livello di puzzle intero vedi `evaluate_full_sequence`."""
    test_ds = ChessPuzzleGraphDataset(df_test, include_timing=include_timing)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    return run_epoch(model, test_loader, optimizer=None)


# ----------------------------------------------------------------------
# 2. Valutazione a livello di PUZZLE INTERO: accuracy SEQUENZA COMPLETA
# ----------------------------------------------------------------------

def solve_puzzle_with_gnn(model, row, include_timing: bool, device_=None):
    """Fa risolvere un intero puzzle al modello, mossa dopo mossa, esattamente
    come si valuta l'LLM in src/llm_baseline.py: ad ogni turno del 'solver'
    il modello predice la mossa migliore sulla posizione corrente; le mosse
    di risposta dell'avversario vengono applicate così come registrate da
    Lichess (il modello non le predice: non tocca a lui giocarle).

    Il puzzle è considerato risolto (`full_sequence_correct=True`) solo se
    OGNI mossa del solver lungo la linea è stata predetta correttamente.
    Ci si ferma alla prima mossa sbagliata (una volta sbagliata la
    posizione diverge dalla linea attesa, non avrebbe senso continuare).
    """
    device_ = device_ or device
    model.eval()
    t_start = time.time()

    steps = solver_moves_sequence(
        row["FEN"], row["Moves"],
        has_opening_move=row.get("has_opening_move", True),
    )

    predicted_moves = []
    confidences = []          # punteggio di confidenza per ogni mossa suggerita
    top3_guidance = []        # le 3 mosse migliori della PRIMA posizione, con confidenza
    first_move_correct = False
    all_correct = True

    for ply_idx, (board, target_move, _opponent_reply) in enumerate(steps):
        # Timing proxy v2 per POSIZIONE (stessa funzione usata nel training)
        think_time_sec = (
            compute_think_time_sec(board, row["rating_norm"], len(steps) - ply_idx)
            if include_timing else None
        )
        data, legal_moves = board_to_graph_data(board, think_time_sec=think_time_sec)
        data = data.to(device_)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device_)

        with torch.no_grad():
            scores, _edge_batch = model(data)

        # Punteggi di confidenza (obiettivo 3 del PDF: "produrre in output le
        # mosse suggerite E i punteggi di confidenza"): softmax sugli score
        # delle sole mosse legali di QUESTA posizione — e' esattamente la
        # distribuzione di probabilita' che il modello ottimizza in training.
        probs = torch.softmax(scores, dim=0)
        pred_local_idx = scores.argmax().item()
        pred_move = legal_moves[pred_local_idx] if pred_local_idx < len(legal_moves) else None
        predicted_moves.append(pred_move.uci() if pred_move is not None else "??")
        confidences.append(float(probs[pred_local_idx].item()))

        if ply_idx == 0:
            k = min(3, len(legal_moves))
            top_vals, top_idx = torch.topk(probs, k)
            top3_guidance = [
                (legal_moves[i].uci(), float(v))
                for v, i in zip(top_vals.tolist(), top_idx.tolist())
                if i < len(legal_moves)
            ]

        is_correct = (pred_move == target_move)
        if ply_idx == 0:
            first_move_correct = is_correct
        if not is_correct:
            all_correct = False
            break

    total_plies = len(steps)
    elapsed = time.time() - t_start
    return {
        "first_move_correct": bool(first_move_correct),
        "full_sequence_correct": bool(all_correct) and total_plies > 0,
        "num_plies": total_plies,
        "predicted_moves": " ".join(predicted_moves),
        "confidences": confidences,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "first_move_confidence": float(confidences[0]) if confidences else 0.0,
        "top3_first_move": top3_guidance,
        "solve_time_sec": elapsed,
    }


def evaluate_full_sequence(model, df: pd.DataFrame, include_timing: bool,
                           checkpoint_csv: str = None, save_every: int = 25):
    """Valuta sequenze complete. Con `checkpoint_csv`, salva risultati a blocchi e
    riusa quelli già calcolati dopo un'interruzione di Colab.

    Il file di ripresa è riferito al campione deterministico fornito dal chiamante;
    eventuali PuzzleId non presenti nel campione corrente vengono ignorati.
    """
    model.eval()
    df = df.reset_index(drop=True)
    rows_by_id = {}
    if checkpoint_csv and os.path.exists(checkpoint_csv):
        try:
            saved = pd.read_csv(checkpoint_csv)
            allowed_ids = set(df["PuzzleId"].astype(str))
            for _, saved_row in saved.iterrows():
                pid = str(saved_row.get("PuzzleId", ""))
                if pid not in allowed_ids:
                    continue
                record = saved_row.to_dict()
                for flag in ("first_move_correct", "full_sequence_correct"):
                    value = record.get(flag, False)
                    record[flag] = str(value).strip().lower() in ("true", "1", "yes")
                rows_by_id[pid] = record
        except Exception as exc:
            print(f"Nota: checkpoint valutazione non leggibile ({exc}); lo rigenero.")
            rows_by_id = {}

    pending_rows = df[~df["PuzzleId"].astype(str).isin(rows_by_id)].reset_index(drop=True)
    if checkpoint_csv:
        print(f"Valutazione sequenza {'timed' if include_timing else 'no-timing'}: "
              f"{len(rows_by_id):,}/{len(df):,} puzzle già completati; "
              f"ne restano {len(pending_rows):,}.")

    newly_saved = []
    iterator = tqdm(pending_rows.iterrows(), total=len(pending_rows),
                    desc=f"Sequenze {'timed' if include_timing else 'no-timing'}",
                    unit="puzzle", leave=False, disable=(len(pending_rows) == 0))
    for _, row in iterator:
        res = solve_puzzle_with_gnn(model, row, include_timing)
        res["PuzzleId"] = str(row["PuzzleId"])
        res["mate_n"] = int(row["mate_n"])
        rows_by_id[str(row["PuzzleId"])] = res
        newly_saved.append(res)
        if checkpoint_csv and len(newly_saved) >= max(1, int(save_every)):
            _save_sequence_eval_checkpoint(checkpoint_csv, rows_by_id)
            newly_saved.clear()
    if checkpoint_csv and newly_saved:
        _save_sequence_eval_checkpoint(checkpoint_csv, rows_by_id)

    # Mantieni l'ordine del campione originale anche dopo aver ripreso da CSV.
    ordered_rows = [rows_by_id[str(row["PuzzleId"])] for _, row in df.iterrows()]
    df_res = pd.DataFrame(ordered_rows)
    for flag in ("first_move_correct", "full_sequence_correct"):
        if flag in df_res:
            df_res[flag] = df_res[flag].map(
                lambda v: v if isinstance(v, (bool, np.bool_)) else
                str(v).strip().lower() in ("true", "1", "yes")
            )
    first_move_acc = df_res["first_move_correct"].mean()
    full_seq_acc = df_res["full_sequence_correct"].mean()
    per_n_full_seq = df_res.groupby("mate_n")["full_sequence_correct"].mean().to_dict()
    per_n_first = df_res.groupby("mate_n")["first_move_correct"].mean().to_dict()

    return {
        "first_move_acc": first_move_acc,
        "full_sequence_acc": full_seq_acc,
        "per_n_full_sequence": per_n_full_seq,
        "per_n_first_move": per_n_first,
        "df_results": df_res,
    }


def _save_sequence_eval_checkpoint(path: str, rows_by_id: dict):
    """Persistenza atomica della valutazione sequenziale (ultima copia completa)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = pd.DataFrame(list(rows_by_id.values()))
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_checkpoint_if_compatible(ckpt_path: str, tag: str):
    """Prova a caricare un checkpoint gia' esistente nell'architettura
    attuale. Ritorna il modello se i pesi sono compatibili, altrimenti None
    (e spiega perche'), cosi' il chiamante puo' decidere di addestrare.

    Serve a riusare i pesi di una sessione Colab precedente senza rifare le
    ore di training: i checkpoint dipendono SOLO dall'architettura
    (HIDDEN_DIM, NUM_LAYERS, NODE_FEAT_DIM/EDGE_FEAT_DIM), non da quanti
    puzzle di valutazione userai dopo. Se pero' hai toccato l'encoding del
    grafo o le dimensioni della rete, i pesi vecchi NON sono piu' validi e
    il caricamento fallisce qui invece di produrre silenziosamente numeri
    senza senso.
    """
    if not os.path.exists(ckpt_path):
        return None

    # Se esiste ANCHE il checkpoint di ripresa (.resume.pt), il training
    # NON e' finito: si e' fermato a meta' (es. sessione Colab disconnessa)
    # dopo che almeno un'epoca aveva gia' migliorato best_val_acc, quindi
    # ckpt_path esiste ma non rappresenta il training completo. train_one_variant
    # cancella il file di ripresa SOLO a fine training (tutte le epoche o
    # early stopping), quindi la sua presenza e' il segnale giusto per
    # NON saltare il training, ma riprenderlo da dove si era fermato.
    resume_path = ckpt_path + ".resume.pt"
    if os.path.exists(resume_path):
        print(f"[{tag}] Trovato sia {os.path.basename(ckpt_path)} sia "
              f"{os.path.basename(resume_path)}: il training si era fermato a meta' "
              "(sessione interrotta prima della fine). Riprendo le epoche mancanti "
              "invece di saltare il training.")
        return None

    model = TimedGNN(NODE_FEAT_DIM, hidden_dim=config.HIDDEN_DIM,
                      num_layers=config.NUM_LAYERS, dropout=config.DROPOUT).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            # checkpoint "resumable" salvato a meta' training
            state = state["model_state"]
        model.load_state_dict(state)
    except Exception as e:
        print(f"[{tag}] Checkpoint {ckpt_path} presente ma NON compatibile con "
              f"l'architettura attuale ({e}).\n"
              f"[{tag}] Quasi sempre significa che hai cambiato HIDDEN_DIM/NUM_LAYERS "
              "o l'encoding del grafo in src/chess_encoding.py: i pesi vecchi vanno "
              "buttati e il training rifatto. Procedo ad addestrare da zero.")
        return None

    print(f"[{tag}] Checkpoint compatibile trovato: {ckpt_path} -> TRAINING SALTATO. "
          "(Per forzare comunque il training: export SKIP_TRAINING=0)")
    return model


def run_phase3(skip_training: bool = None):
    """Fase 3 completa: training (o riuso dei checkpoint) + valutazione.

    `skip_training=None` -> usa config.SKIP_TRAINING_IF_CKPT_EXISTS.
    In ogni caso, se i checkpoint non ci sono o non sono compatibili, il
    training viene eseguito: non si va mai avanti senza un modello.
    """
    if skip_training is None:
        skip_training = config.SKIP_TRAINING_IF_CKPT_EXISTS

    required = [config.TRAIN_PARQUET, config.VAL_PARQUET, config.TEST_PARQUET]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"File mancanti: {missing}. Esegui prima `python -m scripts.run_pipeline`."
        )

    # Prima di tutto: i dati su disco non devono contenere puzzle dell'held-out
    # (succederebbe con file di una versione precedente della pipeline).
    check_no_leakage_from_files()

    df_train = pd.read_parquet(config.TRAIN_PARQUET)
    df_val = pd.read_parquet(config.VAL_PARQUET)
    df_test = pd.read_parquet(config.TEST_PARQUET)
    print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}  (puzzle)")
    print_n_distribution({"train": df_train, "val": df_val, "test": df_test},
                         "Matti in N nei dati di training/validazione/test (numero di PUZZLE)")
    print("Device:", device)

    df_test_run = maybe_subset(df_test, config.SUBSET_SIZE // 5 if config.SUBSET_SIZE else None)

    model_timed = load_checkpoint_if_compatible(config.TIMED_CKPT, "timed") if skip_training else None
    model_no_timing = (load_checkpoint_if_compatible(config.NO_TIMING_CKPT, "no-timing")
                       if skip_training else None)

    history_timed = history_no_timing = None
    if model_timed is None or model_no_timing is None:
        df_train_run = maybe_subset(df_train, config.SUBSET_SIZE)
        df_val_run = maybe_subset(df_val, config.SUBSET_SIZE // 5 if config.SUBSET_SIZE else None)
        print(f"Training su {len(df_train_run):,} puzzle, validazione su {len(df_val_run):,}")
        print_n_distribution(
            {"train": df_train_run, "val": df_val_run},
            "Matti in N effettivamente usati per training e validazione (numero di PUZZLE)",
        )
        print("(NOTA: 'puzzle' != 'esempi di training': ogni puzzle mate-in-n viene "
              "espanso in n esempi (posizione, mossa), uno per ogni mossa della linea "
              "— vedi src/dataset.py.)")

        if model_timed is None:
            model_timed, history_timed, _ = train_one_variant(
                df_train_run, df_val_run, include_timing=True,
                ckpt_path=config.TIMED_CKPT, tag="timed",
            )
        if model_no_timing is None:
            model_no_timing, history_no_timing, _ = train_one_variant(
                df_train_run, df_val_run, include_timing=False,
                ckpt_path=config.NO_TIMING_CKPT, tag="no-timing",
            )

        # Ricarica sempre i pesi MIGLIORI (non quelli dell'ultima epoca)
        model_timed.load_state_dict(torch.load(config.TIMED_CKPT, map_location=device))
        model_no_timing.load_state_dict(torch.load(config.NO_TIMING_CKPT, map_location=device))
    else:
        print("Entrambi i checkpoint sono stati riusati: si passa direttamente "
              "alla valutazione (obiettivi 4 e 5 del PDF).")

    results = evaluate_ablation(model_timed, model_no_timing, df_test_run)
    results.update({
        "model_timed": model_timed,
        "model_no_timing": model_no_timing,
        "history_timed": history_timed,
        "history_no_timing": history_no_timing,
    })
    return results


def evaluate_ablation(model_timed, model_no_timing, df_test_run):
    """Obiettivo 5 del PDF — studio di ablazione sull'informazione di timing,
    ora CON test di significativita' appaiato (prima c'era solo il delta
    nudo, che su un campione finito non dice se la differenza sia reale).
    """
    # --- 1. Accuracy PER MOSSA sul test set interno (veloce, a batch) ---
    test_loss_t, test_acc_t, per_n_t = evaluate_on_test(model_timed, df_test_run, include_timing=True)
    test_loss_nt, test_acc_nt, per_n_nt = evaluate_on_test(model_no_timing, df_test_run, include_timing=False)

    print(f"\nTest accuracy PER MOSSA (con timing):    {test_acc_t:.3f}")
    print(f"Test accuracy PER MOSSA (senza timing):  {test_acc_nt:.3f}")
    print(f"Delta (ablation, obiettivo 5):            {test_acc_t - test_acc_nt:+.3f}")

    # --- 2. Accuracy a PUZZLE INTERO (sequenza completa) su un campione ---
    df_test_seq = maybe_subset(df_test_run, config.FULL_SEQ_EVAL_SIZE)
    print(f"\nValutazione sequenza completa su {len(df_test_seq):,} puzzle "
          "(più lenta: un forward pass per ogni mossa della linea, niente batching)...")

    seq_t = evaluate_full_sequence(
        model_timed, df_test_seq, include_timing=True,
        checkpoint_csv=config.STAGED_SEQ_TIMED_CSV,
    )
    seq_nt = evaluate_full_sequence(
        model_no_timing, df_test_seq, include_timing=False,
        checkpoint_csv=config.STAGED_SEQ_NO_TIMING_CSV,
    )

    print(f"\nTest accuracy SEQUENZA COMPLETA (con timing):   {seq_t['full_sequence_acc']:.3f}")
    print(f"Test accuracy SEQUENZA COMPLETA (senza timing): {seq_nt['full_sequence_acc']:.3f}")
    print("\nAccuracy sequenza completa per n (con timing):", seq_t["per_n_full_sequence"])
    print("Accuracy sequenza completa per n (senza timing):", seq_nt["per_n_full_sequence"])

    # --- 3. Test statistico appaiato sull'ablation ---
    df_t = seq_t["df_results"].set_index("PuzzleId")
    df_nt = seq_nt["df_results"].set_index("PuzzleId")
    common = df_t.index.intersection(df_nt.index)
    print(f"\n=== Significativita' dell'ablation (McNemar appaiato su {len(common):,} "
          "puzzle, sequenza completa) ===")
    abl_stat = mcnemar_from_flags(
        df_t.loc[common, "full_sequence_correct"],
        df_nt.loc[common, "full_sequence_correct"],
        label_1="timed  ", label_2="no-timing",
    )

    # --- 4. Salvataggio tabella riassuntiva per il report ---
    all_n = sorted(set(seq_t["per_n_full_sequence"]) | set(seq_nt["per_n_full_sequence"]))
    df_abl = pd.DataFrame({
        "mate_n": all_n,
        "acc_seq_timed": [seq_t["per_n_full_sequence"].get(n, float("nan")) for n in all_n],
        "acc_seq_no_timing": [seq_nt["per_n_full_sequence"].get(n, float("nan")) for n in all_n],
        "acc_per_mossa_timed": [per_n_t.get(n, float("nan")) for n in all_n],
        "acc_per_mossa_no_timing": [per_n_nt.get(n, float("nan")) for n in all_n],
        "n_puzzle": [int((df_t["mate_n"] == n).sum()) for n in all_n],
    })
    df_abl["delta_seq"] = df_abl["acc_seq_timed"] - df_abl["acc_seq_no_timing"]
    df_abl.to_csv(config.ABLATION_CSV, index=False)
    print(f"\nTabella ablation salvata in {config.ABLATION_CSV}")
    print(df_abl.round(3).to_string(index=False))

    _plot_ablation(df_abl)

    from src.insights import quantify_timing_benefit
    timing_insight = quantify_timing_benefit(df_abl)

    return {
        "test_acc_per_move_timed": test_acc_t,
        "test_acc_per_move_no_timing": test_acc_nt,
        "per_n_per_move_timed": per_n_t,
        "per_n_per_move_no_timing": per_n_nt,
        "test_acc_full_seq_timed": seq_t["full_sequence_acc"],
        "test_acc_full_seq_no_timing": seq_nt["full_sequence_acc"],
        "per_n_full_seq_timed": seq_t["per_n_full_sequence"],
        "per_n_full_seq_no_timing": seq_nt["per_n_full_sequence"],
        "ablation_mcnemar": abl_stat,
        "timing_insight": timing_insight,
        "ablation_table": df_abl,
    }


def _plot_ablation(df_abl):
    """Grafico a barre dell'ablation per valore di n (deliverable: 'visualizzazioni')."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(grafico ablation saltato: {e})")
        return

    x = np.arange(len(df_abl))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, df_abl["acc_seq_timed"], width, label="con timing")
    ax.bar(x + width / 2, df_abl["acc_seq_no_timing"], width, label="senza timing (ablation)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"mate in {int(n)}\n(N={int(c)})"
                        for n, c in zip(df_abl["mate_n"], df_abl["n_puzzle"])])
    ax.set_ylabel("accuracy sequenza completa")
    ax.set_title("Ablation: impatto dell'informazione di timing, per profondita' di matto")
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.ABLATION_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Grafico ablation salvato in {config.ABLATION_PLOT_PATH}")


if __name__ == "__main__":
    run_phase3()
