"""
src/compare.py
-----------------
Fase 4, parte B — Confronto GNN vs LLM sull'held-out (obiettivi 3 e 4 del PDF).

Cosa fa:
1. Esegue il modello GNN "timed" sugli stessi identici puzzle valutati con
   l'LLM (confronto APPAIATO, requisito di McNemar), facendogli risolvere
   l'intera linea mossa per mossa e raccogliendo anche i PUNTEGGI DI
   CONFIDENZA e il tempo di soluzione (PDF, sezione 3: "produrre in output
   le mosse suggerite e i punteggi di confidenza"; metrica "efficienza =
   tempo necessario per la soluzione").
2. McNemar appaiato, con p-value ESATTO oltre al chi-quadro (src/stats.py),
   su UNA sola metrica: accuracy di sequenza completa (la GNN predice solo
   le proprie mosse, all'LLM si accetta la linea alternata o le sole mosse
   del solver: vedi src/llm_baseline.evaluate_llm_solution).
3. Grafico a barre dell'accuratezza per valore di n (PDF, sezione 4).
4. Confronto di efficienza (tempo medio per puzzle).
5. File CSV per la valutazione umana in cieco (rubrica 1-5 correttezza /
   concisione), con etichette "Modello A/B" randomizzate per riga e la
   guida della GNN resa in forma TESTUALE con le confidenze, non come
   stringa UCI nuda: altrimenti il giudizio sulla "chiarezza" sarebbe
   deciso dal formato e non dal contenuto.

Uso da riga di comando:
    python -m scripts.run_llm_comparison
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from src.model import TimedGNN
from src.chess_encoding import NODE_FEAT_DIM
from src.train import solve_puzzle_with_gnn
from src.stats import mcnemar_from_flags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_gnn_guidance(res: dict, mate_n: int) -> str:
    """Rende la predizione della GNN una GUIDA leggibile, con confidenze.

    Serve per due cose: (a) e' cio' che il PDF chiede come output della GNN
    ("mosse suggerite e punteggi di confidenza"); (b) rende sensato il blind
    test umano, che valuta correttezza E concisione: confrontare una stringa
    UCI nuda con un paragrafo in inglese misurerebbe il formato, non la
    qualita' della guida.
    """
    moves = res["predicted_moves"].split()
    confs = res.get("confidences", [])
    righe = [f"Suggerimento per il matto in {mate_n}:"]
    for i, mv in enumerate(moves):
        c = f" (confidenza {confs[i]:.0%})" if i < len(confs) else ""
        righe.append(f"  mossa {i + 1} del solver: {mv}{c}")
    top3 = res.get("top3_first_move") or []
    if top3:
        alt = ", ".join(f"{m} ({p:.0%})" for m, p in top3)
        righe.append(f"Alternative considerate per la prima mossa: {alt}")
    righe.append(f"Tempo di calcolo: {res.get('solve_time_sec', 0):.2f}s")
    return "\n".join(righe)


def add_gnn_predictions(df_llm: pd.DataFrame, df_eval: pd.DataFrame,
                         ckpt_path: str = config.TIMED_CKPT) -> pd.DataFrame:
    """Esegue il modello GNN sugli stessi puzzle valutati dall'LLM (confronto
    appaiato, necessario per McNemar), risolvendo l'INTERA sequenza mossa
    per mossa — non solo la prima mossa.

    IMPORTANTE: il confronto usa SOLO i puzzle presenti in `df_eval`, il
    campione canonico persistito (results/llm_eval_sample.csv). `df_llm`
    (il CSV grezzo accumulato) puo' contenere puzzle in piu' rispetto al
    campione canonico — residui di run passate fatte con un campione
    leggermente diverso, prima che il campione diventasse persistente. Una
    versione precedente di questa funzione cercava i PuzzleId mancanti
    nell'intero pool held-out come fallback: quasi ogni PuzzleId storico
    esiste da qualche parte nel pool intero, quindi quel fallback li faceva
    rientrare comunque, gonfiando silenziosamente il campione oltre i 150
    previsti e alterando la stratificazione per n. Qui vengono invece
    scartati esplicitamente, con un conteggio in output.
    """
    model = TimedGNN(NODE_FEAT_DIM, hidden_dim=config.HIDDEN_DIM,
                      num_layers=config.NUM_LAYERS, dropout=config.DROPOUT).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()

    eval_ids = set(df_eval["PuzzleId"])
    n_before = len(df_llm)
    extra_ids = sorted(set(df_llm["PuzzleId"]) - eval_ids)
    df_llm = df_llm[df_llm["PuzzleId"].isin(eval_ids)].reset_index(drop=True)
    if extra_ids:
        print(f"\nNOTA: {len(extra_ids)} puzzle presenti nel CSV grezzo accumulato non "
              f"fanno parte del campione canonico attuale ({len(df_eval)} puzzle, "
              f"da {config.LLM_EVAL_SAMPLE_CSV}) e sono stati esclusi dal confronto "
              f"(erano {n_before} in totale, ne restano {len(df_llm)}).")
        print("Causa tipica: residui di esecuzioni precedenti a quando il campione e' "
              "diventato persistente. Non richiede alcuna azione: il campione canonico "
              "resta questo da ora in poi.")

    kept_rows = []
    first_move_flags, full_seq_flags, predicted_moves_list = [], [], []
    guidance_list, conf_list, time_list, source_list = [], [], [], []

    for _, row in df_llm.iterrows():
        match = df_eval[df_eval["PuzzleId"] == row["PuzzleId"]]
        if len(match) == 0:
            continue  # non dovrebbe accadere dopo il filtro sopra, ma per sicurezza
        match_row = match.iloc[0]
        kept_rows.append(row)
        res = solve_puzzle_with_gnn(model, match_row, include_timing=True, device_=device)
        first_move_flags.append(res["first_move_correct"])
        full_seq_flags.append(res["full_sequence_correct"])
        predicted_moves_list.append(res["predicted_moves"])
        guidance_list.append(format_gnn_guidance(res, int(row["mate_n"])))
        source_list.append(match_row["source"] if "source" in match_row.index else "")
        conf_list.append(res.get("first_move_confidence", float("nan")))
        time_list.append(res.get("solve_time_sec", float("nan")))

    df_llm = pd.DataFrame(kept_rows).reset_index(drop=True)
    df_llm["gnn_first_move_correct"] = first_move_flags
    df_llm["gnn_full_sequence_correct"] = full_seq_flags
    df_llm["gnn_predicted_moves"] = predicted_moves_list
    df_llm["gnn_guidance"] = guidance_list
    df_llm["gnn_first_move_confidence"] = conf_list
    df_llm["gnn_solve_time_sec"] = time_list
    df_llm["source"] = source_list

    print("=== Accuratezza sullo stesso campione appaiato "
          f"({len(df_llm)} puzzle held-out) ===")
    print(f"GNN — prima mossa:                     {df_llm['gnn_first_move_correct'].mean():.3f}")
    print(f"GNN — sequenza completa:               {df_llm['gnn_full_sequence_correct'].mean():.3f}")
    print(f"LLM — prima mossa:                     {df_llm['first_move_correct'].mean():.3f}")
    print(f"LLM — sequenza completa:               {df_llm['full_sequence_correct'].mean():.3f}")
    print("\nLa metrica di confronto primaria e' la SEQUENZA COMPLETA, non la prima "
          "mossa:\nindovinare solo la prima mossa di un matto in 4 non risolve il puzzle.")
    return df_llm


def efficiency_comparison(df_llm: pd.DataFrame) -> pd.DataFrame:
    """Metrica 'efficienza' del PDF: tempo necessario per la soluzione."""
    rows = [
        {"modello": "Timed GNN", "tempo_medio_sec": df_llm["gnn_solve_time_sec"].mean(),
         "tempo_mediano_sec": df_llm["gnn_solve_time_sec"].median()},
        {"modello": f"LLM ({config.LLM_MODEL_NAME})",
         "tempo_medio_sec": df_llm["response_time_sec"].mean(),
         "tempo_mediano_sec": df_llm["response_time_sec"].median()},
    ]
    df_eff = pd.DataFrame(rows)
    print("\n=== Efficienza (tempo per puzzle) ===")
    print(df_eff.round(3).to_string(index=False))
    ratio = df_llm["response_time_sec"].mean() / max(df_llm["gnn_solve_time_sec"].mean(), 1e-9)
    print(f"L'LLM impiega circa {ratio:.0f}x il tempo della GNN per puzzle "
          "(la GNN gira in locale sulla GPU, l'LLM paga anche la latenza di rete: "
          "e' un confronto indicativo, non una misura di costo computazionale puro).")
    return df_eff


def mcnemar_test(df_llm: pd.DataFrame,
                  gnn_col: str = "gnn_full_sequence_correct",
                  llm_col: str = "full_sequence_correct") -> dict:
    """McNemar appaiato sull'accuratezza di sequenza completa (il confronto
    corretto tra GNN e LLM su questo task). Riporta sia il chi-quadro con
    correzione di continuita' sia il p-value binomiale ESATTO — con poche
    decine di puzzle il primo non e' affidabile, vedi src/stats.py."""
    print("\n=== McNemar: GNN vs LLM (sequenza completa) ===")
    return mcnemar_from_flags(df_llm[gnn_col], df_llm[llm_col],
                              label_1="GNN", label_2="LLM")


def plot_accuracy_by_n(df_llm: pd.DataFrame, model_name: str = config.LLM_MODEL_NAME,
                        save_path: str = None) -> pd.DataFrame:
    """Grafico a barre dell'accuratezza di SEQUENZA COMPLETA per valore di n
    (PDF, sezione 4) + tabella salvata su CSV per il report."""
    save_path = save_path or config.COMPARISON_PLOT_PATH
    acc_by_n = df_llm.groupby("mate_n").agg(
        gnn_acc=("gnn_full_sequence_correct", "mean"),
        llm_acc=("full_sequence_correct", "mean"),
        gnn_first_move=("gnn_first_move_correct", "mean"),
        llm_first_move=("first_move_correct", "mean"),
        n_puzzles=("PuzzleId", "count"),
        fonte=("source", "first"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(acc_by_n))
    ax.bar(x - width / 2, acc_by_n["gnn_acc"], width, label="Timed GNN")
    ax.bar(x + width / 2, acc_by_n["llm_acc"], width, label=f"LLM ({model_name})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"mate in {n}\n(N={c}, {f})" for n, c, f in
                        zip(acc_by_n["mate_n"], acc_by_n["n_puzzles"], acc_by_n["fonte"])])
    ax.set_ylabel("accuracy sequenza completa")
    ax.set_title("GNN vs LLM: accuracy (puzzle risolto per intero) per profondita' di matto")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    acc_by_n.to_csv(config.ACC_BY_N_CSV, index=False)
    print(f"\nGrafico salvato in {save_path}; tabella in {config.ACC_BY_N_CSV}")
    print(acc_by_n.round(3).to_string(index=False))

    sources = sorted(df_llm["source"].dropna().astype(str).unique()) if "source" in df_llm else []
    if len(sources) > 1:
        print("NOTA: il held-out contiene più fonti; nel confronto misto la profondità può "
              "essere confusa con la fonte (stile e difficoltà dei puzzle). Va dichiarato nel report.")
    elif sources == ["lichess"]:
        print("NOTA: tutti i livelli provengono da Lichess; questo confronto riduce la "
              "confusione fra profondità e fonte presente nell'held-out misto.")

    small = acc_by_n[acc_by_n["n_puzzles"] < 10]["mate_n"].tolist()
    if small:
        print(f"ATTENZIONE: per n = {small} il campione ha meno di 10 puzzle. "
              "Le accuracy per quei valori di n sono indicative, non conclusive: "
              "vanno riportate con la numerosita' accanto e non vanno usate per "
              "affermare dove la GNN 'smette' di prevalere.")
    return acc_by_n


def build_blind_eval_file(df_llm: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """CSV per la valutazione umana in cieco (PDF, sezione 3): per ogni
    puzzle le due guide sono etichettate "Modello A/B" in ordine
    randomizzato per riga. La guida GNN e' quella testuale con confidenze
    (vedi format_gnn_guidance)."""
    rng = np.random.default_rng(seed)
    blind_rows = []

    for _, row in df_llm.iterrows():
        model_a_is_gnn = not (rng.random() < 0.5)
        llm_text = row["llm_explanation"]
        gnn_text = row.get("gnn_guidance") or f"Sequenza proposta (UCI): {row['gnn_predicted_moves']}"

        blind_rows.append({
            "PuzzleId": row["PuzzleId"],
            "mate_n": row["mate_n"],
            "FEN": row.get("fen_after_opening", ""),
            "Modello_A": (gnn_text if model_a_is_gnn else llm_text),
            "Modello_B": (llm_text if model_a_is_gnn else gnn_text),
            "A_correttezza_1_5": "",
            "A_concisione_1_5": "",
            "B_correttezza_1_5": "",
            "B_concisione_1_5": "",
            "quale_guida_e_migliore_A_B_pari": "",
            "_true_label_A": "GNN" if model_a_is_gnn else "LLM",
        })

    df_blind = pd.DataFrame(blind_rows)
    df_blind.drop(columns=["_true_label_A"]).to_csv(config.BLIND_EVAL_CSV, index=False)
    df_blind.to_csv(config.BLIND_EVAL_ANSWER_KEY_CSV, index=False)

    print(f"\nFile per il valutatore (senza risposte): {config.BLIND_EVAL_CSV}")
    print("Compila '*_correttezza_1_5', '*_concisione_1_5' (1=pessimo, 5=ottimo) e la "
          "preferenza complessiva, poi ricongiungi con '_true_label_A' dal file "
          "'WITH_ANSWER_KEY' per l'analisi finale.")
    print("NOTA METODOLOGICA: chi valuta non deve sapere quale colonna e' la GNN, e "
          "non deve essere la stessa persona che ha scritto il codice, altrimenti il "
          "test non e' davvero in cieco.")
    return df_blind


def run_phase4_comparison(df_llm: pd.DataFrame, df_eval: pd.DataFrame):
    df_llm = add_gnn_predictions(df_llm, df_eval)
    df_eff = efficiency_comparison(df_llm)
    mcnemar_result = mcnemar_test(df_llm)

    acc_by_n = plot_accuracy_by_n(df_llm)

    from src.insights import plot_confusion_matrices_by_n, quantify_gnn_advantage
    df_confusion = plot_confusion_matrices_by_n(df_llm)
    gnn_insight = quantify_gnn_advantage(acc_by_n)

    df_blind = build_blind_eval_file(df_llm)

    df_llm.to_csv(config.PAIRED_RESULTS_CSV, index=False)
    print(f"\nRisultati appaiati completi salvati in {config.PAIRED_RESULTS_CSV}")

    return {
        "df_llm": df_llm,
        "efficiency": df_eff,
        "mcnemar": mcnemar_result,
        "acc_by_n": acc_by_n,
        "df_confusion": df_confusion,
        "gnn_insight": gnn_insight,
        "df_blind": df_blind,
    }
