"""
Prototipo dell'estensione "integrazione con Stockfish per una risoluzione
ibrida" (PDF, sezione Risultati Attesi). Vedi docs/ESTENSIONI.md per la
discussione completa.

Confronta, sul campione held-out canonico gia' usato per il confronto
GNN-vs-LLM, tre strategie:
  1. GNN pura (gia' presente nel resto della pipeline)
  2. Stockfish puro (limitato a una profondita' fissa, zero-shot come l'LLM)
  3. Ibrida: usa la GNN quando la confidenza della prima mossa supera una
     soglia, altrimenti chiama Stockfish

PREREQUISITO: il binario di Stockfish deve essere installato e nel PATH.
Su Colab:
    !apt-get install -y stockfish -q
Lo script cerca il binario con shutil.which("stockfish"); se non lo trova,
si ferma subito con un messaggio chiaro invece di fallire a meta' esecuzione.

Uso:
    python -m scripts.stockfish_hybrid_demo [--confidence-threshold 0.5]
"""

import argparse
import os
import shutil
import time

import chess
import chess.engine
import pandas as pd
import torch

import config
from src.chess_encoding import NODE_FEAT_DIM
from src.model import TimedGNN
from src.train import solve_puzzle_with_gnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gnn():
    model = TimedGNN(NODE_FEAT_DIM, hidden_dim=config.HIDDEN_DIM,
                      num_layers=config.NUM_LAYERS, dropout=config.DROPOUT).to(device)
    state = torch.load(config.TIMED_CKPT, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def solve_with_stockfish(engine, row, depth: int = 15) -> dict:
    """Fa risolvere l'intera linea a Stockfish, una mossa alla volta,
    esattamente come fa la GNN (confronto equo, nessuna scorciatoia)."""
    from src.chess_encoding import solver_moves_sequence

    t_start = time.time()
    steps = solver_moves_sequence(row["FEN"], row["Moves"],
                                   has_opening_move=row.get("has_opening_move", True))
    predicted = []
    first_move_correct = False
    all_correct = True

    for ply_idx, (board, target_move, _opponent_reply) in enumerate(steps):
        result = engine.play(board, chess.engine.Limit(depth=depth))
        pred_move = result.move
        predicted.append(pred_move.uci() if pred_move else "??")
        correct = (pred_move == target_move)
        if ply_idx == 0:
            first_move_correct = correct
        if not correct:
            all_correct = False

    return {
        "first_move_correct": first_move_correct,
        "full_sequence_correct": all_correct and len(steps) > 0,
        "predicted_moves": " ".join(predicted),
        "solve_time_sec": time.time() - t_start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Sotto questa confidenza della prima mossa GNN, "
                            "la strategia ibrida chiama Stockfish invece.")
    parser.add_argument("--stockfish-depth", type=int, default=15)
    args = parser.parse_args()

    stockfish_path = shutil.which("stockfish")
    if stockfish_path is None:
        raise RuntimeError(
            "Binario 'stockfish' non trovato nel PATH. Su Colab: "
            "!apt-get install -y stockfish -q    poi rilancia questo script."
        )

    if not os.path.exists(config.LLM_EVAL_SAMPLE_CSV):
        raise FileNotFoundError(
            f"{config.LLM_EVAL_SAMPLE_CSV} non trovato: esegui prima "
            "run_llm_comparison almeno una volta, cosi' il campione canonico "
            "e' fissato e questo script lo riusa invece di campionarne uno diverso."
        )
    df_eval = pd.read_csv(config.LLM_EVAL_SAMPLE_CSV)
    print(f"Campione: {len(df_eval)} puzzle da {config.LLM_EVAL_SAMPLE_CSV}")

    model = load_gnn()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    rows = []
    try:
        for i, (_, row) in enumerate(df_eval.iterrows()):
            gnn_res = solve_puzzle_with_gnn(model, row, include_timing=True, device_=device)
            sf_res = solve_with_stockfish(engine, row, depth=args.stockfish_depth)

            use_gnn = gnn_res.get("first_move_confidence", 0.0) >= args.confidence_threshold
            hybrid_correct = (gnn_res["full_sequence_correct"] if use_gnn
                              else sf_res["full_sequence_correct"])

            rows.append({
                "PuzzleId": row["PuzzleId"], "mate_n": row["mate_n"],
                "gnn_full_sequence_correct": gnn_res["full_sequence_correct"],
                "gnn_confidence": gnn_res.get("first_move_confidence", float("nan")),
                "stockfish_full_sequence_correct": sf_res["full_sequence_correct"],
                "hybrid_used_gnn": use_gnn,
                "hybrid_full_sequence_correct": hybrid_correct,
            })
            print(f"[{i+1}/{len(df_eval)}] {row['PuzzleId']} (n={row['mate_n']}): "
                  f"GNN={'OK' if gnn_res['full_sequence_correct'] else 'KO'} "
                  f"Stockfish={'OK' if sf_res['full_sequence_correct'] else 'KO'} "
                  f"ibrido usa {'GNN' if use_gnn else 'Stockfish'}")
    finally:
        engine.quit()

    df = pd.DataFrame(rows)
    summary = df.groupby("mate_n").agg(
        gnn_acc=("gnn_full_sequence_correct", "mean"),
        stockfish_acc=("stockfish_full_sequence_correct", "mean"),
        hybrid_acc=("hybrid_full_sequence_correct", "mean"),
        pct_hybrid_usa_gnn=("hybrid_used_gnn", "mean"),
        n_puzzle=("PuzzleId", "count"),
    ).reset_index()

    print("\n=== Accuratezza per n: GNN vs Stockfish vs Ibrido ===")
    print(summary.round(3).to_string(index=False))

    out_csv = os.path.join(config.RESULTS_DIR, "stockfish_hybrid_results.csv")
    df.to_csv(out_csv, index=False)
    summary.to_csv(os.path.join(config.RESULTS_DIR, "stockfish_hybrid_summary.csv"), index=False)
    print(f"\nSalvato in {out_csv} e stockfish_hybrid_summary.csv")
    print("\nNOTA: Stockfish e' atteso vicino al 100% su questi matti forzati — il punto "
          "del confronto non e' 'chi vince', ma quanto l'ibrido si avvicina a Stockfish "
          "usando la GNN (istantanea) quando e' sicura di se', chiamando Stockfish solo "
          "nei casi dubbi.")


if __name__ == "__main__":
    main()
