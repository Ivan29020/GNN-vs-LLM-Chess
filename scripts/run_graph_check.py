"""
Verifica rapida della Fase 2 (costruzione del grafo) su un piccolo campione,
prima di lanciare il training vero e proprio.

Uso:
    python -m scripts.run_graph_check
"""
import os
import pandas as pd
from torch_geometric.loader import DataLoader

import config
from src.dataset import ChessPuzzleGraphDataset


def main():
    if not os.path.exists(config.TRAIN_PARQUET):
        raise FileNotFoundError(
            f"{config.TRAIN_PARQUET} non trovato. Esegui prima `python -m scripts.run_pipeline`."
        )

    df_train = pd.read_parquet(config.TRAIN_PARQUET)
    sample_df = df_train.sample(min(200, len(df_train)), random_state=0).reset_index(drop=True)
    sample_ds = ChessPuzzleGraphDataset(sample_df, include_timing=True)

    print(f"{len(sample_df)} puzzle campionati -> {len(sample_ds)} esempi (puzzle, mossa) "
          "dopo l'espansione su tutta la linea di matto "
          f"(fattore di espansione: {len(sample_ds) / len(sample_df):.2f}x)")

    invalid_count = 0
    for i in range(len(sample_ds)):
        d = sample_ds.get(i)
        if d.y.item() == -1:
            invalid_count += 1
    print(f"Grafi con target non trovato: {invalid_count} / {len(sample_ds)}")

    d0 = sample_ds.get(0)
    print("\nEsempio di grafo:")
    print(" - x (feature nodi):", d0.x.shape)
    print(" - edge_index:", d0.edge_index.shape)
    print(" - edge_attr:", d0.edge_attr.shape)
    print(" - y (target):", d0.y, "| num mosse candidate:", d0.num_candidate_moves.item())
    print(" - mate_n:", d0.mate_n.item())

    # Prova rapida del DataLoader (batching di grafi a dimensione variabile)
    loader = DataLoader(sample_ds, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print("\nBatch di prova:")
    print(" - num grafi:", batch.num_graphs)
    print(" - x shape (nodi totali nel batch):", batch.x.shape)
    print(" - edge_index shape:", batch.edge_index.shape)


if __name__ == "__main__":
    main()
