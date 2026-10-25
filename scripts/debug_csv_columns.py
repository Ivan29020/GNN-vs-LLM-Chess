"""
scripts/debug_csv_columns.py
------------------------------
Diagnostica indipendente per capire perché `filter_mate_puzzles` trova 0
puzzle "mate in n" sul CSV scaricato — situazione anomala che indica quasi
sempre un problema nella lettura/allineamento delle colonne, non
un'assenza reale di puzzle di matto (il dataset Lichess ne contiene
sempre centinaia di migliaia).

Esegui questo DOPO `python -m scripts.run_pipeline` (anche se quello ha
fallito con errore, il CSV decompresso in data/lichess_db_puzzle.csv
resta comunque su disco ed è quello che leggiamo qui).

Uso:
    python -m scripts.debug_csv_columns
"""

import os
import pandas as pd

import config
from src.data_pipeline import COLUMNS, extract_mate_n


def main():
    if not os.path.exists(config.CSV_PATH):
        raise FileNotFoundError(
            f"{config.CSV_PATH} non trovato. Esegui prima almeno il download "
            "(anche se poi la Fase 1 fallisce più avanti, va bene)."
        )

    print("### 1. Prime righe GREZZE del file (senza pandas, per vedere il reale contenuto) ###")
    with open(config.CSV_PATH, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            print(f"riga {i}: {line[:200]!r}")
            if i >= 3:
                break

    print("\n### 2. Lettura con pandas usando names=COLUMNS, header=0 (come nel progetto) ###")
    df_a = pd.read_csv(config.CSV_PATH, names=COLUMNS, header=0, nrows=1000)
    print("Colonne:", df_a.columns.tolist())
    print(df_a[["PuzzleId", "Themes"]].head(5))
    n_match_a = df_a["Themes"].apply(extract_mate_n).notna().sum()
    print(f"Match 'mateIn' su 1000 righe (metodo attuale, header=0+names): {n_match_a}")

    print("\n### 3. Lettura SENZA specificare names, lasciando l'header reale del file ###")
    df_b = pd.read_csv(config.CSV_PATH, nrows=1000)
    print("Colonne reali nel file:", df_b.columns.tolist())
    if "Themes" in df_b.columns:
        print(df_b[["PuzzleId", "Themes"]].head(5) if "PuzzleId" in df_b.columns else df_b["Themes"].head(5))
        n_match_b = df_b["Themes"].apply(extract_mate_n).notna().sum()
        print(f"Match 'mateIn' su 1000 righe (senza forzare i nomi colonna): {n_match_b}")
    else:
        print("ATTENZIONE: nemmeno con l'header reale del file compare una colonna 'Themes'.")
        print("Questo significa che il file scaricato NON è nel formato atteso: probabilmente")
        print("il download ha restituito una pagina di errore/redirect invece del vero CSV.")
        print("Controlla la dimensione del file:")
        print(f"  {config.CSV_PATH}: {os.path.getsize(config.CSV_PATH):,} byte")
        print("(il vero file decompresso pesa oltre 1 GB; se è molto più piccolo, il download")
        print(" o la decompressione sono falliti silenziosamente: cancella data/*.csv* e")
        print(" ripeti `python -m scripts.run_pipeline`).")


if __name__ == "__main__":
    main()
