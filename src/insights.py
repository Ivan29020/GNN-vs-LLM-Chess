"""
src/insights.py
-----------------
Due deliverable espliciti del PDF che il resto della pipeline non produceva
ancora:

1. "Visualizzazioni (ad es., matrici di confusione divise per n)" — sezione
   Risultati Attesi.
2. "Quantificare i vantaggi delle GNN (ad es., 'la GNN prevale per n<=3 con
   un'accuratezza superiore del 15%') e i benefici del timing (ad es., 'un
   incremento del 10% in scenari blitz')" — stessa sezione.

Il punto 2 in particolare e' una frase-modello che il PDF chiede di
riempire con numeri VERI, calcolati dai dati, non stimati a occhio. Questo
modulo legge i CSV gia' prodotti da compare.py e train.py e genera quelle
frasi in automatico, oltre alle matrici di confusione.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def plot_confusion_matrices_by_n(
    df_paired: pd.DataFrame,
    gnn_col: str = "gnn_full_sequence_correct",
    llm_col: str = "full_sequence_correct",
    save_path: str = None,
) -> pd.DataFrame:
    """Matrice di confusione GNN-corretto x LLM-corretto, UNA per ogni
    valore di n presente nel campione (deliverable esplicito del PDF).

    Nel nostro task l'esito e' binario per modello (puzzle risolto o no),
    non una classificazione multi-classe: la matrice di confusione piu'
    informativa e' quindi la tabella di contingenza 2x2 che mostra, per
    ogni n, quanti puzzle sono stati risolti da entrambi i modelli, solo
    dalla GNN, solo dall'LLM, o da nessuno dei due — sono le stesse quattro
    celle su cui si basa il test di McNemar (src/stats.py), qui rese come
    immagine anziche' come tabella di numeri.
    """
    save_path = save_path or config.CONFUSION_MATRICES_PATH
    ns = sorted(df_paired["mate_n"].unique())
    n_cols = min(4, len(ns))
    n_rows = int(np.ceil(len(ns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows))
    axes = np.array(axes).reshape(-1)

    rows_summary = []
    for i, n in enumerate(ns):
        sub = df_paired[df_paired["mate_n"] == n]
        gnn_ok = sub[gnn_col].astype(bool)
        llm_ok = sub[llm_col].astype(bool)

        both = int((gnn_ok & llm_ok).sum())
        only_gnn = int((gnn_ok & ~llm_ok).sum())
        only_llm = int((~gnn_ok & llm_ok).sum())
        neither = int((~gnn_ok & ~llm_ok).sum())
        matrix = np.array([[both, only_gnn], [only_llm, neither]])

        ax = axes[i]
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(matrix[r, c]), ha="center", va="center",
                       color="white" if matrix[r, c] > matrix.max() / 2 else "black",
                       fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["LLM OK", "LLM KO"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["GNN OK", "GNN KO"], fontsize=8)
        ax.set_title(f"mate in {n}  (N={len(sub)})", fontsize=10)

        rows_summary.append({
            "mate_n": n, "n_puzzle": len(sub),
            "entrambi_corretti": both, "solo_gnn_corretta": only_gnn,
            "solo_llm_corretta": only_llm, "nessuno_corretto": neither,
        })

    for j in range(len(ns), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Matrici di confusione GNN vs LLM, per profondita' di matto "
                 "(sequenza completa)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Matrici di confusione per n salvate in {save_path}")

    df_summary = pd.DataFrame(rows_summary)
    df_summary.to_csv(config.CONFUSION_MATRICES_CSV, index=False)
    return df_summary


def quantify_gnn_advantage(acc_by_n: pd.DataFrame,
                           min_n_for_claim: int = 10) -> dict:
    """Riempie con numeri VERI la frase-modello del PDF: 'la GNN prevale
    per n<=X con un'accuratezza superiore del Y%'.

    Un valore di n entra nell'affermazione solo se ha almeno
    `min_n_for_claim` puzzle nel campione (altrimenti la differenza e'
    aneddotica, non quantificabile con onesta' statistica — coerente con
    l'avviso gia' stampato altrove quando N<10).
    """
    acc_by_n = acc_by_n.copy()
    acc_by_n["delta_pp"] = (acc_by_n["gnn_acc"] - acc_by_n["llm_acc"]) * 100
    reliable = acc_by_n[acc_by_n["n_puzzles"] >= min_n_for_claim].sort_values("mate_n")

    gnn_better_ns = reliable[reliable["delta_pp"] > 0]["mate_n"].tolist()
    print(f"\n=== Quantificazione del vantaggio GNN (PDF: 'la GNN prevale per n<=X "
          f"con un'accuratezza superiore del Y%') — solo n con almeno "
          f"{min_n_for_claim} puzzle ===")

    insight = {"n_affidabili": reliable["mate_n"].tolist(), "gnn_prevale_per_n": gnn_better_ns}

    if gnn_better_ns:
        avg_delta = reliable[reliable["mate_n"].isin(gnn_better_ns)]["delta_pp"].mean()
        n_range = (f"n<={max(gnn_better_ns)}" if gnn_better_ns == list(range(min(gnn_better_ns), max(gnn_better_ns) + 1))
                   else f"n in {gnn_better_ns}")
        frase = (f"La GNN prevale per {n_range}, con un'accuratezza di sequenza completa "
                f"superiore in media di {avg_delta:.1f} punti percentuali rispetto all'LLM.")
        print(frase)
        insight["frase_insight"] = frase
        insight["delta_medio_pp"] = float(avg_delta)
    else:
        print("Nessun valore di n con dati sufficienti mostra un vantaggio della GNN "
              "(verifica se il campione e' abbastanza numeroso).")
        insight["frase_insight"] = None

    for _, row in reliable.iterrows():
        segno = "+" if row["delta_pp"] >= 0 else ""
        print(f"  n={int(row['mate_n'])}: GNN {row['gnn_acc']:.1%}  LLM {row['llm_acc']:.1%}  "
              f"(delta {segno}{row['delta_pp']:.1f} pp, N={int(row['n_puzzles'])})")

    unreliable = acc_by_n[acc_by_n["n_puzzles"] < min_n_for_claim]
    if len(unreliable) > 0:
        print(f"\n  Esclusi dall'affermazione quantitativa per N<{min_n_for_claim}: "
              f"n = {unreliable['mate_n'].tolist()} (dati insufficienti per un claim solido, "
              "riportabili solo come osservazione indicativa).")

    return insight


def quantify_timing_benefit(ablation_table: pd.DataFrame) -> dict:
    """Riempie la seconda frase-modello del PDF: 'un incremento del 10% in
    scenari blitz'. ATTENZIONE: i dati attuali non permettono di isolare
    scenari blitz veri (il timing simulato e' un proxy di difficolta'
    (timing proxy v2, vedi config.TIMING_WEIGHTS), non una simulazione di pressione del tempo — vedi
    docs/RISPOSTE_DOMANDE_PROGETTO.md). Qui si quantifica quindi il
    beneficio medio complessivo e per n, dichiarando esplicitamente questo
    limite invece di affermare un numero sugli 'scenari blitz' che i dati
    non possono davvero sostenere.
    """
    avg_delta = (ablation_table["acc_seq_timed"] - ablation_table["acc_seq_no_timing"]).mean() * 100
    print(f"\n=== Quantificazione del beneficio del timing "
          "(PDF: 'un incremento del 10% in scenari blitz') ===")
    frase = (f"L'informazione di timing produce un incremento medio di {avg_delta:.1f} punti "
            "percentuali nell'accuratezza di sequenza completa, su tutti i valori di n "
            "testati (non isolabile per 'scenari blitz' specifici con i dati attuali: "
            "il timing simulato (proxy v2: rating, mosse legali, scacchi disponibili e "
            "mosse residue al matto) e' un proxy di difficolta', non una simulazione di "
            "pressione del tempo reale; la componente 'mosse residue' porta anche "
            "informazione sulla profondita' — vedi i limiti dichiarati nel report).")
    print(frase)
    return {"delta_medio_pp": float(avg_delta), "frase_insight": frase}
