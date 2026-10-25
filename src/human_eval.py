"""
src/human_eval.py
-------------------
Analisi del blind test umano una volta compilato (PDF, sezione 3:
"qualita' della guida ... valutata da esperti/esseri umani ... rubrica da
1 a 5 per correttezza e concisione" + "Valutazione umana ... blind test").

La compilazione della rubrica resta necessariamente manuale — nessun
automatismo puo' sostituire il giudizio di una persona, ed e' proprio il
punto di un blind test. Questo modulo si occupa di tutto il resto:
ricongiungere le risposte alla chiave A/B->GNN/LLM, calcolare le medie per
modello, un test statistico appaiato sulle valutazioni (Wilcoxon, adatto a
punteggi ordinali 1-5 appaiati sullo stesso puzzle) e un riepilogo pronto
da incollare nel report.

Uso, da riga di comando, DOPO aver compilato results/human_eval_blind.csv:
    python -m scripts.analyze_blind_eval
"""

import os

import numpy as np
import pandas as pd

import config

RATING_COLS_SUFFIX = ["_correttezza_1_5", "_concisione_1_5"]


def _validate_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Controlla che la rubrica sia stata compilata e scarta le righe non
    ancora valutate, segnalandole invece di farle passare come dati validi."""
    rating_cols = [f"{side}{suf}" for side in ("A", "B") for suf in RATING_COLS_SUFFIX]
    missing_cols = [c for c in rating_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Il file non ha le colonne attese {missing_cols}. E' il file giusto "
            f"({config.BLIND_EVAL_CSV}), generato da build_blind_eval_file?"
        )

    for c in rating_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    incomplete = df[rating_cols].isna().any(axis=1)
    n_incomplete = int(incomplete.sum())
    if n_incomplete > 0:
        print(f"ATTENZIONE: {n_incomplete} puzzle su {len(df)} hanno almeno una "
              "colonna di rubrica vuota o non numerica: esclusi dall'analisi. "
              "Se e' tutto il file, probabilmente non e' ancora stato compilato.")
    df_valid = df[~incomplete].copy()

    out_of_range = ((df_valid[rating_cols] < 1) | (df_valid[rating_cols] > 5)).any(axis=1)
    if out_of_range.any():
        print(f"ATTENZIONE: {int(out_of_range.sum())} righe hanno un punteggio fuori "
              "dal range 1-5: controllale a mano, per ora restano incluse.")

    return df_valid


def load_completed_blind_eval(
    filled_csv: str = config.BLIND_EVAL_CSV,
    answer_key_csv: str = config.BLIND_EVAL_ANSWER_KEY_CSV,
) -> pd.DataFrame:
    """Ricarica il file compilato dal valutatore e lo ricongiunge con la
    chiave A/B -> GNN/LLM (che il valutatore non deve mai vedere prima di
    finire, altrimenti il test non e' piu' in cieco)."""
    if not os.path.exists(filled_csv):
        raise FileNotFoundError(
            f"{filled_csv} non trovato. Va prima generato da "
            "run_phase4_comparison (fa parte di run_llm_comparison) e poi "
            "compilato a mano da un valutatore."
        )
    if not os.path.exists(answer_key_csv):
        raise FileNotFoundError(
            f"{answer_key_csv} non trovato: senza la chiave non si puo' sapere "
            "quale colonna (A o B) corrisponde a GNN o LLM."
        )

    df_filled = pd.read_csv(filled_csv)
    df_key = pd.read_csv(answer_key_csv)[["PuzzleId", "_true_label_A"]]

    df = df_filled.merge(df_key, on="PuzzleId", how="left", validate="one_to_one")
    if df["_true_label_A"].isna().any():
        raise ValueError(
            "Alcuni PuzzleId del file compilato non si trovano nella chiave: "
            "il file compilato e la chiave non vengono dalla stessa run. "
            "Rigenera entrambi insieme con run_llm_comparison e non mischiare "
            "compilazioni di run diverse."
        )

    df = _validate_ratings(df)
    if len(df) == 0:
        raise ValueError("Nessuna riga valida da analizzare (rubrica non compilata?).")

    # Riallinea A/B -> GNN/LLM, cosi' il resto dell'analisi lavora sui nomi
    # dei modelli invece che sulle etichette cieche A/B.
    is_a_gnn = df["_true_label_A"] == "GNN"
    for suf in RATING_COLS_SUFFIX:
        df[f"gnn{suf}"] = np.where(is_a_gnn, df[f"A{suf}"], df[f"B{suf}"])
        df[f"llm{suf}"] = np.where(is_a_gnn, df[f"B{suf}"], df[f"A{suf}"])

    def _resolve_preference(row):
        pref = str(row.get("quale_guida_e_migliore_A_B_pari", "")).strip().upper()
        if pref not in ("A", "B", "PARI"):
            return None
        if pref == "PARI":
            return "PARI"
        is_gnn_side = (pref == "A") == bool(row["_true_label_A"] == "GNN")
        return "GNN" if is_gnn_side else "LLM"

    df["preferenza_modello"] = df.apply(_resolve_preference, axis=1)
    return df


def summarize_blind_eval(df: pd.DataFrame) -> dict:
    """Statistiche descrittive + test appaiato (PDF: 'qualita' della guida
    ... valutata da esperti/esseri umani'). Wilcoxon signed-rank e' scelto
    perche' i punteggi sono ordinali (1-5) e appaiati sullo stesso puzzle
    per i due modelli — lo stesso principio di 'confronto appaiato' usato
    per McNemar sull'accuratezza, qui applicato a dati ordinali invece che
    binari."""
    from scipy.stats import wilcoxon

    summary = {"n_puzzle_valutati": len(df)}

    print(f"=== Valutazione umana in cieco — {len(df)} puzzle compilati ===\n")

    for metric, label in [("correttezza_1_5", "Correttezza"), ("concisione_1_5", "Concisione")]:
        gnn_scores = df[f"gnn_{metric}"]
        llm_scores = df[f"llm_{metric}"]
        gnn_mean, llm_mean = gnn_scores.mean(), llm_scores.mean()
        diffs = gnn_scores - llm_scores
        n_nonzero = int((diffs != 0).sum())

        print(f"{label}: GNN {gnn_mean:.2f}/5  vs  LLM {llm_mean:.2f}/5  "
              f"(media su {len(df)} puzzle)")

        summary[f"gnn_{metric}_mean"] = float(gnn_mean)
        summary[f"llm_{metric}_mean"] = float(llm_mean)

        if n_nonzero < 10:
            print(f"  Solo {n_nonzero} puzzle con punteggi diversi tra i due modelli: "
                  "test di Wilcoxon poco affidabile con cosi' poche discordanze, "
                  "riportare la differenza di media con cautela.")
            summary[f"{metric}_wilcoxon_p"] = None
        else:
            stat, p = wilcoxon(gnn_scores, llm_scores)
            summary[f"{metric}_wilcoxon_p"] = float(p)
            verdict = ("differenza statisticamente significativa" if p < 0.05
                       else "nessuna differenza statisticamente significativa")
            print(f"  Wilcoxon signed-rank: p = {p:.4f} -> {verdict} (alpha=0.05)")
        print()

    if "preferenza_modello" in df.columns:
        pref_counts = df["preferenza_modello"].value_counts(dropna=True)
        n_pref = int(pref_counts.sum())
        print(f"Preferenza complessiva ({n_pref} puzzle con risposta valida su {len(df)}):")
        for label in ("GNN", "LLM", "PARI"):
            c = int(pref_counts.get(label, 0))
            pct = c / n_pref if n_pref else 0.0
            print(f"  {label:5}: {c:>3} ({pct:.0%})")
        summary["preferenza_counts"] = pref_counts.to_dict()

        decisive = df[df["preferenza_modello"].isin(["GNN", "LLM"])]
        n_decisive = len(decisive)
        if n_decisive >= 10:
            from src.stats import mcnemar_from_counts
            b = int((decisive["preferenza_modello"] == "GNN").sum())
            c = n_decisive - b
            res = mcnemar_from_counts(b, c)
            print(f"\nTest binomiale sulla preferenza (esclusi i 'PARI'): "
                  f"{b} preferenze GNN, {c} preferenze LLM su {n_decisive} decisive, "
                  f"p esatto = {res['p_exact']:.4f}")
            summary["preferenza_binomiale_p"] = res["p_exact"]

    return summary


def save_summary_report(df: pd.DataFrame, summary: dict,
                        out_csv: str = config.HUMAN_EVAL_SUMMARY_CSV,
                        out_md: str = config.HUMAN_EVAL_SUMMARY_MD) -> None:
    """Salva sia una tabella (per riuso in altri script) sia un breve
    riassunto testuale pronto da incollare nel report finale."""
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    lines = [
        "# Valutazione umana in cieco — riepilogo",
        "",
        f"Puzzle compilati e validi: {summary['n_puzzle_valutati']}",
        "",
        "## Correttezza (1-5)",
        f"- GNN: {summary['gnn_correttezza_1_5_mean']:.2f}",
        f"- LLM: {summary['llm_correttezza_1_5_mean']:.2f}",
    ]
    p = summary.get("correttezza_1_5_wilcoxon_p")
    if p is not None:
        lines.append(f"- Wilcoxon p = {p:.4f} "
                     f"({'significativo' if p < 0.05 else 'non significativo'}, alpha=0.05)")
    lines += [
        "",
        "## Concisione (1-5)",
        f"- GNN: {summary['gnn_concisione_1_5_mean']:.2f}",
        f"- LLM: {summary['llm_concisione_1_5_mean']:.2f}",
    ]
    p = summary.get("concisione_1_5_wilcoxon_p")
    if p is not None:
        lines.append(f"- Wilcoxon p = {p:.4f} "
                     f"({'significativo' if p < 0.05 else 'non significativo'}, alpha=0.05)")

    if "preferenza_counts" in summary:
        lines += ["", "## Preferenza complessiva"]
        for label, cnt in summary["preferenza_counts"].items():
            lines.append(f"- {label}: {cnt}")
        if "preferenza_binomiale_p" in summary:
            lines.append(f"- Test binomiale (GNN vs LLM, esclusi i pari): "
                         f"p = {summary['preferenza_binomiale_p']:.4f}")

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nRiepilogo salvato in {out_csv} e {out_md} (pronto da incollare nel report).")


def run_human_eval_analysis():
    df = load_completed_blind_eval()
    summary = summarize_blind_eval(df)
    save_summary_report(df, summary)
    return summary
