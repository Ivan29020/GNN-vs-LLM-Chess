"""
src/report.py
---------------
Genera il "rapporto di valutazione (evaluation report)" richiesto dal PDF
come deliverable (sezione Risultati Attesi) mettendo insieme tutti i
risultati gia' calcolati dal resto della pipeline: tabelle, grafici,
McNemar, ablation, matrici di confusione, insight quantificati, e — se
presente — il riepilogo del blind test umano.

Questo NON sostituisce la stesura finale del report da parte degli
studenti (il PDF chiede anche di descrivere metodologia, limiti e
discussione in prosa, che restano un lavoro editoriale umano), ma genera
automaticamente la parte "dati e risultati" — numeri, tabelle, grafici
embeddati, e le frasi di insight quantificato — cosi' che il lavoro umano
residuo sia scrivere l'introduzione, commentare i risultati e assemblare
le sezioni descrittive, non ricalcolare o ritrascrivere numeri a mano.

Uso, dopo aver eseguito run_pipeline, run_training e run_llm_comparison
(e, opzionalmente, analyze_blind_eval):
    python -m scripts.generate_evaluation_report
"""

import os
from datetime import datetime

import pandas as pd

import config


def _read_csv_or_none(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.select_dtypes(include="float").columns:
        df[c] = df[c].map(lambda x: float_fmt.format(x) if pd.notna(x) else "")
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([header, sep] + rows)


def generate_evaluation_report(out_path: str = None) -> str:
    out_path = out_path or config.EVALUATION_REPORT_MD
    sections = []

    sections.append(f"# Rapporto di valutazione — Timed GNN per puzzle scacchistici\n\n"
                    f"Generato automaticamente il {datetime.now():%Y-%m-%d %H:%M}.\n\n"
                    "Questo documento raccoglie i risultati numerici prodotti dalla "
                    "pipeline (`run_pipeline`, `run_training`, `run_llm_comparison`, "
                    "`analyze_blind_eval`). La discussione qualitativa e i limiti "
                    "completi sono in `docs/RISPOSTE_DOMANDE_PROGETTO.md`; questo file "
                    "e' pensato per essere incollato o linkato nel report finale del "
                    "progetto.\n")

    # --- Obiettivo 4: valutazione comparativa GNN vs LLM ---
    sections.append("## 4. Valutazione comparativa — GNN vs LLM\n")
    acc_by_n = _read_csv_or_none(config.ACC_BY_N_CSV)
    if acc_by_n is not None:
        sections.append("Accuratezza di sequenza completa per profondita' di matto "
                        "(metrica primaria — vedi `docs/RISPOSTE_DOMANDE_PROGETTO.md` "
                        "per la giustificazione):\n\n" + _md_table(acc_by_n) + "\n")
        sections.append(f"![Accuratezza per n](./{os.path.basename(config.COMPARISON_PLOT_PATH)})\n")

        from src.insights import quantify_gnn_advantage
        insight = quantify_gnn_advantage(acc_by_n)
        if insight.get("frase_insight"):
            sections.append(f"\n**Insight quantificato (PDF, Risultati Attesi):** "
                            f"{insight['frase_insight']}\n")
    else:
        sections.append("_Dati non ancora disponibili: eseguire `run_llm_comparison`._\n")

    confusion = _read_csv_or_none(config.CONFUSION_MATRICES_CSV)
    if confusion is not None:
        sections.append("\n### Matrici di confusione per n\n\n"
                        "Tabella di contingenza GNN-corretta x LLM-corretta, le stesse "
                        "celle su cui si basa il test di McNemar:\n\n"
                        + _md_table(confusion) + "\n\n"
                        f"![Matrici di confusione](./{os.path.basename(config.CONFUSION_MATRICES_PATH)})\n")

    paired = _read_csv_or_none(config.PAIRED_RESULTS_CSV)
    if paired is not None:
        from src.stats import mcnemar_from_flags
        with_capture = []
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mcnemar_from_flags(paired["gnn_full_sequence_correct"],
                               paired["full_sequence_correct"], "GNN", "LLM")
        sections.append("\n### Test statistico (McNemar, campione completo)\n\n```\n"
                        + buf.getvalue() + "```\n")

    # --- Obiettivo 5: ablation timing ---
    sections.append("\n## 5. Studio di ablazione — informazioni di timing\n")
    ablation = _read_csv_or_none(config.ABLATION_CSV)
    if ablation is not None:
        sections.append("Confronto GNN con/senza timing, per n:\n\n"
                        + _md_table(ablation) + "\n\n"
                        f"![Ablation per n](./{os.path.basename(config.ABLATION_PLOT_PATH)})\n")

        from src.insights import quantify_timing_benefit
        insight = quantify_timing_benefit(ablation)
        sections.append(f"\n**Insight quantificato (PDF, Risultati Attesi):** "
                        f"{insight['frase_insight']}\n")
    else:
        sections.append("_Dati non ancora disponibili: eseguire `run_training`._\n")

    # --- Valutazione umana (se compilata) ---
    sections.append("\n## Valutazione umana in cieco (qualita' della guida)\n")
    if os.path.exists(config.HUMAN_EVAL_SUMMARY_MD):
        with open(config.HUMAN_EVAL_SUMMARY_MD) as f:
            sections.append(f.read())
    else:
        sections.append("_Non ancora disponibile: compilare `results/human_eval_blind.csv` "
                        "e poi eseguire `python -m scripts.analyze_blind_eval`._\n")

    # --- Deliverable ed estensioni ---
    sections.append("\n## Deliverable\n\n"
                    "- Repository del codice: questo stesso archivio (`src/`, `scripts/`, "
                    "`config.py`)\n"
                    "- Modello addestrato: `checkpoints/timed_gnn_tv2_best.pt` e "
                    "`checkpoints/no_timing_gnn_best.pt`\n"
                    "- Rapporto di valutazione: questo documento\n"
                    "- Visualizzazioni: grafici per n e matrici di confusione (sopra), "
                    "`results/*.png`\n")

    sections.append("\n## Estensioni possibili\n\n"
                    "Vedi `docs/ESTENSIONI.md` per la discussione dettagliata delle tre "
                    "estensioni indicate dal PDF (applicazione web, integrazione "
                    "Stockfish, input multimodale) applicate a questo progetto "
                    "specifico, incluso un prototipo funzionante per l'integrazione "
                    "Stockfish in `scripts/stockfish_hybrid_demo.py`.\n")

    report_text = "\n".join(sections)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"Rapporto di valutazione generato in {out_path}")
    return report_text
