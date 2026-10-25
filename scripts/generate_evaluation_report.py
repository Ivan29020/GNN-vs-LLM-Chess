"""
Genera results/evaluation_report.md mettendo insieme tutti i risultati gia'
prodotti dalla pipeline: tabelle, grafici, McNemar, ablation, matrici di
confusione, insight quantificati e (se presente) il riepilogo del blind
test umano.

E' il deliverable "rapporto di valutazione (evaluation report)" richiesto
dal PDF (sezione Risultati Attesi). Va eseguito per ultimo, dopo:
  1. python -m scripts.run_pipeline
  2. python -m scripts.run_training
  3. python -m scripts.run_llm_comparison
  4. (opzionale) python -m scripts.analyze_blind_eval

Uso:
    python -m scripts.generate_evaluation_report
"""
from src.report import generate_evaluation_report

if __name__ == "__main__":
    generate_evaluation_report()
