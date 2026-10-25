"""
Chiude il protocollo di confronto del PDF (sezione 3): legge il blind test
umano DOPO che qualcuno lo ha compilato, lo ricongiunge con la chiave
A/B->GNN/LLM, e produce statistiche + un breve riepilogo testuale pronto
per il report finale.

La compilazione della rubrica (correttezza e concisione, 1-5) resta
manuale per forza: e' il punto stesso del blind test. Questo script fa
tutto il resto in automatico.

Uso:
    python -m scripts.analyze_blind_eval

Prerequisito: aver compilato results/human_eval_blind.csv (colonne
A_correttezza_1_5, A_concisione_1_5, B_correttezza_1_5, B_concisione_1_5,
quale_guida_e_migliore_A_B_pari con valori "A", "B" o "pari").
"""
from src.human_eval import run_human_eval_analysis

if __name__ == "__main__":
    run_human_eval_analysis()
