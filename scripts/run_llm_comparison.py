"""
Esegue la Fase 4 (baseline LLM + confronto statistico con la GNN).

Richiede:
- Le Fasi 1-3 già eseguite (servono i checkpoint del modello GNN)
- Una API key valida per il backend scelto (LLM_BACKEND in config.py):
    - "groq"      (default, gratuito, ~14.400 richieste/giorno) -> GROQ_API_KEY
    - "gemini"    (gratuito, quote più basse e variabili)        -> GEMINI_API_KEY
    - "anthropic"                                                -> ANTHROPIC_API_KEY
    - "openai"                                                   -> OPENAI_API_KEY
  impostata come variabile d'ambiente.

Uso (backend di default, Groq gratuito):
    export GROQ_API_KEY="gsk_..."
    python -m scripts.run_llm_comparison

Uso con un altro backend:
    export LLM_BACKEND="gemini"
    export GEMINI_API_KEY="AIza..."
    python -m scripts.run_llm_comparison
"""
import os

import config
from src.llm_baseline import (
    get_llm_client,
    load_evaluation_set,
    run_llm_evaluation,
    print_llm_metrics,
)
from src.compare import run_phase4_comparison


def main():
    if not os.path.exists(config.TIMED_CKPT):
        raise FileNotFoundError(
            f"{config.TIMED_CKPT} non trovato.\n"
            "Hai due strade:\n"
            "  1) addestrare da zero:  python -m scripts.run_training\n"
            "  2) riusare i pesi di una sessione precedente: copia i file\n"
            f"     {os.path.basename(config.TIMED_CKPT)} in {config.CKPT_DIR}/"
        )

    if config.LLM_FRESH_START:
        print("LLM_FRESH_START=1: la valutazione LLM riparte da zero.")
    elif os.path.exists(config.LLM_RAW_CSV):
        print(f"NOTA: {config.LLM_RAW_CSV} esiste gia' e i puzzle gia' valutati verranno "
              "saltati (ripresa). Se hai cambiato il prompt o le metriche, rilancia con "
              "LLM_FRESH_START=1, altrimenti mescoli due esperimenti diversi nello "
              "stesso file.")

    df_eval = load_evaluation_set(n_samples=config.LLM_N_SAMPLES)

    client = get_llm_client(backend=config.LLM_BACKEND)
    df_llm = run_llm_evaluation(df_eval, client, backend=config.LLM_BACKEND,
                                 model_name=config.LLM_MODEL_NAME)
    print_llm_metrics(df_llm)

    run_phase4_comparison(df_llm, df_eval)


if __name__ == "__main__":
    main()
