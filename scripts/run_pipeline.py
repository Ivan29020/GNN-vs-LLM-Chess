"""
Esegue la Fase 1 (download + pulizia + split dei puzzle Lichess).

Uso:
    python -m scripts.run_pipeline
"""
from src.data_pipeline import run_phase1

if __name__ == "__main__":
    run_phase1()
