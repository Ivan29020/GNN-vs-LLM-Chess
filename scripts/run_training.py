"""Entry point storico mantenuto per compatibilità.

Per le nuove esecuzioni su Colab usare gli stage indipendenti:
    python -u -m scripts.run_staged --stage timed
    python -u -m scripts.run_staged --stage no-timing
    python -u -m scripts.run_staged --stage compare

Il flusso combined storico resta disponibile solo con --legacy-combined, ma non
è consigliato per sessioni Colab limitate perché addestra entrambe le varianti
nella stessa esecuzione.
"""
import sys
from src.train import run_phase3


if __name__ == "__main__":
    if "--legacy-combined" not in sys.argv:
        print(
            "Questo è l'entry point storico che può addestrare entrambe le GNN "
            "nella stessa sessione.\n"
            "Per il flusso riprendibile a tre stage usa, una fase per sessione:\n"
            "  python -u -m scripts.run_staged --stage timed\n"
            "  python -u -m scripts.run_staged --stage no-timing\n"
            "  python -u -m scripts.run_staged --stage compare\n"
            "Per forzare intenzionalmente il flusso vecchio: aggiungi --legacy-combined."
        )
        raise SystemExit(2)
    force = "--force-training" in sys.argv or "--no-skip" in sys.argv
    run_phase3(skip_training=False if force else None)
