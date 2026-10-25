"""
config.py
---------
Configurazione centrale del progetto "Timed GNN per puzzle scacchistici".

Tutti gli script leggono i parametri da qui, così puoi cambiare
percorsi/iperparametri in UN solo posto invece che dentro ogni notebook,
come accadeva nei file originali (dove DATA_DIR era ripetuto e ridefinito
in ogni notebook).

Le variabili "sensibili" (API key) NON vanno mai scritte qui: si leggono
da variabili d'ambiente (vedi README.md).
"""

import os

# ---------------------------------------------------------------------
# Percorsi cartelle
# ---------------------------------------------------------------------
# Base del progetto = cartella in cui si trova questo file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# File Fase 1 (pipeline dati)
PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
ZST_PATH = os.path.join(DATA_DIR, "lichess_db_puzzle.csv.zst")
CSV_PATH = os.path.join(DATA_DIR, "lichess_db_puzzle.csv")

TRAIN_PARQUET = os.path.join(DATA_DIR, "puzzles_train.parquet")
VAL_PARQUET = os.path.join(DATA_DIR, "puzzles_val.parquet")
TEST_PARQUET = os.path.join(DATA_DIR, "puzzles_test.parquet")

# Due held-out indipendenti (confronto finale GNN vs LLM), creati dalla pipeline
# e salvati separatamente. Nessun file da scaricare/posizionare a mano.
HELD_OUT_PARQUET = os.path.join(DATA_DIR, "puzzles_held_out.parquet")  # misto Polgar + Lichess
HELD_OUT_LICHESS_PARQUET = os.path.join(DATA_DIR, "puzzles_held_out_lichess.parquet")
# Manifest scritto solo quando train/val/test e TUTTI E DUE gli held-out sono pronti.
DATA_PIPELINE_MANIFEST_JSON = os.path.join(DATA_DIR, "pipeline_manifest.json")
DATA_PIPELINE_SCHEMA_VERSION = 3
POLGAR_PGN_URL = "https://raw.githubusercontent.com/denialromeo/4462-chess-problems/master/polgar.pgn"
POLGAR_PGN_LOCAL = os.path.join(DATA_DIR, "polgar.pgn")

KEEP_COLS = ["PuzzleId", "FEN", "Moves", "Rating", "Popularity", "mate_n",
             "rating_norm", "has_opening_move"]
# Le fasce globali del rating sono salvate nei soli split interni per rendere
# verificabile la stratificazione congiunta mate_n × rating.
TRAINING_COLS = KEEP_COLS + ["rating_bin"]
# Il parquet held-out ha una colonna in piu': la fonte del puzzle.
HELD_OUT_COLS = KEEP_COLS + ["source"]

# ---------------------------------------------------------------------
# Fase 1 — filtro puzzle
# ---------------------------------------------------------------------
MATE_N_MIN, MATE_N_MAX = 1, 5     # range usato per il training (come da metodologia)
SPLIT_TEST_SIZE = 0.20            # quota temporanea: 20% da dividere fra val e test
SPLIT_VAL_OF_TEMP = 0.50          # risultato finale: 80/10/10 (train/val/test)
RANDOM_SEED = 42
SUBSET_RATING_BINS = 5            # fasce quantili GLOBALI del rating puzzle per il subset 300k
SUBSET_SEED = RANDOM_SEED

# ---------------------------------------------------------------------
# Held-out esterno: 30 puzzle per ciascun n da 1 a 5
#   n = 1, 2, 3  -> problemi di Polgar (fonte indipendente da Lichess)
#   n = 4, 5     -> puzzle Lichess, ESTRATTI PRIMA dello split e rimossi
#                   da train/val/test (per PuzzleId e per FEN)
# ---------------------------------------------------------------------
HELD_OUT_PER_N = 30
HELD_OUT_POLGAR_N = (1, 2, 3)
# Nel held-out misto la composizione storica resta Polgar N1-N3 + Lichess N4-N5.
HELD_OUT_MIXED_LICHESS_N = (4, 5)
# Secondo held-out: esclusivamente Lichess, 30 puzzle per ogni N da 1 a 5,
# disgiunti dal held-out misto per PuzzleId e posizione normalizzata.
HELD_OUT_LICHESS_ONLY_N = (1, 2, 3, 4, 5)
HELD_OUT_LICHESS_N = HELD_OUT_MIXED_LICHESS_N  # alias compatibilita' con script precedenti
HELD_OUT_RATING_BINS = 5          # campionamento Lichess stratificato per rating
HELD_OUT_SEED = RANDOM_SEED
HELD_OUT_LICHESS_ONLY_SEED_OFFSET = 10_000

# ---------------------------------------------------------------------
# Fase 2 — encoding del grafo
# ---------------------------------------------------------------------
THINK_TIME_CLIP_SEC = 10.0        # il proxy v2 vale 1-10 s simulati: normalizzato /10

# ---------------------------------------------------------------------
# Timing proxy v2 (feature "tempo di pensiero simulato")
# ---------------------------------------------------------------------
# D(s) = w_R*R + w_B*B(s) + w_F*F(s) + w_K*K(s),   T(s) = clip(1 + 9*D, 1, 10) secondi
#   R    rating del puzzle normalizzato sui percentili P10-P90 del TRAINING SET
#   B(s) numero di mosse legali nella posizione corrente (scala logaritmica)
#   F(s) numero di mosse legali che danno scacco (candidate "forzanti")
#   K(s) mosse del solver che restano fino al matto (profondita' residua)
# Deterministico (nessun rumore) e calcolato PER POSIZIONE (cambia lungo la
# linea), senza motori scacchistici. I pesi sono valori iniziali scelti per
# l'esperimento, NON coefficienti appresi da tempi reali. Mettere un peso a 0
# disattiva la componente (utile per l'ablation: es. TIMING_WEIGHTS["K"] = 0
# per misurare il timing "puro", senza l'informazione sulla profondita').
TIMING_WEIGHTS = {"R": 0.25, "B": 0.20, "F": 0.20, "K": 0.35}
TIMING_RATING_PERCENTILES = (10, 90)
TIMING_LEGAL_MOVES_RANGE = (6, 41)     # B=0 con <=6 mosse legali, B=1 con >=41
TIMING_FORCING_CAP = 5                 # F=1 con >=5 mosse che danno scacco
TIMING_MAX_DEPTH = 5                   # K=1 con 5 mosse del solver ancora da giocare
TIMING_STATS_JSON = os.path.join(DATA_DIR, "timing_stats.json")

# ---------------------------------------------------------------------
# Fase 3 — modello e training
# ---------------------------------------------------------------------
# Iperparametri secondo le linee guida del progetto (layers 3-5, hidden
# dims 128-256, learning rate 1e-3, batch size 32-128, epoche 50-100):
# usati qui i valori piu' alti dei range indicati.
HIDDEN_DIM = 256
NUM_LAYERS = 5
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
# Numero massimo di grafi PyG in cache RAM per dataset. Gli esempi oltre il limite
# sono ricostruiti on-the-fly: 200k è un compromesso più prudente per la RAM di Colab.
# Non aumentare BATCH_SIZE per risolvere problemi di RAM CPU; controllare questo limite.
GRAPH_CACHE_MAX_ITEMS = 200_000
# Configurazione principale della pipeline staged per sessioni Colab separate.
# SUBSET_SIZE è il subset TOTALE, selezionato dopo aver escluso entrambi gli held-out.
# Poi viene diviso 80/10/10: target 240k train, 30k validation, 30k test.
# Il campione è bilanciato per mate_n e, dentro ciascun mate_n, per fasce globali
# quantili del rating del puzzle; le quote impossibili sono ridistribuite.
SUBSET_SIZE = 300_000
NUM_EPOCHS = 100
# Early stopping: se val_acc non migliora di almeno EARLY_STOPPING_MIN_DELTA
# per EARLY_STOPPING_PATIENCE epoche consecutive, il training si ferma
# prima di arrivare a NUM_EPOCHS (il checkpoint salvato resta comunque
# quello della MIGLIOR epoca, non dell'ultima). Vedi train_one_variant.
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# La valutazione "sequenza completa" (src/train.py:evaluate_full_sequence)
# fa un forward pass per OGNI mossa di OGNI puzzle, uno alla volta (non a
# batch, è intrinsecamente sequenziale): tenerla su un campione limitato
# per non rallentare troppo dopo ogni training. Va comunque bene per una
# stima solida dell'accuracy; aumenta se vuoi un numero più stabile.
FULL_SEQ_EVAL_SIZE = 5000

# Il checkpoint "timed" ha il suffisso _tv2: i pesi addestrati con la vecchia
# formula del timing (timed_gnn_best.pt) NON vanno riusati, perche' il
# significato della feature e' cambiato anche se le dimensioni sono uguali.
# Checkpoint dedicati alla pipeline staged con subset=300k. Usiamo nomi nuovi
# per evitare di riutilizzare accidentalmente pesi addestrati con una diversa
# configurazione o su un subset differente.
TIMED_CKPT = os.path.join(CKPT_DIR, "timed_gnn_tv2_subset300k_balanced_rating_v3_best.pt")
NO_TIMING_CKPT = os.path.join(CKPT_DIR, "no_timing_gnn_subset300k_balanced_rating_v3_best.pt")
STAGED_STATE_JSON = os.path.join(RESULTS_DIR, "staged_pipeline_state.json")
STAGED_ABLATION_DONE_JSON = os.path.join(RESULTS_DIR, "staged_ablation_balanced300k_v3_complete.json")
STAGED_COMPARISON_DONE_JSON = os.path.join(RESULTS_DIR, "staged_comparison_balanced300k_v3_complete.json")
STAGED_COMPARISON_MIXED_DONE_JSON = os.path.join(RESULTS_DIR, "staged_comparison_held_out_mixed_balanced300k_v3_complete.json")
STAGED_COMPARISON_LICHESS_DONE_JSON = os.path.join(RESULTS_DIR, "staged_comparison_held_out_lichess_balanced300k_v3_complete.json")
STAGED_LLM_SIGNATURE_JSON = os.path.join(RESULTS_DIR, "staged_llm_signature.json")
STAGED_LLM_SIGNATURE_LICHESS_JSON = os.path.join(RESULTS_DIR, "staged_llm_signature_held_out_lichess.json")
STAGED_SEQ_TIMED_CSV = os.path.join(RESULTS_DIR, "test_sequence_timed_subset300k_balanced_rating_v3.csv")
STAGED_SEQ_NO_TIMING_CSV = os.path.join(RESULTS_DIR, "test_sequence_no_timing_subset300k_balanced_rating_v3.csv")

# Riuso dei checkpoint: se True e i due file sopra esistono gia', la Fase 3
# SALTA il training e passa direttamente alla valutazione. Serve a rifare le
# analisi (ablation, confronto con l'LLM) senza ripagare ogni volta le ~2 ore
# di training su Colab.
# ATTENZIONE: i pesi salvati sono validi SOLO se restano invariati
# HIDDEN_DIM, NUM_LAYERS, DROPOUT e l'encoding del grafo
# (NODE_FEAT_DIM/EDGE_FEAT_DIM in src/chess_encoding.py). Se tocchi una di
# queste cose i checkpoint vanno ributtati e il training rifatto: la verifica
# di compatibilita' e' automatica (vedi src/train.load_checkpoint_if_compatible).
SKIP_TRAINING_IF_CKPT_EXISTS = os.environ.get("SKIP_TRAINING", "1") == "1"

# ---------------------------------------------------------------------
# Fase 4 — baseline LLM
# ---------------------------------------------------------------------
# "anthropic", "openai", "gemini" o "groq".
# Groq (console.groq.com) e' gratuito e molto generoso: circa 14.400
# richieste/giorno sui modelli di punta (contro le 20-1500/giorno di
# Gemini a seconda del modello) — verifica sempre i limiti aggiornati su
# console.groq.com/docs/rate-limits, cambiano nel tempo.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")
# NOTA: Groq dismette e sostituisce spesso i modelli (es. llama-3.3-70b-versatile
# e' stato dismesso ad agosto 2026). Se ricevi un errore 404 "model_not_found",
# controlla l'elenco aggiornato su console.groq.com/docs/models e sovrascrivi
# con `export LLM_MODEL_NAME="nome-attuale"`.
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "openai/gpt-oss-120b")
# Numero di puzzle valutati: l'held-out ha 150 puzzle (30 per ogni n da 1 a 5),
# quindi 150 = tutto l'held-out. Un valore minore sottocampiona restando
# bilanciato per n (vedi src/llm_baseline.allocate_quota).
LLM_N_SAMPLES = int(os.environ.get("LLM_N_SAMPLES", "150"))   # = tutto l'held-out (30 x 5)
# Se True, una nuova run RIPARTE da zero ignorando results/llm_evaluation_raw.csv.
# Da mettere a True ogni volta che cambi il PROMPT o le metriche, altrimenti
# la logica di ripresa salta i puzzle gia' valutati con il prompt vecchio e
# ti ritrovi un CSV che mescola due esperimenti diversi.
LLM_FRESH_START = os.environ.get("LLM_FRESH_START", "0") == "1"
# I modelli "thinking" (es. gemini-3.6-flash) consumano il budget di output
# anche per il ragionamento interno prima di scrivere la risposta visibile:
# con un budget troppo basso la risposta viene TRONCATA prima di arrivare
# alla riga "SOLUTION: ...", e il puzzle risulta erroneamente "sbagliato"
# anche se il modello aveva ragionato bene (e' quello che succedeva con
# LLM_MAX_TOKENS=1024 — controlla sempre 'llm_explanation' nel CSV se
# un'accuracy e' sospettosamente a 0: se e' troncata a meta' frase, il
# problema e' qui, non nella capacita' del modello).
LLM_MAX_TOKENS = 4096
LLM_MAX_RETRIES = 3
# Per Gemini: disattiva il "thinking" esteso (ragionamento nascosto, non
# visibile in llm_explanation ma che consuma comunque token) cosi' tutto
# il budget va nella risposta visibile, che include gia' un ragionamento
# passo-passo richiesto esplicitamente dal prompt.
GEMINI_THINKING_BUDGET = 0
# Per Groq con modelli "reasoning" (openai/gpt-oss-20b, openai/gpt-oss-120b):
# di default usano reasoning_effort="medium", un canale di ragionamento
# INTERNO e NASCOSTO che consuma tempo/token prima di scrivere la risposta
# visibile — se il budget finisce li' dentro, la risposta arriva vuota
# (e' quello che succedeva: 80% di llm_explanation=nan, ~30s di attesa a
# vuoto). "low" riduce drasticamente quel canale nascosto, lasciando tutto
# il budget visibile (LLM_MAX_TOKENS) alla spiegazione breve + SOLUTION.
GROQ_REASONING_EFFORT = "low"

LLM_RAW_CSV = os.path.join(RESULTS_DIR, "llm_evaluation_raw.csv")
LLM_RAW_LICHESS_CSV = os.path.join(RESULTS_DIR, "llm_evaluation_raw_held_out_lichess.csv")
# Il campione di valutazione LLM e' salvato una sola volta e poi RIUSATO
# identico nelle riprese successive (vedi src/llm_baseline.load_evaluation_set),
# cosi' una valutazione spalmata su piu' giorni (quota API) non mescola
# campioni diversi. Con l'held-out attuale (30 x 5 = 150 puzzle) il campione
# coincide con l'intero held-out.
LLM_EVAL_SAMPLE_CSV = os.path.join(RESULTS_DIR, "llm_eval_sample.csv")
LLM_EVAL_SAMPLE_LICHESS_CSV = os.path.join(RESULTS_DIR, "llm_eval_sample_held_out_lichess.csv")
PAIRED_RESULTS_CSV = os.path.join(RESULTS_DIR, "paired_gnn_vs_llm_balanced300k_v3.csv")  # held-out misto
PAIRED_RESULTS_LICHESS_CSV = os.path.join(RESULTS_DIR, "paired_gnn_vs_llm_held_out_lichess_balanced300k_v3.csv")
COMPARISON_GNN_PROGRESS_CSV = os.path.join(RESULTS_DIR, "gnn_predictions_progress_held_out_mixed_balanced300k_v3.csv")
COMPARISON_GNN_PROGRESS_LICHESS_CSV = os.path.join(RESULTS_DIR, "gnn_predictions_progress_held_out_lichess_balanced300k_v3.csv")
ACC_BY_N_CSV = os.path.join(RESULTS_DIR, "accuracy_by_n_balanced300k_v3.csv")  # held-out misto
ACC_BY_N_LICHESS_CSV = os.path.join(RESULTS_DIR, "accuracy_by_n_held_out_lichess_balanced300k_v3.csv")
DISTRIBUTION_CSV = os.path.join(RESULTS_DIR, "n_distribution_balanced300k_v3.csv")
ABLATION_CSV = os.path.join(RESULTS_DIR, "ablation_timing_balanced300k_v3.csv")
ABLATION_PLOT_PATH = os.path.join(RESULTS_DIR, "ablation_timing_by_n_balanced300k_v3.png")
BLIND_EVAL_CSV = os.path.join(RESULTS_DIR, "human_eval_blind_balanced300k_v3.csv")  # held-out misto
BLIND_EVAL_ANSWER_KEY_CSV = os.path.join(RESULTS_DIR, "human_eval_blind_WITH_ANSWER_KEY_balanced300k_v3.csv")
BLIND_EVAL_LICHESS_CSV = os.path.join(RESULTS_DIR, "human_eval_blind_held_out_lichess_balanced300k_v3.csv")
BLIND_EVAL_ANSWER_KEY_LICHESS_CSV = os.path.join(RESULTS_DIR, "human_eval_blind_WITH_ANSWER_KEY_held_out_lichess_balanced300k_v3.csv")
# Output di scripts/analyze_blind_eval.py, da lanciare DOPO aver compilato
# BLIND_EVAL_CSV a mano: chiude il protocollo di valutazione del PDF
# (sezione 3, qualita' della guida + blind test) producendo statistiche e
# un riepilogo pronto per il report.
HUMAN_EVAL_SUMMARY_CSV = os.path.join(RESULTS_DIR, "human_eval_summary.csv")
HUMAN_EVAL_SUMMARY_MD = os.path.join(RESULTS_DIR, "human_eval_summary.md")

# Matrici di confusione per n e rapporto di valutazione automatico
# (deliverable espliciti del PDF, sezione "Risultati Attesi ed Estensioni").
CONFUSION_MATRICES_PATH = os.path.join(RESULTS_DIR, "confusion_matrices_by_n_balanced300k_v3.png")  # held-out misto
CONFUSION_MATRICES_CSV = os.path.join(RESULTS_DIR, "confusion_matrices_by_n_balanced300k_v3.csv")
CONFUSION_MATRICES_LICHESS_PATH = os.path.join(RESULTS_DIR, "confusion_matrices_by_n_held_out_lichess_balanced300k_v3.png")
CONFUSION_MATRICES_LICHESS_CSV = os.path.join(RESULTS_DIR, "confusion_matrices_by_n_held_out_lichess_balanced300k_v3.csv")
EVALUATION_REPORT_MD = os.path.join(RESULTS_DIR, "evaluation_report.md")
COMPARISON_PLOT_PATH = os.path.join(RESULTS_DIR, "gnn_vs_llm_by_n_balanced300k_v3.png")  # held-out misto
COMPARISON_LICHESS_PLOT_PATH = os.path.join(RESULTS_DIR, "gnn_vs_llm_by_n_held_out_lichess_balanced300k_v3.png")
