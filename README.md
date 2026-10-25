# Timed GNN per la risoluzione di puzzle scacchistici (Lichess)

> **VERSIONE staged: subset totale 300k bilanciato per mate_n × rating + due held-out.** La pipeline crea due set esterni disgiunti: (1) il misto storico
> di **150 puzzle** (Polgár N1-N3, Lichess N4-N5) e (2) un held-out **solo Lichess di 150 puzzle**
> (30 per ogni N da 1 a 5). Entrambi vengono esclusi da train/validation/test prima dello split.
> La pipeline salva separatamente i due Parquet e lo Stage 3 esegue due confronti indipendenti con la baseline LLM.
> **v7 — timing proxy v2.** La feature "tempo di pensiero simulato" non è più
> `rating + popolarità + rumore casuale` (uno scalare per puzzle) ma una formula
> deterministica, **calcolata per ogni posizione**, senza Stockfish:
> `D = 0.25·R + 0.20·B + 0.20·F + 0.35·K`, `T = clip(1 + 9·D, 1, 10)` secondi, con
> R = rating normalizzato (P10-P90 del training), B = mosse legali (log), F = mosse
> che danno scacco, K = mosse del solver ancora da giocare. Pesi in
> `config.TIMING_WEIGHTS` (valori iniziali, non appresi da tempi reali). La
> componente K porta anche informazione sulla profondità (che il prompt dell'LLM
> contiene già): per misurare il timing "puro" metti `TIMING_WEIGHTS["K"] = 0` e
> riaddestra il modello timed. Per questa pipeline con due held-out i checkpoint hanno
> suffisso `_subset300k_balanced_rating_v3_best.pt`: i modelli vanno addestrati sui nuovi split.
>
> **I vecchi checkpoint (`checkpoints/*.pt`) non sono più validi**: sono
> stati addestrati su dati che includono i puzzle ora in held-out.
> Cancellali e riaddestra (la fase 3 si ferma con un errore se trova
> leakage nei file di dati).
>
> Documentazione di supporto:
> - `docs/RISPOSTE_DOMANDE_PROGETTO.md` — risposte alle domande di ricerca e
>   al protocollo di confronto del PDF, con i numeri e i limiti
> - `docs/PATCH_APPLICATE.md` — cosa è cambiato e perché
> - `docs/GUIDA_COLAB.md` — sequenza dei comandi per Colab (training da fare)

Questo progetto unisce in codice Python organizzato e collegato i 4 notebook
originali:

| Notebook originale | Diventa | Fase |
|---|---|---|
| `01_data_pipeline.ipynb` | `src/data_pipeline.py` | 1 — dati |
| `02_graph_construction.ipynb` | `src/chess_encoding.py` + `src/dataset.py` | 2 — grafo |
| `03_gnn_model_training.ipynb` | `src/model.py` + `src/train.py` | 3 — training |
| `04_llm_baseline_comparison.ipynb` | `src/llm_baseline.py` + `src/compare.py` | 4 — LLM + confronto |

Rispetto ai notebook, ho eliminato una duplicazione: le funzioni di encoding
(`square_node_features`, `build_edges`, `get_position_and_target`) erano
copiate identiche sia in `02_graph_construction.ipynb` sia in
`03_gnn_model_training.ipynb`. Ora vivono in un solo file
(`src/chess_encoding.py`), importato da entrambi i moduli che ne hanno
bisogno — così non rischi che le due copie si scollegassero in futuro.

---

## 1. Struttura della cartella

```
chess_timed_gnn/
├── config.py                    # TUTTI i percorsi e iperparametri, in un solo posto
├── requirements.txt
├── README.md                    # questo file
├── src/
│   ├── data_pipeline.py         # Fase 1: download, filtro, validazione, timing, held-out, split, anti-leakage
│   ├── held_out.py              # Fase 1: costruzione held-out (Polgár n=1-3 + Lichess n=4-5)
│   ├── chess_encoding.py        # Fase 2: FEN -> feature nodi/archi (funzioni condivise)
│   ├── dataset.py               # Fase 2: ChessPuzzleGraphDataset (PyTorch Geometric)
│   ├── model.py                 # Fase 3: architettura TimedGNN
│   ├── train.py                 # Fase 3: loss, training loop, ablation con/senza timing
│   ├── llm_baseline.py          # Fase 4: prompting LLM, parsing, verifica soluzione
│   ├── compare.py               # Fase 4: GNN vs LLM, McNemar, grafico, file cieco
│   └── stats.py                 # test di McNemar (chi2 + binomiale esatto), condiviso
├── scripts/                     # entry point da riga di comando, uno per fase
│   ├── run_pipeline.py          # -> Fase 1 (include la costruzione dell'held-out)
│   ├── run_graph_check.py       # -> sanity check Fase 2 (facoltativo ma consigliato)
│   ├── run_training.py          # -> Fase 3
│   └── run_llm_comparison.py    # -> Fase 4
├── docs/                        # risposte alle domande del PDF, patch, guida Colab
├── data/                        # output Fase 1 (creata automaticamente)
├── checkpoints/                 # modelli salvati (inizialmente vuota: il training va rifatto)
└── results/                     # CSV, grafici, file di valutazione (creata automaticamente)
```

**Come sono collegati i pezzi (flusso dati):**

```
scripts/run_pipeline.py
   └─> src/data_pipeline.py  (+ src/held_out.py)
          └─> salva data/puzzles_{train,val,test}.parquet
          └─> salva data/puzzles_held_out.parquet         (misto Polgar + Lichess)
          └─> salva data/puzzles_held_out_lichess.parquet (Lichess-only, 30 per N1-N5)
          └─> salva results/n_distribution.csv            (distribuzione dei matti in N)
                     │
                     ▼
scripts/run_graph_check.py  (facoltativo, verifica veloce)
   └─> src/dataset.py (usa src/chess_encoding.py)
                     │
                     ▼
scripts/run_training.py
   └─> src/train.py (usa src/dataset.py + src/model.py)
          └─> salva checkpoints/timed_gnn_tv2_best.pt
          └─> salva checkpoints/no_timing_gnn_best.pt
                     │
                     ▼
scripts/run_llm_comparison.py
   └─> src/llm_baseline.py  (valuta l'held-out: chiama l'API dell'LLM, produce results/llm_evaluation_raw.csv)
   └─> src/compare.py       (carica checkpoints/timed_gnn_tv2_best.pt, fa girare la GNN
                              sugli stessi puzzle, McNemar, grafico, file valutazione cieca)
```

Ogni script si limita a chiamare le funzioni della libreria in `src/`: la
logica vera vive tutta in `src/`, così puoi anche importare i moduli
direttamente in un notebook Colab se preferisci lavorare in modo interattivo
(vedi punto 4 più sotto).

---

## 2. Cosa devi caricare/fare TU (io non posso farlo)

1. **Ambiente con GPU**: consiglio Google Colab (Runtime > Cambia tipo di
   runtime > GPU), come nei notebook originali. Il training su CPU
   funziona ma è molto più lento.

2. **Due set held-out per il confronto finale GNN vs LLM** — *non devi fare
   nulla a mano*: vengono costruiti automaticamente da `python -m scripts.run_pipeline`.

   - Held-out misto: **150 puzzle**, Polgar per N1-N3 e Lichess per N4-N5.
   - Held-out Lichess-only: **150 puzzle**, 30 per ogni N da 1 a 5, disgiunti dal misto.

   | n | Fonte | Come |
   |---|---|---|
   | 1, 2, 3 | Collezione László Polgár (1994), PGN pubblico su GitHub (scaricato in automatico) | 30 per n, campionati a caso (seed fisso) tra i problemi la cui linea è stata riverificata con python-chess; scartati quelli la cui posizione compare già in Lichess |
   | 4, 5 | Puzzle Lichess `mateIn4` / `mateIn5` | 30 per n, **stratificati per rating** (5 fasce di quantili, stessa quota per fascia, seed fisso), estratti dal pool **prima** dello split |

   **Niente leakage.** L'ordine delle operazioni è: filtro e validazione →
   estrazione di entrambi gli held-out disgiunti → rimozione dal pool di ogni puzzle con lo stesso
   `PuzzleId` o la stessa posizione (FEN) → solo dopo, split 80/10/10. A fine
   pipeline un controllo esplicito (`check_no_leakage`) si ferma con un errore
   se trova anche un solo puzzle dell'held-out in train/val/test; lo stesso
   controllo (su ID e FEN) viene rifatto all'inizio del training per entrambi gli held-out.

   **Timing dei puzzle Polgár.** Polgár non ha rating né popolarità, quindi
   per il tempo di pensiero simulato si usa lo stesso meccanismo dei puzzle
   Lichess (`simulate_think_time`) con `Rating` e `Popularity` **imputati**:
   la mediana dei puzzle Lichess con lo stesso n. I valori sono salvati nel
   parquet (colonne `Rating`, `Popularity`), la colonna `source` indica la
   fonte (`polgar` / `lichess`).

   **Limite da dichiarare nel report.** n=1-3 provengono da Polgár e n=4-5
   da Lichess: nel confronto per n, la profondità è quindi confusa con la
   fonte (stile e difficoltà dei puzzle). Per n=4-5 l'held-out è inoltre
   della stessa famiglia del training (Lichess), a differenza di n=1-3. Con
   30 puzzle per n, il test di McNemar per singolo n ha poca potenza: i
   risultati per n vanno letti come indicativi.

   Se per un n Polgár o Lichess avessero meno di 30 puzzle disponibili, la
   pipeline lo segnala e usa quelli che ci sono.

3. **API key per la baseline LLM** (Fase 4). Non va MAI scritta nel
   codice. Il backend di default è **Gemini** (`LLM_BACKEND="gemini"` in
   `config.py`), che ha un livello gratuito generoso:

   - **Gemini (gratuito, default)**: vai su
     [aistudio.google.com](https://aistudio.google.com) -> accedi con un
     account Google -> "Get API key" -> "Create API key". Poi:
     ```bash
     export GEMINI_API_KEY="AIza..."
     python -m scripts.run_llm_comparison
     ```
     Installa il pacchetto necessario: `pip install google-genai`.
     Limiti indicativi del livello gratuito (verifica sempre i valori
     aggiornati su
     [ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)):
     poche richieste al minuto e alcune centinaia al giorno — per la
     Fase 4 va benissimo, dato che `LLM_N_SAMPLES` di default è 20.

   - **Anthropic (Claude)**: `export LLM_BACKEND="anthropic"` e
     `export ANTHROPIC_API_KEY="sk-ant-..."`.

   - **OpenAI (GPT)**: `export LLM_BACKEND="openai"` e
     `export OPENAI_API_KEY="sk-..."`.

   Su Colab, usa i **Secrets** (icona a forma di chiave nella sidebar), es.
   `os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")`, oppure
   lo script te la chiederà in modo sicuro (`getpass`) se non la trova
   nell'ambiente.

4. **Nome del modello LLM aggiornato**: in `config.py` la costante
   `LLM_MODEL_NAME` è impostata su un modello Gemini recente del livello
   gratuito (`gemini-2.5-flash`), ma verifica sempre il nome corrente su
   `https://ai.google.dev/gemini-api/docs/models` (Gemini),
   `https://docs.claude.com` (Anthropic) o
   `https://platform.openai.com/docs/models` (OpenAI) — questi nomi
   cambiano nel tempo. Puoi anche sovrascriverla senza toccare il codice:
   ```bash
   export LLM_MODEL_NAME="nome-modello-attuale"
   ```

5. **Decidere quanti puzzle usare per i primi test.** In `config.py`:
   - `SUBSET_SIZE = 20000` limita il training a un sottoinsieme, per
     verificare velocemente che tutto funzioni prima del training
     "vero" (che con centinaia di migliaia di puzzle e 50-100 epoche può
     richiedere ore). Metti `SUBSET_SIZE = None` per usare tutti i dati.
   - `NUM_EPOCHS = 10` va aumentato a 50-100 per il training completo,
     come indicato dalla metodologia del progetto.
   - `LLM_N_SAMPLES = 20` va aumentato a 100-200 per la valutazione
     finale (attenzione: ogni puzzle = una chiamata API a pagamento).

---

## 3. Come eseguire il progetto (ordine obbligato)

Dalla cartella principale `chess_timed_gnn/`:

```bash
# 0. installa le dipendenze (torch/torch_geometric vanno installati
#    con il comando specifico per la GPU del tuo runtime Colab, es.:
#    !pip install torch --index-url https://download.pytorch.org/whl/cu121
#    !pip install torch_geometric
#    poi installa il resto:
pip install -r requirements.txt

# 1. Fase 1 — scarica e prepara i dati (~250-300MB, alcuni minuti)
python -m scripts.run_pipeline

# 2. (facoltativo ma consigliato) verifica veloce dell'encoding del grafo
python -m scripts.run_graph_check

# 3. Fase 3 — training della GNN (con e senza timing, per l'ablation study)
python -m scripts.run_training

# 4. Fase 4 — baseline LLM + confronto statistico (richiede API key, vedi punto 2.3)
export ANTHROPIC_API_KEY="sk-ant-..."
python -m scripts.run_llm_comparison
```

Ogni fase legge i file prodotti dalla fase precedente e si ferma con un
errore chiaro (`FileNotFoundError`) se mancano — non serve indovinare
l'ordine giusto, te lo dice lo script stesso.

### Se preferisci lavorare in un notebook Colab invece che da riga di comando

Basta importare i moduli invece di riscrivere il codice, per esempio:

```python
import sys
sys.path.append("/content/chess_timed_gnn")   # dove hai caricato la cartella

from src.data_pipeline import run_phase1
run_phase1()

from src.train import run_phase3
risultati = run_phase3()
```

---

## 4. Cosa NON ho potuto verificare io stesso (e perché)

**Aggiornamento v6.** La nuova Fase 1 (costruzione dell'held-out, split,
controllo anti-leakage, distribuzione di n) è stata eseguita end-to-end su un
CSV Lichess *sintetico* con il vero `polgar.pgn` scaricato da GitHub:
30 puzzle per n, nessun leakage, lunghezze delle linee coerenti. Non ho potuto
eseguire il training della GNN con dati reali né chiamare l'API dell'LLM:
quelle parti restano da verificare sul tuo Colab (vedi sotto, il consiglio con
`SUBSET_SIZE` piccolo vale ancora).

Il resto di questa sezione descrive la verifica fatta sulle versioni precedenti.

Ho controllato la correttezza sintattica di tutti i file
(`python -m py_compile`, nessun errore) e testato in isolamento — con dati
sintetici — la logica che non dipende da librerie pesanti: estrazione
`mate_n` dai temi, simulazione del tempo di pensiero, split stratificato,
parsing della risposta LLM (regex), e il test di McNemar.

Il mio ambiente di lavoro qui **non ha accesso alla rete** e quindi non
posso installare `torch`, `torch_geometric`, `python-chess`, `zstandard`,
scaricare il database Lichess, né chiamare l'API Claude/GPT: non ho potuto
eseguire end-to-end la costruzione del grafo, il training della GNN, né la
Fase 4. Il codice è comunque una riorganizzazione fedele (stessa logica,
stesse formule, stessi nomi di variabili dove sensato) di quanto già
scritto e presumibilmente testato nei tuoi notebook originali, con in più:
la deduplicazione dell'encoding (punto sopra) e la centralizzazione dei
parametri in `config.py`.

**Prima di lanciare il training "vero" su tutto il dataset**, ti consiglio
di eseguire in ordine `run_pipeline.py` -> `run_graph_check.py` ->
`run_training.py` con `SUBSET_SIZE` piccolo (es. 2000) e `NUM_EPOCHS = 2`,
per verificare in pochi minuti che l'intera catena funzioni sul tuo
runtime Colab, prima di lanciare la versione completa.

---

## 5. Prossimi passi naturali (fuori dallo scope di questo pacchetto)

- Aumentare `hidden_dim`/`num_layers` e portare `NUM_EPOCHS` a 50-100 per
  il training definitivo (metodologia).
- Ripetere la Fase 4 anche con `checkpoints/no_timing_gnn_best.pt` per
  completare l'ablation study end-to-end.
- Far compilare `results/human_eval_blind.csv` per la valutazione umana
  in cieco richiesta dalla metodologia, poi ricongiungere con
  `results/human_eval_blind_WITH_ANSWER_KEY.csv` per l'analisi finale.
- Mettere insieme risultati/grafici delle 4 fasi in un report finale.

---

## Esecuzione a tre stage su Colab (configurazione 300k)

Per la pipeline persistente tra sessioni Colab, consulta **[`docs/GUIDA_COLAB_STAGED.md`](docs/GUIDA_COLAB_STAGED.md)**. La configurazione staged usa `SUBSET_SIZE=300000`, `NUM_EPOCHS=100` e `EARLY_STOPPING_PATIENCE=10` e i tre comandi sono:

```bash
python -u -m scripts.run_staged --stage timed
python -u -m scripts.run_staged --stage no-timing
python -u -m scripts.run_staged --stage compare
```

Esegui un solo stage per volta e conserva `data/`, `checkpoints/` e `results/` nella stessa cartella persistente su Google Drive.
