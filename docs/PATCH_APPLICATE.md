# Patch applicate — v7 (rispetto alla v6)

## v7.1 Timing proxy v2

**File**: `config.py` (`TIMING_WEIGHTS` e costanti), `src/chess_encoding.py` →
`compute_think_time_sec`, `src/data_pipeline.py` → `compute_rating_stats`,
`add_rating_norm`, `src/dataset.py`, `src/train.py`

Prima: `T = 3 + (rating-1000)/400 - sconto popolarità + rumore N(0, 0.5)`, uno
scalare per puzzle, uguale su tutti i nodi e lungo tutta la linea. Il rumore
rendeva il valore quasi un identificativo del puzzle (possibile memorizzazione) e
non portava informazione sulla mossa da giocare. Nella run v6 il modello senza
timing è risultato migliore del timed (sequenza completa 0.630 contro 0.613).

Ora: `D = 0.25·R + 0.20·B + 0.20·F + 0.35·K`, `T = clip(1 + 9·D, 1, 10)`:

- **R** rating normalizzato `clip((Rating-P10)/(P90-P10), 0, 1)`, percentili dal
  *training set* (salvati in `data/timing_stats.json`); nel parquet è la colonna
  `rating_norm`, che sostituisce `simulated_think_time_sec`;
- **B** numero di mosse legali della posizione, in scala log tra 6 e 41;
- **F** numero di mosse legali che danno scacco (candidate forzanti), cap a 5;
- **K** mosse del solver ancora da giocare fino al matto, su 1-5.

Proprietà: deterministico (nessun rumore), diverso a ogni ply della linea,
nessun motore scacchistico (quindi nessun vantaggio non dichiarato rispetto
all'LLM). I pesi sono valori iniziali, non appresi. Il valore è calcolato al
momento della costruzione del grafo, identico in training e in inferenza.

**Limite da dichiarare**: B è già implicito nel grafo (gli archi sono le mosse
legali). K è l'unica componente che dà alla GNN qualcosa che non può vedere dalla
scacchiera; è un'informazione che anche il prompt dell'LLM riceve ("mate in n"),
quindi il confronto resta alla pari, ma un eventuale miglioramento del modello
timed va attribuito in parte alla profondità e non al "tempo". Per isolare il
timing puro: `TIMING_WEIGHTS["K"] = 0` e riaddestra il solo modello timed.

**Checkpoint**: il timed si chiama `timed_gnn_tv2_best.pt` (il vecchio
`timed_gnn_best.pt` non viene caricato: stesse dimensioni, significato diverso).
Il `no_timing_gnn_best.pt` della v6 non dipende dal timing e dallo split (seed
fissi), quindi si può riusare dopo aver rilanciato `run_pipeline`; il controllo
anti-leakage verifica comunque i dati.

**Dopo `run_pipeline`**: i parquet hanno `rating_norm` al posto di
`simulated_think_time_sec`; `llm_eval_sample.csv` viene aggiornato da solo.

---

# Storico: patch della v6

# Patch applicate — v6 (rispetto alla v5)

Sei modifiche. **Toccano i dati, quindi i checkpoint della v5 NON sono più
validi** (vedi in fondo). Le sette patch della v5 sono riportate sotto, come storico.

## v6.1 Held-out: 30 puzzle per n da 1 a 5 (Polgár n=1-3, Lichess n=4-5)

**File**: `src/held_out.py` (nuovo), `src/data_pipeline.py` → `run_phase1`, `config.py`

- n=1, 2, 3: 30 problemi Polgár per n (campione casuale con seed fisso tra
  quelli riverificati con python-chess; scartati quelli la cui posizione è
  già in Lichess).
- n=4, 5: 30 puzzle Lichess per n, **stratificati per rating** (5 fasce di
  quantili, stessa quota per fascia, seed fisso).
- **Ordine**: i 60 puzzle Lichess dell'held-out sono estratti *prima* dello
  split e tolti dal pool per `PuzzleId` **e** per FEN (posizione di partenza e
  posizione da risolvere). Lo split 80/10/10 si fa dopo, sul pool ridotto.
- `check_no_leakage` verifica a fine pipeline che nessun puzzle dell'held-out
  sia in train/val/test, e si ferma con un errore altrimenti. Una versione
  leggera dello stesso controllo (su file) gira all'inizio di `run_phase3`.
- **Timing Polgár**: Polgár non ha Rating/Popularity; vengono imputati con la
  mediana Lichess dello stesso n e poi si applica la stessa
  `simulate_think_time` dei puzzle Lichess (prima ricadeva su un default
  identico per tutti). Colonna `source` (`polgar`/`lichess`) nel parquet.
- **Eliminato tutto WTHarvey**: `scripts/build_held_out_wtharvey.py`, e anche
  `build_held_out_polgar.py`, `build_held_out_dataset.py` e
  `fetch_held_out_dataset.py` (Kaggle), sostituiti da `src/held_out.py` chiamato
  da `run_pipeline`. Rimossa la dipendenza `kagglehub`.

## v6.2 Distribuzione dei matti in N

**File**: `src/data_pipeline.py` → `print_n_distribution`, `src/train.py`, `src/dataset.py`

`run_pipeline` stampa e salva (`results/n_distribution.csv`) i puzzle N1..N5
di train/val/test/held-out. `run_training` ripete la tabella per i dati
effettivamente usati (dopo `SUBSET_SIZE`) e stampa anche il numero di *esempi*
(posizione, mossa) per n.

## v6.3 Una sola metrica LLM

**File**: `src/llm_baseline.py`, `src/compare.py`, `src/insights.py`, `src/report.py`

Rimossa `solver_moves_correct` e il McNemar "di controllo" con la metrica
stretta. Resta `full_sequence_correct`: vera se la risposta coincide con la
linea alternata o con le sole mosse del solver (la GNN predice solo le
proprie), cioè la stessa definizione della vecchia "permissiva". Un CSV
`llm_evaluation_raw.csv` di una versione precedente viene convertito alla
lettura. Il grafico e `accuracy_by_n.csv` riportano anche la fonte (Polgár/Lichess)
di ogni n.

## v6.4 Validazioni ridondanti eliminate

- L'held-out non viene più validato due volte (prima negli script di
  costruzione, poi di nuovo in `load_held_out_if_present`): Polgár è validato
  una volta in `parse_polgar_pgn`, i puzzle Lichess dell'held-out vengono dal
  pool già validato.
- La deduplicazione non è più un passo a parte con un file intermedio: è
  inclusa nell'estrazione.
- `run_pipeline` si lancia **una volta** (prima andava lanciato due volte,
  con tre script in mezzo).
- `validate_dataframe` e `add_simulated_timing` usano cicli su liste / numpy
  invece di `df.apply(axis=1)`, molto più veloci su centinaia di migliaia di
  puzzle. La posizione "da risolvere" si ricava nello stesso passaggio della
  validazione, senza replay aggiuntivi.

## v6.5 Documentazione

`README.md`, `docs/GUIDA_COLAB.md` e `checkpoints/README.md` aggiornati.
`docs/RISPOSTE_DOMANDE_PROGETTO.md` ha un avviso in testa: i numeri della
"run 1" sono di versioni precedenti e vanno rigenerati. I notebook in
`notebooks_originali/` sono lo storico da cui è nato il codice in `src/`:
non sono stati aggiornati, la pipeline valida è quella in `src/` + `scripts/`.

## Checkpoint

**Vanno cancellati e il training rifatto**: l'held-out v6 contiene 60 puzzle
Lichess (n=4-5) che nella v5 erano nel training, quindi i vecchi pesi li
hanno visti. `run_phase3` ferma il training con un errore se i file di dati
sono quelli vecchi.

---

# Storico: patch della v5

Sette modifiche, tutte tracciabili. Nessuna tocca l'architettura del modello,
quindi **i checkpoint della sessione precedente restano validi** (vedi in
fondo).

---

## 1. Bug: tutti i matti in 1 dell'held-out venivano scartati

**File**: `src/data_pipeline.py` → `validate_puzzle`

Prima, per ogni sorgente:

```python
if len(moves) < 2:
    return False, None
```

Ora la soglia dipende dal formato della sorgente:

```python
min_len = 2 if row.get("has_opening_move", True) else 1
if len(moves) < min_len:
    return False, None
```

I puzzle Lichess hanno una mossa di apertura dell'avversario prima della
linea, quindi anche un matto in 1 ha 2 mosse; i puzzle classici no, e un
matto in 1 ha una mossa sola. La vecchia soglia li eliminava in blocco: 306
su 2.523.

Aggiunta anche una diagnostica in `validate_dataframe` che ripartisce gli
scarti per valore di n e avvisa quando un intero gruppo viene scartato al
100%, che è la firma tipica di un bug di formato.

**Effetto**: l'held-out passa da 2.217 a 2.523 problemi e recupera n = 1,
cioè il caso più favorevole all'ipotesi del progetto.

## 2. Il campione di valutazione si fermava a 46 puzzle invece di 100-200

**File**: `src/llm_baseline.py` → nuova `allocate_quota` + `load_evaluation_set`
**File**: `config.py` → `LLM_N_SAMPLES` da 100 a 150

Prima la quota era fissa per gruppo (`n_samples // n_gruppi`) e quella non
spendibile dai gruppi rari andava persa. Ora un round-robin la redistribuisce
ai gruppi che hanno ancora materiale.

Ripartizione attesa con l'held-out corretto e 150 richiesti:
n=1 → 46, n=2 → 45, n=3 → 45, n=4 → 8, n=5 → 1, n=6 → 3, n=7 → 2. Totale 150.

L'output ora stampa anche quanti puzzle erano disponibili per ogni n e
avvisa quando il campione resta sotto il richiesto, spiegando perché.

## 3. Il confronto con l'LLM non era alla pari (la patch più importante)

**File**: `src/llm_baseline.py` → `PROMPT_TEMPLATE`, `evaluate_llm_solution`

Il target di riferimento è la linea completa alternata (mossa del solver,
risposta avversario, ...) ma il prompt non lo diceva. Una risposta corretta
che elencava le sole mosse del solver veniva segnata sbagliata **e** illegale.
Da qui la accuracy di sequenza a 0.000 e il 45,7% di "risposte legali" della
run 1: artefatti di formato.

Tre interventi:

- **prompt esplicito** sul formato della riga `SOLUTION:`, con regole
  numerate e un esempio di formattazione per un matto in 2 (riguarda la
  punteggiatura, non la soluzione scacchistica: resta zero-shot);
- **doppia metrica**: `full_sequence_correct` (stretta) e
  `solver_moves_correct` (permissiva, accetta anche le sole mosse del
  solver). Il confronto principale con la GNN usa la permissiva, che è
  l'unica che confronta la stessa quantità nei due modelli;
- **legalità** verificata sotto entrambe le interpretazioni, con una nuova
  colonna `interpretation` che dice quale il modello ha usato.

## 4. Statistica: McNemar esatto, e McNemar anche sull'ablation

**File nuovo**: `src/stats.py`
**File**: `src/compare.py`, `src/train.py`

- Il p-value ora viene calcolato anche in forma **binomiale esatta**, valida
  con qualunque numero di discordanze. Sui dati della run 1 (b=7, c=0):
  chi-quadro p = 0.0233, esatto **p = 0.0156**. Il codice segnala quando
  b + c < 25 e indica quale usare.
- L'ablation timing/no-timing, che prima riportava il delta nudo, ora ha il
  suo McNemar appaiato sul test set interno, più una tabella
  (`results/ablation_timing.csv`) e un grafico
  (`results/ablation_timing_by_n.png`).

## 5. Punteggi di confidenza ed efficienza (richiesti dal PDF)

**File**: `src/train.py` → `solve_puzzle_with_gnn`; `src/compare.py`

La GNN ora restituisce, oltre alle mosse: la confidenza di ciascuna mossa
(softmax sulle sole mosse legali della posizione), le tre alternative
migliori per la prima mossa e il tempo di soluzione. `efficiency_comparison`
confronta i tempi dei due modelli.

Il blind test riceve una guida GNN **testuale con confidenze** invece di una
stringa UCI nuda: altrimenti il voto di "concisione" avrebbe misurato il
formato e non la qualità della guida.

## 6. Riuso dei checkpoint: il training si può saltare

**File**: `src/train.py` → `load_checkpoint_if_compatible`, `run_phase3`
riscritta; `scripts/run_training.py`; `config.py` → `SKIP_TRAINING_IF_CKPT_EXISTS`

Se in `checkpoints/` ci sono già `timed_gnn_best.pt` e
`no_timing_gnn_best.pt` compatibili con l'architettura attuale, il training
viene saltato e si passa direttamente alla valutazione. Se i pesi non sono
compatibili il caricamento fallisce con un messaggio esplicito e il training
riparte da zero, invece di produrre numeri senza senso in silenzio.

## 7. Scraper WTHarvey: più tentativi e diagnostica

**File**: `scripts/build_held_out_wtharvey.py`

La resa di verifica era del 3,4% (52 puzzle su 1.521 candidati), ed è il
motivo per cui l'held-out è quasi privo di n ≥ 4. Il parser ora prova più
segmentazioni del testo (linea principale, testo completo, ogni diramazione
separatamente) e, soprattutto, stampa i motivi di scarto aggregati e i primi
esempi falliti con il testo grezzo, così il problema è diagnosticabile in
pochi minuti invece che a occhio.

---

## Altre modifiche minori

- `config.py`: nuovo flag `LLM_FRESH_START` (variabile d'ambiente
  `LLM_FRESH_START=1`) per ignorare `results/llm_evaluation_raw.csv` e
  ripartire da zero. **Va usato dopo ogni modifica del prompt o delle
  metriche**, altrimenti la logica di ripresa salta i puzzle già valutati con
  il prompt vecchio e il CSV mescola due esperimenti. La run precedente viene
  spostata in `llm_evaluation_precedente.csv`, non cancellata.
- Nuovi file di output per il report: `results/paired_gnn_vs_llm.csv`,
  `results/accuracy_by_n.csv`, `results/ablation_timing.csv`,
  `results/ablation_timing_by_n.png`.
- `plot_accuracy_by_n` avvisa quando un valore di n ha meno di 10 puzzle, per
  evitare di riportare accuracy costruite su 1-3 casi come se fossero
  risultati.

---

## Validità dei checkpoint inclusi

Nello zip, in `checkpoints/`, ci sono i pesi della sessione precedente:
`timed_gnn_best.pt` (val accuracy per mossa 0.832, epoca 34) e
`no_timing_gnn_best.pt` (0.827, epoca 22). Sono `state_dict` puliti, 5 layer,
hidden 256.

**Restano validi** perché nessuna patch tocca l'architettura né l'encoding
del grafo. Tutte le modifiche riguardano la preparazione dei dati di
valutazione, il prompt, le metriche e la statistica — nessuna delle quali
entra nei pesi.

**Vanno buttati e il training rifatto** solo se cambi una di queste:
`HIDDEN_DIM`, `NUM_LAYERS`, `DROPOUT`, oppure `NODE_FEAT_DIM` /
`EDGE_FEAT_DIM` / la struttura del grafo in `src/chess_encoding.py`. Il
controllo è automatico: `load_checkpoint_if_compatible` rifiuta i pesi
incompatibili e lo dice.

Attenzione a un punto: i checkpoint riflettono un training fatto **senza** i
matti in 1 nell'held-out — irrilevante, perché l'held-out non entra nel
training, che usa solo Lichess. Il training set non è cambiato affatto.
