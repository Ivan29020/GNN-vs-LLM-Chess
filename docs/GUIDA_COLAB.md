> **Pipeline persistente aggiornata:** per la nuova esecuzione a tre stage su Google Drive
> (Timed GNN → GNN senza timing → confronto), con `SUBSET_SIZE=300000`, 100 epoche e
> early stopping a 10 epoche, usa [`GUIDA_COLAB_STAGED.md`](GUIDA_COLAB_STAGED.md).
> Questa guida descrive il flusso storico non staged.

# Guida Colab — run completa da zero, con ripresa multi-giorno

Questa versione (v7) dello zip ha i checkpoint **vuoti** e va riaddestrata: la
prima esecuzione fa il training vero con i valori di `SUBSET_SIZE` e `NUM_EPOCHS`
in `config.py`. **Non riusare checkpoint di versioni precedenti**: sono stati
addestrati su puzzle che ora sono nell'held-out.
Con questi numeri il training prende via diverse ore — è molto probabile
che serva più di una sessione Colab per finire tutto. Questa guida copre
sia la run "tutto in una sessione" sia la ripresa il giorno dopo.

## Perché può servire più di un giorno

Due colli di bottiglia diversi, non confonderli:

- **Il training** non ha un limite di quota: può volerci più tempo di
  quanto Colab tenga la sessione viva (disconnessioni per inattività, o il
  limite di sessione del piano gratuito).
- **La valutazione LLM** (Groq) ha invece una **quota giornaliera fissa**
  (~66 chiamate nei test fatti finora): con 150 puzzle richiesti, quasi
  certamente servono 2-3 giorni solo per quella parte.

Il codice gestisce entrambe le interruzioni con una logica di ripresa, ma
**solo se scarichi e ricarichi le cartelle giuste** — il disco di Colab è
effimero, si svuota quando la sessione finisce.

## 0. Setup (ogni sessione, anche le successive)

```python
from google.colab import files
uploaded = files.upload()   # chess_timed_gnn_patched_v7.zip
```

```python
!unzip -q chess_timed_gnn_patched_v7.zip
%cd chess_timed_gnn
!pip install -q -r requirements.txt
```

**Se è la PRIMA sessione**, salta al punto 1.

**Se stai riprendendo da un giorno precedente**, ricarica qui tutto quello
che avevi scaricato a fine sessione precedente, prima di lanciare
qualunque script:

```python
uploaded2 = files.upload()   # lo zip di backup della sessione precedente (vedi punto 6)
!unzip -oq *.zip -d .        # sovrascrive results/ e checkpoints/ con quelli salvati
```

Verifica cosa hai recuperato:

```python
!ls checkpoints/
!ls results/
```

- Se vedi `timed_gnn_tv2_best.pt.resume.pt` (o l'equivalente no-timing):
  il training si era fermato a metà, riprenderà da lì.
- Se vedi `llm_evaluation_raw.csv` e `llm_eval_sample.csv`: la valutazione
  LLM riprenderà dai puzzle già fatti.

## 1. Dati Lichess + held-out (va rifatto ogni sessione, non persiste)

```python
!python -m scripts.run_pipeline
```

Un solo comando: scarica Lichess e `polgar.pgn`, valida, **estrae l'held-out
(30 puzzle per n=1..5: Polgár n=1-3, Lichess n=4-5)**, lo toglie dal pool,
fa lo split, controlla che non ci sia leakage e **stampa la distribuzione
N1..N5** di train/val/test/held-out (salvata in `results/n_distribution.csv`).
Controlla subito che la tabella dica 30 per ogni n nella colonna `held-out`.

~10-15 minuti, indipendentemente dal giorno.

> **Se hai già fatto girare la v6 nella stessa sessione**: dopo aver decompresso lo zip v7
> rilancia `run_pipeline` (le colonne dei dati sono cambiate: `rating_norm`). Il modello
> `no_timing_gnn_best.pt` della v6 si riusa; il timed (`timed_gnn_tv2_best.pt`) va
> addestrato di nuovo, e `run_training` lo fa da solo saltando il no-timing.

## 2. Training — riprende da solo se interrotto

```python
!python -m scripts.run_training
```

Tre scenari possibili, gestiti automaticamente:

- **Checkpoint non presenti** (prima sessione): training da zero,
  epoca 1/50.
- **Solo `.resume.pt` presente** (sessione precedente interrotta a metà):
  stampa `Ripresa da checkpoint: riparto dall'epoca N` e continua da lì,
  con la stessa history e lo stesso stato di early stopping.
- **Solo i `.pt` finali presenti, nessun `.resume.pt`** (training già
  completato in una sessione precedente): stampa `TRAINING SALTATO` e
  passa diretto alla valutazione — zero tempo sprecato.

Se per qualunque motivo vuoi forzare un training completo da zero
ignorando tutto: `!python -m scripts.run_training --force-training`.

**Importante**: se questa sessione si interrompe PRIMA che il training
finisca, al passo 6 scarica comunque `checkpoints/` — i file `.resume.pt`
sono lì anche a metà training, salvati dopo ogni epoca.

## 3. Valutazione LLM — riprende da sola se la quota si esaurisce

```python
import os
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
```

```python
!python -m scripts.run_llm_comparison
```

La prima volta salva il campione canonico (150 puzzle = tutto l'held-out) in
`results/llm_eval_sample.csv` — da quel momento resta fisso. Se la quota
Groq si esaurisce a metà, lo script si ferma da solo con un messaggio
chiaro e tutto il fatto fino a quel punto resta salvato in
`results/llm_evaluation_raw.csv`.

**Non mettere `LLM_FRESH_START=1`** quando riprendi un giorno diverso: è
quello che butterebbe via il lavoro già fatto e ti farebbe rifare tutte le
chiamate da capo. Usalo SOLO se cambi volutamente il prompt, le metriche,
o il numero di puzzle richiesti.

## 4. Matrici di confusione, insight e report (dentro lo step 3)

`run_llm_comparison` genera già in automatico
`results/confusion_matrices_by_n.png` e gli insight quantificati. Per il
documento finale:

```python
!python -m scripts.generate_evaluation_report
```

## 5. Blind test umano (una volta sola, a dati completi)

```python
!python -m scripts.analyze_blind_eval
```

da lanciare solo dopo aver fatto compilare `results/human_eval_blind.csv`
a qualcuno — non è legato alla ripresa multi-giorno, fallo per ultimo.

## 6. A FINE SESSIONE, sempre — anche se non hai finito tutto

Questo è il passo che rende possibile riprendere il giorno dopo. Scarica
**tutto**, non solo `results/`:

```python
!zip -qr backup_sessione.zip results/ checkpoints/ data/puzzles_held_out.parquet data/puzzles_train.parquet data/puzzles_val.parquet data/puzzles_test.parquet
from google.colab import files
files.download("backup_sessione.zip")
```

`checkpoints/` include i `.resume.pt` se il training non è finito;
`results/` include `llm_eval_sample.csv` e `llm_evaluation_raw.csv` se la
valutazione LLM non è finita. Senza questo scaricamento, tutto quello che
non è ancora completo va perso quando la VM si ricicla.

La prossima sessione, segui di nuovo da "0. Setup", punto "Se stai
riprendendo", caricando questo stesso `backup_sessione.zip`.

## Problemi frequenti

**"model_not_found" da Groq.** Il modello è stato dismesso: controlla
console.groq.com/docs/models e imposta
`os.environ["LLM_MODEL_NAME"] = "nome-attuale"` prima dello step 3.

**`run_training` dice "TRAINING SALTATO" ma non volevi.** Hai ricaricato
dei checkpoint completi da una sessione precedente. Usa
`--force-training` se vuoi rifare tutto da zero apposta.

**Il training riparte da zero anche se avevi il `.resume.pt`.** Controlla
di averlo ricaricato nel posto giusto: deve stare in
`checkpoints/timed_gnn_tv2_best.pt.resume.pt` (stesso nome, dentro
`checkpoints/`), non in una sottocartella diversa generata da un unzip
annidato — verifica con `!find . -name "*.resume.pt"`.

**"PuzzleId presente nei risultati LLM ma non nel campione".** Il
`llm_eval_sample.csv` caricato non corrisponde al `llm_evaluation_raw.csv`
caricato (vengono da sessioni diverse con parametri diversi). Ricarica
entrambi dallo stesso backup.
