# Guida Colab — pipeline a tre stage, persistente su Google Drive

Questa modalità divide l'esperimento in tre comandi indipendenti:

1. `timed`: prepara i dataset solo se mancano e addestra la Timed GNN.
2. `no-timing`: addestra la GNN senza la feature temporale.
3. `compare`: esegue l'ablation sul test set e il confronto Timed GNN vs LLM.

Configurazione inclusa: `SUBSET_SIZE=300000`, `NUM_EPOCHS=100`, `EARLY_STOPPING_PATIENCE=10`. Dopo aver escluso entrambi gli held-out, la pipeline seleziona un subset TOTALE di 300.000 puzzle: quote bilanciate per `mate_n`, poi fasce quantili globali del rating dentro ciascun `mate_n`. Se i gruppi rari non possono raggiungere la quota, quella parte viene ridistribuita. Solo dopo il campionamento il subset viene diviso in 80% train (240.000), 10% validation (30.000) e 10% test (30.000), stratificando congiuntamente per profondità e fascia di rating.

## Una sola configurazione persistente su Drive

1. Carica lo ZIP del progetto in `MyDrive`.
2. Monta Drive e decomprimi lo ZIP una sola volta in `MyDrive`. Il contenuto deve essere in `MyDrive/chess_timed_gnn/`.
3. Conserva sempre `data/`, `checkpoints/` e `results/` nella stessa cartella su Drive. Non lavorare su una copia sotto `/content`, perché è temporanea.

Esempio, se hai caricato lo ZIP con il nome `chess_timed_gnn_staged_300k_balanced_rating_v3_two_heldouts.zip` (o il nome del tuo archivio aggiornato):

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!unzip -oq /content/drive/MyDrive/chess_timed_gnn_staged_300k_balanced_rating_v3_two_heldouts.zip -d /content/drive/MyDrive/
%cd /content/drive/MyDrive/chess_timed_gnn
```

Esegui l'estrazione iniziale una volta. Se devi aggiornare il codice in una cartella già usata, estrai lo ZIP aggiornato sovrascrivendo i file di codice ma **non cancellare** `data/`, `checkpoints/` o `results/`.

Se nella cartella esistono tentativi staged precedenti e vuoi partire da un esperimento davvero pulito, esegui UNA SOLA VOLTA questa cella prima dello Stage 1. Cancella soltanto checkpoint e risultati staged, non i Parquet già preparati; quindi evita di rieseguirla nei giorni successivi, altrimenti elimineresti i progressi da riprendere.

```bash
!rm -f checkpoints/timed_gnn_tv2_subset300k_balanced_rating_v3_best.pt* checkpoints/no_timing_gnn_subset300k_balanced_rating_v3_best.pt*
!rm -f results/staged_pipeline_state.json results/staged_ablation_balanced300k_v3_complete.json results/staged_comparison_balanced300k_v3_complete.json results/staged_comparison_held_out_mixed_balanced300k_v3_complete.json results/staged_comparison_held_out_lichess_balanced300k_v3_complete.json
!rm -f results/test_sequence_timed_subset300k_balanced_rating_v3.csv results/test_sequence_no_timing_subset300k_balanced_rating_v3.csv
!rm -f results/paired_gnn_vs_llm_balanced300k_v3.csv results/paired_gnn_vs_llm_held_out_lichess_balanced300k_v3.csv results/ablation_timing_balanced300k_v3.csv
```

Se vuoi invece conservare risultati LLM già validi e riprendere la loro valutazione, non cancellare i relativi CSV.

## Installazione in ogni nuovo runtime Colab

Ogni nuovo runtime va reinstallato; i file su Drive, invece, rimangono.

```bash
%cd /content/drive/MyDrive/chess_timed_gnn
!pip -q install python-chess zstandard pyarrow tqdm scikit-learn requests pandas numpy matplotlib scipy groq
!pip -q install torch_geometric
```

Colab fornisce normalmente già PyTorch con la build adatta al runtime. Per evitare di sostituire accidentalmente la build CUDA, questa guida non reinstalla `torch` genericamente. Controlla prima che il runtime sia impostato su GPU: **Runtime → Change runtime type → T4 GPU** (o la GPU disponibile).

## Stage 1 — dati + Timed GNN

```bash
%cd /content/drive/MyDrive/chess_timed_gnn
!python -u -m scripts.run_staged --stage timed
```

La pipeline riusa i Parquet soltanto se sono presenti train/validation/test, entrambi gli held-out e il manifest schema v3. Il subset da 300.000 viene selezionato DOPO aver escluso i 300 puzzle riservati e POI diviso 240.000/30.000/30.000; il training non effettua un secondo sottocampionamento. Se i Parquet appartengono alla vecchia metodologia, viene rigenerato lo split. Poi addestra solo la Timed GNN, salva il modello migliore e il checkpoint di ripresa dopo ogni epoca e termina.

Se Colab si disconnette durante il training, il giorno dopo monta Drive, reinstalla le dipendenze e rilancia **lo stesso comando**. Riparte dall'ultima epoca completata; può essere ricalcolata solo l'epoca interrotta prima del salvataggio.

## Stage 2 — GNN senza timing

Dopo che Stage 1 ha stampato `COMPLETATO`:

```bash
%cd /content/drive/MyDrive/chess_timed_gnn
!python -u -m scripts.run_staged --stage no-timing
```

Questa fase verifica che il checkpoint Timed sia completo, quindi addestra solo la variante no-timing. Anche qui, dopo una disconnessione, si rilancia lo stesso comando.

## Stage 3 — confronto e risultati

Solo dopo che Stage 2 ha stampato `COMPLETATO`:

```bash
%cd /content/drive/MyDrive/chess_timed_gnn
!python -u -m scripts.run_staged --stage compare
```

La prima parte confronta Timed GNN e no-timing sul test set interno. La valutazione sequenza completa conserva i risultati a blocchi in `results/test_sequence_*_subset300k_balanced_rating_v3.csv` e li riusa se Colab si interrompe. Poi esegue due confronti appaiati separati: prima sul held-out misto, poi sul held-out solo Lichess, con output versionati `*_balanced300k_v3.csv`. Le risposte LLM, i campioni, i grafici e le tabelle delle due valutazioni vengono mantenuti in file distinti e il comando si può rilanciare dopo una disconnessione.

Per il backend predefinito Groq serve `GROQ_API_KEY`. È preferibile salvarla in **Colab → Secrets** e caricarla senza scriverla nel notebook:

```python
from google.colab import userdata
import os
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
```

Se la valutazione LLM resta incompleta (ad esempio per quota giornaliera), il comando termina senza segnare il confronto come completo. Il giorno dopo rilancia lo stesso comando: il CSV `results/llm_evaluation_raw.csv` contiene le risposte già ottenute e quelle non ancora presenti vengono riprovate.

Se cambi backend, modello o prompt LLM e vuoi iniziare un esperimento nuovo, esegui **una sola volta** `LLM_FRESH_START=1 python -u -m scripts.run_staged --stage compare`; le risposte precedenti vengono archiviate. Nelle sessioni successive rimuovi `LLM_FRESH_START=1`, altrimenti la baseline ripartirebbe sempre da zero.

## Output e ripresa

- `checkpoints/*_subset300k_balanced_rating_v3_best.pt`: pesi della migliore epoca, addestrati sui nuovi split senza i due held-out.
- `checkpoints/*.resume.pt`: stato completo per continuare il training; rimane solo durante una fase non ancora completata.
- `checkpoints/*.complete.json`: marker che prova il completamento del training.
- `results/staged_pipeline_state.json`: stato delle fasi.
- `results/test_sequence_*_subset300k_balanced_rating_v3.csv`: valutazione sequenziale progressiva sul test.
- `results/llm_evaluation_raw.csv`: risultati LLM progressivi.
- Gli altri CSV e grafici di confronto sono in `results/`.

Lascia la cartella del progetto su Drive invariata tra i giorni. Non copiare solo il checkpoint `.pt`: la ripresa richiede anche i relativi `.resume.pt` quando presenti e la cartella dati/risultati. Se modifichi gli iperparametri dopo aver iniziato, il programma si ferma per evitare di mescolare esperimenti: crea una nuova cartella progetto per un esperimento diverso.

La barra di training appare solo durante la prima epoca eseguita nella sessione; le epoche successive stampano esclusivamente la riga di metriche a fine epoca. La valutazione sequenziale e la baseline LLM usano una sola barra di progresso ciascuna, invece di stampare una riga per ogni elemento.
