# Piano di debugging e risoluzione — GNN timed vs LLM

## Scopo e regola di sicurezza

Questa procedura indaga i risultati attuali senza modificare training, dataset o checkpoint. Non riaddestrare ancora e non eliminare risultati precedenti: prima vanno verificati le linee di riferimento e il criterio di successo. Il nuovo script `scripts/run_diagnostics.py` scrive unicamente in `results/diagnostics/`.

## 1. Prima sessione: installazione/verifica e audit rapido

Dalla root del progetto in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/chess_timed_gnn
```

Assicurati che siano presenti i file `.parquet` in `data/` e i CSV in `results/`. Installa solo le dipendenze mancanti:

```bash
!pip -q install python-chess pyarrow pandas numpy scipy
```

Esegui l'audit completo dei held-out, CSV LLM, predizioni GNN e ablation:

```bash
!python -u -m scripts.run_diagnostics --section all
```

Il report è in `results/diagnostics/summary.txt`. I CSV più importanti sono:

- `data_line_validation.csv`: legalità dei puzzle dei due held-out, matto finale, mosse solver effettive e coerenza con `mate_n`;
- `data_split_leakage.csv`: sovrapposizioni rapide per ID/FEN originale tra held-out e train/validation/test;
- `source_sample_line_validation.csv`: replay dei campioni LLM canonici a partire da FEN e Moves originali;
- `llm_row_audit_mixed.csv` e `llm_row_audit_lichess_only.csv`: valutazione indipendente delle linee grezze dell'LLM, inclusi legalità, matto finale, linea vuota e lunghezze;
- `llm_summary_by_n.csv`: riepilogo LLM per N;
- `gnn_first_error_per_puzzle.csv`: posizione del primo errore rispetto alla linea target, per ogni puzzle del campione test;
- `gnn_move_accuracy_by_step.csv`: accuratezza condizionata per indice della mossa del solver;
- `gnn_error_summary_by_n.csv` e `gnn_first_error_distribution.csv`: errori per profondità e distribuzione dell'indice del primo errore;
- `ablation_paired_overall.csv` e `ablation_paired_by_n.csv`: McNemar esatto, differenza appaiata e bootstrap CI per timed vs no-timing;
- `csv_inventory.csv`: inventario dei CSV dei risultati.

### Audit integrale del dataset

Dopo l'audit rapido, se hai tempo, esegui anche il replay di tutte le linee nei Parquet train/validation/test. Può richiedere diversi minuti:

```bash
!python -u -m scripts.run_diagnostics --section data --full-dataset
```

Questo fa un controllo più forte della contaminazione: oltre a ID e FEN originale, confronta anche la posizione dopo la mossa di apertura (solve-FEN). Non modifica gli split.

## 2. Come interpretare l'audit delle linee

Per una linea di riferimento valida ci aspettiamo:

- `all_moves_legal = True`;
- `ends_in_checkmate = True`;
- `line_length_matches_mate_n = True`.

`line_length_matches_mate_n=False` è un allarme metodologico: la profondità indicata dal tema non coincide con il numero di mosse del solver nella linea memorizzata. Non correggere automaticamente l'etichetta né eliminare questi esempi prima di vedere quanti sono e in quali profondità/fonti compaiono. Se il mismatch è diffuso, va fatta una nuova pipeline con una regola esplicita e poi va riaddestrato il modello per mantenere coerente il benchmark.

Una linea legale che termina in scacco matto dimostra solo che quella linea è una linea vincente. Non dimostra da sola che sia forzata contro tutte le difese: per certificare una soluzione alternativa serve un motore scacchistico/ricerca sul gioco, oppure una verifica formale delle risposte avversarie.

## 3. Interpretare le metriche LLM

Il vecchio campo `proposed_legal` registra che il testo parsato è giocabile secondo uno dei due formati supportati dal valutatore. **Non** prova che il modello abbia risolto il puzzle. Nel report nuovo leggere separatamente:

- `proposal_legal_as_alternating_line`;
- `proposal_ends_in_checkmate_as_alternating_line`;
- `proposal_legal_as_solver_only_with_reference_replies`;
- `proposal_ends_in_mate_with_reference_replies`;
- `exact_full_target` e `exact_solver_moves`.

Le risposte solver-only testate con le risposte dell'avversario prese dalla linea target sono soltanto un controllo diagnostico. Non certificano una soluzione alternativa forzata.

`solution_tag_missing_count` e `empty_proposal_count` aiutano a identificare output senza una riga `SOLUTION:` o non estraibili dal parser. Gli errori di chiamata API che non furono salvati nei vecchi CSV non possono essere ricostruiti retroattivamente. `llm_missing_from_raw_*.csv` identifica gli ID presenti nel campione canonico ma assenti dal CSV grezzo; quelli andranno distinti da output vuoti.

## 4. Interpretare il primo errore della GNN

`gnn_first_error_per_puzzle.csv` confronta le mosse predette con le mosse del solver nella linea registrata. L'indice è 1-based. `first_error_solver_move_index_1based=1` significa prima mossa errata; `NaN` significa che la sequenza predetta coincide esattamente con la sequenza solver target. In `gnn_move_accuracy_by_step.csv`, l'accuracy è **condizionata al fatto che il modello abbia prodotto una predizione a quel passo**: il codice attuale si ferma al primo errore, quindi i passi successivi non vengono predetti per quel puzzle.

Anche qui la corrispondenza esatta al target non riconosce automaticamente linee alternative corrette. La legalità e il matto finale vengono riportati separatamente, senza chiamarli una prova di matto forzato.

## 5. Interpretare l'ablation timing

`ablation_paired_overall.csv` usa lo stesso PuzzleId per i due modelli, McNemar esatto sulle coppie discordanti, differenza di accuratezza e bootstrap percentile CI appaiato. `ablation_paired_by_n.csv` presenta gli stessi dati per profondità; i p-value per N sono esplorativi e includono anche una correzione Holm per i cinque confronti entro metrica.

Non cambiare i pesi temporali se l'intervallo della differenza comprende zero o il vantaggio è piccolo senza replicare il training con più seed. Un singolo split/seed misura solo una realizzazione dell'esperimento.

## 6. Azioni dopo aver ricevuto il report

1. **Se le linee reference non sono legali o non finiscono in matto:** fermarsi e correggere pipeline/dati prima di usare accuracy o fare training nuovo.
2. **Se `mate_n` non coincide spesso con le mosse del solver:** non riutilizzare l'attuale risultato aggregato per profondità come se N fosse corretto. Decidere una regola di filtraggio/riclassificazione e creare un esperimento nuovo, con manifest e checkpoint separati.
3. **Se le linee target sono valide ma le proposte LLM sono vuote o non parsabili:** correggere parser/prompt e rifare solo l'evaluation LLM, usando un nuovo file/versione di risultati per non mescolare protocolli.
4. **Se l'LLM produce linee legali che non terminano in matto:** rafforzare la generazione e verificare eventuali alternative con un motore; non contare `proposed_legal` come solved.
5. **Se il divario Polgar resta dopo i controlli:** fare error analysis per posizione e valutare training multi-source/augmentation; mantenere Polgar fuori dal train per misurare ancora la generalizzazione.
6. **Solo dopo:** confrontare più seed per timed/no-timing e decidere se timing, architettura o distribuzione del training richiedono modifiche.

## 7. Cosa inviare per il debugging

Per evitare di condividere segreti o checkpoint pesanti, invia:

1. `results/diagnostics/summary.txt`;
2. l'intera cartella `results/diagnostics/` compressa in ZIP;
3. se il report mostra mismatch, `data_line_validation.csv`, `source_sample_line_validation.csv` e `gnn_error_summary_by_n.csv`;
4. gli eventuali errori completi di Colab da quando parte `run_diagnostics.py`.

Non inviare `GROQ_API_KEY`, token API, credenziali o file Secrets. Non c'è bisogno di trasferire i checkpoint per questa prima fase di debugging.
