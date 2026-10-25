# Fix training subset grandi su Colab

## Diagnosi

La versione precedente manteneva in RAM tutti gli oggetti `Data` di PyG in
`ChessPuzzleGraphDataset._data_cache` e tutti gli stati `chess.Board` intermedi.
Con il subset bilanciato da 500.000 puzzle, il log mostra circa 719.000 esempi di
training e 134.000 di validation; conservare grafi completi float32/int64 per
tutti gli esempi può far crescere molto la RAM.

Inoltre, `loss_and_metrics()` calcolava l'argmax con `.item()` dentro un ciclo per
ogni grafo. Su centinaia di migliaia di esempi questo forza molte sincronizzazioni
CPU/GPU inutili.

`^C` indica che il processo ha ricevuto un interrupt (SIGINT/KeyboardInterrupt):
non è un messaggio che il modello ha scelto di terminare. La sola riga `^C` non
permette di distinguere l'interruzione del runtime da una interruzione esterna;
la pressione RAM e il calcolo lento sono però problemi reali del codice precedente
e possono rendere il run vulnerabile ai limiti del runtime.

## Modifiche applicate

- Gli stati della linea vengono memorizzati come FEN + mossa UCI, anziché come
  oggetti `chess.Board` completi per ogni esempio.
- La cache dei grafi ha un limite (`GRAPH_CACHE_MAX_ITEMS = 500_000`) e memorizza
  gli indici degli archi in int16 e gli attributi one-hot in uint8, mantenendo le
  feature dei nodi in float32. I grafi che superano la capienza vengono ricostruiti
  quando servono; la RAM non cresce senza limite.
- L'accuracy per batch usa operazioni scatter vettoriali anziché molte chiamate
  `.item()` in un ciclo Python.
- Soltanto durante la prima epoca eseguita in ciascun run viene mostrata una
  singola barra `tqdm` dal 0% al 100%, condivisa tra training e validation.
  Il postfix mostra fase, esempi elaborati e numero di grafi in cache; si aggiorna
  ogni 500 batch senza creare nuove righe. Dalla seconda epoca eseguita in poi
  viene stampata soltanto la riga riepilogativa a fine epoca.

## Impostazioni consigliate

- Non aumentare `BATCH_SIZE` per risolvere la RAM CPU: il batch size riguarda
  soprattutto il carico per batch e la memoria GPU. Lascia 128 se non hai errori
  CUDA; usa 64 (o 32) solo se compare un errore GPU out-of-memory.
- Parti con `GRAPH_CACHE_MAX_ITEMS = 300_000` se il runtime Colab è configurato
  con poca RAM. Il valore 500.000 è il compromesso predefinito per il subset
  descritto sopra; abbassarlo riduce l'uso di RAM ma aumenta il lavoro di
  ricostruzione dei grafi tra un'epoca e l'altra.
- Controlla che Colab usi GPU dal menu Runtime > Change runtime type; il codice
  stampa `Device: cuda` se una GPU è disponibile e selezionata correttamente.
- Avvia lo script con `python -u -m scripts.run_training --force-training` se
  vuoi forzare il training e vedere i log senza buffering. Se esiste un file
  `.resume.pt` da un run precedente, il codice riprende dall'ultima epoca salvata.

## Interpretare l'eventuale nuovo arresto

- Durante la prima epoca eseguita, se la barra continua ad avanzare, il processo sta lavorando.
- Se compare `torch.cuda.OutOfMemoryError`, riduci `BATCH_SIZE` a 64.
- Se il runtime si riavvia o Colab segnala che la RAM è esaurita, riduci
  `GRAPH_CACHE_MAX_ITEMS` a 200.000–300.000.
- Se compare ancora `^C` senza un traceback o un riavvio, controlla il pannello
  di sistema/esecuzione di Colab: la riga da sola non identifica chi abbia inviato
  l'interrupt.
