# Checkpoint della pipeline staged

Questa cartella ospita il training separato in due fasi. La configurazione staged di riferimento è `SUBSET_SIZE=300000`, `NUM_EPOCHS=100`, `EARLY_STOPPING_PATIENCE=10`.

- `timed_gnn_tv2_subset300k_balanced_rating_v3_best.pt`: pesi migliori della Timed GNN.
- `timed_gnn_tv2_subset300k_balanced_rating_v3_best.pt.resume.pt`: stato di ripresa aggiornato dopo ogni epoca finché la fase non è completa.
- `timed_gnn_tv2_subset300k_balanced_rating_v3_best.pt.complete.json`: prova che il training timed è terminato (per epoche complete o early stopping).
- `no_timing_gnn_subset300k_balanced_rating_v3_best.pt`: pesi migliori della GNN senza timing.
- `no_timing_gnn_subset300k_balanced_rating_v3_best.pt.resume.pt`: stato di ripresa del training no-timing.
- `no_timing_gnn_subset300k_balanced_rating_v3_best.pt.complete.json`: prova che il training no-timing è terminato.

Dopo un'interruzione, ripeti lo stesso comando di stage: il checkpoint resume fa ripartire dall'ultima epoca salvata. Non cancellare i file `.resume.pt` mentre un training è incompleto. Il programma scrive i checkpoint in modo atomico per limitare il rischio di file parziali.

I checkpoint storici `timed_gnn_tv2_best.pt` e `no_timing_gnn_best.pt` non vengono usati dalla pipeline staged, per evitare di riutilizzare silenziosamente pesi addestrati con un'altra configurazione.
