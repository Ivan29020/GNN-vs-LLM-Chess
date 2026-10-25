# Risposte alle domande del progetto

Documento di riferimento per il report finale. Copre le due domande di
ricerca del PDF, il protocollo di confronto (sezione 3) e gli esperimenti
(sezione 4), più le domande pratiche emerse leggendo l'output della prima
run.

> **AVVISO v6.** Questo documento è stato scritto per la v5. Nella v6 l'held-out
> è cambiato (30 puzzle per n=1..5: Polgár n=1-3, Lichess n=4-5; niente
> WTHarvey), la valutazione LLM ha una sola metrica di sequenza completa
> (niente stretta/permissiva) e i checkpoint vanno riaddestrati. Tutti i numeri
> e i riferimenti a WTHarvey, ai 2.523 problemi dell'held-out, alla metrica
> "permissiva/stretta" e ai checkpoint inclusi sono **storici**: vedi
> `docs/PATCH_APPLICATE.md` e rigenera i risultati con la v6/v7. **Nella v7** il timing non è più
> `rating+popolarità+rumore` ma un proxy v2 per posizione (vedi
> `docs/PATCH_APPLICATE.md`): il paragrafo sul timing qui sotto descrive la formula v5.

I numeri citati come "run 1" vengono dalla prima esecuzione completa, quella
**precedente alle patch**. Dove una patch li invalida è scritto
esplicitamente: vanno rigenerati prima di finire nel report.

---

## Parte A — Le due domande di ricerca

### Domanda 1: per quali valori di n la GNN dà una guida superiore all'LLM?

**Ipotesi del PDF**: la GNN vince per n piccoli (1-3), dove la struttura del
grafo cattura direttamente il motivo tattico; l'LLM recupera per n grandi
(4-10), dove conta il ragionamento linguistico.

**Come il progetto risponde**: accuratezza di sequenza completa, stratificata
per n, sullo stesso identico campione held-out per i due modelli (confronto
appaiato), con test di McNemar e grafico a barre per n.
Codice: `src/compare.py` → `plot_accuracy_by_n`, `mcnemar_test`.
Output: `results/accuracy_by_n.csv`, `results/gnn_vs_llm_by_n.png`.

**Risultato run 1** (46 puzzle, metrica stretta, *pre-patch*):

| n | accuracy GNN | accuracy LLM | N puzzle |
|---|---|---|---|
| 2 | 0.250 | 0.000 | 16 |
| 3 | 0.188 | 0.000 | 16 |
| 4 | 0.000 | 0.000 | 8 |
| 5 | 0.000 | 0.000 | 1 |
| 6 | 0.000 | 0.000 | 3 |
| 7 | 0.000 | 0.000 | 2 |

Complessivo: GNN 0.152, LLM 0.000. McNemar b=7, c=0, p(chi2)=0.023.

**Come va letto, onestamente**: la direzione è coerente con l'ipotesi — la
GNN prevale a n=2 e n=3 e crolla a zero da n=4 in poi — ma la run 1 **non
dimostra** quell'ipotesi, per tre ragioni:

1. La accuracy dell'LLM a 0.000 ovunque era in buona parte un artefatto del
   formato della risposta, non della sua forza scacchistica (vedi
   Domanda 3 più sotto). Con il prompt corretto e la metrica equa il numero
   dell'LLM salirà: di quanto, lo dirà la run 2.
2. La metà "alta" dell'ipotesi (n ≥ 4) è non testata: 8 + 1 + 3 + 2 = 14
   puzzle in tutto, e la GNN è a zero anche perché è stata addestrata solo
   su n = 1-5. Non si può affermare *né* che l'LLM recuperi *né* che non
   recuperi.
3. n = 1, il caso più favorevole all'ipotesi, mancava del tutto per un bug
   (vedi Parte B).

**Formulazione difendibile per il report**, dopo la run 2: "sui valori di n
per cui l'held-out ha una numerosità sufficiente (n = 1-3), la GNN
temporizzata risolve una frazione significativamente maggiore di puzzle
rispetto alla baseline LLM zero-shot (McNemar esatto, p = ...). Per n ≥ 4 la
numerosità dell'held-out (N = 14) non consente di verificare la seconda metà
dell'ipotesi."

### Domanda 2: l'informazione di timing migliora le prestazioni?

**Come il progetto risponde**: due modelli identici per architettura, seed,
subset di training e numero di epoche, addestrati uno con la feature di
timing attiva e uno con la stessa feature azzerata
(`include_timing` in `src/dataset.py`). Confronto su due metriche e ora
anche con McNemar appaiato sul test set interno.
Codice: `src/train.py` → `evaluate_ablation`.
Output: `results/ablation_timing.csv`, `results/ablation_timing_by_n.png`.

**Risultato run 1**:

| metrica | con timing | senza timing | delta |
|---|---|---|---|
| accuracy per mossa (test) | 0.829 | 0.824 | +0.005 |
| accuracy sequenza completa (4.620 puzzle) | 0.615 | 0.595 | +0.020 |
| val accuracy migliore | 0.832 (ep. 34) | 0.827 (ep. 22) | +0.005 |

Accuracy sequenza completa per n: con timing {1: 0.892, 2: 0.743, 3: 0.627,
4: 0.441, 5: 0.219}; senza {1: 0.873, 2: 0.717, 3: 0.622, 4: 0.407,
5: 0.213}. Il timing aiuta più a n bassi che a n alti.

**Come va letto**: il delta è **piccolo ma sistematicamente positivo** su
tutte le metriche e su tutti i valori di n. Non è rumore casuale nel senso
banale (un rumore cambierebbe segno da un n all'altro), ma la run 1 non
forniva nessun test di significatività. La patch lo aggiunge: con ~4.600
puzzle appaiati un +0.020 dovrebbe risultare significativo, ma va misurato,
non assunto.

**Limite importante da dichiarare nel report**: così com'è implementata, la
feature di timing è **uno scalare per puzzle**, derivato in modo
deterministico da `Rating` e `Popularity` (`simulate_think_time` in
`src/data_pipeline.py`), replicato identico su tutti i 64 nodi e costante
lungo l'intera linea. Non è una dinamica temporale: è un **proxy di
difficoltà**. Che dia un +0.5/+2 punti è quindi plausibile e coerente — il
modello sta usando un indizio sulla difficoltà del puzzle, non un "senso di
urgenza".

Conseguenza pratica (v5): sull'held-out classico Polgár non esistono Rating e
Popularity, quindi `simulate_think_time` ricadeva su valori di default e
produceva quasi lo stesso numero per tutti i puzzle. **Nella v6** questi due
valori sono imputati con la mediana Lichess dello stesso n, ma restano
costanti all'interno di ogni n per i puzzle Polgár: la feature di timing sui
puzzle n=1-3 dell'held-out è comunque poco informativa. L'ablation va
valutata sul test set interno Lichess, dove il Rating esiste davvero.

L'ipotesi del PDF ("il timing aiuta nei puzzle che simulano partite blitz")
non è testabile con i dati attuali, perché i puzzle Lichess non hanno tempi
reali e la simulazione non distingue blitz da lento. Da dire nel report come
limite, con l'estensione naturale: usare il database delle partite d'élite
(PGN con orologi reali) invece dei soli puzzle.

---

## Parte B — Le anomalie della run 1 e cosa le causava

### Perché la valutazione partiva da n = 2 e mancava il matto in 1

Bug in `src/data_pipeline.py::validate_puzzle`: il controllo di lunghezza
minima della linea era `len(moves) < 2` per tutte le sorgenti. Per i puzzle
Lichess è corretto (la FEN precede la mossa di apertura dell'avversario,
quindi anche un matto in 1 ha 2 mosse), ma per i puzzle classici
(`has_opening_move=False`) la FEN è già la posizione da risolvere e un matto
in 1 ha **una sola mossa**: tutti scartati.

Verifica numerica: l'held-out combinato aveva 2.523 problemi di cui 306 con
n = 1; la validazione ne teneva 2.217, cioè esattamente 2.523 − 306.

Corretto: la soglia ora dipende da `has_opening_move`, come già faceva
`chess_encoding.solver_moves_sequence`. Aggiunta anche una diagnostica che
segnala quando un intero gruppo di n viene scartato al 100%.

### Perché 46 puzzle valutati invece di 100-200

`load_evaluation_set` assegnava a ogni valore di n una quota fissa
`LLM_N_SAMPLES // n_gruppi = 100 // 6 = 16`, prendendo `min(quota,
disponibili)`. I gruppi profondi non riuscivano a spendere la quota e
nessuno la raccoglieva: 16 + 16 + 8 + 1 + 3 + 2 = **46**.

Corretto con un'allocazione round-robin che redistribuisce la quota
inutilizzata. Con il matto in 1 recuperato e `LLM_N_SAMPLES = 150`, la
ripartizione attesa è 46 / 45 / 45 / 8 / 1 / 3 / 2 = **150 puzzle**, dentro
la forchetta del PDF.

### Perché prima 2.217 puzzle e poi solo 46

Sono due cose diverse e vanno chiamate con nomi diversi nel report:

- **pool held-out**: 2.217 (ora 2.523) problemi classici verificati e
  deduplicati per FEN contro il training Lichess. È il bacino.
- **campione di valutazione**: il sottoinsieme stratificato per n su cui si
  interroga l'LLM. È qui che il PDF chiede 100-200, perché ogni puzzle costa
  una chiamata API e ~11 secondi.

Avere un pool più grande del campione è corretto e utile: permette di
stratificare e di rivalutare la sola GNN su tutti i 2.523 (costa pochi
minuti in locale) come analisi di supporto.

Il problema vero non è il numero ma la **composizione**: il pool è quasi
tutto Polgár (n = 2 e 3). Lo scraper WTHarvey, che doveva coprire gli n
profondi, ha verificato 52 puzzle su 1.521 candidati (3,4%). Quella resa è
troppo bassa per essere spiegata dalla prudenza: è un problema di parsing. Le
patch aggiungono al parser altri tentativi di segmentazione e una diagnostica
che stampa i motivi di scarto e i primi esempi falliti, così da poterlo
correggere empiricamente.

### Perché 117.916 puzzle di training ma 298.306 esempi

Non è un'incoerenza, sono due unità di misura.

- **117.916 puzzle**: `SUBSET_SIZE = 150000` viene ripartito in parti uguali
  fra i 5 valori di n (30.000 ciascuno), ma su Lichess i mate-in-4 e
  mate-in-5 sono meno di 30.000 nello split di training, quindi si prende
  tutto il disponibile e il totale scende. Stessa logica per validazione
  (21.490) e per i 4.620 della valutazione a sequenza completa.
- **298.306 esempi (puzzle, mossa)**: ogni puzzle viene espanso in **un
  esempio per ogni mossa del solver** lungo la linea. Un mate-in-3 produce 3
  grafi. 298.306 / 117.916 ≈ 2,53 mosse per puzzle, coerente con un mix
  bilanciato di n = 1-5.

È una scelta necessaria: allenando solo sulla prima mossa, il modello non
vedrebbe mai le posizioni a metà linea e la sequenza completa crollerebbe.

Da dichiarare nel report: 117.916 puzzle usati su 1.549.599 disponibili
(7,6%), per vincoli di tempo su Colab gratuito.

### Perché ha senso la sequenza completa e non la prima mossa

Hai ragione: indovinare la prima mossa di un matto in 4 non risolve il
puzzle. La metrica primaria è e resta la **sequenza completa**, ed è quella
usata per McNemar e per il grafico per n. La prima mossa è una **diagnostica**
utile per distinguere due fallimenti diversi — "non ha visto il tema" contro
"ha visto il tema ma si è perso nel seguito" — e va riportata come
secondaria. Il PDF, sezione 3, dice appunto "Accuratezza (sequenza di matto
corretta)".

---

## Parte C — Il protocollo di confronto (PDF sezione 3), punto per punto

### Guida della GNN: mosse suggerite **e punteggi di confidenza**

Implementato in `src/train.py::solve_puzzle_with_gnn`: per ogni posizione
della linea si applica una softmax sugli score delle sole mosse legali
(esattamente la distribuzione ottimizzata in training) e si restituisce la
confidenza della mossa scelta, le tre alternative migliori per la prima
mossa e il tempo di calcolo. `src/compare.py::format_gnn_guidance` li rende
in forma leggibile.

### Baseline LLM zero-shot

Backend Groq, modello `openai/gpt-oss-120b`, zero-shot, nessun fine-tuning,
prompt che chiede di spiegare passo dopo passo e poi produrre la soluzione —
in linea con il PDF. Il prompt è stato reso **esplicito sul formato** (vedi
sotto), che non è few-shot sul contenuto scacchistico: l'unico esempio
mostrato riguarda la punteggiatura della riga di output, non come si risolve
un matto. La natura zero-shot è preservata.

### Il problema di equità del confronto, e come è stato risolto

Nella run 1 il confronto non era alla pari, per due motivi cumulativi:

1. `get_position_after_opening` restituisce come target la linea **completa
   e alternata** (mia mossa, sua risposta, mia mossa, ...), e la correttezza
   era `proposed == target`. Ma il prompt chiedeva solo "trova il matto in n
   mosse": un modello che rispondeva con le sole proprie mosse — lettura
   perfettamente ragionevole — veniva segnato sbagliato.
2. La stessa lista veniva spinta tutta di seguito sulla scacchiera per il
   controllo di legalità, quindi la stessa risposta risultava anche
   *illegale*.

Da qui i due numeri sospetti della run 1: sequenza completa 0.000 e "risposte
con mosse legali" 0.457. Non misuravano la forza dell'LLM, misuravano quanto
indovinava un formato mai specificato.

**Soluzione adottata** (fra le tre valutate, questa è la più fedele al PDF e
la più economica):

- **prompt esplicito**: si chiede la linea completa alternata, in UCI, con un
  esempio di formattazione per un matto in 2. Costo zero, e toglie di mezzo
  l'ambiguità alla radice.
- **doppia metrica**: `full_sequence_correct` (stretta, linea alternata) e
  `solver_moves_correct` (permissiva, accetta anche le sole mosse del
  solver). Il confronto principale con la GNN usa la **permissiva**, perché
  la GNN per costruzione predice solo le proprie mosse: è l'unico modo di
  confrontare la stessa quantità nei due modelli.
- **legalità valutata sotto entrambe le interpretazioni**, con una colonna
  `interpretation` che registra quale delle due il modello ha usato — è la
  diagnostica che dice quanto il prompt viene rispettato.
- Viene stampato **anche** il McNemar con la metrica stretta: se le due
  conclusioni divergono, la differenza dipende dal formato e non dagli
  scacchi, e questo va scritto nel report.

**Perché non la valutazione turno per turno.** L'alternativa più pulita
sarebbe stata far giocare l'LLM una mossa alla volta, ridandogli la FEN dopo
ogni risposta forzata, esattamente come fa la GNN. Elimina anche l'ultima
asimmetria residua (la GNN riceve gratis le risposte dell'avversario). È
stata scartata perché moltiplica le chiamate API per la profondità del matto
e soprattutto perché **si allontana dal PDF**, che descrive una guida
zero-shot generata in un colpo solo ("trova il matto in n mosse. Spiega passo
dopo passo"). Resta la candidata numero uno fra le estensioni, e l'asimmetria
residua va dichiarata nei limiti del report.

### Metriche: efficienza

`src/compare.py::efficiency_comparison` confronta il tempo medio e mediano
per puzzle dei due modelli. Nella run 1 l'LLM stava su 11,2 s medi per
puzzle. Va riportato notando che la GNN gira in locale su GPU mentre l'LLM
paga anche la latenza di rete: è un confronto di esperienza d'uso, non di
costo computazionale puro.

### Metriche: qualità della guida e blind test

`results/human_eval_blind.csv` contiene per ogni puzzle le due guide con
etichette Modello A/B randomizzate per riga, le quattro colonne della rubrica
1-5 (correttezza e concisione per ciascun modello) e una colonna di
preferenza complessiva. La chiave è in
`human_eval_blind_WITH_ANSWER_KEY.csv`.

Due accorgimenti metodologici aggiunti:

- la guida della GNN è ora **testuale e con confidenze**, non una stringa UCI
  nuda. Confrontare "d8h4 g5h4 c1g5" con un paragrafo in inglese sulla scala
  "concisione" avrebbe misurato il formato, non la qualità.
- chi compila non deve sapere quale colonna è quale, e non dovrebbe essere la
  stessa persona che ha scritto il codice.

### Test statistico

Il PDF ammette "t-test accoppiati o il test di McNemar". McNemar è la scelta
corretta qui, perché l'esito è binario (puzzle risolto / non risolto) sugli
stessi puzzle; il t-test accoppiato sarebbe appropriato per misure continue.

La run 1 usava solo il chi-quadro con correzione di continuità, che con
b + c = 7 non è affidabile (regola pratica: serve b + c ≥ 25). Ora
`src/stats.py` calcola **anche il p-value binomiale esatto**, che su b = 7,
c = 0 dà **p = 0.0156** contro lo 0.0233 del chi-quadro. La conclusione della
run 1 reggeva, ma il numero da riportare è quello esatto.

---

## Parte D — Cosa manca ancora, in ordine di importanza

1. **Puzzle con n ≥ 4 nell'held-out.** *(Risolto nella v6: 30 puzzle Lichess
   per n=4 e n=5. Resta il limite che n=1-3 e n=4-5 hanno fonti diverse.)*
2. **Compilare il blind test.** Il file c'è, i dati ci sono, servono i
   valutatori umani: è un deliverable esplicito del PDF.
3. **Il gap di generalizzazione** merita di stare nel report come risultato a
   sé: la GNN fa 0.743 di sequenza completa a n = 2 sul test Lichess e 0.250
   sull'held-out classico. Posizioni composte/storiche contro posizioni da
   partite online reali. È un risultato onesto e interessante, non un
   fallimento.
4. **La libreria Wong-Damiani** citata dal PDF non è stata usata: qui c'è
   un'implementazione propria in PyTorch Geometric (GINEConv + policy head
   sugli archi). Va dichiarato apertamente nel report, con il motivo.
