# Estensioni possibili

Il PDF (sezione Risultati Attesi) suggerisce tre estensioni generiche.
Questo documento le cala sul progetto concreto: cosa servirebbe davvero,
quanto lavoro richiede, e — per quella piu' fattibile nel tempo residuo —
un prototipo funzionante incluso in questo zip.

---

## 1. Integrazione Stockfish per una risoluzione ibrida — PROTOTIPO INCLUSO

E' l'estensione piu' naturale e meno costosa: Stockfish e' deterministico
e quasi perfetto sui matti forzati, quindi non serve "batterlo", serve
usarlo per capire DOVE la GNN fallisce e perche'.

### Due modalita' d'uso, entrambe utili

**A) Stockfish come arbitro, non come concorrente.** Per ogni puzzle che
la GNN sbaglia, si chiede a Stockfish la mossa migliore e si confronta con
quella del dataset (il test della verita', cioe' se davvero esiste un'unica
mossa che porta al matto in n, o se la posizione ammette alternative che
la metrica attuale (match esatto con la sequenza del dataset) penalizza
ingiustamente). Questo e' un controllo di qualita' sui dati, non solo
un'estensione del modello: rivelerebbe se parte dell'accuratezza "persa"
dalla GNN sia in realta' un falso negativo della metrica.

**B) Sistema ibrido reale.** La GNN propone una mossa con un punteggio di
confidenza (gia' implementato — vedi `solve_puzzle_with_gnn` in
`src/train.py`); se la confidenza e' sotto una soglia, si chiama Stockfish
come fallback. Utile soprattutto per n grandi, dove la GNN crolla quasi a
zero (vedi i risultati per n>=4): un sistema del genere avrebbe
un'accuratezza complessiva molto piu' alta di entrambi i componenti presi
singolarmente, al costo di perdere la "purezza" del confronto GNN-vs-LLM
richiesto dal PDF (per questo va presentato come estensione separata, non
mescolato ai risultati della domanda di ricerca 1).

### Prototipo: `scripts/stockfish_hybrid_demo.py`

Incluso in questo zip. Confronta, sullo stesso campione held-out gia'
usato per il confronto GNN-vs-LLM, tre strategie: GNN pura, Stockfish puro,
e ibrida (GNN se confidenza > soglia, altrimenti Stockfish). Produce una
tabella di accuratezza per n per le tre strategie.

**Non e' stato esguito in questo ambiente** perche' richiede il binario di
Stockfish, non incluso di default su Colab. Per eseguirlo:

```python
!apt-get install -y stockfish -q
!python -m scripts.stockfish_hybrid_demo
```

Tempo stimato: pochi secondi per puzzle (Stockfish e' molto piu' veloce di
un LLM), quindi l'intero campione di 150 puzzle gira in meno di un minuto.

---

## 2. Applicazione web per l'allenamento scacchistico

Idea: una pagina dove l'utente carica una posizione (FEN o mossa per
mossa su una scacchiera interattiva) e riceve la guida della GNN — mossa
suggerita, confidenza, alternative — nello stesso formato testuale gia'
prodotto da `format_gnn_guidance` in `src/compare.py`.

**Componenti gia' pronti, da riusare direttamente:**
- `solve_puzzle_with_gnn` (src/train.py): dato un FEN, restituisce mossa,
  confidenza, alternative — e' gia' la funzione che servirebbe come
  backend.
- `format_gnn_guidance` (src/compare.py): rende l'output leggibile.

**Cosa manca, in ordine di sforzo:**
1. Un endpoint che accetta un FEN e chiama `solve_puzzle_with_gnn` — poche
   righe, il grosso del lavoro e' gia' scritto.
2. Una scacchiera interattiva lato client (librerie JS pronte come
   chessboard.js o react-chessboard la rendono questione di ore, non
   giorni).
3. Caricamento del checkpoint una volta sola all'avvio del server, non per
   ogni richiesta (il modello e' piccolo, ~3 MB, quindi l'inferenza e'
   gia' istantanea — si vede dal tempo medio di 0.01s per puzzle nei
   risultati).

**Stima**: un prototipo minimo (FEN in input, mossa suggerita in output,
nessuna gestione di partite multi-mossa) e' fattibile in una sessione di
lavoro, usando il backend gia' scritto. Una versione completa con
scacchiera cliccabile e cronologia delle mosse e' un progetto a parte.

---

## 3. Input multimodale — immagini della scacchiera

Idea: invece di dare in input un FEN testuale, l'utente fotografa una
scacchiera reale (o uno screenshot) e il sistema la converte in FEN prima
di passarla alla GNN.

**Questa e' l'estensione piu' impegnativa delle tre**, perche' introduce un
problema di computer vision completamente separato dal resto del progetto
(rilevamento della griglia, classificazione del contenuto di ogni casella
in 13 classi — 6 pezzi x 2 colori + vuota). Non e' qualcosa che si innesta
sul codice esistente: servirebbe un modello di visione a parte (una CNN
semplice per la classificazione per-casella e' il punto di partenza piu'
comune in letteratura) e un dataset di immagini di scacchiere etichettate.

**Scorciatoia realistica**: esistono gia' librerie/modelli open-source per
"board-to-FEN" (es. progetti come chessvision.ai o modelli pubblicati su
HuggingFace per questo compito specifico) che si potrebbero integrare come
componente esterno piuttosto che riaddestrare da zero — la GNN riceverebbe
comunque un FEN in input, esattamente come ora, quindi tutto il resto della
pipeline resterebbe invariato.

**Stima**: non fattibile come prototipo nel tempo residuo di questo
progetto. Va presentata nel report come direzione futura, con la
scorciatoia del modello esterno come percorso piu' realistico rispetto ad
addestrarne uno da zero.

---

## Riepilogo per il report

| Estensione | Sforzo per un prototipo | Stato in questo zip |
|---|---|---|
| Ibrido GNN+Stockfish | Basso (riusa codice esistente) | Script incluso, da eseguire con Stockfish installato |
| App web | Medio (backend pronto, manca l'interfaccia) | Non implementata, backend riusabile documentato sopra |
| Input multimodale (immagini) | Alto (dominio nuovo, serve CV) | Non implementata, solo analisi di fattibilita' |
