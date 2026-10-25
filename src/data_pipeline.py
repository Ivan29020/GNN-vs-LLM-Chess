"""
src/data_pipeline.py
---------------------
Fase 1 — Pipeline dati (equivalente a 01_data_pipeline.ipynb).

Cosa fa (tutto con UN solo comando: `python -m scripts.run_pipeline`):
1. Scarica il database puzzle di Lichess (.csv.zst) se non già presente
2. Filtra i puzzle "mate in n" (n = 1..5)
3. Valida FEN e mosse con python-chess, scarta le righe corrotte
4. Normalizza il rating (P10-P90 del training set): e' la componente
   puzzle-level del "timing proxy v2" (Lichess non ha timing reale; il resto
   del proxy si calcola per posizione, vedi src/chess_encoding.py)
5. COSTRUISCE due held-out disgiunti prima dello split: misto Polgar N1-N3 +
   Lichess N4-N5, e Lichess-only N1-N5 (30 puzzle per ogni N)
6. Rimuove entrambi dal pool, seleziona un subset complessivo bilanciato da 300k
   per `mate_n` e fasce di rating, poi crea lo split 80/10/10 stratificato su entrambi
7. Controllo anti-leakage su entrambi gli held-out, per PuzzleId e FEN
8. Salva train/val/test e i due held-out in Parquet

Uso da riga di comando:
    python -m scripts.run_pipeline
"""

import os
import json
import re
import numpy as np
import pandas as pd
import chess
import requests
import zstandard as zstd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import config
from src.held_out import (
    normalize_fen, select_lichess_held_out, remove_from_pool,
    download_polgar_pgn, parse_polgar_pgn, select_polgar_held_out,
)

MATE_THEME_RE = re.compile(r"^mateIn(\d+)$")

COLUMNS = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
           "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags",
           "DailyDate"]


# ---------------------------------------------------------------------
# 1. Download + decompressione
# ---------------------------------------------------------------------
def download_file(url: str, path: str, chunk_size: int = 1024 * 1024) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"File gia' presente: {path}")
        return
    print(f"Download da {url} ...")
    tmp_path = path + ".part"
    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print("Download completato.")


def decompress_zst(zst_path: str, out_path: str) -> None:
    # Decomprime SEMPRE da zero e pubblica il CSV solo a decompressione finita.
    tmp_path = out_path + ".part"
    try:
        dctx = zstd.ZstdDecompressor()
        with open(zst_path, "rb") as ifh, open(tmp_path, "wb") as ofh:
            dctx.copy_stream(ifh, ofh)
            ofh.flush()
            os.fsync(ofh.fileno())
        os.replace(tmp_path, out_path)
    except Exception as exc:
        for bad_path in (tmp_path, out_path):
            if os.path.exists(bad_path):
                os.remove(bad_path)
        # Se la sorgente è troncata, rimuoverla consente al successivo avvio
        # di scaricarla di nuovo invece di riprovare sullo stesso file guasto.
        if os.path.exists(zst_path):
            try:
                os.remove(zst_path)
            except OSError:
                pass
        raise RuntimeError(
            "Decompressione Zstandard fallita: il file sorgente potrebbe essere "
            "incompleto. È stato rimosso; rilancia lo stesso comando per riscaricarlo."
        ) from exc
    print("Decompressione completata.")


# ---------------------------------------------------------------------
# 2. Filtro "mate in n"
# ---------------------------------------------------------------------
def extract_mate_n(themes_str):
    if not isinstance(themes_str, str):
        return None
    for theme in themes_str.split():
        m = MATE_THEME_RE.match(theme)
        if m:
            return int(m.group(1))
    return None


def diagnose_empty_mate_pool(df: pd.DataFrame) -> None:
    """Stampa informazioni diagnostiche quando 0 puzzle 'mate in n' vengono
    trovati — situazione anomala che di solito indica un problema nella
    lettura del CSV (colonne disallineate, encoding, ecc.) più che
    l'assenza reale di puzzle di matto nel dataset."""
    print("\n" + "=" * 70)
    print("DIAGNOSTICA: 0 puzzle 'mate in n' trovati, non dovrebbe succedere.")
    print(f"Colonne lette dal CSV: {list(df.columns)}")
    if "Themes" in df.columns:
        print(f"Dtype colonna 'Themes': {df['Themes'].dtype}")
        print(f"Valori NaN in 'Themes': {df['Themes'].isna().mean():.1%}")
        print("Primi 5 valori di 'Themes':")
        print(df["Themes"].head(5).to_list())
    else:
        print("La colonna 'Themes' non esiste nel dataframe letto!")
    print("Prime 2 righe complete del dataframe:")
    print(df.head(2).to_string())
    print("=" * 70 + "\n")


def filter_mate_puzzles(df: pd.DataFrame, n_min=config.MATE_N_MIN,
                         n_max=config.MATE_N_MAX) -> pd.DataFrame:
    df = df.copy()
    df["mate_n"] = df["Themes"].apply(extract_mate_n)
    df_mate = df[df["mate_n"].notna()].copy()
    df_mate["mate_n"] = df_mate["mate_n"].astype(int)
    print(f"Puzzle di tipo 'mate in n': {len(df_mate):,} su {len(df):,} totali")

    if len(df_mate) == 0:
        diagnose_empty_mate_pool(df)

    df_pool = df_mate[df_mate["mate_n"].between(n_min, n_max)].copy()
    print(f"Puzzle mate in {n_min}-{n_max} disponibili per il training set: {len(df_pool):,}")
    return df_pool


# ---------------------------------------------------------------------
# 3. Validazione FEN/mosse con python-chess
# ---------------------------------------------------------------------
def _validate_line(fen: str, moves_str: str, has_opening_move: bool = True) -> tuple:
    """Ritorna (is_valid, ends_in_checkmate, solve_fen_norm).

    `solve_fen_norm` e' la posizione che il solver si trova davanti (dopo
    la mossa di apertura per i puzzle Lichess), normalizzata: serve a
    togliere dal training le posizioni identiche a quelle dell'held-out.
    Si ricava nello STESSO passaggio in cui si validano le mosse (nessun
    replay aggiuntivo).

    ATTENZIONE alla lunghezza minima della linea, che dipende dalla sorgente:
    - puzzle Lichess (`has_opening_move=True`): la FEN e' la posizione PRIMA
      della mossa di apertura dell'avversario, quindi anche un matto in 1 ha
      2 mosse (apertura + matto) -> minimo 2;
    - puzzle classici Polgar (`has_opening_move=False`): la FEN e' gia' la
      posizione da risolvere e un matto in 1 ha UNA sola mossa -> minimo 1.
    Stessa convenzione di src/chess_encoding.solver_moves_sequence.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return False, None, None

    moves = moves_str.split()
    min_len = 2 if has_opening_move else 1
    if len(moves) < min_len:
        return False, None, None

    solve_fen = fen
    try:
        for i, uci in enumerate(moves):
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return False, None, None
            board.push(move)
            if i == 0 and has_opening_move:
                solve_fen = board.fen()
    except (ValueError, AssertionError):
        return False, None, None

    return True, board.is_checkmate(), normalize_fen(solve_fen)


def validate_puzzle(row) -> tuple:
    """Versione 'per riga' (compatibilita'): vedi `_validate_line`."""
    return _validate_line(row["FEN"], row["Moves"], row.get("has_opening_move", True))


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Valida tutti i puzzle di `df` (legalita' della linea + matto finale) e
    aggiunge le colonne `fen_norm` e `solve_fen_norm`.

    Usa un ciclo su liste invece di `df.apply(axis=1)` (che crea una Series
    per riga ed e' molto piu' lento su centinaia di migliaia di puzzle)."""
    df = df.copy()
    if len(df) == 0:
        print("Nessuna riga da validare (dataframe vuoto in ingresso).")
        for c in ("is_valid", "ends_in_checkmate"):
            df[c] = pd.Series(dtype=bool)
        df["fen_norm"] = pd.Series(dtype=str)
        df["solve_fen_norm"] = pd.Series(dtype=str)
        return df

    has_open = (df["has_opening_move"].tolist() if "has_opening_move" in df.columns
                else [True] * len(df))
    results = [
        _validate_line(f, m, h)
        for f, m, h in tqdm(zip(df["FEN"].tolist(), df["Moves"].tolist(), has_open),
                            total=len(df), desc="Validazione puzzle")
    ]
    df["is_valid"] = [r[0] for r in results]
    df["ends_in_checkmate"] = [r[1] for r in results]
    df["solve_fen_norm"] = [r[2] for r in results]
    df["fen_norm"] = df["FEN"].map(normalize_fen)

    n_before = len(df)
    keep = df["is_valid"] & (df["ends_in_checkmate"] == True)  # noqa: E712
    df_kept = df[keep].copy()
    n_after = len(df_kept)
    print(f"Puzzle validi e che terminano in scacco matto: {n_after:,} / {n_before:,}")

    n_discarded = n_before - n_after
    if n_discarded > 0 and "mate_n" in df.columns:
        discarded_by_n = df[~keep]["mate_n"].value_counts().sort_index()
        print(f"Scartati {n_discarded:,}, ripartiti per profondita' di matto:")
        for n_val, cnt in discarded_by_n.items():
            tot_n = int((df["mate_n"] == n_val).sum())
            print(f"  - mate in {n_val}: {cnt:,} scartati su {tot_n:,} "
                  f"({cnt / tot_n:.0%})")
    return df_kept


# ---------------------------------------------------------------------
# 5. Rating normalizzato (componente puzzle-level del timing proxy v2)
# ---------------------------------------------------------------------
def ensure_opening_flag(df: pd.DataFrame) -> pd.DataFrame:
    """I puzzle Lichess hanno sempre una mossa di apertura dell'avversario
    prima della linea di matto (vedi src/chess_encoding.solver_moves_sequence).
    Per i puzzle Polgar la colonna esiste gia' (False) e resta intatta."""
    df = df.copy()
    if "has_opening_move" not in df.columns:
        df["has_opening_move"] = True
    return df


def compute_rating_stats(df_train: pd.DataFrame) -> dict:
    """Percentili P10 e P90 del rating NEL TRAINING SET (non su val/test/
    held-out, per non far entrare informazione di quei dati nella
    normalizzazione). Salvati in data/timing_stats.json."""
    lo_p, hi_p = config.TIMING_RATING_PERCENTILES
    lo = float(np.percentile(df_train["Rating"].astype(float), lo_p))
    hi = float(np.percentile(df_train["Rating"].astype(float), hi_p))
    stats = {"rating_p_lo": lo, "rating_p_hi": hi,
             "percentiles": [lo_p, hi_p], "n_train_puzzles": int(len(df_train))}
    with open(config.TIMING_STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Timing proxy v2: rating P{lo_p}={lo:.0f}, P{hi_p}={hi:.0f} "
          f"(dal training set, salvati in {config.TIMING_STATS_JSON})")
    return stats


def add_rating_norm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """R = clip((Rating - P10) / (P90 - P10), 0, 1). Rating mancante -> mediana
    (R = 0.5). Il resto del timing proxy (mosse legali, scacchi, profondita'
    residua) dipende dalla posizione e si calcola al momento di costruire il
    grafo: vedi src/chess_encoding.compute_think_time_sec."""
    df = df.copy()
    lo, hi = stats["rating_p_lo"], stats["rating_p_hi"]
    r = (df["Rating"].astype(float) - lo) / max(hi - lo, 1e-9)
    df["rating_norm"] = r.clip(0.0, 1.0).fillna(0.5)
    return df


# ---------------------------------------------------------------------
# 6. Subset complessivo bilanciato e split 80/10/10 per mate_n × rating_bin
# ---------------------------------------------------------------------
def _balanced_quotas(sizes: dict, target: int) -> dict:
    """Quote il più uniformi possibile; la disponibilità limita i gruppi rari."""
    keys = sorted(sizes, key=lambda x: str(x))
    sizes = {k: max(0, int(sizes[k])) for k in keys}
    target = min(max(0, int(target)), sum(sizes.values()))
    quota = {k: 0 for k in keys}
    remaining = target
    # Water-filling: evita un ciclo per ogni elemento anche con subset grandi.
    while remaining > 0:
        active = [k for k in keys if quota[k] < sizes[k]]
        if not active:
            break
        share = max(1, remaining // len(active))
        added = 0
        for k in active:
            if remaining <= 0:
                break
            take = min(share, sizes[k] - quota[k], remaining)
            if take > 0:
                quota[k] += take
                remaining -= take
                added += take
        if added == 0:
            break
    return quota


def _global_rating_bins(df: pd.DataFrame, n_bins: int = config.SUBSET_RATING_BINS) -> pd.Series:
    """Crea fasce quantili GLOBALI sul rating puzzle, deterministiche e senza bin vuoti."""
    if len(df) == 0:
        return pd.Series(index=df.index, dtype="int64", name="rating_bin")
    ratings = pd.to_numeric(df["Rating"], errors="coerce")
    median = float(ratings.median()) if ratings.notna().any() else 1500.0
    ratings = ratings.fillna(median if np.isfinite(median) else 1500.0)
    q = max(1, min(int(n_bins), len(df)))
    # Il rank evita che rating ripetuti rendano impossibile qcut con quantili uguali.
    ranked = ratings.rank(method="first")
    bins = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    return bins.astype(int).rename("rating_bin")


def select_balanced_subset(df_pool: pd.DataFrame, target_size: int = config.SUBSET_SIZE,
                           n_rating_bins: int = config.SUBSET_RATING_BINS,
                           seed: int = config.SUBSET_SEED) -> pd.DataFrame:
    """Seleziona PRIMA dello split il subset totale, bilanciando mate_n e rating.

    Prima assegna quote il più possibile uniformi fra le profondità N1-N5. Per
    ciascun N ripartisce poi la quota uniformemente fra le fasce di rating
    globali disponibili. Se N4/N5 o qualche fascia sono troppo piccoli, le quote
    inutilizzate vengono ridistribuite agli altri gruppi.
    """
    if df_pool.empty:
        raise ValueError("Il pool dopo l'esclusione degli held-out è vuoto.")
    target = len(df_pool) if target_size is None else min(int(target_size), len(df_pool))
    if target < int(target_size or 0):
        print(f"ATTENZIONE: dopo i filtri restano solo {len(df_pool):,} puzzle; "
              f"il subset sarà inferiore al target {int(target_size):,}.")

    work = df_pool.copy().reset_index(drop=True)
    work["rating_bin"] = _global_rating_bins(work, n_rating_bins).to_numpy()
    sizes_by_n = {int(n): int(len(g)) for n, g in work.groupby("mate_n", sort=True)}
    quotas_by_n = _balanced_quotas(sizes_by_n, target)
    selected = []

    for n in sorted(quotas_by_n):
        quota_n = quotas_by_n[n]
        if quota_n <= 0:
            continue
        group_n = work[work["mate_n"].astype(int) == int(n)]
        sizes_by_bin = {int(b): int(len(g)) for b, g in group_n.groupby("rating_bin", sort=True)}
        quotas_by_bin = _balanced_quotas(sizes_by_bin, quota_n)
        for b in sorted(quotas_by_bin):
            take = quotas_by_bin[b]
            if take <= 0:
                continue
            group_bin = group_n[group_n["rating_bin"] == b]
            selected.append(group_bin.sample(
                n=take, random_state=int(seed) + int(n) * 100 + int(b)
            ))

    if not selected:
        raise RuntimeError("Il campionamento bilanciato non ha selezionato puzzle.")
    result = pd.concat(selected, ignore_index=True)
    if len(result) != target:
        raise RuntimeError(f"Subset errato: selezionati {len(result):,}, attesi {target:,}.")
    result = result.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    by_n = result.groupby("mate_n").size().sort_index()
    print(f"\nSubset totale selezionato: {len(result):,} puzzle (target {target:,})")
    print("Distribuzione per mate_n:")
    print(by_n.to_string())
    joint = pd.crosstab(result["mate_n"], result["rating_bin"], margins=True)
    print("Distribuzione congiunta mate_n × fascia globale Rating:")
    print(joint.to_string())
    return result


def _split_strata(df: pd.DataFrame) -> pd.Series | None:
    """Preferisce mate_n × rating_bin; fallback solo se dati piccoli rendono
    matematicamente impossibile stratificare ogni cella in tutti gli split."""
    if "rating_bin" in df.columns:
        joint = df["mate_n"].astype(str) + "|R" + df["rating_bin"].astype(str)
        if len(joint.value_counts()) and int(joint.value_counts().min()) >= 10:
            return joint
        print("ATTENZIONE: alcune celle mate_n × rating_bin sono troppo piccole; "
              "per lo split uso la stratificazione per mate_n.")
    by_n = df["mate_n"].astype(str)
    if len(by_n.value_counts()) and int(by_n.value_counts().min()) >= 10:
        return by_n
    print("ATTENZIONE: campione piccolo per una stratificazione robusta; split casuale.")
    return None


def stratified_split(df: pd.DataFrame, test_size=config.SPLIT_TEST_SIZE,
                      val_of_temp=config.SPLIT_VAL_OF_TEMP, seed=config.RANDOM_SEED):
    """Split 80/10/10 preservando per quanto possibile mate_n e fasce di rating."""
    strata = _split_strata(df)
    df_train, temp = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strata
    )
    temp_strata = strata.loc[temp.index] if strata is not None else None
    df_val, df_test = train_test_split(
        temp, test_size=val_of_temp, random_state=seed + 1, stratify=temp_strata
    )
    return (df_train.reset_index(drop=True), df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


def joint_rating_distribution(df: pd.DataFrame) -> dict:
    """Conteggi serializzabili per verificare la distribuzione mate_n × rating_bin."""
    if "rating_bin" not in df.columns:
        return {}
    return {
        f"N{int(n)}_R{int(b)}": int(count)
        for (n, b), count in df.groupby(["mate_n", "rating_bin"]).size().items()
    }



# ---------------------------------------------------------------------
# 7. Distribuzione di n e controllo anti-leakage
# ---------------------------------------------------------------------
def n_distribution_table(frames: dict) -> pd.DataFrame:
    """Tabella: una riga per ogni n (1..5), una colonna per insieme di dati
    (es. train/val/test/held-out), con i conteggi di puzzle, piu' un totale."""
    ns = sorted({int(n) for df in frames.values() for n in df["mate_n"].unique()})
    table = pd.DataFrame(
        {name: [int((df["mate_n"] == n).sum()) for n in ns] for name, df in frames.items()},
        index=[f"N{n}" for n in ns],
    )
    table.loc["TOTALE"] = table.sum()
    return table


def print_n_distribution(frames: dict, title: str, csv_path: str = None) -> pd.DataFrame:
    """Stampa (e opzionalmente salva) quanti puzzle N1..N5 ci sono in ogni
    insieme di dati."""
    table = n_distribution_table(frames)
    print(f"\n=== {title} ===")
    print(table.to_string())
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        table.to_csv(csv_path)
        print(f"(salvata in {csv_path})")
    return table


def check_no_leakage(df_held: pd.DataFrame, splits: dict) -> None:
    """Verifica che NESSUN puzzle dell'held-out compaia in train/val/test, ne'
    per PuzzleId ne' per posizione (FEN normalizzato). Solleva RuntimeError
    se trova una sovrapposizione: meglio fermarsi che confrontare GNN e LLM
    su puzzle che la GNN ha gia' visto."""
    ids = set(df_held["PuzzleId"])
    fens = set(df_held["fen_norm"]) | set(df_held["solve_fen_norm"])
    for name, d in splits.items():
        id_ov = int(d["PuzzleId"].isin(ids).sum())
        fen_ov = int((d["fen_norm"].isin(fens) | d["solve_fen_norm"].isin(fens)).sum())
        if id_ov or fen_ov:
            raise RuntimeError(
                f"LEAKAGE: {name} contiene {id_ov} PuzzleId e {fen_ov} posizioni "
                "presenti nell'held-out."
            )
    print("Controllo anti-leakage OK: nessun puzzle dell'held-out (per ID o per FEN) "
          "in train/val/test.")


def check_no_leakage_from_files() -> None:
    """Controlla train/val/test contro ENTRAMBI gli held-out persistenti."""
    held_paths = [config.HELD_OUT_PARQUET, config.HELD_OUT_LICHESS_PARQUET]
    missing = [p for p in held_paths if not os.path.exists(p)]
    if missing:
        print("ATTENZIONE: manca uno degli held-out; esegui `python -m scripts.run_pipeline` "
              f"prima del training. File mancanti: {missing}")
    existing = [p for p in held_paths if os.path.exists(p)]
    if not existing:
        print("ATTENZIONE: nessun held-out trovato, salto il controllo anti-leakage.")
        return

    held_frames = [pd.read_parquet(p, columns=["PuzzleId", "FEN"])
                   for p in existing]
    held = pd.concat(held_frames, ignore_index=True).drop_duplicates(subset=["PuzzleId"])
    ids = set(held["PuzzleId"].astype(str))
    fens = {normalize_fen(f) for f in held["FEN"]}
    for name, path in [("train", config.TRAIN_PARQUET), ("val", config.VAL_PARQUET),
                       ("test", config.TEST_PARQUET)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet {name} non trovato: {path}")
        d = pd.read_parquet(path, columns=["PuzzleId", "FEN"])
        id_ov = int(d["PuzzleId"].astype(str).isin(ids).sum())
        fen_ov = int(d["FEN"].map(normalize_fen).isin(fens).sum())
        if id_ov or fen_ov:
            raise RuntimeError(
                f"LEAKAGE: {name} contiene {id_ov} PuzzleId e {fen_ov} posizioni "
                "presenti in uno dei due held-out. I file di dati sono di una "
                "versione precedente: rilancia `python -m scripts.run_pipeline` "
                "e riaddestra i modelli sui nuovi split."
            )
    print("Controllo anti-leakage OK: train/val/test non contengono puzzle "
          "di nessuno dei due held-out.")


# ---------------------------------------------------------------------
# Orchestratore Fase 1
# ---------------------------------------------------------------------
def run_phase1(skip_download: bool = False) -> None:
    if not skip_download:
        download_file(config.PUZZLE_URL, config.ZST_PATH)
        decompress_zst(config.ZST_PATH, config.CSV_PATH)

    # Usa l'header reale del CSV: lo schema di Lichess può aggiungere colonne.
    df = pd.read_csv(config.CSV_PATH)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Colonne attese mancanti nel CSV Lichess: {missing}. "
            f"Colonne trovate: {list(df.columns)}. Lo schema del file e' "
            "probabilmente cambiato: aggiorna COLUMNS in src/data_pipeline.py."
        )
    print(f"Totale puzzle: {len(df):,}")

    df_pool = filter_mate_puzzles(df)
    if len(df_pool) == 0:
        raise RuntimeError(
            "0 puzzle 'mate in n' trovati: la lettura del CSV non ha "
            "funzionato come atteso (vedi diagnostica sopra). Esegui "
            "`python -m scripts.debug_csv_columns` per indagare prima di "
            "andare avanti — non ha senso proseguire con un dataset vuoto."
        )
    df_valid = validate_dataframe(df_pool)
    df_timed = ensure_opening_flag(df_valid)

    # ---- HELD-OUT 1: composizione mista storica, invariata ----
    print("\n--- HELD-OUT misto: Polgar N1-N3 + Lichess N4-N5 (30 per N) ---")
    df_held_lichess_mixed = select_lichess_held_out(
        df_timed, ns=config.HELD_OUT_MIXED_LICHESS_N,
        k=config.HELD_OUT_PER_N, seed=config.HELD_OUT_SEED,
    )

    download_polgar_pgn()
    df_polgar = parse_polgar_pgn()
    if len(df_polgar) == 0:
        raise RuntimeError("Nessun problema Polgar valido: impossibile costruire l'held-out n=1-3.")
    df_held_polgar = select_polgar_held_out(df_polgar, df_timed)
    df_held_polgar = ensure_opening_flag(df_held_polgar)

    df_held_mixed = pd.concat([df_held_polgar, df_held_lichess_mixed], ignore_index=True)
    df_held_mixed = df_held_mixed.sort_values(["mate_n", "PuzzleId"]).reset_index(drop=True)

    # ---- HELD-OUT 2: solo Lichess, N1-N5, disgiunto dal misto ----
    # Rimuoviamo prima TUTTO l'held-out misto dal pool dei candidati: così i
    # puzzle N4-N5 del nuovo held-out non possono coincidere con quelli misti,
    # e neanche le posizioni N1-N3 possono sovrapporsi ai puzzle Polgar.
    df_pool_after_mixed, n_removed_mixed = remove_from_pool(df_timed, df_held_mixed)
    print("\n--- HELD-OUT solo Lichess: N1-N5 (30 per N), senza sovrapposizioni col misto ---")
    df_held_lichess_only = select_lichess_held_out(
        df_pool_after_mixed,
        ns=config.HELD_OUT_LICHESS_ONLY_N,
        k=config.HELD_OUT_PER_N,
        seed=config.HELD_OUT_SEED + config.HELD_OUT_LICHESS_ONLY_SEED_OFFSET,
    )
    df_held_lichess_only = df_held_lichess_only.sort_values(
        ["mate_n", "PuzzleId"]
    ).reset_index(drop=True)
    if not (df_held_lichess_only["source"] == "lichess").all():
        raise RuntimeError("L'held-out Lichess-only contiene una fonte diversa da Lichess.")

    # Il pool di training esclude entrambi gli held-out, per PuzzleId e per
    # posizione normalizzata, PRIMA di eseguire train/validation/test split.
    df_pool_train, n_removed_lichess_only = remove_from_pool(
        df_pool_after_mixed, df_held_lichess_only
    )
    print(f"Rimossi dal pool {n_removed_mixed:,} puzzle per l'held-out misto e "
          f"{n_removed_lichess_only:,} per l'held-out Lichess-only.")
    print(f"Held-out misto: {len(df_held_mixed):,} puzzle; "
          f"held-out solo Lichess: {len(df_held_lichess_only):,} puzzle.")

    # ---- SUBSET COMPLESSIVO 300k DOPO L'ESCLUSIONE DEI DUE HELD-OUT ----
    # Il target è il totale prima dello split, NON 300k soltanto per il train.
    df_subset = select_balanced_subset(
        df_pool_train, target_size=config.SUBSET_SIZE,
        n_rating_bins=config.SUBSET_RATING_BINS, seed=config.SUBSET_SEED,
    )

    # ---- SPLIT del subset selezionato: 80% train, 10% validation, 10% test ----
    df_train, df_val, df_test = stratified_split(df_subset)
    print(f"\nSplit finale del subset: train={len(df_train):,} ({len(df_train)/len(df_subset):.1%}) | "
          f"validation={len(df_val):,} ({len(df_val)/len(df_subset):.1%}) | "
          f"test={len(df_test):,} ({len(df_test)/len(df_subset):.1%})")

    # Rating normalizzato ricavato SOLTANTO dal training set e applicato a tutti.
    rating_stats = compute_rating_stats(df_train)
    df_train = add_rating_norm(df_train, rating_stats)
    df_val = add_rating_norm(df_val, rating_stats)
    df_test = add_rating_norm(df_test, rating_stats)
    df_held_mixed = add_rating_norm(df_held_mixed, rating_stats)
    df_held_lichess_only = add_rating_norm(df_held_lichess_only, rating_stats)

    # Unione usata SOLO per controllare anti-leakage; i due held-out restano
    # salvati separatamente e sono disgiunti per ID e posizione.
    all_held_ids = set(df_held_mixed["PuzzleId"].astype(str))
    all_held_fens = set(df_held_mixed["fen_norm"]) | set(df_held_mixed["solve_fen_norm"])
    overlap_ids = set(df_held_lichess_only["PuzzleId"].astype(str)) & all_held_ids
    overlap_fens = (set(df_held_lichess_only["fen_norm"]) |
                    set(df_held_lichess_only["solve_fen_norm"])) & all_held_fens
    if overlap_ids or overlap_fens:
        raise RuntimeError(
            f"I due held-out non sono disgiunti: {len(overlap_ids)} PuzzleId e "
            f"{len(overlap_fens)} posizioni sovrapposte. Interrompo per evitare leakage."
        )
    df_held_union = pd.concat([df_held_mixed, df_held_lichess_only], ignore_index=True)
    check_no_leakage(df_held_union, {"train": df_train, "val": df_val, "test": df_test})

    print_n_distribution(
        {"train": df_train, "val": df_val, "test": df_test,
         "held-out misto": df_held_mixed,
         "held-out Lichess-only": df_held_lichess_only},
        "Distribuzione dei matti in N per insieme (numero di PUZZLE)",
        csv_path=config.DISTRIBUTION_CSV,
    )
    print("(Ogni puzzle mate-in-n genera n esempi (posizione, mossa) in training.)")

    # Salvataggi Parquet. Scriviamo il manifest per dichiarare che i due
    # held-out e gli split appartengono alla stessa generazione della pipeline.
    df_train[config.TRAINING_COLS].to_parquet(config.TRAIN_PARQUET, index=False)
    df_val[config.TRAINING_COLS].to_parquet(config.VAL_PARQUET, index=False)
    df_test[config.TRAINING_COLS].to_parquet(config.TEST_PARQUET, index=False)
    df_held_mixed[config.HELD_OUT_COLS].to_parquet(config.HELD_OUT_PARQUET, index=False)
    df_held_lichess_only[config.HELD_OUT_COLS].to_parquet(
        config.HELD_OUT_LICHESS_PARQUET, index=False
    )

    manifest = {
        "schema_version": config.DATA_PIPELINE_SCHEMA_VERSION,
        "held_out_design": "mixed_polgar_n1_n3_lichess_n4_n5_plus_lichess_only_n1_n5",
        "counts": {
            "subset_total": int(len(df_subset)),
            "train": int(len(df_train)), "validation": int(len(df_val)),
            "test": int(len(df_test)), "held_out_mixed": int(len(df_held_mixed)),
            "held_out_lichess_only": int(len(df_held_lichess_only)),
        },
        "subset_design": {
            "target_total": int(config.SUBSET_SIZE),
            "rating_bins_global_quantiles": int(config.SUBSET_RATING_BINS),
            "seed": int(config.SUBSET_SEED),
            "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "balance_method": "mate_n first; within mate_n balanced across global rating quantile bins; joint-stratified split",
            "subset_counts_by_mate_n": {str(int(n)): int(count) for n, count in df_subset.groupby("mate_n").size().items()},
            "subset_counts_by_mate_n_rating_bin": joint_rating_distribution(df_subset),
            "train_counts_by_mate_n_rating_bin": joint_rating_distribution(df_train),
            "validation_counts_by_mate_n_rating_bin": joint_rating_distribution(df_val),
            "test_counts_by_mate_n_rating_bin": joint_rating_distribution(df_test),
        },
        "held_out_lichess_only_by_n": {
            str(int(n)): int(count)
            for n, count in df_held_lichess_only.groupby("mate_n").size().items()
        },
    }
    manifest_tmp = config.DATA_PIPELINE_MANIFEST_JSON + ".tmp"
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(manifest_tmp, config.DATA_PIPELINE_MANIFEST_JSON)

    print(f"\nSalvati train/val/test + held-out misto ({len(df_held_mixed)} puzzle, "
          f"{config.HELD_OUT_PARQUET}) + held-out Lichess-only "
          f"({len(df_held_lichess_only)} puzzle, {config.HELD_OUT_LICHESS_PARQUET}).")
    print(f"Manifest pipeline: {config.DATA_PIPELINE_MANIFEST_JSON}")
    print("ATTENZIONE: i checkpoint precedenti devono essere riaddestrati, perché "
          "train/validation/test ora escludono anche il nuovo held-out Lichess-only.")


if __name__ == "__main__":
    run_phase1()
