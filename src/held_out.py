"""
src/held_out.py
-----------------
Costruzione del set HELD-OUT esterno (quello usato SOLO per il confronto
finale GNN vs LLM). Struttura richiesta: 30 puzzle per ciascun n da 1 a 5.

    n = 1, 2, 3  ->  problemi di László Polgár (raccolta del 1994, fonte
                     indipendente da Lichess; PGN pubblico su GitHub)
    n = 4, 5     ->  puzzle Lichess, estratti dal pool PRIMA dello split
                     train/val/test e rimossi dal pool (per PuzzleId e per
                     FEN), cosi' non possono finire nel training.

Tutto viene eseguito da `python -m scripts.run_pipeline` (vedi
src/data_pipeline.run_phase1): non ci sono piu' script separati da lanciare
a mano, ne' file esterni da scaricare/posizionare.

Perche' questo ordine (held-out PRIMA dello split) e' importante: se prima
si fa lo split e poi si pesca l'held-out, i puzzle pescati sono gia' finiti
in train/val/test (leakage). Qui il pool viene prima ridotto e solo dopo
diviso.

Convenzione per `has_opening_move`:
- Lichess: True (la FEN e' PRIMA della mossa di apertura dell'avversario)
- Polgar:  False (la FEN e' gia' la posizione da risolvere)
"""

import os

import chess
import chess.pgn
import numpy as np
import pandas as pd
import requests

import config


# ---------------------------------------------------------------------
# Utilita'
# ---------------------------------------------------------------------
def normalize_fen(fen: str) -> str:
    """FEN 'normalizzato' per i confronti: disposizione pezzi, lato al
    tratto e diritti di arrocco (ignora en passant e contatori mosse, che
    non cambiano l'identita' della posizione ai fini della deduplicazione)."""
    parts = str(fen).split()
    return " ".join(parts[:3]) if len(parts) >= 3 else str(fen)


def _round_robin_quota(sizes: dict, k: int) -> dict:
    """Ripartisce k fra i gruppi il piu' equamente possibile, ridistribuendo
    la quota dei gruppi troppo piccoli per spenderla."""
    quota = {g: 0 for g in sizes}
    remaining = int(k)
    while remaining > 0 and any(quota[g] < sizes[g] for g in sizes):
        for g in sorted(sizes):
            if remaining == 0:
                break
            if quota[g] < sizes[g]:
                quota[g] += 1
                remaining -= 1
    return quota


def stratified_rating_sample(df: pd.DataFrame, k: int, n_bins: int, seed: int) -> pd.DataFrame:
    """Estrae k righe da `df` stratificando per RATING (n_bins quantili,
    stessa quota per fascia). Un campione casuale semplice pescherebbe quasi
    solo puzzle di rating medio/basso (i piu' numerosi); cosi' invece il
    campione copre l'intera gamma di difficolta'. Deterministico (seed fisso,
    righe ordinate per PuzzleId prima di campionare)."""
    df = df.sort_values("PuzzleId").reset_index(drop=True)
    if len(df) <= k:
        return df
    q = max(1, min(n_bins, len(df)))
    bins = pd.qcut(df["Rating"].astype(float).rank(method="first"), q=q, labels=False)
    groups = {int(b): df[bins == b] for b in sorted(bins.unique())}
    quota = _round_robin_quota({b: len(g) for b, g in groups.items()}, k)
    parts = [groups[b].sample(quota[b], random_state=seed) for b in groups if quota[b] > 0]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------
# n = 4, 5: puzzle Lichess estratti PRIMA dello split
# ---------------------------------------------------------------------
def select_lichess_held_out(df_pool: pd.DataFrame, ns=config.HELD_OUT_LICHESS_N,
                            k: int = config.HELD_OUT_PER_N,
                            n_bins: int = config.HELD_OUT_RATING_BINS,
                            seed: int = config.HELD_OUT_SEED) -> pd.DataFrame:
    """k puzzle per ciascun n in `ns`, stratificati per rating. `df_pool`
    deve essere gia' validato (colonne `solve_fen_norm`, `fen_norm`)."""
    frames = []
    for n in ns:
        cand = df_pool[df_pool["mate_n"] == n]
        # niente candidati con la stessa posizione di partenza: l'held-out
        # non deve contenere duplicati al suo interno
        cand = cand.drop_duplicates(subset="solve_fen_norm")
        if len(cand) < k:
            print(f"  ATTENZIONE: per n={n} ci sono solo {len(cand)} puzzle Lichess "
                  f"(richiesti {k}): uso tutti quelli disponibili.")
        sel = stratified_rating_sample(cand, k, n_bins, seed + n)
        sel = sel.copy()
        sel["source"] = "lichess"
        frames.append(sel)
        print(f"  n={n}: {len(sel)} puzzle Lichess estratti (rating {sel['Rating'].min():.0f}"
              f"-{sel['Rating'].max():.0f}, mediana {sel['Rating'].median():.0f})")
    return pd.concat(frames, ignore_index=True)


def remove_from_pool(df_pool: pd.DataFrame, df_held: pd.DataFrame):
    """Toglie dal pool OGNI puzzle che coincide con un puzzle dell'held-out,
    per PuzzleId oppure per posizione (FEN normalizzato, sia quello di
    partenza che quello da risolvere, su entrambi i lati del confronto).
    Ritorna (pool_ridotto, numero_di_righe_rimosse)."""
    ids = set(df_held["PuzzleId"])
    held_fens = set(df_held["fen_norm"]) | set(df_held["solve_fen_norm"])
    bad = (
        df_pool["PuzzleId"].isin(ids)
        | df_pool["fen_norm"].isin(held_fens)
        | df_pool["solve_fen_norm"].isin(held_fens)
    )
    return df_pool[~bad].reset_index(drop=True), int(bad.sum())


# ---------------------------------------------------------------------
# n = 1, 2, 3: Polgar
# ---------------------------------------------------------------------
def download_polgar_pgn() -> None:
    if os.path.exists(config.POLGAR_PGN_LOCAL) and os.path.getsize(config.POLGAR_PGN_LOCAL) > 0:
        print(f"Polgar PGN gia' presente: {config.POLGAR_PGN_LOCAL}")
        return
    print(f"Download Polgar da {config.POLGAR_PGN_URL} ...")
    resp = requests.get(config.POLGAR_PGN_URL, timeout=60)
    resp.raise_for_status()
    tmp_path = config.POLGAR_PGN_LOCAL + ".part"
    with open(tmp_path, "wb") as f:
        f.write(resp.content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, config.POLGAR_PGN_LOCAL)
    print("Download completato.")


def parse_polgar_pgn() -> pd.DataFrame:
    """Legge polgar.pgn e tiene SOLO i problemi la cui linea e' interamente
    legale e termina in scacco matto (verifica con python-chess: e' l'UNICA
    validazione dell'held-out Polgar, non viene ripetuta altrove). La
    profondita' n e' calcolata dalla linea verificata, non dall'etichetta
    del libro."""
    results = []
    discard = {"no_fen_header": 0, "invalid_fen": 0, "zero_moves": 0,
               "illegal_move": 0, "not_checkmate": 0}

    with open(config.POLGAR_PGN_LOCAL, "r", encoding="utf-8", errors="replace") as f:
        idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            idx += 1

            fen = game.headers.get("FEN")
            if not fen:
                discard["no_fen_header"] += 1
                continue
            try:
                board = chess.Board(fen)
            except ValueError:
                discard["invalid_fen"] += 1
                continue

            mainline = list(game.mainline_moves())
            if not mainline:
                discard["zero_moves"] += 1
                continue

            uci_moves, ok = [], True
            for move in mainline:
                if move not in board.legal_moves:
                    ok = False
                    break
                uci_moves.append(move.uci())
                board.push(move)
            if not ok:
                discard["illegal_move"] += 1
                continue
            if not board.is_checkmate():
                discard["not_checkmate"] += 1
                continue

            results.append({
                "PuzzleId": f"polgar_{idx:04d}",
                "FEN": fen,
                "Moves": " ".join(uci_moves),
                "mate_n": (len(uci_moves) + 1) // 2,
                "has_opening_move": False,
                "source": "polgar",
            })

    df = pd.DataFrame(results)
    print(f"Polgar: {len(df)} problemi validi, {sum(discard.values())} scartati {discard}")
    if len(df):
        df["fen_norm"] = df["FEN"].map(normalize_fen)
        df["solve_fen_norm"] = df["fen_norm"]   # nessuna mossa di apertura
    return df


def select_polgar_held_out(df_polgar: pd.DataFrame, df_pool_all: pd.DataFrame,
                           ns=config.HELD_OUT_POLGAR_N, k: int = config.HELD_OUT_PER_N,
                           seed: int = config.HELD_OUT_SEED) -> pd.DataFrame:
    """k problemi Polgar per ciascun n in `ns`. Scarta i candidati la cui
    posizione compare gia' tra i puzzle Lichess (indipendenza dal training),
    poi campiona a caso con seed fisso. Aggiunge Rating/Popularity IMPUTATI
    (Polgar non li ha): mediana dei puzzle Lichess con lo stesso n, cosi' il
    rating normalizzato (e quindi il timing proxy v2) segue la STESSA regola
    dei puzzle Lichess invece di un default arbitrario."""
    lichess_fens = set(df_pool_all["fen_norm"]) | set(df_pool_all["solve_fen_norm"])
    n_dup = int(df_polgar["solve_fen_norm"].isin(lichess_fens).sum())
    if n_dup:
        print(f"  Polgar: {n_dup} candidati scartati perche' la posizione e' gia' in Lichess")
    df_polgar = df_polgar[~df_polgar["solve_fen_norm"].isin(lichess_fens)]
    df_polgar = df_polgar.drop_duplicates(subset="solve_fen_norm")

    frames = []
    for n in ns:
        cand = df_polgar[df_polgar["mate_n"] == n].sort_values("PuzzleId")
        if len(cand) < k:
            print(f"  ATTENZIONE: per n={n} Polgar ha solo {len(cand)} problemi "
                  f"(richiesti {k}): uso tutti quelli disponibili.")
        sel = cand.sample(min(k, len(cand)), random_state=seed + n).copy()
        pool_n = df_pool_all[df_pool_all["mate_n"] == n]
        sel["Rating"] = float(pool_n["Rating"].median()) if len(pool_n) else 1500.0
        sel["Popularity"] = float(pool_n["Popularity"].median()) if len(pool_n) else 0.0
        frames.append(sel)
        print(f"  n={n}: {len(sel)} problemi Polgar (Rating imputato = "
              f"{sel['Rating'].iloc[0] if len(sel) else float('nan'):.0f}, mediana Lichess per questo n)")
    return pd.concat(frames, ignore_index=True)
