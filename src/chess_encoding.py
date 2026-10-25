"""
src/chess_encoding.py
----------------------
Funzioni di encoding condivise FEN/mosse -> feature per il grafo.

Rappresentazione:
- Nodi: le 64 caselle della scacchiera. Feature = tipo di pezzo (6, one-hot)
  x colore (2) + flag "casella vuota" + (file, rank) normalizzati + timing.
- Archi: tipo 0 = mossa legale (from_square -> to_square), tipo 1 = relazione
  di attacco (per catturare tattica/minacce anche oltre le mosse legali,
  es. pezzi inchiodati).
- Target: indice, tra le mosse legali, della mossa di soluzione.

Nota sul formato Lichess: la FEN è la posizione *prima* della mossa di
apertura dell'avversario. `Moves` è quindi una sequenza alternata:
    [mossa_avversario, mossa_solver, mossa_avversario, mossa_solver, ...]
Le mosse del "solver" (quello che deve trovare il matto) sono quindi agli
indici dispari 1, 3, 5, ... della lista originale (o, guardando solo la
parte dopo la mossa di apertura, agli indici pari 0, 2, 4, ...).

`solver_moves_sequence` espone TUTTE le mosse del solver lungo la linea di
matto (non solo la prima), con la posizione-prima-di-ciascuna-mossa e
l'eventuale mossa di risposta forzata dell'avversario. Questo è il pezzo
chiave per poter allenare/valutare la GNN sull'intera sequenza e non solo
sulla prima mossa.
"""

from typing import List, Optional, Tuple

import math

import chess
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import config

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
NUM_PIECE_FEATS = len(PIECE_TYPES) * 2 + 1     # 6 tipi x 2 colori + flag 'vuota'
NUM_POS_FEATS = 2                               # file, rank normalizzati
NUM_TIMING_FEATS = 1                            # tempo di pensiero simulato (broadcast)
NODE_FEAT_DIM = NUM_PIECE_FEATS + NUM_POS_FEATS + NUM_TIMING_FEATS
EDGE_FEAT_DIM = 2                               # one-hot: mossa legale / attacco


def square_node_features(board: chess.Board, square: int) -> List[float]:
    """Vettore di feature (senza timing) per una singola casella."""
    feat = [0.0] * NUM_PIECE_FEATS
    piece = board.piece_at(square)
    if piece is None:
        feat[-1] = 1.0
    else:
        idx = PIECE_TYPES.index(piece.piece_type)
        color_offset = 0 if piece.color == chess.WHITE else len(PIECE_TYPES)
        feat[idx + color_offset] = 1.0

    file_ = chess.square_file(square) / 7.0
    rank_ = chess.square_rank(square) / 7.0
    return feat + [file_, rank_]


def build_edges(board: chess.Board):
    """Ritorna (edge_index, edge_attr, legal_moves).

    - edge_index: tensore [2, num_archi]
    - edge_attr: one-hot [num_archi, 2] (tipo 0 = mossa legale, tipo 1 = attacco)
    - legal_moves: lista delle mosse legali, nello stesso ordine dei primi
      `len(legal_moves)` archi (quelli di tipo 0). L'ordine di questa lista
      corrisponde 1:1 all'ordine degli score prodotti dal modello per le
      mosse candidate (vedi src/model.py), quindi va sempre usata per
      tradurre l'indice predetto dal modello in una mossa UCI reale.
    """
    edge_src, edge_dst, edge_type = [], [], []

    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        edge_src.append(move.from_square)
        edge_dst.append(move.to_square)
        edge_type.append(0)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        for target_sq in board.attacks(square):
            edge_src.append(square)
            edge_dst.append(target_sq)
            edge_type.append(1)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type_t = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type_t, num_classes=2).float()
    return edge_index, edge_attr, legal_moves


def compute_think_time_sec(board: chess.Board, rating_norm: float, moves_left: int) -> float:
    """Timing proxy v2: tempo di pensiero SIMULATO (secondi, tra 1 e 10) per la
    posizione `board`. Deterministico (niente rumore) e calcolato per posizione.

        D = w_R*R + w_B*B + w_F*F + w_K*K        T = clip(1 + 9*D, 1, 10)

    - R: `rating_norm`, rating del puzzle normalizzato (P10-P90 del training).
    - B: numero di mosse legali (scala log) tra TIMING_LEGAL_MOVES_RANGE.
    - F: numero di mosse legali che danno scacco (candidate forzanti).
    - K: `moves_left` = mosse del solver ancora da giocare fino al matto
         (1 = ultima mossa), normalizzato su TIMING_MAX_DEPTH.
    Pesi in config.TIMING_WEIGHTS: valori iniziali, non appresi da tempi reali.
    NB: K e' un'informazione che anche il prompt dell'LLM contiene (il "mate in
    n"); per misurare il timing "puro" metti TIMING_WEIGHTS["K"] = 0.
    """
    w = config.TIMING_WEIGHTS
    legal = list(board.legal_moves)
    lo, hi = config.TIMING_LEGAL_MOVES_RANGE
    b = (math.log(1 + len(legal)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    b = min(max(b, 0.0), 1.0)
    n_checks = sum(1 for m in legal if board.gives_check(m))
    f = min(n_checks / config.TIMING_FORCING_CAP, 1.0)
    k = min(max((moves_left - 1) / (config.TIMING_MAX_DEPTH - 1), 0.0), 1.0)
    r = min(max(float(rating_norm), 0.0), 1.0)
    d = w["R"] * r + w["B"] * b + w["F"] * f + w["K"] * k
    return float(min(max(1.0 + 9.0 * d, 1.0), 10.0))


def board_to_graph_data(board: chess.Board, target_move: Optional[chess.Move] = None,
                         think_time_sec: Optional[float] = None) -> Tuple[Data, List[chess.Move]]:
    """Costruisce l'oggetto Data (nodi/archi/feature) per UNA posizione.

    Funzione condivisa da:
    - src/dataset.py (costruzione del dataset di training/validation/test)
    - src/train.py (valutazione della GNN mossa-per-mossa sull'intera linea
      di matto, in inferenza, posizione dopo posizione)

    `think_time_sec=None` equivale a "niente timing" (feature azzerata),
    usato per la variante no-timing dell'ablation study.

    Ritorna (data, legal_moves): `legal_moves[i]` è la mossa UCI reale
    corrispondente allo score in posizione `i` prodotto dal modello.
    """
    node_feats = [square_node_features(board, sq) for sq in chess.SQUARES]
    node_feats = torch.tensor(node_feats, dtype=torch.float)

    if think_time_sec is not None:
        think_time_norm = min(think_time_sec, config.THINK_TIME_CLIP_SEC) / config.THINK_TIME_CLIP_SEC
    else:
        think_time_norm = 0.0
    timing_col = torch.full((node_feats.size(0), 1), think_time_norm, dtype=torch.float)
    node_feats = torch.cat([node_feats, timing_col], dim=1)

    edge_index, edge_attr, legal_moves = build_edges(board)

    data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
    data.num_candidate_moves = torch.tensor([len(legal_moves)], dtype=torch.long)

    if target_move is not None:
        try:
            target_local_idx = legal_moves.index(target_move)
        except ValueError:
            # Non dovrebbe succedere se la Fase 1 ha validato correttamente i dati.
            target_local_idx = -1
        data.y = torch.tensor([target_local_idx], dtype=torch.long)

    return data, legal_moves


def solver_moves_sequence(fen: str, moves_str: str, has_opening_move: bool = True):
    """Ricostruisce TUTTA la linea di matto e la spezzetta per mossa del
    solver (non solo la prima).

    `has_opening_move`: True per i puzzle Lichess (la FEN è la posizione
    PRIMA della mossa che apre il puzzle, quindi la prima mossa in
    `moves_str` va giocata e scartata). False per i puzzle "classici"
    (Polgár), dove la FEN è già la posizione da risolvere
    e TUTTE le mosse in `moves_str` fanno parte della linea di matto.

    Ritorna una lista di tuple (board_before, target_move, opponent_reply):
    - board_before: chess.Board nella posizione PRIMA che il solver giochi
      la sua mossa a questo punto della linea
    - target_move: la mossa che il solver deve trovare in questa posizione
    - opponent_reply: la mossa di risposta (forzata, presa dai dati) che
      l'avversario gioca subito dopo, oppure None se `target_move` è la
      mossa di matto finale

    Esempio per un "mate in 2" Lichess (Moves ha 4 mosse: apertura
    avversario, mossa1 solver, risposta avversario, mossa2 solver=matto):
        steps[0] = (board dopo apertura,           mossa1, risposta_avversario)
        steps[1] = (board dopo apertura+1+risposta, mossa2 (matto), None)

    Per un "mate in 2" classico (Moves ha SOLO le 3 mosse della linea:
    mossa1 solver, risposta avversario, mossa2 solver=matto), identico
    ma senza consumare una prima mossa come apertura.
    """
    board = chess.Board(fen)
    moves_uci = moves_str.split()
    min_len = 2 if has_opening_move else 1
    if len(moves_uci) < min_len:
        return []

    moves = [chess.Move.from_uci(u) for u in moves_uci]
    if has_opening_move:
        board.push(moves[0])             # mossa di apertura dell'avversario
        solver_and_replies = moves[1:]   # alterna: solver, avversario, solver, avversario, ...
    else:
        solver_and_replies = moves       # già tutta la linea del solver, dal primo tratto

    steps = []
    i = 0
    while i < len(solver_and_replies):
        target_move = solver_and_replies[i]
        opponent_reply = solver_and_replies[i + 1] if i + 1 < len(solver_and_replies) else None
        steps.append((board.copy(), target_move, opponent_reply))

        board.push(target_move)
        if opponent_reply is not None:
            board.push(opponent_reply)
        i += 2

    return steps


def get_position_after_opening(fen: str, moves_str: str, has_opening_move: bool = True) -> Tuple[str, List[str]]:
    """Ritorna la FEN dopo la mossa di apertura (se presente) + TUTTE le
    mosse di soluzione (usato dalla baseline LLM, che deve proporre
    l'intera sequenza, non solo la prima mossa).

    `has_opening_move`: vedi solver_moves_sequence — False per i puzzle
    "classici" dove la FEN è già la posizione da risolvere.
    """
    board = chess.Board(fen)
    moves = moves_str.split()
    if has_opening_move and len(moves) >= 2:
        board.push(chess.Move.from_uci(moves[0]))
        target_moves = moves[1:]
    else:
        target_moves = moves
    return board.fen(), target_moves
