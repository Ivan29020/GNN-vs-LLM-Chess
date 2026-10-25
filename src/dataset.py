"""
src/dataset.py
----------------
Fase 2 — Costruzione del grafo.

Definisce `ChessPuzzleGraphDataset`, che converte on-the-fly ogni puzzle in
UNO O PIU' oggetti torch_geometric.data.Data: uno per ciascuna mossa del
solver lungo l'intera linea di matto (non solo la prima mossa).

Esempio: un puzzle "mate in 3" ha 3 mosse del solver -> genera 3 esempi di
training, uno per ogni posizione della linea (la 1a mossa, poi la posizione
dopo 1a mossa + risposta avversario, poi la posizione dopo 2a mossa +
risposta avversario). Questo è necessario perché il modello, per risolvere
un puzzle intero in valutazione (src/train.py:solve_puzzle_with_gnn), deve
saper giocare bene anche le posizioni che si incontrano più avanti nella
linea, non solo la prima — se lo alleniamo solo sulla prima mossa, non ha
mai visto (durante il training) le posizioni successive e fallirà quasi
sempre la sequenza completa.

I grafi vengono mantenuti in una cache RAM LIMITATA e compressa: gli indici
arco sono int16 e gli attributi arco uint8. Oltre al limite di configurazione,
si ricostruiscono on-the-fly per evitare crescita illimitata della RAM.

Il flag `include_timing` implementa l'ablation study (Fase 5 del progetto:
modello con vs senza feature temporale).
"""

import pandas as pd
import torch
import chess
from torch_geometric.data import Data, Dataset

import config

from src.chess_encoding import board_to_graph_data, solver_moves_sequence, compute_think_time_sec


class ChessPuzzleGraphDataset(Dataset):
    """Dataset espanso senza cache illimitata degli oggetti grafo.

    Le posizioni intermedie vengono conservate come FEN + UCI (molto più
    compatte di centinaia di migliaia di oggetti chess.Board). I grafi hanno
    una cache compressa e con limite di capienza; gli esempi oltre il limite
    vengono ricostruiti on-the-fly, senza crescita illimitata della RAM.
    """
    def __init__(self, df: pd.DataFrame, include_timing: bool = True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.include_timing = include_timing

        # Per ogni mossa memorizziamo solo (FEN, target UCI, mosse del solver
        # rimanenti), non l'intero oggetto Board copiato per ogni posizione.
        self._index = []
        self._solver_cache = {}
        for row_idx in range(len(self.df)):
            row = self.df.iloc[row_idx]
            steps = solver_moves_sequence(
                row["FEN"], row["Moves"],
                has_opening_move=row.get("has_opening_move", True),
            )
            compact_steps = []
            n_steps = len(steps)
            for ply_idx, (board, target_move, _opponent_reply) in enumerate(steps):
                compact_steps.append((board.fen(), target_move.uci(), n_steps - ply_idx))
                self._index.append((row_idx, ply_idx))
            self._solver_cache[row_idx] = compact_steps
            # `steps` contieneva i Board completi: una volta terminato questo
            # giro, non li conserviamo più in memoria.

        # Cache limitata e compressa. Il vecchio dict conservava tutti i grafi
        # in float32/int64, facendo crescere la RAM fino a diversi GB.
        # x resta float32 (nessuna perdita di precisione); edge_index usa
        # int16 (le caselle sono 0..63) ed edge_attr uint8 (one-hot 0/1).
        self._data_cache = {}
        self._max_graph_cache_items = max(
            0, int(getattr(config, "GRAPH_CACHE_MAX_ITEMS", 500_000))
        )

    def len(self):
        return len(self._index)

    def examples_per_n(self) -> dict:
        """Numero di esempi posizione-mossa per ogni valore di mate_n."""
        counts = {}
        mate_n = self.df["mate_n"].to_numpy()
        for row_idx, _ply in self._index:
            n = int(mate_n[row_idx])
            counts[n] = counts.get(n, 0) + 1
        return dict(sorted(counts.items()))

    def _restore_cached_graph(self, idx, row_idx, ply_idx):
        x, edge_index_i16, edge_attr_u8, candidate_count, target_idx = self._data_cache[idx]
        data = Data(
            x=x,
            edge_index=edge_index_i16.to(torch.long),
            edge_attr=edge_attr_u8.to(torch.float32),
            num_candidate_moves=torch.tensor([candidate_count], dtype=torch.long),
            y=torch.tensor([target_idx], dtype=torch.long),
        )
        row = self.df.iloc[row_idx]
        data.mate_n = torch.tensor([int(row["mate_n"])], dtype=torch.long)
        data.ply_idx = torch.tensor([ply_idx], dtype=torch.long)
        data.puzzle_id = row["PuzzleId"]
        return data

    def get(self, idx):
        row_idx, ply_idx = self._index[idx]
        if idx in self._data_cache:
            return self._restore_cached_graph(idx, row_idx, ply_idx)

        row = self.df.iloc[row_idx]
        fen, target_uci, moves_left = self._solver_cache[row_idx][ply_idx]
        board = chess.Board(fen)
        target_move = chess.Move.from_uci(target_uci)
        think_time_sec = (
            compute_think_time_sec(board, row["rating_norm"], moves_left)
            if self.include_timing else None
        )
        data, _legal_moves = board_to_graph_data(
            board, target_move=target_move, think_time_sec=think_time_sec
        )
        data.mate_n = torch.tensor([int(row["mate_n"])], dtype=torch.long)
        data.ply_idx = torch.tensor([ply_idx], dtype=torch.long)
        data.puzzle_id = row["PuzzleId"]

        # Conserva al massimo GRAPH_CACHE_MAX_ITEMS grafi, comprimendo solo
        # gli indici/attributi degli archi. La capienza evita che un subset
        # grande occupi indefinitamente tutta la RAM di Colab. Quando la cache
        # è piena, gli esempi rimanenti sono calcolati on-the-fly.
        if len(self._data_cache) < self._max_graph_cache_items:
            self._data_cache[idx] = (
                data.x.contiguous(),
                data.edge_index.to(torch.int16).contiguous(),
                data.edge_attr.to(torch.uint8).contiguous(),
                int(data.num_candidate_moves.item()),
                int(data.y.item()),
            )
            # Usa la stessa rappresentazione anche al primo accesso per
            # mantenere identici i tensori tra prima epoca e cache hit.
            return self._restore_cached_graph(idx, row_idx, ply_idx)
        return data

