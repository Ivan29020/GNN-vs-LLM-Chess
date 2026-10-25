"""
src/model.py
-------------
Fase 3 — Modello `TimedGNN` (equivalente alla sezione 2 di
03_gnn_model_training.ipynb).

- Encoder: message passing con GINEConv (supporta nativamente edge_attr,
  quindi distingue archi "mossa legale" da archi "attacco"), con residual
  + layer norm.
- Policy head: per ogni arco "mossa legale", concatena gli embedding dei
  nodi sorgente/destinazione e produce uno score; la mossa scelta è quella
  con score piu' alto TRA le mosse legali della stessa posizione (softmax
  raggruppata per grafo, non globale sul batch — vedi src/train.py).

`include_timing` è una proprietà del DATASET (src/dataset.py), non del
modello: la stessa architettura viene addestrata due volte (con e senza
la feature di timing azzerata) per l'ablation study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

from src.chess_encoding import EDGE_FEAT_DIM


class TimedGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim=EDGE_FEAT_DIM,
                 hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_feat_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch_vec = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h = F.relu(h_new) + h  # residual
            h = self.dropout(h)

        candidate_mask = edge_attr[:, 0] == 1
        cand_src = edge_index[0][candidate_mask]
        cand_dst = edge_index[1][candidate_mask]

        edge_repr = torch.cat([h[cand_src], h[cand_dst]], dim=1)
        scores = self.edge_scorer(edge_repr).squeeze(-1)
        edge_batch = batch_vec[cand_src]

        return scores, edge_batch
