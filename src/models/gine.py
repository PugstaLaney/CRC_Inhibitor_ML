"""GINE-based GNN architectures.

`MultiModalGINE` is the Phase 4 architecture: a GINE molecular encoder fused
with a projected protein-language-model embedding (ESM-2). One model handles
any target via its protein embedding — no retraining per target.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool


class MultiModalGINE(nn.Module):
    """Molecule (GINE) + target (ESM-2 embedding) → predicted pIC50.

    Args:
        atom_dim:    Atom-feature dimensionality of the input molecular graphs.
        edge_dim:    Bond-feature dimensionality of the input molecular graphs.
        target_dim:  Dimensionality of the protein embedding (e.g., 480 for ESM-2 t12_35M).
        hidden_dim:  Internal representation dimensionality (default 128).
        n_layers:    Number of GINE message-passing layers (default 3).
        dropout:     Dropout probability applied after each conv (default 0.2).
    """

    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        target_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(atom_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        target_emb = data.target_emb  # (batch_size, target_dim)

        # Molecule branch
        h = self.input_proj(x)
        e = self.edge_proj(edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, e)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
        mol_emb = global_add_pool(h, batch)  # (batch_size, hidden_dim)

        # Target branch
        tgt_emb = self.target_proj(target_emb)  # (batch_size, hidden_dim)

        # Fuse and predict
        combined = torch.cat([mol_emb, tgt_emb], dim=1)
        return self.predictor(combined).squeeze(-1)


def load_multi_modal_gine(checkpoint_path, device=None):
    """Load a saved MultiModalGINE checkpoint and return the model in eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultiModalGINE(
        atom_dim=ckpt["atom_dim"],
        edge_dim=ckpt["edge_dim"],
        target_dim=ckpt["target_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
