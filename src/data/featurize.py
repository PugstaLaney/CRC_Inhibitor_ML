"""SMILES → PyTorch Geometric molecular graph featurization.

Atom-level features: 24-dim (element one-hot, hybridization one-hot, charge,
hydrogen count, degree, aromaticity flag, ring-membership flag).

Bond-level features: 6-dim (bond type one-hot, conjugation flag, ring flag).

Compatible with the GINEConv layer (PyG) — both atom and bond features are
projected to the hidden dim inside the model.
"""
from __future__ import annotations

from typing import Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data

ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "H", "Other"]
HYB_TYPES = [
    Chem.HybridizationType.S,
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

N_ATOM_FEATURES = len(ATOM_TYPES) + len(HYB_TYPES) + 5
N_BOND_FEATURES = len(BOND_TYPES) + 2


def _one_hot(value, options):
    return [int(value == o) for o in options]


def atom_features(atom):
    sym = atom.GetSymbol() if atom.GetSymbol() in ATOM_TYPES else "Other"
    return (
        _one_hot(sym, ATOM_TYPES)
        + _one_hot(atom.GetHybridization(), HYB_TYPES)
        + [
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetDegree(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ]
    )


def bond_features(bond):
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    )


def smiles_to_pyg(
    smiles: str,
    target_emb: Optional[torch.Tensor] = None,
    y: float = 0.0,
) -> Optional[Data]:
    """Convert a SMILES string to a PyG Data object.

    Returns None if the SMILES is unparseable. Set `target_emb` to attach a
    target representation (e.g., from ESM-2) for multi-target models.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        eattr += [bf, bf]

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, N_BOND_FEATURES), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)

    kwargs = dict(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.float),
    )
    if target_emb is not None:
        kwargs["target_emb"] = target_emb.unsqueeze(0)
    return Data(**kwargs)
