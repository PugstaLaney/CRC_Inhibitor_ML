"""CRC_Inhibitor_ML — Phase 5 CLI for scoring molecule libraries against any target.

Given a target identifier (ChEMBL ID like CHEMBL203 or UniProt accession like
P00533) and a file of candidate SMILES strings, this script:

  1. Resolves the target to a UniProt accession (if needed)
  2. Fetches its protein sequence from UniProt
  3. Embeds the sequence with ESM-2 (cached after first run)
  4. Loads the trained multi-target Phase 4 model
  5. Scores every candidate SMILES against the target
  6. Writes a CSV ranked by predicted pIC50 (high = potent)

Example:
    python predict.py \\
        --target  CHEMBL203 \\
        --smiles  examples/sample_smiles.smi \\
        --output  predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.data.featurize import smiles_to_pyg
from src.data.proteins import get_target_embedding
from src.models.gine import load_multi_modal_gine

# Resolve project-relative defaults from the script's own location
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "gine_esm2_multi_target.pt"
DEFAULT_MAPPING = PROJECT_ROOT / "data" / "raw" / "chembl_uniprot_mapping.txt"
DEFAULT_EMB_CACHE = PROJECT_ROOT / "data" / "processed" / "target_esm2_embeddings.pt"


def load_smiles(path: Path) -> list[str]:
    """Read SMILES strings from a file, one per line. Blanks and comments stripped."""
    smiles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # Allow header / comment lines starting with #
        if not line or line.startswith("#"):
            continue
        # Allow optional second column (name) — take only the SMILES
        smiles.append(line.split()[0])
    return smiles


def score(model, smiles_list, target_emb, device, batch_size=64):
    """Featurize SMILES, batch through the model, return predictions + failures."""
    data_objects = []
    smiles_kept = []
    failed = []
    for smi in smiles_list:
        graph = smiles_to_pyg(smi, target_emb=target_emb, y=0.0)
        if graph is None:
            failed.append(smi)
        else:
            data_objects.append(graph)
            smiles_kept.append(smi)

    if not data_objects:
        return pd.DataFrame(columns=["smiles", "predicted_pic50"]), failed

    loader = DataLoader(data_objects, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).cpu().numpy())

    return pd.DataFrame({
        "smiles": smiles_kept,
        "predicted_pic50": np.concatenate(preds),
    }), failed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score molecule libraries against any ChEMBL / UniProt target.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python predict.py --target CHEMBL203 --smiles examples/sample_smiles.smi "
            "--output predictions.csv"
        ),
    )
    parser.add_argument("--target", required=True,
                        help="Target identifier: ChEMBL ID (e.g. CHEMBL203) or UniProt accession (e.g. P00533).")
    parser.add_argument("--smiles", required=True, type=Path,
                        help="Path to a text file with one SMILES per line.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV path. Columns: smiles, predicted_pic50.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"Path to the trained Phase 4 checkpoint (default: {DEFAULT_MODEL.relative_to(PROJECT_ROOT)}).")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING,
                        help=f"Path to chembl_uniprot_mapping.txt (default: {DEFAULT_MAPPING.relative_to(PROJECT_ROOT)}).")
    parser.add_argument("--cache", type=Path, default=DEFAULT_EMB_CACHE,
                        help=f"Path to ESM-2 embedding cache (default: {DEFAULT_EMB_CACHE.relative_to(PROJECT_ROOT)}).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many top predictions to print to stdout (default: 10).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Sanity checks
    if not args.model.exists():
        sys.exit(
            f"ERROR: Model checkpoint not found at {args.model}\n"
            f"Train the Phase 4 model first by running notebooks/04_multi_target_esm2.ipynb."
        )
    if not args.smiles.exists():
        sys.exit(f"ERROR: SMILES file not found at {args.smiles}")
    if not args.mapping.exists() and args.target.startswith("CHEMBL"):
        sys.exit(
            f"ERROR: ChEMBL→UniProt mapping file not found at {args.mapping}\n"
            f"Either provide --mapping or pass --target as a UniProt accession directly."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Load candidate molecules
    smiles_list = load_smiles(args.smiles)
    print(f"\nLoaded {len(smiles_list)} candidate SMILES from {args.smiles}")

    # Resolve target → embedding (with caching)
    print(f"\nResolving target '{args.target}'...")
    target_emb = get_target_embedding(
        target=args.target,
        mapping_path=args.mapping,
        cache_path=args.cache,
        device=device,
    )

    # Load model
    print(f"\nLoading model from {args.model.relative_to(PROJECT_ROOT)}...")
    model, ckpt = load_multi_modal_gine(args.model, device=device)
    if "targets" in ckpt:
        trained_on = sorted(ckpt["targets"].values())
        print(f"  Model was trained on: {', '.join(trained_on)}")

    # Predict
    print(f"\nScoring {len(smiles_list)} molecules...")
    results, failed = score(model, smiles_list, target_emb, device, batch_size=args.batch_size)

    # Sort by predicted potency (highest first)
    results = results.sort_values("predicted_pic50", ascending=False).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nWrote {len(results)} predictions to {args.output}")
    if failed:
        print(f"WARNING: {len(failed)} SMILES failed to parse and were skipped.")

    # Show top-K to stdout
    if args.top_k > 0 and len(results) > 0:
        k = min(args.top_k, len(results))
        print(f"\nTop {k} predicted hits (highest pIC50 = most potent):\n")
        print(results.head(k).to_string(index=False))


if __name__ == "__main__":
    main()
