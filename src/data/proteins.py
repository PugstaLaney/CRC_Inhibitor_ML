"""Protein sequence retrieval (UniProt) and ESM-2 embedding.

Pipeline:
  ChEMBL target ID  ──(via chembl_uniprot_mapping.txt)──►  UniProt accession
  UniProt accession ──(via UniProt REST API)──►            amino acid sequence
  Sequence          ──(via ESM-2 transformer)──►           480-dim embedding

ESM-2 embeddings are cached per-target so the user doesn't pay the inference
cost on every CLI invocation.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import torch
from Bio import SeqIO

ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
ESM_DIM = 480


def chembl_to_uniprot(chembl_id: str, mapping_path: Path) -> str:
    """Resolve a ChEMBL target ID (e.g. CHEMBL203) to a UniProt accession.

    Reads the ChEMBL-shipped tab-separated mapping file. If the target maps to
    multiple UniProt accessions (e.g., protein complexes), returns the first.
    """
    mapping = pd.read_csv(
        mapping_path, sep="\t", comment="#", header=None,
        names=["uniprot", "target_chembl_id", "name", "target_type"],
    )
    matches = mapping[mapping["target_chembl_id"] == chembl_id]
    if len(matches) == 0:
        raise ValueError(
            f"No UniProt mapping found for ChEMBL ID '{chembl_id}'. "
            f"Check the ID exists in ChEMBL, or pass the UniProt accession directly."
        )
    return matches.iloc[0]["uniprot"]


def fetch_uniprot_sequence(accession: str, timeout: int = 30) -> str:
    """Fetch the canonical amino acid sequence from UniProt's REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    record = next(SeqIO.parse(io.StringIO(response.text), "fasta"))
    return str(record.seq)


def embed_sequence(sequence: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """Run ESM-2 on a protein sequence and return a mean-pooled 480-dim embedding.

    First call downloads ~150 MB of model weights from HuggingFace; cached after.
    """
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM_MODEL_NAME).to(device).eval()

    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Mean-pool over residues (skip CLS at position 0 and EOS at last position)
        embedding = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu()

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return embedding


def get_target_embedding(
    target: str,
    mapping_path: Path,
    cache_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Resolve a target (ChEMBL ID or UniProt accession) to an ESM-2 embedding.

    If `cache_path` is provided and the file exists, embeddings are looked up
    (and newly computed ones are written back to the cache).
    """
    # Resolve identifier
    if target.startswith("CHEMBL"):
        uniprot = chembl_to_uniprot(target, mapping_path)
        if verbose:
            print(f"  {target} → UniProt {uniprot}")
    else:
        uniprot = target

    # Check cache
    cache = {}
    if cache_path is not None and cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        # Cache may be keyed by ChEMBL ID or UniProt
        for key in (target, uniprot):
            if key in cache:
                if verbose:
                    print(f"  Cache hit: {key}")
                return cache[key]

    # Compute embedding
    sequence = fetch_uniprot_sequence(uniprot)
    if verbose:
        print(f"  Fetched sequence: {len(sequence)} residues")
        print(f"  Embedding with ESM-2 ({ESM_MODEL_NAME})...")
    embedding = embed_sequence(sequence, device=device)

    # Update cache
    if cache_path is not None:
        cache[uniprot] = embedding
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        if verbose:
            print(f"  Cached embedding → {cache_path.name}")

    return embedding
