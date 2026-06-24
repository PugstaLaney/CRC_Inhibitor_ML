"""CRC_Inhibitor_ML — FastAPI backend for the web tool.

Run with:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health                 sanity check + GPU/model status
    GET  /presets                preset target list (the 4 CRC targets)
    POST /predict                body: {target, smiles[]} → ranked predictions
    GET  /target/{id}/info       target metadata (UniProt accession, name)
    GET  /target/{id}/pdb        AlphaFold PDB text (server-side fetch, no CORS issues)
    GET  /molecule/3d?smi=...    3D mol block (SDF format) for a SMILES
    GET  /molecule/png?smi=...   2D structure PNG for a SMILES

CORS is wide-open (`allow_origins=["*"]`) for local dev. Tighten for production.
"""
from __future__ import annotations

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

# Project setup so `from src...` works
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.featurize import smiles_to_pyg  # noqa: E402
from src.data.proteins import (  # noqa: E402
    chembl_to_uniprot, get_target_embedding,
)
from src.models.gine import load_multi_modal_gine  # noqa: E402
from torch_geometric.loader import DataLoader as PyGDataLoader  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_PATH    = PROJECT_ROOT / "models"             / "gine_esm2_multi_target.pt"
MAPPING_PATH  = PROJECT_ROOT / "data" / "raw"       / "chembl_uniprot_mapping.txt"
EMB_CACHE     = PROJECT_ROOT / "data" / "processed" / "target_esm2_embeddings.pt"

TARGET_PRESETS = {
    "CHEMBL2189121": {"chembl_id": "CHEMBL2189121", "short": "KRAS",
                      "long_name": "GTPase KRas",
                      "blurb": "Mutated in ~40% of CRC. G12C inhibitors are the recent oncology breakthrough."},
    "CHEMBL5145":    {"chembl_id": "CHEMBL5145",    "short": "BRAF",
                      "long_name": "Serine/threonine-protein kinase B-raf",
                      "blurb": "V600E mutation in ~10% of CRC. Target of vemurafenib, dabrafenib."},
    "CHEMBL203":     {"chembl_id": "CHEMBL203",     "short": "EGFR",
                      "long_name": "Epidermal growth factor receptor",
                      "blurb": "Target of cetuximab and panitumumab. The most studied kinase target."},
    "CHEMBL4005":    {"chembl_id": "CHEMBL4005",    "short": "PIK3CA",
                      "long_name": "PI3K-alpha catalytic subunit",
                      "blurb": "Frequently mutated in CRC. Downstream of EGFR signaling."},
}

# -----------------------------------------------------------------------------
# Model loaded once at startup
# -----------------------------------------------------------------------------
_state = {"model": None, "ckpt": None, "device": None, "target_emb_cache": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state["device"] = device
    if MODEL_PATH.exists():
        model, ckpt = load_multi_modal_gine(MODEL_PATH, device=device)
        _state["model"] = model
        _state["ckpt"] = ckpt
        gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        print(f"[api] Model loaded on {device} ({gpu})")
    else:
        print(f"[api] WARNING: model checkpoint not found at {MODEL_PATH}")
        print(f"[api]   Run notebooks/04_multi_target_esm2.ipynb to train it.")
    yield
    # No teardown needed


app = FastAPI(title="CRC Inhibitor ML API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_target_to_embedding(target_id: str) -> torch.Tensor:
    cache = _state["target_emb_cache"]
    if target_id in cache:
        return cache[target_id]
    emb = get_target_embedding(
        target=target_id,
        mapping_path=MAPPING_PATH,
        cache_path=EMB_CACHE,
        device=_state["device"],
        verbose=False,
    )
    cache[target_id] = emb
    return emb


# -----------------------------------------------------------------------------
# Models (request / response shapes)
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    target: str
    smiles: list[str]
    batch_size: int = 64


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    device = _state["device"]
    return {
        "status":       "ok",
        "device":       str(device) if device else "unknown",
        "gpu":          torch.cuda.get_device_name(0) if device and device.type == "cuda" else None,
        "model_loaded": _state["model"] is not None,
        "trained_on":   list((_state["ckpt"] or {}).get("targets", {}).values()),
    }


@app.get("/presets")
def get_presets():
    return {"presets": list(TARGET_PRESETS.values())}


@app.get("/target/{target_id}/info")
def get_target_info(target_id: str):
    """Resolve a target to its UniProt accession and short metadata."""
    try:
        if target_id.startswith("CHEMBL"):
            uniprot = chembl_to_uniprot(target_id, MAPPING_PATH)
        else:
            uniprot = target_id
    except Exception as e:
        raise HTTPException(404, f"Could not resolve target: {e}")

    info = TARGET_PRESETS.get(target_id, {
        "chembl_id": target_id if target_id.startswith("CHEMBL") else None,
        "short":     target_id,
        "long_name": "(unknown — custom target)",
        "blurb":     "Not in training set. Predictions are extrapolated via ESM-2 protein embedding.",
    })
    info = {**info, "uniprot": uniprot}
    return info


@app.get("/target/{target_id}/pdb")
def get_target_pdb(target_id: str):
    """Server-side fetch of an AlphaFold PDB. Avoids the browser hitting external APIs directly."""
    try:
        if target_id.startswith("CHEMBL"):
            uniprot = chembl_to_uniprot(target_id, MAPPING_PATH)
        else:
            uniprot = target_id

        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
        meta_resp = requests.get(api_url, timeout=30)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        if not meta:
            raise HTTPException(404, f"No AlphaFold prediction for UniProt {uniprot}")

        pdb_url = meta[0]["pdbUrl"]
        pdb_text = requests.get(pdb_url, timeout=60).text
        return Response(content=pdb_text, media_type="chemical/x-pdb")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch PDB: {e}")


@app.get("/molecule/3d")
def get_molecule_3d(smi: str = Query(..., min_length=1)):
    """Return a 3D mol block (SDF format) for a SMILES string. RDKit-embedded + MMFF94-optimized."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception as e:
        raise HTTPException(500, f"3D embedding failed: {e}")
    mol_block = Chem.MolToMolBlock(mol)
    return Response(content=mol_block, media_type="chemical/x-mdl-sdfile")


@app.get("/molecule/png")
def get_molecule_png(smi: str = Query(..., min_length=1), size: int = Query(300, ge=100, le=800)):
    """Return a 2D PNG image of a SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise HTTPException(400, "Invalid SMILES")
    img = Draw.MolToImage(mol, size=(size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/predict")
def predict(req: PredictRequest):
    """Score a batch of SMILES against a target. Returns predictions sorted descending by pIC50."""
    model = _state["model"]
    device = _state["device"]
    if model is None:
        raise HTTPException(503, "Model not loaded. Train notebook 04 first.")
    if not req.smiles:
        raise HTTPException(400, "No SMILES provided")

    # Resolve target → embedding (cached)
    try:
        target_emb = _resolve_target_to_embedding(req.target)
    except Exception as e:
        raise HTTPException(400, f"Failed to resolve / embed target '{req.target}': {e}")

    # Featurize, keeping track of failures
    data_objects, keep_smiles, failed = [], [], []
    for smi in req.smiles:
        d = smiles_to_pyg(smi, target_emb=target_emb, y=0.0)
        if d is None:
            failed.append(smi)
        else:
            data_objects.append(d)
            keep_smiles.append(smi)

    if not data_objects:
        return {"target": req.target, "predictions": [], "failed": failed}

    # Batch through model
    loader = PyGDataLoader(data_objects, batch_size=req.batch_size, shuffle=False)
    preds_arr = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds_arr.append(model(batch).cpu().numpy())
    preds = np.concatenate(preds_arr)

    # Sort descending by predicted pIC50
    predictions = sorted(
        [{"smiles": s, "predicted_pic50": float(p)} for s, p in zip(keep_smiles, preds)],
        key=lambda x: x["predicted_pic50"],
        reverse=True,
    )

    return {
        "target":      req.target,
        "n_input":     len(req.smiles),
        "n_scored":    len(predictions),
        "n_failed":    len(failed),
        "predictions": predictions,
        "failed":      failed,
    }
