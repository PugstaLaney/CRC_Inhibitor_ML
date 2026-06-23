# CRC_Inhibitor_ML

Machine-learning pipeline for predicting small-molecule inhibitor potency against colorectal cancer (CRC) oncogenic targets. Built as a **target-agnostic tool**: point it at any ChEMBL target ID and get back a trained graph neural network, evaluation metrics, and structure-activity interpretation.

## Targets

Four major CRC oncogene drivers from the ChEMBL 37 bioactivity database:

| Target | ChEMBL ID | CRC relevance |
|---|---|---|
| **KRAS** | CHEMBL2189121 | Mutated in ~40% of CRC; G12C inhibitors (sotorasib, adagrasib) are the hottest oncology drug class |
| **BRAF** | CHEMBL5145 | V600E mutation in ~10% of CRC |
| **EGFR** | CHEMBL203 | Target of cetuximab and panitumumab |
| **PIK3CA** | CHEMBL4005 | Frequently mutated; downstream of EGFR |

## Documentation

- **[docs/chembl-cheatsheet.md](docs/chembl-cheatsheet.md)** — the 5 ChEMBL tables that matter for bioactivity work, canonical query template, common variations, filter value reference, quick exploration commands. The one file to open when you need to write a ChEMBL query.

## Project structure

```
CRC_Inhibitor_ML/
├── data/
│   ├── chembl_37_schema.pdf      ChEMBL ER diagram (reference)
│   ├── raw/                      ChEMBL extracts + .db symlink (gitignored)
│   ├── interim/                  Standardized SMILES + harmonized pIC50 (gitignored)
│   └── processed/                Featurized graphs, train/val/test splits (gitignored)
├── docs/
│   └── chembl-cheatsheet.md      Working reference for ChEMBL queries
├── notebooks/
│   ├── 00_chembl_setup_and_eda.ipynb   Phase 0: data pull + EDA
│   └── 01_curate.ipynb                 Phase 1: cleanup + standardization
├── src/
│   ├── data/                     Downloaders, cleaners, featurizers
│   ├── models/                   GNN architectures, ESM-2 wrappers
│   ├── training/                 Train loops, eval metrics
│   └── utils/
├── models/                       Trained checkpoints (gitignored)
├── reports/figures/              Generated plots (gitignored)
├── configs/                      YAML hyperparameter configs
├── tests/
├── .gitignore
└── README.md
```

## Tech stack

- **Python 3.11**, PyTorch 2.6.0 + CUDA 12.4
- **PyTorch Geometric** for graph neural networks
- **RDKit** for cheminformatics (SMILES standardization, fingerprints, molecular graph extraction)
- **HuggingFace Transformers** for ESM-2 protein embeddings (Phase 4)
- **SQLite** for ChEMBL bioactivity data
- **scikit-learn** for the random-forest baseline
- Trained on an NVIDIA RTX 3060 (12 GB VRAM)

## Setup

### 1. Python environment

Requires Python 3.11. PyTorch is pinned to 2.6.0 on Windows (newer versions hit a `WinError 193` on `shm.dll` import).

```powershell
py -3.11 -m venv "$env:USERPROFILE\venvs\CRC_Inhibitor_ML"
& "$env:USERPROFILE\venvs\CRC_Inhibitor_ML\Scripts\Activate.ps1"

python -m pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scikit-learn matplotlib seaborn jupyter tqdm pyyaml
pip install rdkit torch_geometric transformers biopython chembl_webresource_client
```

The venv lives outside the project folder to avoid OneDrive sync issues with large binary dependencies.

### 2. ChEMBL database

Download the SQLite dump from EBI (~6 GB compressed, ~30 GB extracted):

```
https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_sqlite.tar.gz
```

Extract to any non-OneDrive location (the project assumes `E:\ml_data\chembl\chembl_37\chembl_37_sqlite\chembl_37.db`). Update `CHEMBL_DB_PATH` at the top of `notebooks/00_chembl_setup_and_eda.ipynb` if you put it elsewhere.

## Running

Open `notebooks/00_chembl_setup_and_eda.ipynb`, select the `CRC_Inhibitor_ML` venv as the kernel, run cells top to bottom. Output: `data/raw/chembl_crc_targets_raw.csv` — the input to Phase 1.

## Roadmap

| Phase | Status | Output |
|---|---|---|
| 0 — Env setup + ChEMBL extraction | In progress | Raw labeled CSV |
| 1 — SMILES standardization + pIC50 harmonization | Planned | Curated CSV |
| 2 — Random-forest-on-fingerprints baseline | Planned | Baseline metrics |
| 3 — Single-target GNN (PyG) | Planned | Per-target trained models |
| 4 — Multi-target ESM-2 + GNN | Planned | Multi-modal model |
| 5 — CLI tool refactor | Planned | `python -m crc_inhibitor_ml.pipeline --target <ID>` |
| 6 — SAR interpretation + writeup | Planned | Attention heatmaps, counterfactuals, portfolio writeup |

## Why this project

Demonstrates an end-to-end ML pipeline for drug discovery — not a single trained model, but a reusable system that turns any ChEMBL target into a trained inhibitor-potency predictor with interpretable predictions. Designed for a small-team commercial drug discovery workflow where each new target should produce a working model within an afternoon.

## Data provenance

ChEMBL 37 (release May 2026), licensed CC BY-SA 3.0. Cite: Mendez et al., *Nucleic Acids Research* 2024.
