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
│   └── processed/                ESM-2 embeddings, featurized graphs (gitignored)
├── docs/
│   └── chembl-cheatsheet.md      Working reference for ChEMBL queries
├── examples/
│   └── sample_smiles.smi         Demo SMILES library for predict.py
├── notebooks/
│   ├── 00_chembl_setup_and_eda.ipynb   Phase 0: data pull + EDA
│   ├── 01_curate.ipynb                 Phase 1: cleanup + standardization
│   ├── 02_baseline_rf.ipynb            Phase 2: random forest baseline
│   ├── 03_gnn_single_target.ipynb      Phase 3: single-target GIN
│   ├── 03b_gnn_v2_fixed_split.ipynb    Phase 3.5: GINE + scaffold-val
│   └── 04_multi_target_esm2.ipynb      Phase 4: multi-target ESM-2 + GINE
├── src/
│   ├── data/
│   │   ├── featurize.py          SMILES → PyG molecular graphs
│   │   └── proteins.py           UniProt lookup + ESM-2 embedding
│   └── models/
│       └── gine.py               MultiModalGINE architecture + checkpoint loader
├── predict.py                    Phase 5 CLI: score molecules against any target
├── models/                       Trained checkpoints (gitignored)
├── reports/                      Per-phase metrics JSON
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

### Running the notebooks (Phases 0–4)

Open the notebooks in order (`00_…` through `04_…`), select the `CRC_Inhibitor_ML` venv as the kernel, run cells top to bottom. Each phase produces inputs for the next:

- `00_chembl_setup_and_eda.ipynb` → `data/raw/chembl_crc_targets_raw.csv`
- `01_curate.ipynb` → `data/interim/chembl_crc_targets_clean.csv`
- `02_baseline_rf.ipynb` → `reports/baseline_rf_metrics.json`
- `03_gnn_single_target.ipynb` → per-target GIN checkpoints + metrics
- `03b_gnn_v2_fixed_split.ipynb` → per-target GINE checkpoints + metrics
- `04_multi_target_esm2.ipynb` → `models/gine_esm2_multi_target.pt` + `reports/phase4_multi_target_metrics.json`

### Predicting against any target (Phase 5 — the CLI tool)

The Phase 4 multi-target model can score molecule libraries against any target — including targets the model wasn't explicitly trained on, by feeding in the target's amino acid sequence via ESM-2 embedding.

```powershell
python predict.py `
    --target  CHEMBL203 `
    --smiles  examples/sample_smiles.smi `
    --output  predictions.csv
```

Arguments:

- `--target` — ChEMBL target ID (e.g. `CHEMBL203` for EGFR) **or** UniProt accession (e.g. `P00533` for EGFR). The script resolves ChEMBL IDs via the bundled mapping file, fetches the protein sequence from UniProt, and embeds it with ESM-2 (cached after first use).
- `--smiles` — text file with one SMILES per line. Lines starting with `#` are comments; an optional second whitespace-separated column is treated as a molecule name (ignored for scoring).
- `--output` — destination CSV with columns `smiles, predicted_pic50`, sorted by predicted potency (highest pIC50 first).

Output preview is printed to stdout (top 10 by default; configurable via `--top-k`).

The CLI is a thin wrapper around three reusable modules: [src/data/featurize.py](src/data/featurize.py), [src/data/proteins.py](src/data/proteins.py), [src/models/gine.py](src/models/gine.py).

## Roadmap

| Phase | Status | Output |
|---|---|---|
| 0 — Env setup + ChEMBL extraction | Done | 41,343-row raw labeled CSV |
| 1 — SMILES standardization + pIC50 harmonization | Done | 25,170-row curated CSV |
| 2 — Random-forest-on-fingerprints baseline | Done | Scaffold-split R² 0.45–0.61 across targets |
| 3 — Single-target GIN | Done | Underperformed baseline; diagnosed val-split bug |
| 3.5 — GINE + three-way scaffold split | Done | Val→test gap closed; absolute scores still trail RF |
| 4 — Multi-target ESM-2 + GINE | Done | Closed most of gap to RF; zero-shot target generalization |
| 5 — CLI tool (`predict.py`) | Done | Score any SMILES library against any ChEMBL / UniProt target |
| 6 — SAR interpretation + writeup | Next | Attention heatmaps, counterfactuals, portfolio writeup |

## Why this project

Demonstrates an end-to-end ML pipeline for drug discovery — not a single trained model, but a reusable system that turns any ChEMBL target into a trained inhibitor-potency predictor with interpretable predictions. Designed for a small-team commercial drug discovery workflow where each new target should produce a working model within an afternoon.

## Data provenance

ChEMBL 37 (release May 2026), licensed CC BY-SA 3.0. Cite: Mendez et al., *Nucleic Acids Research* 2024.
