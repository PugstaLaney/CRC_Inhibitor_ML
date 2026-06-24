"""CRC_Inhibitor_ML — Streamlit web UI for the multi-target inhibitor potency predictor.

Wraps the same Phase 4 model the predict.py CLI uses, but with a browser-based
interface aimed at medicinal chemists / pharma users who don't want to live in
the terminal.

Run with:
    streamlit run app.py

Opens automatically at http://localhost:8501
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch
import py3Dmol

# Resolve project root so the script works from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski, QED  # noqa: E402
from torch_geometric.loader import DataLoader as PyGDataLoader  # noqa: E402

from src.data.featurize import smiles_to_pyg  # noqa: E402
from src.data.proteins import (  # noqa: E402
    chembl_to_uniprot, get_target_embedding,
)
from src.models.gine import load_multi_modal_gine  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_PATH         = PROJECT_ROOT / "models"             / "gine_esm2_multi_target.pt"
MAPPING_PATH       = PROJECT_ROOT / "data" / "raw"       / "chembl_uniprot_mapping.txt"
EMB_CACHE          = PROJECT_ROOT / "data" / "processed" / "target_esm2_embeddings.pt"
CLEAN_CSV          = PROJECT_ROOT / "data" / "interim"   / "chembl_crc_targets_clean.csv"
DESCRIPTORS_CACHE  = PROJECT_ROOT / "data" / "processed" / "molecule_descriptors.parquet"

# Short names for the four trained targets, used by the Browse tab filter
TARGET_SHORT_NAMES = {
    "CHEMBL2189121": "KRAS",
    "CHEMBL5145":    "BRAF",
    "CHEMBL203":     "EGFR",
    "CHEMBL4005":    "PIK3CA",
}

PRESET_TARGETS = {
    "KRAS — GTPase KRas (CRC, mutated in ~40%)":              "CHEMBL2189121",
    "BRAF — Serine/threonine-protein kinase B-raf":           "CHEMBL5145",
    "EGFR — Epidermal growth factor receptor":                "CHEMBL203",
    "PIK3CA — PI3K-alpha catalytic subunit":                  "CHEMBL4005",
}

SAMPLE_SMILES = """# Known EGFR inhibitors (FDA approved)
COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1                            gefitinib
COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC                                    erlotinib
CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1    lapatinib

# Multi-kinase inhibitors
Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1              imatinib
CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1                sorafenib

# KRAS G12C inhibitor (approximate)
CC#CC(=O)N1CCC(N2C(=O)N(c3cc(C(F)(F)F)ccc3C)C(=N2)C2CCNCC2)CC1           adagrasib_like

# Inert reference compounds (should rank low)
CC(=O)Oc1ccccc1C(=O)O                                                     aspirin
CN1C=NC2=C1C(=O)N(C(=O)N2C)C                                              caffeine
CCO                                                                       ethanol
"""

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CRC Inhibitor ML",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Cached resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Phase 4 model checkpoint...")
def _load_model():
    if not MODEL_PATH.exists():
        return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_multi_modal_gine(MODEL_PATH, device=device)
    return model, ckpt, device


@st.cache_data(show_spinner="Resolving target and embedding with ESM-2...")
def _get_target_embedding(target_id: str) -> Optional[torch.Tensor]:
    _, _, device = _load_model()
    if device is None:
        return None
    return get_target_embedding(
        target=target_id,
        mapping_path=MAPPING_PATH,
        cache_path=EMB_CACHE,
        device=device,
        verbose=False,
    )


@st.cache_data(show_spinner="Resolving ChEMBL ID to UniProt...")
def _resolve_to_uniprot(target_id: str) -> str:
    if target_id.startswith("CHEMBL"):
        return chembl_to_uniprot(target_id, MAPPING_PATH)
    return target_id


@st.cache_data(show_spinner="Fetching protein structure from AlphaFold...")
def _fetch_alphafold_pdb(uniprot_accession: str) -> str:
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}"
    metadata = requests.get(api_url, timeout=30).json()
    if not metadata:
        raise ValueError(f"No AlphaFold prediction for {uniprot_accession}")
    pdb_url = metadata[0]["pdbUrl"]
    return requests.get(pdb_url, timeout=60).text


def _compute_descriptors_for_smiles(smiles_list: list[str]) -> pd.DataFrame:
    """Compute molecular descriptors for a list of unique SMILES (with a progress bar)."""
    rows = []
    progress = st.progress(0.0, text="Computing molecular descriptors (one-time, takes ~30 sec)...")
    n = len(smiles_list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            rows.append({
                "canonical_smiles_std": smi,
                "mw":              float(Descriptors.MolWt(mol)),
                "logp":            float(Descriptors.MolLogP(mol)),
                "hbd":             int(Lipinski.NumHDonors(mol)),
                "hba":             int(Lipinski.NumHAcceptors(mol)),
                "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
                "tpsa":            float(Descriptors.TPSA(mol)),
                "qed":             float(QED.qed(mol)),
                "n_heavy_atoms":   int(Descriptors.HeavyAtomCount(mol)),
            })
        except Exception:
            continue
        if i % 500 == 0:
            progress.progress(i / n, text=f"Computing descriptors... {i:,} / {n:,}")
    progress.empty()
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_browse_dataset() -> Optional[pd.DataFrame]:
    """Load the curated dataset + molecular descriptors. Cached to parquet on first call."""
    if not CLEAN_CSV.exists():
        return None
    df = pd.read_csv(CLEAN_CSV)

    # Compute descriptors if not cached
    if DESCRIPTORS_CACHE.exists():
        descriptors = pd.read_parquet(DESCRIPTORS_CACHE)
    else:
        unique_smiles = df["canonical_smiles_std"].dropna().unique().tolist()
        descriptors = _compute_descriptors_for_smiles(unique_smiles)
        DESCRIPTORS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        descriptors.to_parquet(DESCRIPTORS_CACHE, index=False)

    df = df.merge(descriptors, on="canonical_smiles_std", how="left")
    df["target_short"] = (
        df["target_chembl_id"].map(TARGET_SHORT_NAMES).fillna(df["target_chembl_id"])
    )
    # Reorder columns
    cols = [
        "target_short", "target_chembl_id", "pic50", "pic50_std", "n_measurements",
        "canonical_smiles_std",
        "mw", "logp", "hbd", "hba", "rotatable_bonds", "tpsa", "qed", "n_heavy_atoms",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


# -----------------------------------------------------------------------------
# Helpers — parsing, scoring, rendering
# -----------------------------------------------------------------------------
def parse_smiles_text(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        smi = parts[0]
        name = parts[1].strip() if len(parts) > 1 else ""
        out.append((smi, name))
    return out


def score_batch(smiles_pairs, target_emb, batch_size: int = 64):
    model, _, device = _load_model()
    data_objects, keep_smiles, keep_names, failed = [], [], [], []

    for smi, name in smiles_pairs:
        d = smiles_to_pyg(smi, target_emb=target_emb, y=0.0)
        if d is None:
            failed.append(smi)
        else:
            data_objects.append(d)
            keep_smiles.append(smi)
            keep_names.append(name)

    if not data_objects:
        return pd.DataFrame(columns=["name", "smiles", "predicted_pic50"]), failed

    loader = PyGDataLoader(data_objects, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).cpu().numpy())

    return pd.DataFrame({
        "name":            keep_names,
        "smiles":          keep_smiles,
        "predicted_pic50": np.concatenate(preds),
    }), failed


def make_3d_mol_html(smiles: str, width: int = 280, height: int = 240) -> Optional[str]:
    """Generate standalone HTML for a 3D molecule viewer (RDKit-embedded conformer)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        mol_block = Chem.MolToMolBlock(mol)
    except Exception:
        return None
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol_block, "sdf")
    viewer.setStyle({}, {"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    return viewer._make_html()


def make_3d_protein_html(pdb_text: str, style: str, width: int = 700, height: int = 400) -> str:
    """Generate standalone HTML for a 3D protein viewer."""
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(pdb_text, "pdb")
    if style == "cartoon":
        viewer.setStyle({}, {"cartoon": {
            "colorscheme": {"prop": "b", "gradient": "roygb", "min": 50, "max": 100}
        }})
    elif style == "residues":
        viewer.setStyle({}, {"cartoon": {"color": "lightgray", "opacity": 0.6}})
        viewer.addStyle({}, {"stick": {"radius": 0.12, "colorscheme": "default"}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    return viewer._make_html()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🧬 CRC Inhibitor ML")
st.markdown(
    "**Target-agnostic small-molecule potency predictor.** Pick any ChEMBL or UniProt target, "
    "paste candidate molecules, get back a ranked list with predicted pIC50 and rotatable 3D structures."
)

model, ckpt, device = _load_model()
if model is None:
    st.error(
        f"Model checkpoint not found at `{MODEL_PATH.relative_to(PROJECT_ROOT)}`. "
        f"Run `notebooks/04_multi_target_esm2.ipynb` to train the Phase 4 model first."
    )
    st.stop()

device_label = f"NVIDIA {torch.cuda.get_device_name(0)}" if device.type == "cuda" else "CPU"
trained_on   = ", ".join(sorted(ckpt.get("targets", {}).values()))
st.caption(
    f"Running on {device_label}  •  Model: Phase 4 multi-modal GINE + ESM-2  •  "
    f"Trained on: {trained_on}"
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎯 Target")
    target_mode = st.radio("Input mode", ["Preset", "Custom"], horizontal=True)

    if target_mode == "Preset":
        choice = st.selectbox("Pre-loaded CRC targets", list(PRESET_TARGETS.keys()))
        target_id = PRESET_TARGETS[choice]
    else:
        target_id = st.text_input(
            "ChEMBL ID or UniProt accession",
            value="CHEMBL203",
            help="e.g. CHEMBL203 (EGFR) or P00533 (also EGFR, by UniProt). "
                 "Targets not in training are scored via the ESM-2 protein-language-model pathway.",
        )

    st.caption(f"Active target: `{target_id}`")

    st.divider()

    st.header("⚙️ Settings")
    batch_size = st.slider("Batch size", 16, 256, 64, 16,
                           help="Larger = faster on GPU, more memory. Lower if you hit OOM.")
    top_k = st.slider("Top molecules to highlight", 3, 20, 5)
    show_target_protein = st.checkbox("Show target protein 3D viewer", value=True)

    st.divider()

    st.header("ℹ️ About")
    st.markdown(
        "- Model: Phase 4 of [CRC_Inhibitor_ML](https://github.com/PugstaLaney/CRC_Inhibitor_ML)\n"
        "- Architecture: GINE molecule encoder fused with frozen ESM-2 protein embedding\n"
        "- Training data: ChEMBL 37, four CRC oncogenic targets\n"
        "- Predicts pIC50 (higher = more potent inhibitor)\n"
        "- For novel targets, scores via ESM-2 (no per-target retraining)\n"
        "- Target 3D structures from AlphaFold; molecule 3D from RDKit + MMFF94"
    )

# -----------------------------------------------------------------------------
# Main area — Predict + Browse Dataset tabs
# -----------------------------------------------------------------------------
tab_predict, tab_browse = st.tabs(["🧬 Predict", "📚 Browse Dataset"])

with tab_predict:
  col_in, col_out = st.columns([2, 3])

  with col_in:
    st.subheader("Input molecules")
    input_mode = st.radio("How to provide SMILES", ["Paste", "Upload file"], horizontal=True)

    smiles_text = ""

    if input_mode == "Paste":
        if st.button("📋 Load sample library"):
            st.session_state["smiles_textarea"] = SAMPLE_SMILES

        smiles_text = st.text_area(
            "One SMILES per line. Optional second column = molecule name. "
            "Lines starting with `#` are ignored.",
            value=st.session_state.get("smiles_textarea", SAMPLE_SMILES),
            height=400,
            key="smiles_textarea",
        )
    else:
        uploaded = st.file_uploader(
            "Upload .smi, .csv, or .txt with one SMILES per line",
            type=["smi", "csv", "txt"],
        )
        if uploaded:
            smiles_text = uploaded.read().decode("utf-8")
            st.caption(f"Loaded {len(smiles_text):,} characters from `{uploaded.name}`")

    predict = st.button("🧬 Predict", type="primary", use_container_width=True)

  with col_out:
    if predict:
        smiles_pairs = parse_smiles_text(smiles_text)
        if not smiles_pairs:
            st.warning("No valid SMILES found in input.")
            st.stop()

        # Resolve target → ESM-2 embedding
        try:
            target_emb = _get_target_embedding(target_id)
        except Exception as e:
            st.error(f"Failed to fetch / embed target `{target_id}`: {e}")
            st.stop()

        if target_emb is None:
            st.error("Could not obtain target embedding.")
            st.stop()

        # ----- Target protein 3D viewer -----
        if show_target_protein:
            with st.expander(f"🎯 Target protein structure: `{target_id}` (AlphaFold)", expanded=True):
                try:
                    uniprot = _resolve_to_uniprot(target_id)
                    st.caption(f"UniProt accession: `{uniprot}`")
                    pdb_text = _fetch_alphafold_pdb(uniprot)
                    tab1, tab2 = st.tabs(["Cartoon (colored by pLDDT)", "Residues (full atomic detail)"])
                    with tab1:
                        html = make_3d_protein_html(pdb_text, style="cartoon", width=700, height=400)
                        components.html(html, height=420)
                    with tab2:
                        html = make_3d_protein_html(pdb_text, style="residues", width=700, height=400)
                        components.html(html, height=420)
                    st.caption(
                        "AlphaFold-predicted structure (not experimental). Color gradient on cartoon view "
                        "= pLDDT confidence: blue = high (well-folded), orange = low (flexible / disordered)."
                    )
                except Exception as e:
                    st.warning(f"Couldn't load target structure: {e}")

        # ----- Score molecules -----
        with st.spinner(f"Scoring {len(smiles_pairs)} molecules..."):
            results, failed = score_batch(smiles_pairs, target_emb, batch_size=batch_size)

        if results.empty:
            st.error(f"All {len(smiles_pairs)} SMILES failed to parse.")
            st.stop()

        if failed:
            st.warning(f"{len(failed)} SMILES failed to parse and were skipped.")

        # Sort + 1-index
        results = results.sort_values("predicted_pic50", ascending=False).reset_index(drop=True)
        results.index = results.index + 1
        results.index.name = "rank"

        # ----- Ranked table -----
        st.subheader(f"Ranked predictions against `{target_id}`")
        st.dataframe(
            results.style.format({"predicted_pic50": "{:.3f}"}),
            use_container_width=True,
            height=min(400, 60 + 36 * len(results)),
        )

        # ----- Download -----
        csv_bytes = results.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download predictions CSV",
            data=csv_bytes,
            file_name=f"predictions_{target_id}.csv",
            mime="text/csv",
        )

        # ----- Distribution chart -----
        st.subheader("Predicted pIC50 distribution")
        chart_labels = results["name"].where(results["name"] != "", results["smiles"].str[:25])
        chart_df = pd.DataFrame({
            "label": chart_labels.values,
            "pIC50": results["predicted_pic50"].values,
        }).set_index("label")
        st.bar_chart(chart_df, height=300)

        # ----- Top-K 3D structures -----
        st.subheader(f"Top {min(top_k, len(results))} structures (interactive 3D, click + drag to rotate)")
        top = results.head(top_k)
        cols_per_row = 3   # 3D viewers are bigger than 2D thumbnails, so fewer per row

        for row_start in range(0, len(top), cols_per_row):
            row_slice = top.iloc[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for i, (rank, row) in enumerate(row_slice.iterrows()):
                with cols[i]:
                    label = row["name"] if row["name"] else row["smiles"][:30] + ("…" if len(row["smiles"]) > 30 else "")
                    st.markdown(f"**#{rank}**  •  pIC50 = `{row['predicted_pic50']:.2f}`  •  {label}")
                    html = make_3d_mol_html(row["smiles"], width=300, height=260)
                    if html is not None:
                        components.html(html, height=280)
                    else:
                        # Fallback to 2D if 3D embedding fails
                        mol = Chem.MolFromSmiles(row["smiles"])
                        if mol is not None:
                            st.image(Draw.MolToImage(mol, size=(280, 220)), use_container_width=True)
    else:
      st.info("👈 Enter target + SMILES in the left panel, then click **Predict** to run.")


# =============================================================================
# Browse Dataset tab — searchable / filterable library of 25,170 curated rows
# =============================================================================
with tab_browse:
  st.subheader("Browse curated ChEMBL dataset")
  st.caption(
      "25,170 high-confidence (target, molecule) measurements after Phase 1 curation. "
      "Filter by target, potency, and drug-likeness properties. "
      "Select rows and send them to the Predict tab to score with the model."
  )

  browse_df = load_browse_dataset()
  if browse_df is None:
      st.error(
          f"Curated dataset not found at `{CLEAN_CSV.relative_to(PROJECT_ROOT)}`. "
          f"Run `notebooks/01_curate.ipynb` first."
      )
      st.stop()

  # -------- Filters --------
  filt_col, table_col = st.columns([1, 3])

  with filt_col:
      st.markdown("**Filters**")
      target_choices = sorted(browse_df["target_short"].unique().tolist())
      selected_targets = st.multiselect(
          "Targets", target_choices, default=target_choices,
      )

      pic50_min, pic50_max = float(browse_df["pic50"].min()), float(browse_df["pic50"].max())
      pic50_range = st.slider(
          "pIC50 range", pic50_min, pic50_max, (pic50_min, pic50_max),
          step=0.1,
          help="Higher pIC50 = more potent inhibitor. 7 = 100 nM, 8 = 10 nM, 9 = 1 nM.",
      )

      mw_range = st.slider("MW range (Da)", 50.0, 1500.0, (100.0, 700.0), step=10.0)
      logp_range = st.slider(
          "LogP range", -5.0, 12.0, (-2.0, 7.0), step=0.5,
          help="Octanol-water partition. Negative = hydrophilic, positive = lipophilic. Lipinski limit is ≤ 5.",
      )

      qed_min = st.slider(
          "Min QED (drug-likeness)", 0.0, 1.0, 0.0, step=0.05,
          help="Quantitative Estimate of Drug-likeness. 0 = poor, 1 = ideal drug-like profile.",
      )

      lipinski_only = st.checkbox(
          "Lipinski compliant only",
          value=False,
          help="MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10. The classic 'orally bioavailable drug' filter.",
      )

      search_text = st.text_input(
          "SMILES contains",
          value="",
          placeholder="e.g. c1ccccc1 for benzene",
          help="Substring match against the canonical SMILES string. Case insensitive.",
      )

  # -------- Apply filters --------
  filtered = browse_df[
      browse_df["target_short"].isin(selected_targets)
      & browse_df["pic50"].between(*pic50_range)
      & browse_df["mw"].between(*mw_range)
      & browse_df["logp"].between(*logp_range)
      & (browse_df["qed"] >= qed_min)
  ]

  if lipinski_only:
      filtered = filtered[
          (filtered["mw"] <= 500)
          & (filtered["logp"] <= 5)
          & (filtered["hbd"] <= 5)
          & (filtered["hba"] <= 10)
      ]

  if search_text.strip():
      filtered = filtered[
          filtered["canonical_smiles_std"].str.contains(
              search_text.strip(), case=False, regex=False, na=False
          )
      ]

  with table_col:
      st.markdown(f"**{len(filtered):,} molecules** match filters  •  "
                  f"sorted by predicted pIC50 (descending). "
                  f"Click row checkboxes to select.")

      # Sort descending by pIC50 by default
      filtered_sorted = filtered.sort_values("pic50", ascending=False).reset_index(drop=True)

      # Format for display: round floats
      display_df = filtered_sorted.copy()
      for col in ["mw", "logp", "tpsa", "qed", "pic50", "pic50_std"]:
          if col in display_df.columns:
              display_df[col] = display_df[col].round(2)

      event = st.dataframe(
          display_df,
          use_container_width=True,
          height=500,
          hide_index=True,
          on_select="rerun",
          selection_mode="multi-row",
          key="browse_table",
      )

      # -------- Selection actions --------
      selected_idx = event.selection.rows if hasattr(event, "selection") else []
      selected_count = len(selected_idx)

      col_a, col_b, col_c = st.columns([1, 1, 2])

      with col_a:
          if st.button(
              f"🧬 Send {selected_count} to Predict",
              disabled=(selected_count == 0),
              use_container_width=True,
          ):
              picked = filtered_sorted.iloc[selected_idx]
              lines = [
                  f"{smi}\t{tgt}_pIC50={pic:.2f}"
                  for smi, tgt, pic in zip(
                      picked["canonical_smiles_std"],
                      picked["target_short"],
                      picked["pic50"],
                  )
              ]
              st.session_state["smiles_textarea"] = "\n".join(lines)
              st.success(
                  f"Sent {selected_count} molecules to the Predict tab. "
                  f"Switch tabs to see them and click Predict."
              )

      with col_b:
          csv_bytes = filtered_sorted.to_csv(index=False).encode("utf-8")
          st.download_button(
              "📥 Download CSV",
              data=csv_bytes,
              file_name="browse_filtered.csv",
              mime="text/csv",
              use_container_width=True,
          )

      with col_c:
          if selected_count > 0:
              st.caption(
                  f"Selected {selected_count} row(s). The 'Send' button will overwrite "
                  f"the Predict tab's SMILES input with these molecules."
              )
