"""Generate static demo data for the public-facing drugbinding.html page.

Outputs into Colo/HTML/assets/portfolio/:
  - drugbinding_data.json — full sample data for 5 sample diseases
                             (targets, drugs w/ SMILES + canonical RDKit
                              molecular properties + pre-computed 3D mol blocks)
  - drugbinding_<uniprot>.pdb — AlphaFold predicted PDB for each disease's
                                primary target

The HTML page reads the JSON and PDBs directly via fetch(), so the public
deployment needs zero backend. Run this script once on a machine that has
the CRC_Inhibitor_ML venv (for RDKit) and internet (for AlphaFold).
"""
from __future__ import annotations
import json
from pathlib import Path

import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED

RDLogger.DisableLog("rdApp.*")

OUT_DIR = Path(r"C:\Users\palla\OneDrive\Documents\Coding Projects\Colo\HTML\assets\portfolio")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Sample diseases. For each:
#   primary_target: the AlphaFold structure shown in the modal-style protein view
#   targets:        a small fleet of relevant targets the user can pick in Step 1
#   molecules:      (name, smiles, max_phase, [target_chembl_ids drug acts on])
# Real names + canonical-enough SMILES; the demo doesn't have to be
# bioactivity-accurate, just chemically valid and recognizable.
# -----------------------------------------------------------------------------
SAMPLES = {
    "Colorectal Neoplasms": {
        "display_name":   "Colorectal Cancer",
        "primary_target": {"chembl_id": "CHEMBL203", "pref_name": "Epidermal growth factor receptor (EGFR)", "uniprot": "P00533"},
        "targets": [
            {"chembl_id": "CHEMBL203",     "pref_name": "Epidermal growth factor receptor (EGFR)",      "target_type": "SINGLE PROTEIN", "n_drugs": 47},
            {"chembl_id": "CHEMBL2189121", "pref_name": "GTPase KRas",                                   "target_type": "SINGLE PROTEIN", "n_drugs": 12},
            {"chembl_id": "CHEMBL5145",    "pref_name": "Serine/threonine-protein kinase B-raf (BRAF)", "target_type": "SINGLE PROTEIN", "n_drugs":  6},
            {"chembl_id": "CHEMBL4005",    "pref_name": "PI3-kinase alpha catalytic subunit (PIK3CA)",  "target_type": "SINGLE PROTEIN", "n_drugs": 23},
            {"chembl_id": "CHEMBL279",     "pref_name": "Vascular endothelial growth factor receptor 2","target_type": "SINGLE PROTEIN", "n_drugs": 18},
        ],
        "molecules": [
            ("gefitinib",    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",                            4, ["CHEMBL203"]),
            ("erlotinib",    "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",                                    4, ["CHEMBL203"]),
            ("lapatinib",    "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",     4, ["CHEMBL203"]),
            ("imatinib",     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",              4, ["CHEMBL279"]),
            ("sorafenib",    "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",                4, ["CHEMBL5145", "CHEMBL279"]),
            ("fluorouracil", "O=c1[nH]cc(F)c(=O)[nH]1",                                                   4, []),
        ],
    },

    "Alzheimer Disease": {
        "display_name":   "Alzheimer Disease",
        "primary_target": {"chembl_id": "CHEMBL220", "pref_name": "Acetylcholinesterase (AChE)", "uniprot": "P22303"},
        "targets": [
            {"chembl_id": "CHEMBL220",  "pref_name": "Acetylcholinesterase (AChE)",                  "target_type": "SINGLE PROTEIN",  "n_drugs": 28},
            {"chembl_id": "CHEMBL2331", "pref_name": "Butyrylcholinesterase (BChE)",                 "target_type": "SINGLE PROTEIN",  "n_drugs": 12},
            {"chembl_id": "CHEMBL1907601", "pref_name": "NMDA receptor (glutamate ionotropic)",     "target_type": "PROTEIN COMPLEX", "n_drugs":  4},
        ],
        "molecules": [
            ("donepezil",    "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",  4, ["CHEMBL220", "CHEMBL2331"]),
            ("galantamine",  "COc1ccc2CC3N(C)CCC34C=CC(O)CC4Oc1c2",            4, ["CHEMBL220"]),
            ("rivastigmine", "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1",                4, ["CHEMBL220", "CHEMBL2331"]),
            ("memantine",    "CC12CC3CC(C)(C1)CC(N)(C3)C2",                    4, ["CHEMBL1907601"]),
            ("tacrine",      "Nc1c2c(nc3ccccc13)CCCC2",                        4, ["CHEMBL220"]),
        ],
    },

    "Diabetes Mellitus, Type 2": {
        "display_name":   "Type 2 Diabetes",
        "primary_target": {"chembl_id": "CHEMBL284", "pref_name": "Dipeptidyl peptidase IV (DPP-4)", "uniprot": "P27487"},
        "targets": [
            {"chembl_id": "CHEMBL284",  "pref_name": "Dipeptidyl peptidase IV (DPP-4)",          "target_type": "SINGLE PROTEIN", "n_drugs": 11},
            {"chembl_id": "CHEMBL2074", "pref_name": "Sodium glucose cotransporter 2 (SGLT2)",    "target_type": "SINGLE PROTEIN", "n_drugs":  6},
            {"chembl_id": "CHEMBL1798", "pref_name": "Alpha-glucosidase",                         "target_type": "SINGLE PROTEIN", "n_drugs":  3},
            {"chembl_id": "CHEMBL2851", "pref_name": "Glucagon-like peptide-1 receptor (GLP-1R)", "target_type": "SINGLE PROTEIN", "n_drugs":  5},
        ],
        "molecules": [
            ("metformin",     "CN(C)C(=N)N=C(N)N",                                                                                          4, []),
            ("sitagliptin",   "NC(CC(=O)N1CCn2c(C1)nnc2C(F)(F)F)Cc1cc(F)c(F)cc1F",                                                          4, ["CHEMBL284"]),
            ("dapagliflozin", "OCC1OC(c2ccc(Cl)c(Cc3ccc(OCC)cc3)c2)C(O)C(O)C1O",                                                            4, ["CHEMBL2074"]),
            ("empagliflozin", "OCC1OC(c2ccc(Cl)c(Cc3ccc(OC4CCOCC4)cc3)c2)C(O)C(O)C1O",                                                      4, ["CHEMBL2074"]),
            ("linagliptin",   "Cn1c(CC2(N=NN2C)c2nc3ccccc3n2CC)nc2c1c(=O)n(C)c(=O)n2C",                                                     4, ["CHEMBL284"]),
        ],
    },

    "Hypertension": {
        "display_name":   "Hypertension",
        "primary_target": {"chembl_id": "CHEMBL1808", "pref_name": "Angiotensin-converting enzyme (ACE)", "uniprot": "P12821"},
        "targets": [
            {"chembl_id": "CHEMBL1808", "pref_name": "Angiotensin-converting enzyme (ACE)", "target_type": "SINGLE PROTEIN",  "n_drugs": 18},
            {"chembl_id": "CHEMBL227",  "pref_name": "Angiotensin II receptor type 1",      "target_type": "SINGLE PROTEIN",  "n_drugs":  9},
            {"chembl_id": "CHEMBL1980", "pref_name": "L-type calcium channel",              "target_type": "PROTEIN COMPLEX", "n_drugs":  7},
            {"chembl_id": "CHEMBL210",  "pref_name": "Beta-1 adrenergic receptor",          "target_type": "SINGLE PROTEIN",  "n_drugs": 11},
        ],
        "molecules": [
            ("lisinopril", "NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O",                4, ["CHEMBL1808"]),
            ("enalapril",  "CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O",                  4, ["CHEMBL1808"]),
            ("captopril",  "CC(CS)C(=O)N1CCCC1C(=O)O",                                      4, ["CHEMBL1808"]),
            ("amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",               4, ["CHEMBL1980"]),
            ("losartan",   "OCc1ncnc(-c2ccc(Cn3nnnc3-c3ccccc3-c3ccccc3)cc2)n1",              4, ["CHEMBL227"]),
            ("valsartan",  "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)C(C(C)C)C(=O)O",      4, ["CHEMBL227"]),
        ],
    },

    "Parkinson Disease": {
        "display_name":   "Parkinson Disease",
        "primary_target": {"chembl_id": "CHEMBL2039", "pref_name": "Monoamine oxidase B (MAO-B)", "uniprot": "P27338"},
        "targets": [
            {"chembl_id": "CHEMBL2039",    "pref_name": "Monoamine oxidase B (MAO-B)",     "target_type": "SINGLE PROTEIN", "n_drugs":  8},
            {"chembl_id": "CHEMBL217",     "pref_name": "Dopamine D2 receptor",            "target_type": "SINGLE PROTEIN", "n_drugs": 15},
            {"chembl_id": "CHEMBL1907604", "pref_name": "Catechol O-methyltransferase",    "target_type": "SINGLE PROTEIN", "n_drugs":  3},
        ],
        "molecules": [
            ("levodopa",   "NC(Cc1ccc(O)c(O)c1)C(=O)O",         4, []),
            ("selegiline", "C#CCN(C)C(C)Cc1ccccc1",             4, ["CHEMBL2039"]),
            ("rasagiline", "C#CNC1Cc2ccccc2C1",                 4, ["CHEMBL2039"]),
            ("safinamide", "CC(NCc1ccc(OCc2cccc(F)c2)cc1)C(=O)N", 4, ["CHEMBL2039"]),
        ],
    },
}


def fetch_alphafold_pdb(uniprot):
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
    meta = requests.get(api_url, timeout=30).json()
    pdb_url = meta[0]["pdbUrl"]
    return requests.get(pdb_url, timeout=60).text


def mol_properties(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mw   = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    hbd  = int(Lipinski.NumHDonors(mol))
    hba  = int(Lipinski.NumHAcceptors(mol))
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "mw":              round(mw, 1),
        "logp":            round(logp, 2),
        "hbd":             hbd,
        "hba":             hba,
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "tpsa":            round(float(Descriptors.TPSA(mol)), 1),
        "qed":             round(float(QED.qed(mol)), 3),
        "n_heavy_atoms":   int(Descriptors.HeavyAtomCount(mol)),
        "lipinski_violations": violations,
        "lipinski_pass":   violations == 0,
    }


def mol_to_3d_block(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def main():
    output = {"diseases": {}}

    for disease_name, info in SAMPLES.items():
        print(f"\n[{disease_name}]")

        # Fetch + save the primary target's AlphaFold PDB
        primary = info["primary_target"]
        pdb_filename = f"drugbinding_{primary['uniprot'].lower()}.pdb"
        pdb_path = OUT_DIR / pdb_filename
        if pdb_path.exists():
            print(f"  [skip] {pdb_filename} already present")
        else:
            try:
                print(f"  fetching AlphaFold structure for {primary['uniprot']}...")
                pdb_text = fetch_alphafold_pdb(primary["uniprot"])
                pdb_path.write_text(pdb_text, encoding="utf-8")
                print(f"  saved {pdb_filename} ({len(pdb_text):,} chars)")
            except Exception as e:
                print(f"  WARNING: failed to fetch PDB for {primary['uniprot']}: {e}")

        # Build per-molecule entries
        molecules = []
        for name, smi, max_phase, target_acts in info["molecules"]:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"  WARNING: {name} SMILES failed to parse, skipping")
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            props = mol_properties(canonical)
            mol_block = mol_to_3d_block(canonical)
            # Make sure the primary target is on the drug's acted-on list so the
            # 'Acting on selected target' filter in the UI shows a sensible
            # default selection.
            target_chembl_ids = sorted(set(target_acts + [primary["chembl_id"]]))
            molecules.append({
                "name":              name,
                "smiles":            canonical,
                "max_phase":         max_phase,
                "target_chembl_ids": target_chembl_ids,
                "properties":        props,
                "mol_block_3d":      mol_block,
            })
            print(f"  + {name:14s}  MW={props['mw']:>6}  QED={props['qed']:.3f}  Lipinski={'pass' if props['lipinski_pass'] else 'fail'}")

        output["diseases"][disease_name] = {
            "display_name":   info["display_name"],
            "primary_target": {**primary, "pdb_filename": pdb_filename},
            "targets":        info["targets"],
            "molecules":      molecules,
        }

    out_json = OUT_DIR / "drugbinding_data.json"
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json} ({out_json.stat().st_size:,} bytes)")
    print("\nDone. Check the file then deploy alongside drugbinding.html.")


if __name__ == "__main__":
    main()
