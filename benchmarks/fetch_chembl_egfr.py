"""Fetch EGFR bioactivity data from ChEMBL and compute RDKit descriptors.

Target: EGFR (Epidermal Growth Factor Receptor, CHEMBL203)
Measurement: IC50 / Ki in nM → pIC50 = 9 - log10(IC50_nM), capped at [3, 10]
Features: 18 RDKit physicochemical descriptors (same as ZINC benchmark)

Output: benchmarks/data/chembl_egfr.csv
Columns: smiles, pic50, n_measurements, mw, logp, tpsa, hbd, hba, rotb, rings,
         arom_rings, aliph_rings, fsp3, heavy_atoms, molmr, bertz,
         labute_asa, max_charge, min_charge, n_stereo, n_hetero

Usage:
    python -m benchmarks.fetch_chembl_egfr
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

OUT = os.path.join(os.path.dirname(__file__), "data", "chembl_egfr.csv")

DESC_COLS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings",
             "arom_rings", "aliph_rings", "fsp3", "heavy_atoms", "molmr",
             "bertz", "labute_asa", "max_charge", "min_charge",
             "n_stereo", "n_hetero"]


def fetch_bioactivities():
    """Query ChEMBL for EGFR (CHEMBL203) IC50/Ki measurements."""
    from chembl_webresource_client.new_client import new_client

    activity = new_client.activity
    print("Querying ChEMBL for EGFR (CHEMBL203) bioactivities...")

    res = activity.filter(
        target_chembl_id="CHEMBL203",
        standard_type__in=["IC50", "Ki"],
        standard_relation="=",
        standard_units="nM",
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_type",
        "standard_units",
        "pchembl_value",
    ])

    records = []
    for i, r in enumerate(res):
        if i % 5000 == 0:
            print(f"  fetched {i} records...")
        smiles = r.get("canonical_smiles")
        val    = r.get("standard_value")
        pid    = r.get("molecule_chembl_id")
        pchembl = r.get("pchembl_value")  # ChEMBL's own pIC50
        if smiles and val:
            try:
                nM = float(val)
                if nM > 0:
                    records.append({
                        "chembl_id": pid,
                        "smiles": smiles,
                        "ic50_nM": nM,
                        "pchembl_value": float(pchembl) if pchembl else None,
                    })
            except (ValueError, TypeError):
                pass

    print(f"  Total raw records: {len(records)}")
    return pd.DataFrame(records)


def compute_descriptors(mol):
    AllChem.ComputeGasteigerCharges(mol)
    charges = [mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge")
               for i in range(mol.GetNumAtoms())]
    charges = [c for c in charges if np.isfinite(c)]
    max_q = max(charges) if charges else 0.0
    min_q = min(charges) if charges else 0.0
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
        Descriptors.MolMR(mol),
        Descriptors.BertzCT(mol),
        Descriptors.LabuteASA(mol),
        max_q,
        min_q,
        n_stereo,
        rdMolDescriptors.CalcNumHeteroatoms(mol),
    ]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # ---- Step 1: Fetch from ChEMBL ----
    df_raw = fetch_bioactivities()
    print(f"\nRaw records: {len(df_raw)}")

    # ---- Step 2: Compute pIC50 ----
    # pIC50 = 9 - log10(IC50_nM)  [since IC50_nM = IC50_M × 1e9]
    df_raw["pic50"] = 9.0 - np.log10(df_raw["ic50_nM"].clip(lower=1e-3))

    # ---- Step 3: Deduplicate — keep median pIC50 per canonical SMILES ----
    # Standardise SMILES via RDKit
    print("Standardising SMILES and deduplicating...")
    inchi_map = {}
    for smi in df_raw["smiles"].unique():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            inchi_map[smi] = Chem.MolToInchiKey(mol)
    df_raw["inchi_key"] = df_raw["smiles"].map(inchi_map)
    df_raw = df_raw.dropna(subset=["inchi_key"])

    # Keep one canonical SMILES per InChI key, median pIC50
    agg = df_raw.groupby("inchi_key").agg(
        smiles=("smiles", "first"),
        pic50=("pic50", "median"),
        n_measurements=("pic50", "count"),
    ).reset_index()
    print(f"Unique compounds: {len(agg)}")

    print("Computing RDKit descriptors...")
    rows = []
    skipped = 0
    for i, row in agg.iterrows():
        if i % 2000 == 0:
            print(f"  {i}/{len(agg)}...")
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            skipped += 1
            continue
        try:
            desc = compute_descriptors(mol)
        except Exception:
            skipped += 1
            continue
        if not all(np.isfinite(v) for v in desc):
            skipped += 1
            continue
        rows.append([row["smiles"], row["pic50"], row["n_measurements"]] + desc)

    print(f"Valid: {len(rows)}, skipped: {skipped}")

    cols = ["smiles", "pic50", "n_measurements"] + DESC_COLS
    out_df = pd.DataFrame(rows, columns=cols)

    # Remove pIC50 < 3 (IC50 > 1mM, essentially inactive noise)
    # Remove pIC50 > 10 (IC50 < 0.1 nM — beyond reliable assay range, likely artifacts)
    before = len(out_df)
    out_df = out_df[(out_df["pic50"] >= 3.0) & (out_df["pic50"] <= 10.0)]
    print(f"After pIC50 filter [3, 10]: {len(out_df)} (removed {before - len(out_df)})")

    out_df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}  ({len(out_df)} compounds, {len(DESC_COLS)} descriptors)")

    # ---- Stats ----
    p = out_df["pic50"]
    print(f"\npIC50 stats:")
    print(f"  mean={p.mean():.2f}, std={p.std():.2f}")
    print(f"  min={p.min():.2f}, max={p.max():.2f}")
    print(f"  >7 (IC50<100nM): {(p>7).sum()} ({100*(p>7).mean():.1f}%)")
    print(f"  >8 (IC50<10nM):  {(p>8).sum()} ({100*(p>8).mean():.1f}%)")
    print(f"  >9 (IC50<1nM):   {(p>9).sum()} ({100*(p>9).mean():.1f}%)")
    q95 = p.quantile(0.95)
    print(f"  top 5% threshold: {q95:.2f}")
    print(f"  top 5% count: {(p>=q95).sum()}")


if __name__ == "__main__":
    main()
