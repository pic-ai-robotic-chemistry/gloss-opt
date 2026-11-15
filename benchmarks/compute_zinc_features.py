"""Compute 18 RDKit descriptors + QED for ZINC250k.

Saves: benchmarks/data/zinc250k_features.csv
Columns: smiles, qed, mw, logp, tpsa, hbd, hba, rotb, rings,
         arom_rings, aliph_rings, fsp3, heavy_atoms, molmr, bertz,
         labute_asa, max_charge, min_charge, n_stereo, n_hetero

Usage:
    python -m benchmarks.compute_zinc_features
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

RAW  = os.path.join(os.path.dirname(__file__), "data", "zinc250k_raw.csv")
OUT  = os.path.join(os.path.dirname(__file__), "data", "zinc250k_features.csv")

COLS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings",
        "arom_rings", "aliph_rings", "fsp3", "heavy_atoms", "molmr",
        "bertz", "labute_asa", "max_charge", "min_charge",
        "n_stereo", "n_hetero"]


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
    print(f"Reading {RAW} ...")
    df = pd.read_csv(RAW)
    df["smiles"] = df["smiles"].str.strip()
    total = len(df)
    print(f"  {total} molecules")

    rows = []
    skipped = 0
    for i, row in df.iterrows():
        if i % 10000 == 0:
            print(f"  {i}/{total} ({100*i/total:.1f}%)...")
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
        rows.append([row["smiles"], row["qed"]] + desc)

    print(f"  Done. {len(rows)} valid, {skipped} skipped.")
    out_df = pd.DataFrame(rows, columns=["smiles", "qed"] + COLS)
    out_df.to_csv(OUT, index=False)
    print(f"Saved: {OUT}  ({len(out_df)} rows, {len(COLS)} descriptors)")

    # Quick stats
    print(f"\nQED: mean={out_df['qed'].mean():.4f}, "
          f"max={out_df['qed'].max():.4f}, "
          f"top5%_thresh={out_df['qed'].quantile(0.95):.4f}, "
          f"top5%_count={(out_df['qed'] >= out_df['qed'].quantile(0.95)).sum()}")


if __name__ == "__main__":
    main()
