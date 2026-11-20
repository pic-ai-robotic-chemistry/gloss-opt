from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from reproduction.paths import TABLES

MAIN = {"qm9": "QM9-gap", "arr": "Arrhenius-2D-cont", "bh": "Buchwald-Hartwig"}
RESAMPLES = 100_000
SEED = 0
ALPHA = 0.05
BASE = "GLOSS(4:2:2)"
COMPARISONS = (
    "qPO",
    "Thompson",
    "UCB-BO",
    "BO(EI)",
    "GA",
    "Random",
    "GLOSS(8:0:0)",
    "GLOSS(8:0:0)+div",
)
DATASETS = ("qm9", "bh", "arr")
EXPECTED_SEEDS = 20


def generator(dataset: str, arm: str) -> np.random.Generator:
    """A generator that depends only on the contrast, not on loop order."""
    key = hashlib.sha256(f"{SEED}|{dataset}|{arm}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "little"))


def bootstrap(diffs: np.ndarray, rng: np.random.Generator, *, archived_method=False) -> tuple[float, float, float, bool]:
    """Paired percentile interval and twice the smaller empirical tail probability.

    The default counts means equal to zero in both tails. Strict tails
    reproduce the historical analysis. Identical pairs give p = 1.
    """
    n = len(diffs)
    means = rng.choice(diffs, (RESAMPLES, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    if np.all(diffs == 0):
        return 0.0, 0.0, 1.0, False
    if archived_method:
        below, above = (means < 0).sum(), (means > 0).sum()
    else:
        below, above = (means <= 0).sum(), (means >= 0).sum()
    tail = min(below, above)
    floor = tail == 0
    p = 2.0 * tail / RESAMPLES
    if floor:
        p = 1.0 / RESAMPLES
    return float(lo), float(hi), float(min(p, 1.0)), floor


def holm(pvals: list[float], alpha: float) -> list[bool]:
    """Holm step-down decisions, in the order of the input list."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    decision = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            decision[i] = True
        else:
            break
    return decision


def contrast_units(units: pd.DataFrame, dataset: str, arm: str) -> np.ndarray:
    """Paired differences for one contrast, with guards against silent failure."""
    base = units[(units["dataset"] == MAIN[dataset]) &
                 (units["algorithm"] == BASE)].set_index("seed")["t95"]
    other = units[(units["dataset"] == MAIN[dataset]) &
                  (units["algorithm"] == arm)].set_index("seed")["t95"]
    if base.empty or other.empty:
        raise SystemExit(f"missing arm for {dataset} / {arm}")
    if set(base.index) != set(other.index):
        raise SystemExit(f"seed sets differ for {dataset} / {arm}")
    if len(base) != EXPECTED_SEEDS:
        raise SystemExit(f"{len(base)} seeds for {dataset} / {arm}, expected {EXPECTED_SEEDS}")
    diffs = (other.sort_index() - base.sort_index()).values
    if np.isnan(diffs).any():
        raise SystemExit(f"NaN in the difference for {dataset} / {arm}")
    return diffs


def write_contrasts(units, *, archived_method=False):
    rows = []
    for ds in DATASETS:
        for arm in COMPARISONS:
            diffs = contrast_units(units, ds, arm)
            lo, hi, p, floor = bootstrap(diffs, generator(ds, arm), archived_method=archived_method)
            rows.append({
                "dataset": ds,
                "baseline": arm,
                "diff": float(diffs.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "p_floor": floor,
                "n_seeds": len(diffs),
            })
    out = pd.DataFrame(rows)

    raw_p = out["p_value"].tolist()
    out["holm_significant"] = holm(raw_p, ALPHA)
    out["holm_significant_dataset"] = False
    for ds in DATASETS:
        idx = out.index[out["dataset"] == ds].tolist()
        for i, flag in zip(idx, holm([raw_p[i] for i in idx], ALPHA)):
            out.loc[i, "holm_significant_dataset"] = flag

    # Format for the CSV: two decimals for rounds, and a p value that never
    # prints as 0.00.
    def pfmt(p: float, floor: bool, adjusted: bool) -> str:
        if floor and adjusted:
            return f"<{2.0 / RESAMPLES:.0e}"
        return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"

    out["p_value"] = [pfmt(p, f, a) for p, f, a in
                      zip(raw_p, out["p_floor"], out["holm_significant"])]
    for column in ("diff", "ci_low", "ci_high"):
        out[column] = out[column].map(lambda x: f"{x:.2f}")
    out["resolved"] = [float(lo) > 0 or float(hi) < 0
                       for lo, hi in zip(out["ci_low"], out["ci_high"])]

    path = TABLES / ("main_contrasts_archived_method.csv" if archived_method else "main_contrasts.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\nfamily size {len(out)}, Holm survivors {int(out['holm_significant'].sum())}")
    print(f"per-dataset family, survivors {int(out['holm_significant_dataset'].sum())}")
    print(f"wrote {path}")
    return 0
