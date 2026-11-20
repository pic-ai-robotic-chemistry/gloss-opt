"""Measure within-batch distance and spread for SI Table S13."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from benchmarks.algorithms import GLOSSAlgorithm, RandomAlgorithm  # noqa: E402
from benchmarks.bench_main import continuous_landscape, run_continuous  # noqa: E402

RESULTS = os.path.join(ROOT, "build", "paper", "reruns", "radius")

# (ndim, n_peaks, peak_width) — the three levels the earlier run used.
LEVELS = [(2, 3, 0.1), (4, 5, 0.08), (10, 7, 0.2)]
RATIOS = ["2:3:3", "4:2:2", "6:1:1"]
SEED = 47
ROUNDS = 10


def ratio_arm(ratio: str, seed: int) -> GLOSSAlgorithm:
    """GLOSS with the batch share named by the ratio, radius at its default."""
    g, l, u = (int(part) for part in ratio.split(":"))
    return GLOSSAlgorithm(
        strategy_points={"global_best": g, "local_best": l,
                         "unexplored": u, "unconverged": 0},
        ucb_kappa=2.0, diversity_radius=0.02, seed=seed)


def batches(points, ndim: int) -> list[np.ndarray]:
    """One array per round, holding that round's picks."""
    out: dict[int, list[list[float]]] = {}
    for point in points:
        if point["round"] < 0:
            continue
        out.setdefault(int(point["round"]), []).append(
            [point[f"x{d}"] for d in range(ndim)])
    return [np.asarray(rows, dtype=float) for rows in out.values()]


def batch_stats(points, ndim: int, key_dims: list[int]) -> dict[str, float]:
    """Smallest within-batch distance, and the two spreads, over the run.

    The smallest distance is the minimum over rounds: one batch that puts two
    picks on top of each other is the case the radius exists to prevent, so an
    average would hide it. The spreads are means over rounds, raw (not divided
    by the dimension), so that the ratio of two spreads is dimension free.
    """
    rest = [d for d in range(ndim) if d not in key_dims]
    smallest, key, other = [], [], []
    for arr in batches(points, ndim):
        if len(arr) < 2:
            continue
        smallest.append(float(pdist(arr).min()))
        key.append(float(pdist(arr[:, key_dims]).mean()))
        if rest:
            other.append(float(pdist(arr[:, rest]).mean()))
    return {"min_batch_distance": min(smallest) if smallest else float("nan"),
            "key_spread": float(np.mean(key)) if key else float("nan"),
            "other_spread": float(np.mean(other)) if other else float("nan")}


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for ndim, n_peaks, width in LEVELS:
        land = continuous_landscape("nd_clusters", ndim=ndim, n_peaks=n_peaks,
                                    peak_width=width, noise_sigma=0.0, seed=42)
        key_dims = land["key_dims"]
        for ratio in RATIOS + ["Random"]:
            make = (lambda s: RandomAlgorithm(seed=s)) if ratio == "Random" \
                else (lambda s, r=ratio: ratio_arm(r, s))
            label = "Random" if ratio == "Random" else f"GLOSS({ratio})"
            _summary, points = run_continuous(land, f"C{ndim}", label, make(SEED),
                                              SEED, n_rounds=ROUNDS)
            stats = batch_stats(points, ndim, key_dims)
            rows.append({"ndim": ndim, "n_peaks": n_peaks, "arm": ratio,
                         "seed": SEED, "rounds": ROUNDS, "n_key_dims": len(key_dims),
                         "min_batch_distance": round(stats["min_batch_distance"], 6),
                         "key_spread": round(stats["key_spread"], 6),
                         "other_spread": round(stats["other_spread"], 6)})
            print(f"  {ndim:2d}D  {ratio:10s} min {stats['min_batch_distance']:.4f}  "
                  f"key {stats['key_spread']:.4f}  other {stats['other_spread']:.4f}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "radius_by_dimension.csv")
    df.to_csv(path, index=False)
    print(f"\nwritten to {os.path.relpath(path, ROOT)}")

    print("\nsmallest distance inside a batch, over the three ratio arms:")
    for ndim, _, _ in LEVELS:
        sub = df[(df["ndim"] == ndim) & (df.arm != "Random")]
        print(f"  {ndim:2d}D  {sub.min_batch_distance.min():.4f} to "
              f"{sub.min_batch_distance.max():.4f}")

    print("\nGLOSS(4:2:2) key-axis spread against random (below 1 is tighter):")
    for ndim, _, _ in LEVELS:
        g = df[(df["ndim"] == ndim) & (df.arm == "4:2:2")].iloc[0]
        r = df[(df["ndim"] == ndim) & (df.arm == "Random")].iloc[0]
        print(f"  {ndim:2d}D  {g.key_spread / r.key_spread:.3f}   "
              f"off the key axes {g.other_spread / r.other_spread:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
