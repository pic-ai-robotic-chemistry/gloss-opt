"""Run the local-penalisation comparison. Each unit saves its trajectory and occupancy diagnostics."""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.bench_main import (  # noqa: E402
    continuous_landscape, make_arms, run_continuous, run_discrete,
)
from benchmarks.datasets import load_buchwald_hartwig, load_qm9  # noqa: E402

OUT = os.path.join(ROOT, "build", "paper", "reruns", "local_penalty")

#: The twenty seeds, frozen, first ten published.
SEEDS = (47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
         97, 101, 103, 107, 109, 113, 127, 131, 137, 139)

ARMS = ("LocalPenalty(EI)", "LocalPenalty(UCB)")
DATASETS = ("qm9", "bh", "arr")

_CACHE = {}


def dataset_payload(name):
    """The dataset as the published studies build it, cached per process."""
    if name in _CACHE:
        return _CACHE[name]
    if name == "qm9":
        ds = load_qm9(n_samples=100000, target="gap")
        _CACHE[name] = {"kind": "discrete", "ds": ds, "name": ds["name"]}
    elif name == "bh":
        ds = load_buchwald_hartwig()
        _CACHE[name] = {"kind": "discrete", "ds": ds, "name": ds["name"]}
    elif name == "arr":
        land = continuous_landscape("arrhenius", n_peaks=3, peak_width=0.08,
                                    noise_sigma=0.0, seed=42)
        _CACHE[name] = {"kind": "continuous", "land": land,
                        "name": "Arrhenius-2D-cont"}
    else:
        raise KeyError(name)
    return _CACHE[name]


def unit_path(arm, ds, seed):
    safe = arm.replace("(", "_").replace(")", "").replace(":", "")
    return os.path.join(OUT, f"{safe}_{ds}_s{seed}.csv")


def run_unit(arm, ds, seed):
    payload = dataset_payload(ds)
    algo = make_arms(seed, [arm])[arm]
    t0 = time.time()
    if payload["kind"] == "discrete":
        summary, points = run_discrete(payload["ds"], arm, algo, seed)
    else:
        summary, points = run_continuous(payload["land"], payload["name"],
                                         arm, algo, seed)
    diag = algo.diagnostics_frame()
    # Diagnostics are one row per round, in order, starting at round 0.
    diag.insert(0, "round", np.arange(len(diag)))
    diag.insert(0, "seed", seed)
    diag.insert(0, "dataset", payload["name"])
    diag.insert(0, "algorithm", arm)
    return summary, points, diag, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", default=None)
    args = ap.parse_args()

    seeds = SEEDS if (args.seeds is None or args.seeds == ["all"]) else tuple(
        int(s) for s in args.seeds)
    os.makedirs(OUT, exist_ok=True)
    todo = [(arm, ds, seed) for arm in args.arms for ds in args.datasets
            for seed in seeds]
    print(f"{len(todo)} units queued", flush=True)
    done = 0
    for arm, ds, seed in todo:
        path = unit_path(arm, ds, seed)
        if os.path.exists(path):
            done += 1
            continue
        summary, points, diag, secs = run_unit(arm, ds, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(points).to_csv(path.replace(".csv", "_points.csv"),
                                    index=False)
        diag.to_csv(path.replace(".csv", "_diag.csv"), index=False)
        pd.DataFrame(summary).to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        done += 1
        best = max(r["pct_opt"] for r in summary)
        print(f"[{done}/{len(todo)}] {arm:18s} {ds:3s} s{seed}: "
              f"pct={best:5.1f} ({secs:.0f}s)", flush=True)
    print("all units done", flush=True)


if __name__ == "__main__":
    main()
