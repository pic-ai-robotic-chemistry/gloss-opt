"""Scaling benchmark on the *quantile-stratified* QM9-gap subsets.

Reads benchmarks/data/qm9_strat/qm9_strat_n*.npz built by
benchmarks.build_qm9_strat. Same algorithm protocol as
benchmarks.bench_main.study_scaling (5 algos x 5 seeds x N_ROUNDS rounds).

Distinct outputs (no overwrite of paper Figure 3 data):
  benchmarks/results/bench_scaling_strat_summary.csv

Run (background recommended):
  PYTHONUNBUFFERED=1 python -u -m benchmarks.bench_scaling_strat \\
      > /tmp/bench_scaling_strat.log 2>&1 &
"""

import glob
import os
import time

import numpy as np
import pandas as pd

from benchmarks.bench_main import (
    SEEDS,
    RESULTS_DIR,
    make_algorithms,
    run_discrete,
    N_ROUNDS as DEFAULT_N_ROUNDS,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "qm9_strat")
OUT_CSV = os.path.join(RESULTS_DIR, "bench_scaling_strat_summary.csv")

# Match the existing paper Figure 3 horizon so the new run is directly
# comparable (uniform vs stratified, same n_rounds).
N_ROUNDS = DEFAULT_N_ROUNDS  # = 20


def load_subset(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    n_real = int(d["y"].shape[0])
    name = os.path.basename(path).replace("qm9_strat_n", "").replace(".npz", "")
    return {
        "name": f"QM9-gap-strat-{n_real // 1000}k",
        "candidates": d["X"].astype(np.float64),
        "y_true": d["y"].astype(np.float64),
        "direction": "maximize",
    }


def main() -> None:
    if os.path.exists(OUT_CSV):
        raise SystemExit(
            f"refusing to overwrite existing {OUT_CSV}; "
            "rename or remove it before re-running")

    paths = sorted(glob.glob(os.path.join(DATA_DIR, "qm9_strat_n*.npz")))
    if not paths:
        raise SystemExit(
            f"no stratified subsets in {DATA_DIR}; "
            "run `python -m benchmarks.build_qm9_strat` first")
    print(f"Found {len(paths)} stratified subsets; n_rounds={N_ROUNDS}")

    all_summary = []
    t_start = time.time()

    for path in paths:
        t_scale = time.time()
        ds = load_subset(path)
        n = len(ds["y_true"])
        print(f"\n--- {ds['name']} (n={n}) ---", flush=True)
        for seed in SEEDS:
            algos = make_algorithms(seed)
            for aname, algo in algos.items():
                t0 = time.time()
                print(f"  {ds['name']} / {aname} / seed={seed}", flush=True)
                rows, _ = run_discrete(ds, aname, algo, seed, N_ROUNDS)
                all_summary.extend(rows)
                print(f"    finished in {time.time() - t0:.1f}s", flush=True)
        print(
            f"  scale total {time.time() - t_scale:.1f}s  "
            f"(elapsed {time.time() - t_start:.0f}s)",
            flush=True,
        )

    df = pd.DataFrame(all_summary)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Total wall-clock: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
