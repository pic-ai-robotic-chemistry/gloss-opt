"""Re-run the scaling study with n_rounds = 100 (vs default 20 in bench_main).

Identical sweep otherwise: QM9-gap at n = 5k -> 100k step 5k, 5 algorithms,
5 seeds. Outputs land in distinct files so the original Figure 3 data stays
untouched.

Outputs (all NEW, no overwrite):
  benchmarks/results/bench_scaling_v100_summary.csv

Run (background recommended; ~2-3 hours wall-clock on Apple silicon):
  PYTHONUNBUFFERED=1 python -u -m benchmarks.bench_scaling_v100 \
      > /tmp/bench_scaling_v100.log 2>&1 &
"""

import os
import time

import pandas as pd

from benchmarks.bench_main import (
    SEEDS,
    RESULTS_DIR,
    make_algorithms,
    run_discrete,
)
from benchmarks.datasets import load_qm9


N_ROUNDS_V100 = 100
OUT_CSV = os.path.join(RESULTS_DIR, "bench_scaling_v100_summary.csv")


def main() -> None:
    print("=" * 60)
    print(f"SCALING STUDY (v100) — n_rounds = {N_ROUNDS_V100}")
    print("=" * 60)
    if os.path.exists(OUT_CSV):
        raise SystemExit(
            f"refusing to overwrite existing {OUT_CSV}; "
            "rename or remove it before re-running"
        )

    all_summary = []
    scales = list(range(5000, 105000, 5000))  # 5k -> 100k step 5k
    t_start = time.time()

    for n in scales:
        t_scale_start = time.time()
        print(f"\n--- Scale n={n} ---", flush=True)
        ds = load_qm9(n_samples=n, target="gap")
        ds["name"] = f"QM9-gap-{n // 1000}k"
        for seed in SEEDS:
            algos = make_algorithms(seed)
            for aname, algo in algos.items():
                t0 = time.time()
                print(
                    f"  {ds['name']} / {aname} / seed={seed} ...",
                    flush=True,
                )
                rows, _ = run_discrete(ds, aname, algo, seed, N_ROUNDS_V100)
                all_summary.extend(rows)
                print(
                    f"    finished in {time.time() - t0:.1f}s",
                    flush=True,
                )
        print(
            f"  scale n={n} total {time.time() - t_scale_start:.1f}s "
            f"(elapsed {time.time() - t_start:.0f}s)",
            flush=True,
        )

    df = pd.DataFrame(all_summary)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Total wall-clock: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
