"""Three-arm QM9 scaling experiment with GLOSS(4:2:2), qPO and UCB-BO.

Each library is sampled with load_qm9. The experiment uses eight initial
points from the bottom quintile and requests eight points per round for
twenty rounds. See reproduction.studies.scaling for the seven-arm study.

Usage:
    python -m benchmarks.bench_scaling_batch_aware --scales 20000 --seeds 1
    python -m benchmarks.bench_scaling_batch_aware
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from benchmarks.bench_main import (
    make_arms as bench_main_make_arms,
    N_ROUNDS, SEEDS, SEEDS_10,
    RESULTS_DIR, run_discrete,
)
from benchmarks.datasets import load_qm9

THRESHOLD = 95.0
CENSOR_VALUE = N_ROUNDS + 1
DEFAULT_SCALES = list(range(5000, 105000, 5000))

# The scaling experiment uses the first ten paper seeds.
ALL_SEEDS = SEEDS_10


# The names and their order are kept exactly as this script first ran them;
# what changed is only where the objects come from. One registry in
# bench_main now builds every arm.
ARMS = ["GLOSS(4:2:2)", "qPO", "UCB-BO"]


def make_arms(seed):
    return bench_main_make_arms(seed, ARMS)


def report(df):
    print("\n" + "=" * 62)
    print(f"t95 by pool size; a run that never reaches 95% counts as "
          f"{CENSOR_VALUE}")
    print("=" * 62)
    arms = list(df["algorithm"].unique())
    print(f"  {'n':>8s} " + " ".join(f"{a:>16s}" for a in arms))
    for ds_name in sorted(df["dataset"].unique(),
                          key=lambda s: int(s.split("-")[-1].rstrip("k"))):
        sub = df[df["dataset"] == ds_name]
        cells = []
        for arm in arms:
            runs = sub[sub["algorithm"] == arm]
            censored, n_ok = [], 0
            for seed in sorted(runs["seed"].unique()):
                run = runs[runs["seed"] == seed]
                hit = run[run["pct_opt"] >= THRESHOLD]
                if len(hit):
                    censored.append(int(hit["round"].min()) + 1)
                    n_ok += 1
                else:
                    censored.append(CENSOR_VALUE)
            cells.append(f"{np.mean(censored):6.1f} ({n_ok}/{len(censored)})")
        print(f"  {ds_name.split('-')[-1]:>8s} " +
              " ".join(f"{c:>16s}" for c in cells))


def run(scales, seeds, n_rounds, tag):
    all_summary = []
    t_start = time.time()

    for n in scales:
        t_load = time.time()
        ds = load_qm9(n_samples=n, target="gap")
        ds["name"] = f"QM9-gap-{n // 1000}k"
        load_s = time.time() - t_load
        print(f"\n--- n={n} ({len(ds['candidates'])} candidates after "
              f"filtering, loaded in {load_s:.0f}s) ---", flush=True)

        for seed in seeds:
            arms = make_arms(seed)
            for aname, algo in arms.items():
                t0 = time.time()
                s, _ = run_discrete(ds, aname, algo, seed, n_rounds)
                all_summary.extend(s)
                print(f"  {ds['name']} / {aname} / seed={seed} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(all_summary)
    out = os.path.join(RESULTS_DIR, f"bench_scaling_batch_aware_{tag}.csv")
    df.to_csv(out, index=False)
    print(f"\nElapsed: {(time.time() - t_start) / 60:.1f} min")
    print(f"Saved: {os.path.basename(out)}")
    report(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", default=DEFAULT_SCALES,
                        help="pool sizes to sweep")
    parser.add_argument("--seeds", type=int, default=len(SEEDS),
                        help=f"how many seeds to use, up to {len(ALL_SEEDS)}; "
                             "the first five are the published ones")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    if args.seeds > len(ALL_SEEDS):
        parser.error(f"--seeds at most {len(ALL_SEEDS)}")
    seeds = ALL_SEEDS[:args.seeds]
    tag = args.tag or ("probe" if len(args.scales) < len(DEFAULT_SCALES)
                       else "summary")
    run(args.scales, seeds, N_ROUNDS, tag)
