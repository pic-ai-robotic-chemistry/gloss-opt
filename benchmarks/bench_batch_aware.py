"""GLOSS against batch-aware baselines, requested by Reviewer 2.

The published benchmark compared GLOSS with BO baselines that build a batch by
taking the top q values of a pointwise acquisition function. Reviewer 2 asked
for baselines that construct the batch as a batch, naming qPO and Thompson
sampling. Both are added here, both driven by the same random forest surrogate
as every other algorithm, so the comparison still isolates the selection rule.

Arms:
    GLOSS(4:2:2)   the published algorithm
    qPO            multipoint probability of optimality (Fromer et al. 2025)
    Thompson       parallel Thompson sampling (Hernandez-Lobato et al. 2017)
    UCB-BO         greedy top-q UCB, kept so the table ties back to the paper
    BO(EI)         greedy top-q expected improvement, the published baseline

Protocol matches the main study: 8 initial points drawn from the bottom 20% of
each pool, batches of 8, 20 rounds, the same five seeds.

Usage:
    python -m benchmarks.bench_batch_aware --study pilot
    python -m benchmarks.bench_batch_aware --study full --dataset qm9
"""

import argparse
import os
import time

import pandas as pd

from benchmarks.bench_main import (
    make_arms as bench_main_make_arms,
    N_ROUNDS, SEEDS,
    RESULTS_DIR, run_discrete,
)
from benchmarks.datasets import load_qm9, load_buchwald_hartwig
from benchmarks.virtual_functions import get_virtual_dataset

THRESHOLD = 95.0
CENSOR_VALUE = N_ROUNDS + 1


# The names and their order are kept exactly as this script first ran them;
# what changed is only where the objects come from. One registry in
# bench_main now builds every arm.
ARMS = ["GLOSS(4:2:2)", "qPO", "Thompson", "UCB-BO", "BO(EI)"]


def make_arms(seed):
    return bench_main_make_arms(seed, ARMS)


def load_datasets(which):
    out = []
    if which in ("all", "bh"):
        print("Loading Buchwald-Hartwig...")
        out.append(load_buchwald_hartwig())
    if which in ("all", "arrhenius"):
        print("Loading Arrhenius-2D (discrete, 10k)...")
        ds = get_virtual_dataset("arrhenius", n_candidates=10000, n_peaks=3,
                                 noise_sigma=0.0, peak_width=0.08, seed=42)
        ds["name"] = "Arrhenius-2D"
        out.append(ds)
    if which in ("all", "qm9"):
        print("Loading QM9-gap (100k)...")
        out.append(load_qm9(n_samples=100000, target="gap"))
    return out


def report(df):
    print("\n" + "=" * 70)
    print(f"t95 = rounds to 95% of pool optimum; a run that never reaches it "
          f"counts as {CENSOR_VALUE}")
    print("=" * 70)
    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name]
        print(f"\n--- {ds_name} ---")
        print(f"  {'arm':14s} {'t95':>6s} {'reached':>9s} {'final %opt':>12s}")
        for arm in sub["algorithm"].unique():
            runs = sub[sub["algorithm"] == arm]
            censored, finals, n_ok = [], [], 0
            for seed in sorted(runs["seed"].unique()):
                run = runs[runs["seed"] == seed]
                hit = run[run["pct_opt"] >= THRESHOLD]
                if len(hit):
                    censored.append(int(hit["round"].min()) + 1)
                    n_ok += 1
                else:
                    censored.append(CENSOR_VALUE)
                finals.append(float(run["pct_opt"].max()))
            n = runs["seed"].nunique()
            print(f"  {arm:14s} {sum(censored)/len(censored):6.1f} "
                  f"{n_ok:>5d}/{n:<3d} {sum(finals)/len(finals):>11.1f}%")


def run(seeds, n_rounds, which, tag):
    datasets = load_datasets(which)
    all_summary, all_points = [], []
    t_start = time.time()

    for ds in datasets:
        print(f"\n=== {ds['name']} ===")
        for seed in seeds:
            arms = make_arms(seed)
            for aname, algo in arms.items():
                print(f"  {ds['name']} / {aname} / seed={seed}")
                s, p = run_discrete(ds, aname, algo, seed, n_rounds)
                for row in p:
                    row["dataset"] = ds["name"]
                    row["algorithm"] = aname
                    row["seed"] = seed
                all_summary.extend(s)
                all_points.extend(p)

    df_sum = pd.DataFrame(all_summary)
    sum_path = os.path.join(RESULTS_DIR, f"bench_batch_aware_{tag}_summary.csv")
    pts_path = os.path.join(RESULTS_DIR, f"bench_batch_aware_{tag}_points.csv")
    df_sum.to_csv(sum_path, index=False)
    pd.DataFrame(all_points).to_csv(pts_path, index=False)

    print(f"\nElapsed: {(time.time() - t_start) / 60:.1f} min")
    print(f"Saved: {os.path.basename(sum_path)}, {os.path.basename(pts_path)}")
    report(df_sum)
    return df_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="pilot", choices=["pilot", "full"])
    parser.add_argument("--dataset", default="all",
                        choices=["all", "qm9", "bh", "arrhenius"])
    args = parser.parse_args()

    suffix = "" if args.dataset == "all" else f"_{args.dataset}"
    if args.study == "pilot":
        run(SEEDS[:1], 5, args.dataset, f"pilot{suffix}")
    else:
        run(SEEDS, N_ROUNDS, args.dataset, f"full{suffix}")
