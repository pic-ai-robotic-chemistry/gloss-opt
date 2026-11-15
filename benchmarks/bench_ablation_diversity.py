"""Ablation: batch diversification vs. strategy decomposition.

Reviewer 2 (ci-2026-02855a) noted that the BO baselines form a batch by taking
the top-q values of a pointwise acquisition function, while GLOSS both
decomposes the batch across strategies AND enforces a minimum distance between
batch points (diversity_radius). The reported advantage could therefore come
from batch diversification alone.

This study separates the two effects. All four arms share the same RF
surrogate, the same bottom-20% initialisation, q = 8, 20 rounds and the same
five seeds as the main study:

    GLOSS(4:2:2)        decomposition + diversification  (the published algorithm)
    GLOSS(8:0:0)+div    diversification only             <- the arm that matters
    GLOSS(8:0:0)        neither                          (sanity check: should
                                                          reproduce UCB-BO)
    UCB-BO              neither                         (greedy top-q)

Reading the result:
    8:0:0+div close to 4:2:2       -> the gain is diversification, not decomposition
    8:0:0+div close to UCB-BO      -> the gain is decomposition, as published
    in between                     -> both contribute; report the split

Usage:
    python -m benchmarks.bench_ablation_diversity --study pilot   # 1 seed, 5 rounds
    python -m benchmarks.bench_ablation_diversity --study full
    python -m benchmarks.bench_ablation_diversity --study full --dataset qm9
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

# Four arms differing only in decomposition and diversity radius. The names
# and their order are kept exactly as this script first ran them; what
# changed is only where the objects come from. One registry in bench_main
# now builds every arm, because this list used to exist in nine places.
ARMS = ["GLOSS(4:2:2)", "GLOSS(8:0:0)+div", "GLOSS(8:0:0)", "UCB-BO"]


def make_arms(seed):
    return bench_main_make_arms(seed, ARMS)


def load_datasets(which):
    """Return the datasets to run, in increasing cost order."""
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


def run(seeds, n_rounds, which, tag):
    datasets = load_datasets(which)

    all_summary = []
    all_points = []
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
    df_pts = pd.DataFrame(all_points)
    sum_path = os.path.join(RESULTS_DIR, f"bench_ablation_diversity_{tag}_summary.csv")
    pts_path = os.path.join(RESULTS_DIR, f"bench_ablation_diversity_{tag}_points.csv")
    df_sum.to_csv(sum_path, index=False)
    df_pts.to_csv(pts_path, index=False)

    print(f"\nElapsed: {(time.time() - t_start) / 60:.1f} min")
    print(f"Saved: {os.path.basename(sum_path)}, {os.path.basename(pts_path)}")
    report(df_sum, n_rounds)
    return df_sum


def report(df, n_rounds):
    """Print t95 and final pct_opt per arm.

    t95 is the first round reaching 95% of the pool optimum, 1-indexed to match
    the paper. Runs that never reach it are reported separately as a success
    rate instead of being folded into the mean, which is what Reviewer 2 asked
    for.
    """
    print("\n" + "=" * 68)
    print("t95 = rounds to 95% of pool optimum (lower is better; budget = "
          f"{n_rounds} rounds)")
    print("mean t95 is over the runs that reached it; n/N is the success rate")
    print("=" * 68)

    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name]
        print(f"\n--- {ds_name} ---")
        print(f"  {'arm':20s} {'mean t95':>10s} {'reached':>9s} {'final %opt':>12s}")
        for aname in sub["algorithm"].unique():
            arm = sub[sub["algorithm"] == aname]
            t95s = []
            finals = []
            for seed in arm["seed"].unique():
                run_df = arm[arm["seed"] == seed]
                reached = run_df[run_df["pct_opt"] >= 95.0]
                if len(reached):
                    t95s.append(int(reached["round"].min()) + 1)
                finals.append(float(run_df["pct_opt"].max()))
            n_seeds = arm["seed"].nunique()
            mean_t95 = f"{sum(t95s) / len(t95s):.1f}" if t95s else "never"
            print(f"  {aname:20s} {mean_t95:>10s} {len(t95s):>5d}/{n_seeds:<3d} "
                  f"{sum(finals) / len(finals):>11.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="pilot", choices=["pilot", "full"])
    parser.add_argument("--dataset", default="all",
                        choices=["all", "qm9", "bh", "arrhenius"])
    args = parser.parse_args()

    # Tag by dataset so separate invocations do not overwrite each other's CSV.
    suffix = "" if args.dataset == "all" else f"_{args.dataset}"

    if args.study == "pilot":
        run(seeds=SEEDS[:1], n_rounds=5, which=args.dataset,
            tag=f"pilot{suffix}")
    else:
        run(seeds=SEEDS, n_rounds=N_ROUNDS, which=args.dataset,
            tag=f"full{suffix}")
