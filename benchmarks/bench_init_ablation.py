"""Does the GLOSS advantage survive a different starting set?

Reviewer 2 pointed out that every published run starts from eight points drawn
from the bottom 20% of the pool, and that this is exactly the starting
condition under which forced exploration pays off, so it may not represent a
real campaign. This script repeats the comparison from three starting sets.

    bottom20   eight points from the worst 20% by value, as published
    random     eight points drawn uniformly from the whole pool
    lhs        a Latin hypercube in the feature box, each point snapped to its
               nearest unused candidate

A Latin hypercube spreads points by splitting every feature into equal strata
and taking one point per stratum, so the start covers the space evenly instead
of landing wherever chance puts it. It is defined for a continuous box, while
these pools are finite sets of real molecules or conditions, so each generated
point is moved to the nearest candidate that is still free. Two consequences
are worth stating rather than hiding:

  - snapping pulls points off their strata, so the start is only approximately
    even, and the more crowded the pool the larger the pull;
  - with eight points in twenty or more features, stratification buys little
    over uniform sampling, because eight points cannot cover that many
    dimensions either way. The comparison that carries weight here is
    bottom20 against random.

Buchwald-Hartwig is run with bottom20 and random only. Its features are 44
one-hot columns, so the feature box is a corner set, not a volume, and a Latin
hypercube in it has no meaning.

Usage:
    python -m benchmarks.bench_init_ablation --dataset arrhenius
    python -m benchmarks.bench_init_ablation --dataset qm9 --seeds 10
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

from benchmarks.algorithms import GLOSSAlgorithm, UCBOAlgorithm, QPOAlgorithm
from benchmarks.bench_main import (
    N_INIT, N_ROUNDS, SEEDS, GLOSS_STRATEGY, GLOSS_DIVERSITY_RADIUS,
    RESULTS_DIR, run_discrete, sample_init,
)
from benchmarks.bench_scaling_batch_aware import ALL_SEEDS
from benchmarks.datasets import load_qm9, load_buchwald_hartwig
from benchmarks.virtual_functions import get_virtual_dataset

THRESHOLD = 95.0
CENSOR_VALUE = N_ROUNDS + 1


def make_arms(seed):
    return {
        "GLOSS(4:2:2)": GLOSSAlgorithm(
            strategy_points=GLOSS_STRATEGY, ucb_kappa=2.0,
            diversity_radius=GLOSS_DIVERSITY_RADIUS, seed=seed),
        "qPO": QPOAlgorithm(seed=seed),
        "UCB-BO": UCBOAlgorithm(kappa=2.0, seed=seed),
    }


def init_bottom20(candidates, y_true, n_init, seed):
    return sample_init(candidates, y_true, n_init, seed)


def init_random(candidates, y_true, n_init, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(len(candidates), size=n_init, replace=False).tolist()


def init_lhs(candidates, y_true, n_init, seed):
    """Latin hypercube in the feature box, snapped to nearest free candidates."""
    lo = candidates.min(axis=0)
    hi = candidates.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)

    sampler = qmc.LatinHypercube(d=candidates.shape[1], seed=seed)
    targets = lo + sampler.random(n_init) * span

    chosen = []
    taken = np.zeros(len(candidates), dtype=bool)
    for target in targets:
        dists = np.linalg.norm(candidates - target, axis=1)
        dists[taken] = np.inf
        idx = int(np.argmin(dists))
        taken[idx] = True
        chosen.append(idx)
    return chosen


INITS = {
    "bottom20": init_bottom20,
    "random": init_random,
    "lhs": init_lhs,
}


def load_datasets(which):
    """Returns (dataset, list of init modes to run on it)."""
    out = []
    if which in ("all", "bh"):
        print("Loading Buchwald-Hartwig...")
        # One-hot features: no continuous box, so no Latin hypercube.
        out.append((load_buchwald_hartwig(), ["bottom20", "random"]))
    if which in ("all", "arrhenius"):
        print("Loading Arrhenius-2D (discrete, 10k)...")
        ds = get_virtual_dataset("arrhenius", n_candidates=10000, n_peaks=3,
                                 noise_sigma=0.0, peak_width=0.08, seed=42)
        ds["name"] = "Arrhenius-2D"
        out.append((ds, ["bottom20", "random", "lhs"]))
    if which in ("all", "qm9"):
        print("Loading QM9-gap (100k)...")
        out.append((load_qm9(n_samples=100000, target="gap"),
                    ["bottom20", "random", "lhs"]))
    return out


def report(df):
    print("\n" + "=" * 74)
    print(f"t95 by starting set; a run that never reaches 95% counts as "
          f"{CENSOR_VALUE}")
    print("=" * 74)
    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name]
        print(f"\n--- {ds_name} ---")
        modes = [m for m in INITS if m in set(sub["init"])]
        print(f"  {'arm':14s} " + " ".join(f"{m:>17s}" for m in modes))
        for arm in sub["algorithm"].unique():
            cells = []
            for mode in modes:
                runs = sub[(sub["algorithm"] == arm) & (sub["init"] == mode)]
                censored, n_ok = [], 0
                for seed in sorted(runs["seed"].unique()):
                    run = runs[runs["seed"] == seed]
                    hit = run[run["pct_opt"] >= THRESHOLD]
                    if len(hit):
                        censored.append(int(hit["round"].min()) + 1)
                        n_ok += 1
                    else:
                        censored.append(CENSOR_VALUE)
                cells.append(f"{np.mean(censored):5.1f} ({n_ok}/{len(censored)})")
            print(f"  {arm:14s} " + " ".join(f"{c:>17s}" for c in cells))


def run(seeds, n_rounds, which, tag):
    datasets = load_datasets(which)
    all_summary = []
    t_start = time.time()

    for ds, modes in datasets:
        print(f"\n=== {ds['name']} ===", flush=True)
        for mode in modes:
            make_init = INITS[mode]
            for seed in seeds:
                init_idx = make_init(ds["candidates"], ds["y_true"],
                                     N_INIT, seed)
                arms = make_arms(seed)
                for aname, algo in arms.items():
                    s, _ = run_discrete(ds, aname, algo, seed, n_rounds,
                                        init_idx=init_idx)
                    for row in s:
                        row["init"] = mode
                    all_summary.extend(s)
                print(f"  {ds['name']} / init={mode} / seed={seed} done",
                      flush=True)

    df = pd.DataFrame(all_summary)
    out = os.path.join(RESULTS_DIR, f"bench_init_ablation_{tag}.csv")
    df.to_csv(out, index=False)
    print(f"\nElapsed: {(time.time() - t_start) / 60:.1f} min")
    print(f"Saved: {os.path.basename(out)}")
    report(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "qm9", "bh", "arrhenius"])
    parser.add_argument("--seeds", type=int, default=len(SEEDS),
                        help=f"how many seeds, up to {len(ALL_SEEDS)}")
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    args = parser.parse_args()

    if args.seeds > len(ALL_SEEDS):
        parser.error(f"--seeds at most {len(ALL_SEEDS)}")
    suffix = "" if args.dataset == "all" else f"_{args.dataset}"
    run(ALL_SEEDS[:args.seeds], args.rounds, args.dataset, f"full{suffix}")
