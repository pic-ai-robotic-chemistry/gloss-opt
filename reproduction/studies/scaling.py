"""Repeat the RF QM9 scaling and ratio scans on the paper's sampled libraries."""
import argparse

import pandas as pd

from benchmarks.bench_main import N_ROUNDS, SEEDS_10, load_qm9, make_arms, run_discrete
from reproduction.paths import OUTPUT

ARMS = ["GLOSS(4:2:2)", "qPO", "Thompson", "UCB-BO", "BO(EI)", "GA", "Random"]
RATIOS = ["GLOSS(2:3:3)", "GLOSS(4:2:2)", "GLOSS(6:1:1)"]
SIZES = range(5000, 100001, 5000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["scaling", "ratios"], default="scaling")
    parser.add_argument("--sizes", nargs="+", type=int, choices=SIZES, default=list(SIZES))
    parser.add_argument("--seeds", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--rounds", type=int, choices=range(1, 21), default=N_ROUNDS)
    args = parser.parse_args()
    output = OUTPUT / "reruns" / args.study / f"{args.rounds}_rounds"
    output.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        ds = load_qm9(n_samples=size, target="gap")
        ds["name"] = f"QM9-gap-{size // 1000}k"
        for seed in SEEDS_10[:args.seeds]:
            rows = []
            for arm, algo in make_arms(seed, RATIOS if args.study == "ratios" else ARMS).items():
                summary, _ = run_discrete(ds, arm, algo, seed, args.rounds)
                rows.extend(summary)
            pd.DataFrame(rows).to_csv(output / f"{size}_{seed}.csv", index=False)


if __name__ == "__main__":
    main()
