"""Repeat the four shared starting designs on three datasets."""
import argparse

import numpy as np
import pandas as pd

from benchmarks.bench_main import RERUN_ARMS, SEEDS_20, make_arms, run_continuous, run_discrete
from reproduction.paths import OUTPUT
from .init_designs import continuous_design, discrete_design
from .local_penalty import DATASETS, dataset_payload

DESIGNS = ["bottom20", "uniform", "lhs", "maxmin"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
    parser.add_argument("--designs", choices=DESIGNS, nargs="+", default=DESIGNS)
    parser.add_argument("--seeds", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--rounds", type=int, choices=range(1, 21), default=20)
    args = parser.parse_args()
    output = OUTPUT / "reruns/initialization" / f"{args.rounds}_rounds"
    output.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        payload = dataset_payload(ds)
        for design in args.designs:
            for seed in SEEDS_20[:args.seeds]:
                if payload["kind"] == "discrete":
                    pool = payload["ds"]
                    start, _ = discrete_design(design, pool["candidates"], pool["y_true"], seed)
                    expected = np.max(pool["y_true"][start])
                else:
                    land = payload["land"]
                    start, _ = continuous_design(design, land, seed)
                    expected = np.max(land["surface"](start))
                rows = []
                for name, arm in make_arms(seed, RERUN_ARMS).items():
                    if payload["kind"] == "discrete":
                        summary, _ = run_discrete(pool, name, arm, seed, args.rounds, init_idx=start)
                    else:
                        summary, _ = run_continuous(land, payload["name"], name, arm, seed,
                                                     args.rounds, init_X=start, init_kind=design)
                    if abs(summary[0]["best_so_far"] - expected) > 1e-9:
                        raise ValueError(f"Starting design mismatch for {ds}/{name}/{seed}")
                    rows.extend(dict(row, init=design) for row in summary)
                pd.DataFrame(rows).to_csv(output / f"{ds}_{design}_{seed}.csv", index=False)


if __name__ == "__main__":
    main()
