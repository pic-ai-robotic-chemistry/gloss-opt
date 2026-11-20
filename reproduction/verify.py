"""Check the rebuilt numerical outputs against the SI and independent analysis exports."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from reproduction.paths import ROOT, DATA, TABLES

REFERENCE = ROOT / "reproduction/reference"


def verify():
    printed = pd.read_csv(REFERENCE / "si_printed.csv", keep_default_na=False, dtype=str)
    for table, expected in printed.groupby("table"):
        result = pd.read_csv(TABLES / (table + ".csv"))
        if table.startswith("S15"):
            result = result.rename(columns={"arm": "algorithm", "size": "dataset"})
        if table.startswith("S4"):
            result["dataset"] = result.dataset.str.split("-").str[0]
        if "dataset" not in result:
            result["dataset"] = ""
        result["dataset"] = result.dataset.astype(str)
        indexed = result.set_index(["dataset", "algorithm"], verify_integrity=True)
        for row in expected.itertuples():
            got = indexed.loc[(row.dataset_or_size, row.algorithm), row.metric]
            # Endpoints displayed to one decimal can sit exactly halfway between tenths.
            tolerance = 0.05000001 if "." in row.printed else 1e-8
            if abs(got - float(row.printed)) > tolerance or not np.isfinite(got):
                raise ValueError(f"{table}, {row.algorithm}, {row.metric}: {got} vs {row.printed}")
        print(f"{table}: {len(expected)} printed values agree")
    for name, archive, keys, values in [
        ("S5_initialization_contrasts", "initialization/init1_gateA", ["dataset_slug", "init"], ["point", "lo", "hi", "p", "p_holm", "n_seeds"]),
        ("S11_local_penalisation", "local_penalty/family_step11", ["dataset", "arm"], ["D", "lo", "hi", "p_two_sided", "p_holm", "occ_k8_mean"]),
    ]:
        expected = pd.read_csv(DATA / (archive + ".csv.gz"))
        if "family" in expected:
            expected = expected[expected.family == "same acquisition"]
        actual = pd.read_csv(TABLES / (name + ".csv"))
        pd.testing.assert_frame_equal(actual.set_index(keys)[values].sort_index(),
                                      expected.set_index(keys)[values].sort_index(),
                                      check_dtype=False, atol=1e-10, rtol=0)
        print(f"{name}: {len(actual)} bootstrap contrasts agree with the archived analysis")
    archived = pd.read_csv(REFERENCE / "main_contrasts.csv")
    historical = pd.read_csv(TABLES / "main_contrasts_archived_method.csv")
    pd.testing.assert_frame_equal(historical, archived)
    corrected = pd.read_csv(TABLES / "main_contrasts.csv")
    # Counting ties changes some p values. For these records the confidence
    # intervals and both families of Holm decisions agree with the paper.
    columns = archived.columns.drop(["p_value", "p_floor"])
    pd.testing.assert_frame_equal(corrected[columns], archived[columns])
    print("24 archived contrasts reproduced; tie-inclusive intervals and Holm decisions verified")


def check_rerun(prefix, *, full=False):
    """Check every recorded point and round in a partial or complete main rerun."""
    key = ["dataset", "algorithm", "seed", "round"]
    from benchmarks.bench_main import N_ROUNDS, PAPER_ARMS, SEEDS_20
    from reproduction.statistics import MAIN

    for suffix, fields in [("summary", ["pct_opt", "best_so_far"]),
                           ("points", ["candidate_idx", "x0", "x1", "y", "strategy"])]:
        ids = key + (["point_idx"] if suffix == "points" else [])
        fresh = pd.read_csv(f"{prefix}_{suffix}.csv", usecols=ids + fields).set_index(ids, verify_integrity=True)
        if fresh.empty:
            raise ValueError(f"Empty {suffix} rerun")
        if suffix == "summary":
            seeds = fresh.index.unique("seed")
            rounds = fresh.index.unique("round")
            if full and (set(seeds) != set(SEEDS_20) or set(rounds) != set(range(-1, N_ROUNDS))):
                raise ValueError("The full paper benchmark requires all twenty seeds and rounds")
            if not set(seeds).issubset(SEEDS_20):
                raise ValueError("Unrecognised paper seed")
            if max(rounds) < 0 or max(rounds) >= N_ROUNDS or set(rounds) != set(range(-1, max(rounds) + 1)):
                raise ValueError("Expected consecutive rounds starting at initialization (-1)")
            cells = pd.MultiIndex.from_product(
                [list(MAIN.values()), PAPER_ARMS, seeds, rounds], names=key)
            pd.testing.assert_index_equal(fresh.index.sort_values(), cells.sort_values())
        # Continuous point archives have coordinates but no candidate index.
        files = sorted((DATA / "main").glob(f"*_{suffix}.csv.gz"))
        archived = pd.concat([pd.read_csv(p, low_memory=False) for p in files])
        archived = archived.set_index(ids, verify_integrity=True)
        expected = archived[archived.index.get_level_values("seed").isin(seeds) &
                            archived.index.get_level_values("round").isin(rounds)]
        if suffix == "points":
            # The archived point IDs define the actual size of each batch,
            # including short QM9 batches. Both files must cover the same rounds.
            pd.testing.assert_index_equal(
                expected.index.droplevel("point_idx").unique().sort_values(), cells.sort_values())
        pd.testing.assert_frame_equal(fresh[fields].sort_index(), expected[fields].sort_index(),
                                      check_dtype=False, atol=1e-8, rtol=0)
        print(f"{len(fresh)} {suffix} records agree")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-prefix", type=Path, help="Also check a completed paper run, omitting _summary.csv")
    parser.add_argument("--full", action="store_true",
                        help="Require all twenty seeds and twenty recommendation rounds")
    args = parser.parse_args()
    if args.full and not args.rerun_prefix:
        parser.error("--full requires --rerun-prefix")
    verify()
    if args.rerun_prefix:
        check_rerun(args.rerun_prefix, full=args.full)
