"""Reconstruct the numerical SI tables from the paper records."""
import pandas as pd

from benchmarks.bench_main import PAPER_ARMS, SEEDS, SEEDS_10, SEEDS_20
from reproduction.figures import convergence, gp, scaling
from reproduction.paths import DATA, ROOT, TABLES
from reproduction.statistics import write_contrasts

KEY = ["dataset", "algorithm", "seed"]


def endpoints(rows, seeds):
    """One value per run. Round -1 is initialization. A miss counts as 21."""
    if rows.duplicated(KEY + ["round"]).any() or rows["pct_opt"].isna().any():
        raise ValueError("Duplicate rounds or missing objective values")
    by_run = rows.groupby(KEY)
    if set(rows["round"]) != set(range(-1, 20)) or not by_run.size().eq(21).all():
        raise ValueError("A complete run needs initialization and twenty rounds")
    if not rows.groupby(KEY[:2])["seed"].agg(set).map(lambda x: x == set(seeds)).all():
        raise ValueError("Unexpected seed set")
    final = rows.loc[rows["round"] == 19].set_index(KEY)["pct_opt"].rename("final_pct")
    hits = rows.loc[rows["pct_opt"] >= 95].groupby(KEY)["round"].min() + 1
    return pd.concat([final, hits.rename("t95")], axis=1).assign(
        reached=lambda df: df.t95.notna()).fillna({"t95": 21}).reset_index()


def summarize(units):
    return units.groupby(KEY[:2], as_index=False).agg(
        mean_t95=("t95", "mean"), successes=("reached", "sum"),
        seeds=("seed", "size"), mean_final_pct=("final_pct", "mean"))


def save(data, name):
    data.to_csv(TABLES / (name + ".csv"), index=False)


def tables(main_summary=None):
    TABLES.mkdir(exist_ok=True, parents=True)
    raw = pd.read_csv(main_summary) if main_summary else convergence.load()
    units = endpoints(raw, SEEDS_20)
    if len(units) != 660 or set(units.algorithm) != set(PAPER_ARMS):
        raise ValueError("Expected eleven methods on three datasets and twenty seeds")
    save(summarize(units), "S14_main")
    write_contrasts(units)
    write_contrasts(units, archived_method=True)
    for prefix, dataset in zip(("S7_BH", "S8_QM9", "S9_Arrhenius"),
                               ("Buchwald-Hartwig", "QM9-gap", "Arrhenius-2D-cont")):
        subset = raw[(raw.dataset == dataset) & raw["round"].isin([-1, 0, 1, 2, 3, 4, 9, 14, 19])]
        wide = subset.pivot_table(index="algorithm", columns="round", values="pct_opt")
        wide.columns = wide.columns + 1
        save(wide.reset_index(), prefix)
    save(summarize(endpoints(gp.load_main(), SEEDS_20)), "S1_GP")
    ratios = pd.read_csv(DATA / "scaling/step13_ratio_scaling_summary.csv.gz")
    save(summarize(endpoints(ratios, SEEDS_10)), "S3_ratio_scaling")
    saved = pd.read_csv(ROOT / "benchmarks/results/bench_complexity_summary.csv")
    saved["algorithm"] = saved.algorithm.replace({"GA": "UCB tournament"})
    save(summarize(endpoints(saved, SEEDS)), "S4_complexity")
    save(scaling.load_table(), "S15_QM9_scaling")

    from reproduction import supplementary
    raw_init = pd.read_csv(DATA / "initialization/rounds.csv.gz", low_memory=False)
    init = pd.concat([endpoints(rows, SEEDS_20).assign(init=design)
                      for design, rows in raw_init.groupby("init")], ignore_index=True)
    init_contrasts, init_cells = supplementary.initialization(init)
    save(init_contrasts, "S5_initialization_contrasts")
    save(init_cells, "S12_initialization")
    penalty = endpoints(pd.read_csv(DATA / "local_penalty/rounds.csv.gz"), SEEDS_20)
    save(supplementary.penalisation(penalty, init), "S11_local_penalisation")
    distances = pd.read_csv(DATA / "dimensions/radius_by_dimension.csv.gz")
    limits = distances[distances.arm != "Random"].groupby("ndim").min_batch_distance.agg(["min", "max"])
    spreads = distances.pivot(index="ndim", columns="arm", values="key_spread")
    limits["spread_vs_random"] = spreads["4:2:2"] / spreads["Random"]
    save(limits.reset_index(), "S13_batch_distances")
