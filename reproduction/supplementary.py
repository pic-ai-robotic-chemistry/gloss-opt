"""Paired inference for the initialization and local-penalisation studies."""
import numpy as np
import pandas as pd

from benchmarks.bench_main import RERUN_ARMS
from reproduction.paths import DATA

DATASETS = {"qm9": "QM9-gap", "bh": "Buchwald-Hartwig", "arr": "Arrhenius-2D-cont"}
DESIGNS = ("bottom20", "uniform", "lhs", "maxmin")
BASE = "GLOSS(4:2:2)"
B = 10_000


def adjusted_pvalues(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def initialization(data):
    rng = np.random.default_rng(20261026)
    contrasts, cells = [], []
    for slug, dataset in DATASETS.items():
        for design in DESIGNS:
            subset = data[(data.dataset == dataset) & (data.init == design)]
            wide = subset.pivot(index="seed", columns="algorithm", values="t95")
            means = wide.mean()
            # Resolve equal means in the order of the paper's experimental arms.
            fastest = means.reindex(RERUN_ARMS).idxmin()
            cells.append((dataset, design, means[BASE], int((wide[BASE] < 21).sum()), fastest, means[fastest]))
            conditional = wide.loc[wide[BASE] > 0, ["GLOSS(8:0:0)+div", BASE]].to_numpy()
            n = len(conditional)
            def draw():
                pairs = conditional[rng.integers(n, size=(B, n))]
                return pairs[:, :, 0].mean(axis=1) - pairs[:, :, 1].mean(axis=1)
            lo, hi = np.quantile(draw(), [0.025, 0.975])
            # The recorded analysis drew a BCa sensitivity sample before the one-sided test.
            draw()
            p = (1 + (draw() <= 0).sum()) / (B + 1)
            contrasts.append((slug, design, n, (conditional[:, 0] - conditional[:, 1]).mean(), lo, hi, p))
    summary = pd.DataFrame(contrasts, columns=["dataset_slug", "init", "n_seeds", "point", "lo", "hi", "p"])
    summary["p_holm"] = adjusted_pvalues(summary.p)
    return summary, pd.DataFrame(cells, columns=["dataset", "init", "t95_mean_gloss", "reach_k", "algorithm", "t95_mean_fastest"])


def penalisation(units, init):
    rng = np.random.default_rng(20261025)
    diagnostics = pd.read_csv(DATA / "local_penalty/diagnostics.csv.gz")
    rows = []
    for arm, reference in [("LocalPenalty(EI)", "BO(EI)"), ("LocalPenalty(UCB)", "UCB-BO")]:
        for slug, dataset in DATASETS.items():
            group = units[(units.algorithm == arm) & (units.dataset == dataset)].set_index("seed")
            # The original unit files were processed in lexical filename order.
            group = group.loc[sorted(group.index, key=lambda s: f"s{s}.csv")]
            control = init[(init.init == "bottom20") & (init.algorithm == reference) & (init.dataset == dataset)].set_index("seed").loc[group.index]
            diff = (control.t95 - group.t95).to_numpy()
            def draw():
                return diff[rng.integers(len(diff), size=(B, len(diff)))].mean(axis=1)
            lo, hi = np.percentile(draw(), [2.5, 97.5])
            lower = (1 + (draw() <= 0).sum()) / (B + 1)
            diag = diagnostics[(diagnostics.algorithm == arm) & (diagnostics.dataset == dataset)]
            rows.append((slug, arm, reference, len(diff), group.t95.mean(), control.t95.mean(),
                         diff.mean(), lo, hi, 2 * min(lower, 1-lower), diag.lp_occ_k8.mean()))
    out = pd.DataFrame(rows, columns=["dataset", "arm", "reference", "n_seeds", "arm_mean_t95", "reference_mean_t95", "D", "lo", "hi", "p_two_sided", "occ_k8_mean"])
    out["p_holm"] = adjusted_pvalues(out.p_two_sided)
    out["not_activated"] = out.occ_k8_mean < 0.01
    return out
