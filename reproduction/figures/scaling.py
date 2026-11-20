"""Generate scaling from the paper records."""

import glob
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from reproduction.paths import DATA, FIGURES

OUT = FIGURES

CORE = DATA / "scaling" / "bench_scaling_batch_aware_rerun1021.csv.gz"
EXTRA_DIR = DATA / "scaling"

ARMS = [
    ("GLOSS(4:2:2)", "GLOSS 4:2:2", "#2166ac", "o"),
    ("qPO", "qPO", "#e08214", "D"),
    ("Thompson", "Thompson", "#7b3294", "v"),
    ("BO-UCB", "BO-UCB", "#b2182b", "s"),
    ("BO(EI)", "BO-EI", "#d6604d", "^"),
    ("GA", "GA", "#4dac26", "D"),
    ("Random", "Random sampling", "#969696", "v"),
]
# Both scaling files name the greedy baseline "UCB-BO"; the figure calls it
# BO-UCB, the name the manuscript uses.
RENAME = {"UCB-BO": "BO-UCB"}
CENSOR = 21


def t95_by_size(df):
    """Mean, standard deviation and success count of t95 per arm and pool size."""
    out = []
    for (arm, size), g in df.groupby(["algorithm", "dataset"]):
        hit = g[pd.to_numeric(g["pct_opt"], errors="coerce") >= 95].copy()
        hit["round"] = pd.to_numeric(hit["round"], errors="coerce")
        first = hit.groupby("seed")["round"].min()
        seeds = g["seed"].unique()
        # A seed that never reaches the target counts as the budget plus one,
        # which is the convention of the paper.
        t = (first + 1).reindex(seeds).fillna(CENSOR)
        out.append({"arm": RENAME.get(arm, arm),
                    "size": int(str(size).split("-")[-1].rstrip("k")),
                    "t95": float(t.mean()), "sd": float(t.std(ddof=1)),
                    "seeds": len(seeds), "success": int((t < CENSOR).sum())})
    return pd.DataFrame(out)


def load_scaling_runs():
    core = pd.read_csv(CORE)
    shards = sorted(glob.glob(os.path.join(EXTRA_DIR, "step13_scaling_extra*.csv.gz")))
    if not shards:
        raise SystemExit("no scaling-extra files found")
    extra = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    keys = ["dataset", "algorithm", "seed", "round"]
    overlap = extra[extra.duplicated(subset=keys, keep=False)]
    if (overlap.groupby(keys)["pct_opt"].nunique(dropna=False) > 1).any():
        raise ValueError("Conflicting outcomes in overlapping scaling shards")
    extra = extra.drop_duplicates(subset=keys)
    data = pd.concat([core, extra], ignore_index=True)
    keep = [a for a, *_ in ARMS] + list(RENAME)
    return data[data["algorithm"].isin(keep)]


def load_table():
    table = t95_by_size(load_scaling_runs())
    coverage = table.groupby("arm")["size"].nunique()
    if coverage.min() != coverage.max():
        raise SystemExit(f"arms cover different numbers of sizes: {coverage.to_dict()}")
    if table["seeds"].min() != table["seeds"].max():
        raise SystemExit("arms carry different seed counts")
    return table


def bar_panel(ax, sub, label, color):
    x = np.arange(len(sub))
    failed = sub["success"].to_numpy() < sub["seeds"].to_numpy() / 2
    for i, row in enumerate(sub.itertuples()):
        weak = row.success < row.seeds / 2
        ax.bar(i, row.t95, width=0.82, color="white" if weak else color,
               hatch="////" if weak else None, edgecolor=color, linewidth=0.5)
    if not failed.any():
        ax.text(0.5, 0.93, "", transform=ax.transAxes)
    ax.errorbar(x, sub["t95"], yerr=sub["sd"], fmt="none", ecolor="#333333",
                elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=4)
    ax.axhline(CENSOR, color="black", lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.set_ylim(0, 23.4)
    ax.set_xlim(-0.7, len(sub) - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sub["size"]], fontsize=5.6, rotation=90)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_title(label, fontsize=8.4, color=color, pad=13, fontweight="bold")
    ax.text(0.5, 1.015, f"{int(sub['success'].sum())} of "
            f"{int(sub['seeds'].sum())} runs reached the target",
            transform=ax.transAxes, fontsize=6.2, ha="center", va="bottom",
            color="#333333")



def main():
    table = load_table()
    fig = plt.figure(figsize=(11.2, 5.6))
    grid = fig.add_gridspec(2, 4, hspace=0.52, wspace=0.30, left=0.055,
                            right=0.985, top=0.915, bottom=0.115)
    for index, (arm, label, color, _) in enumerate(ARMS):
        ax = fig.add_subplot(grid[index // 4, index % 4])
        sub = table[table["arm"] == arm].sort_values("size")
        bar_panel(ax, sub, label, color)
        if index % 4 == 0:
            ax.set_ylabel("mean $t_{95}$ (rounds)", fontsize=7.4)
    legend_ax = fig.add_subplot(grid[1, 3])
    legend_ax.axis("off")
    handles = [Patch(facecolor="#888888", label="mean over ten seeds"),
               Patch(facecolor="white", edgecolor="#888888", hatch="////",
                     label="hatched: fewer than half\nof the seeds reached target"),
               Patch(facecolor="none", edgecolor="#333333", label="error bar: 1 s.d.")]
    legend_ax.legend(handles=handles, fontsize=7, loc="center", frameon=False,
                     handlelength=1.8, handletextpad=0.6)
    legend_ax.text(0.5, 0.12, "x axis: pool size in thousands\nof candidates",
                   fontsize=7, ha="center", color="#333333")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig3_scaling_bar.jpg")
    fig.savefig(path, dpi=300, format="jpeg", pil_kwargs={"quality": 95})
    print("wrote", path)
    print(table.groupby("arm")["success"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
