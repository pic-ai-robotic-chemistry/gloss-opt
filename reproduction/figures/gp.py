"""Generate the three Gaussian-process sensitivity figures."""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproduction.paths import DATA, FIGURES

OUT = FIGURES
from . import convergence as _mod

#: Colour, line style and marker come from the main-text figure, so the same
#: arm keeps the same identity in the SI.
STYLE = {key: (color, style, marker)
         for key, _label, color, style, marker in _mod.ARMS}
LABEL = {key: label for key, label, *_ in _mod.ARMS}

MAIN_ARMS = ("GLOSS(2:3:3)", "GLOSS(4:2:2)", "GLOSS(6:1:1)",
             "GLOSS(8:0:0)+div", "GLOSS(8:0:0)",
             "UCB-BO", "BO(EI)", "GA", "Random")
SCALING_ARMS = ("GLOSS(4:2:2)", "UCB-BO", "BO(EI)", "GA", "Random")
TRAJ_ARMS = ("GLOSS(4:2:2)", "GLOSS(8:0:0)+div", "GLOSS(8:0:0)",
             "UCB-BO", "BO(EI)", "GA", "Random")

GP_DIR = DATA / "gp"
RERUN = DATA / "main"

DATASETS = [
    ("Buchwald-Hartwig", "gp_main_bh_summary.csv.gz",
     "Buchwald-Hartwig (experimental)", "A"),
    ("QM9-gap", "gp_main_qm9_summary.csv.gz", "QM9-gap (100k, DFT)", "B"),
    ("Arrhenius-2D-cont", "gp_main_arr_summary.csv.gz",
     "Arrhenius-2D (continuous)", "C"),
]

CENSOR = 21.0
TARGET = 95.0


def load_main():
    frames = [pd.read_csv(os.path.join(GP_DIR, name)) for _, name, _, _ in DATASETS]
    data = pd.concat(frames, ignore_index=True)
    missing = set(MAIN_ARMS) - set(data["algorithm"].unique())
    if missing:
        raise SystemExit(f"arms missing from the GP summaries: {sorted(missing)}")
    return data


def panel(ax, data, dataset, letter, title, show_ylabel):
    sub = data[data["dataset"] == dataset]
    rounds = sorted(sub["round"].unique())
    x = [r + 1 for r in rounds]
    for key in MAIN_ARMS:
        color, style, marker = STYLE[key]
        asub = sub[sub["algorithm"] == key]
        mean = np.array([asub[asub["round"] == r]["pct_opt"].mean() for r in rounds])
        std = np.array([asub[asub["round"] == r]["pct_opt"].std() for r in rounds])
        ax.plot(x, mean, color=color, ls=style, lw=1.7, marker=marker, ms=3.2,
                markevery=2, label=LABEL[key], zorder=3)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, lw=0)
    ax.axhline(TARGET, color="black", ls="--", lw=1.4, alpha=0.55, zorder=1)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_title(title, fontsize=11.5, pad=6, fontweight="bold")
    ax.set_xlabel("Round", fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel("% of optimum", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    ax.text(0.015, 0.965, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left", zorder=5)


def fig_main():
    data = load_main()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.05))
    for ax, (dataset, _name, title, letter) in zip(axes, DATASETS):
        panel(ax, data, dataset, letter, title, show_ylabel=(letter == "A"))
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10.0,
               frameon=False, handlelength=2.4, columnspacing=2.0,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(wspace=0.17, left=0.062, right=0.995, top=0.93,
                        bottom=0.245)
    path = os.path.join(OUT, "si_fig_gp_main.jpg")
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(path, dpi=300, format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print("wrote", path)


def t95_table(df):
    """Mean t95, sd and success count per arm and pool size, censored at 21."""
    rows = []
    for (arm, size), g in df.groupby(["algorithm", "dataset"]):
        hit = g[pd.to_numeric(g["pct_opt"], errors="coerce") >= TARGET]
        hit = hit.assign(round=pd.to_numeric(hit["round"], errors="coerce"))
        first = hit.groupby("seed")["round"].min()
        seeds = g["seed"].unique()
        t = (first + 1).reindex(seeds).fillna(CENSOR)
        rows.append({"arm": arm,
                     "size": int(str(size).split("-")[-1].rstrip("k")),
                     "t95": float(t.mean()), "sd": float(t.std(ddof=1)),
                     "seeds": len(seeds), "success": int((t < CENSOR).sum())})
    return pd.DataFrame(rows)


def fig_scaling():
    df = pd.read_csv(os.path.join(GP_DIR, "gp_scaling_summary.csv.gz"))
    table = t95_table(df)
    missing = set(SCALING_ARMS) - set(table["arm"].unique())
    if missing:
        raise SystemExit(f"arms missing from the GP scaling table: {sorted(missing)}")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for key in SCALING_ARMS:
        color, style, marker = STYLE[key]
        sub = table[table["arm"] == key].sort_values("size")
        ax.errorbar(sub["size"], sub["t95"], yerr=sub["sd"],
                    color=color, ls=style, lw=1.6, marker=marker, ms=4.0,
                    capsize=2.4, elinewidth=0.9, label=LABEL[key], zorder=3)
        lost = sub[sub["success"] < sub["seeds"]]
        if len(lost):
            ax.plot(lost["size"], lost["t95"], ls="none", marker="o",
                    ms=8.0, mfc="white", mec=color, mew=1.1, zorder=4)
    ax.axhline(CENSOR, color="black", ls=":", lw=1.0, alpha=0.5)
    ax.set_xlabel("Candidate pool size ($10^3$)", fontsize=10.5)
    ax.set_ylabel("$t_{95}$ (rounds)", fontsize=10.5)
    ax.set_ylim(0, CENSOR + 1.0)
    ax.tick_params(labelsize=9.5)
    ax.legend(fontsize=9.0, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), handlelength=2.2, columnspacing=1.4)
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.965, bottom=0.27)
    path = os.path.join(OUT, "si_fig_gp_scaling.jpg")
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(path, dpi=300, format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print("wrote", path)


def _traj(ax, data, dataset, arms, letter, title):
    sub = data[data["dataset"] == dataset]
    rounds = sorted(sub["round"].unique())
    x = [r + 1 for r in rounds]
    for key in arms:
        color, style, marker = STYLE[key]
        asub = sub[sub["algorithm"] == key]
        mean = np.array([asub[asub["round"] == r]["pct_opt"].mean() for r in rounds])
        ax.plot(x, mean, color=color, ls=style, lw=1.7, marker=marker, ms=3.0,
                markevery=2, label=LABEL[key], zorder=3)
    ax.axhline(TARGET, color="black", ls="--", lw=1.4, alpha=0.55, zorder=1)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_title(title, fontsize=11.5, pad=6, fontweight="bold")
    ax.set_xlabel("Round", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    ax.text(0.015, 0.965, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left", zorder=5)


def fig_trajectory():
    seed = 127
    rf = pd.read_csv(os.path.join(RERUN, "rerun_arrhenius_step13_summary.csv.gz"))
    gp = pd.read_csv(os.path.join(GP_DIR, "gp_main_arr_summary.csv.gz"))
    rf = rf[(rf["seed"] == seed) & (rf["algorithm"].isin(TRAJ_ARMS))]
    gp = gp[(gp["seed"] == seed) & (gp["algorithm"].isin(TRAJ_ARMS))]
    if rf.empty or gp.empty:
        raise SystemExit(f"seed {seed} missing from one of the two Arrhenius tables")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    _traj(axes[0], rf, "Arrhenius-2D-cont", TRAJ_ARMS, "A",
          "Random forest surrogate")
    _traj(axes[1], gp, "Arrhenius-2D-cont", TRAJ_ARMS, "B",
          "Gaussian-process surrogate")
    axes[0].set_ylabel("% of optimum", fontsize=10.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.4,
               frameon=False, handlelength=2.2, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(wspace=0.08, left=0.082, right=0.99, top=0.90,
                        bottom=0.30)
    path = os.path.join(OUT, "si_fig_gp_trajectory.jpg")
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(path, dpi=300, format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print("wrote", path)


def main():
    fig_main()
    fig_scaling()
    fig_trajectory()


if __name__ == "__main__":
    main()
