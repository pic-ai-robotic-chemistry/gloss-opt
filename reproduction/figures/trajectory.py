"""Generate trajectory from the paper records."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from reproduction.paths import DATA, FIGURES

OUT = FIGURES

POINTS = DATA / "main" / "rerun_arrhenius_step13_points.csv.gz"
SUMMARY = DATA / "main" / "rerun_arrhenius_step13_summary.csv.gz"
DATASET = "Arrhenius-2D-cont"

ARMS = [
    ("GLOSS(4:2:2)", "GLOSS 4:2:2", "o"),
    ("GLOSS(8:0:0)+div", "GLOSS 8:0:0+div", "s"),
    ("GLOSS(8:0:0)", "GLOSS 8:0:0", "^"),
    ("qPO", "qPO", "D"),
    ("Thompson", "Thompson", "v"),
    ("UCB-BO", "BO-UCB", "s"),
    ("BO(EI)", "BO-EI", "^"),
    ("GA", "GA", "D"),
    ("Random", "Random sampling", "v"),
]


def continuous_denominator():
    """The surface optimum and minimum the continuous percentages divide by."""
    from benchmarks import reference_values
    ref = reference_values.get("arrhenius", n_peaks=3, peak_width=0.08,
                               noise_sigma=0.0, seed=42)
    return float(ref["y_opt"]), float(ref["y_min"])


def pick_seed(summary):
    """The most typical seed of the twenty, by a rule that names no arm.

    For each seed, take the nine final percentages of the optimum and sum their
    absolute deviations from the per-arm median over the twenty seeds. The seed
    with the smallest total is the one whose outcome is closest to what the
    twenty seeds say on average for all nine arms at once. The rule never looks
    at which arm wins, so it cannot be tuned to favour GLOSS.

    """
    final = summary[summary["round"] == summary["round"].max()].pivot_table(
        index="seed", columns="algorithm", values="pct_opt")
    arms = [a for a, _, _ in ARMS if a in final.columns]
    if len(arms) != len(ARMS):
        raise SystemExit(f"arms missing from the summary: {sorted(set(a for a,_,_ in ARMS) - set(final.columns))}")
    deviation = (final[arms] - final[arms].median()).abs().sum(axis=1)
    return int(deviation.idxmin()), final


def panel(ax, df, arm, label, marker, Z, xs, y_opt, y_min):
    ax.pcolormesh(xs, xs, Z, cmap="Spectral_r", shading="auto", rasterized=True)
    init = df[df["round"] == -1]
    ib = init.loc[init["y"].idxmax()]
    rounds = df[df["round"] >= 0]
    best = [rounds[rounds["round"] == r].loc[rounds[rounds["round"] == r]["y"].idxmax()]
            for r in sorted(rounds["round"].unique())]
    best = pd.DataFrame(best).reset_index(drop=True)
    pct = (np.maximum.accumulate(np.concatenate([[ib["y"]], best["y"].to_numpy()]))[1:]
           - y_min) / (y_opt - y_min) * 100
    hit = np.where(pct >= 95.0)[0]
    first = int(hit[0]) if len(hit) else None

    lx = [ib["x0"]] + best["x0"].tolist()
    ly = [ib["x1"]] + best["x1"].tolist()
    ax.plot(lx, ly, color="black", lw=1.0, zorder=2, alpha=0.85)
    ax.scatter(ib["x0"], ib["x1"], marker="P", s=60, color="white",
               edgecolors="black", linewidths=0.7, zorder=5)
    ax.scatter(best["x0"], best["x1"], marker=marker, s=26, color="white",
               alpha=0.95, edgecolors="black", linewidths=0.4, zorder=3)
    if first is not None:
        ax.scatter(best.iloc[first]["x0"], best.iloc[first]["x1"], marker=marker,
                   s=30, color="yellow", edgecolors="black", linewidths=0.5, zorder=4)
    peak = float(pct.max())
    note = f"{peak:.1f}% optimum, 95% at round {int(best.iloc[first]['round']) + 1}" if first is not None else \
           f"{peak:.1f}% optimum, target not reached"
    ax.set_title(f"{label}\n{note}", fontsize=8.2, pad=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=6.5)
    return peak


def main():
    from benchmarks.virtual_functions import arrhenius_yield

    points = pd.read_csv(POINTS)
    points = points[points["dataset"] == DATASET]
    summary = pd.read_csv(SUMMARY)
    summary = summary[summary["dataset"] == DATASET]
    y_opt, y_min = continuous_denominator()
    seed, final = pick_seed(summary)
    print(f"typical seed {seed}; final percentages: "
          + ", ".join(f"{a} {final.loc[seed, a]:.1f}" for a, _, _ in ARMS))

    xs = np.linspace(0, 1, 300)
    XX, YY = np.meshgrid(xs, xs)
    Z = arrhenius_yield(np.column_stack([XX.ravel(), YY.ravel()]),
                        n_peaks=3, peak_width=0.08, noise_sigma=0.0,
                        seed=42).reshape(XX.shape)

    fig, axes = plt.subplots(3, 3, figsize=(10.4, 10.2))
    for letter, (ax, (arm, label, marker)) in zip("ABCDEFGHI", zip(axes.ravel(), ARMS)):
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        try:
            df = points[(points["algorithm"] == arm) & (points["seed"] == seed)]
            panel(ax, df, arm, label, marker, Z, xs, y_opt, y_min)
        except Exception as exc:  # a missing arm must not hide the other eight
            ax.text(0.5, 0.5, f"{label}\nmissing: {exc}", ha="center",
                    va="center", fontsize=7, transform=ax.transAxes)
    for ax in axes[-1]:
        ax.set_xlabel("$x_1$", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("$x_2$", fontsize=8)
    fig.subplots_adjust(wspace=0.18, hspace=0.34, left=0.055, right=0.92,
                        top=0.955, bottom=0.06)
    cax = fig.add_axes([0.935, 0.30, 0.014, 0.42])
    mappable = plt.cm.ScalarMappable(cmap="Spectral_r")
    mappable.set_array(Z)
    mappable.set_clim(float(Z.min()), float(Z.max()))
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label("$f(x)$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    handles = [Line2D([0], [0], marker="P", color="none", markerfacecolor="white",
                      markeredgecolor="black", markersize=7, label="initial"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                      markeredgecolor="black", markersize=6, label="per-round best"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="yellow",
                      markeredgecolor="black", markersize=6,
                      label="first round at 95%")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig4_trajectory.jpg")
    fig.savefig(path, dpi=300, format="jpeg", pil_kwargs={"quality": 95})
    print("wrote", path)


if __name__ == "__main__":
    main()
