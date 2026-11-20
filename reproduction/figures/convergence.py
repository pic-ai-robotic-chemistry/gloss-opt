"""Generate convergence from the paper records."""

import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproduction.paths import DATA, FIGURES

OUT = FIGURES

# Order runs from the primary arm outward: the five GLOSS variants (three ratio
# arms, then the two diversity ablations), then the batch-aware arms, then the
# pointwise baselines, then the reference arms.
ARMS = [
    ("GLOSS(2:3:3)", "GLOSS 2:3:3", "#08306b", "-", "P"),
    ("GLOSS(4:2:2)", "GLOSS 4:2:2", "#2171b5", "-", "o"),
    ("GLOSS(6:1:1)", "GLOSS 6:1:1", "#6baed6", "-", "s"),
    ("GLOSS(8:0:0)+div", "GLOSS 8:0:0+div", "#4292c6", "--", "^"),
    ("GLOSS(8:0:0)", "GLOSS 8:0:0", "#9ecae1", "-.", "v"),
    ("qPO", "qPO", "#e08214", "-", "D"),
    ("Thompson", "Thompson", "#7b3294", "--", "v"),
    ("UCB-BO", "BO-UCB", "#b2182b", "-", "s"),
    ("BO(EI)", "BO-EI", "#d6604d", "--", "^"),
    ("GA", "GA", "#4dac26", ":", "D"),
    ("Random", "Random sampling", "#969696", ":", "v"),
]

DATASETS = [
    ("Buchwald-Hartwig", "Buchwald-Hartwig (experimental)", "G"),
    ("QM9-gap", "QM9-gap (100k, DFT)", "H"),
    ("Arrhenius-2D-cont", "Arrhenius-2D (continuous)", "I"),
]

def load():
    frames = [pd.read_csv(path) for path in sorted((DATA / "main").glob("*_summary.csv.gz"))]
    data = pd.concat(frames, ignore_index=True)
    missing = set(a for a, *_ in ARMS) - set(data["algorithm"].unique())
    if missing:
        raise SystemExit(f"arms missing from the rerun summaries: {sorted(missing)}")
    return data


def panel(ax, data, dataset, letter, title, show_ylabel):
    sub = data[data["dataset"] == dataset]
    rounds = sorted(sub["round"].unique())
    x = [r + 1 for r in rounds]  # init is -1, the first recommendation is 1
    for key, label, color, style, marker in ARMS:
        asub = sub[sub["algorithm"] == key]
        mean = np.array([asub[asub["round"] == r]["pct_opt"].mean() for r in rounds])
        std = np.array([asub[asub["round"] == r]["pct_opt"].std() for r in rounds])
        ax.plot(x, mean, color=color, ls=style, lw=1.7, marker=marker, ms=3.2,
                markevery=2, label=label, zorder=3)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, lw=0)
    ax.axhline(95, color="black", ls="--", lw=1.4, alpha=0.55, zorder=1)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_title(title, fontsize=11.5, pad=6, fontweight="bold")
    ax.set_xlabel("Round", fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel("% of optimum", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    ax.text(0.015, 0.965, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left", zorder=5)


def main():
    data = load()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.05))
    for ax, (dataset, title, letter) in zip(axes, DATASETS):
        panel(ax, data, dataset, letter, title, show_ylabel=(letter == "G"))
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10.0,
               frameon=False, handlelength=2.4, columnspacing=2.0,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(wspace=0.17, left=0.062, right=0.995, top=0.93, bottom=0.245)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig2_panels_G-I_v3.jpg")
    fig.savefig(path, dpi=300, format="jpeg", pil_kwargs={"quality": 95})
    print("wrote", path)


if __name__ == "__main__":
    main()
