"""Regenerate benchmarks/plots/bench_fig2_panelH.jpg — standalone QM9-gap
convergence panel (mean +/- std over 5 seeds), legend in lower right.

Reconstructed to match the original raster (the original ad-hoc generator was
never saved):
  * canvas exactly 3000 x 1500 px  (dpi=300, figsize=(10, 5),
    NO bbox_inches='tight' so the pixel size is exact)
  * margins kept as fractions measured from the original 2371x1168 raster
    (left 206px, right 2342px, top 39px, bottom 1012px)
  * all fonts 17 pt
  * legend labels:  UCB-BO -> BO-UCB,  BO(EI) -> BO-EI

Output path can be overridden with env PANELH_OUT (used for preview before
overwriting the real figure).

Run:
    python -m benchmarks.plot_fig2_panelH
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"   # all text in Arial
import matplotlib.pyplot as plt

import benchmarks.plot_benchmark as pb

RESULTS = pb.RESULTS_DIR
OUT = os.environ.get(
    "PANELH_OUT", os.path.join("benchmarks", "plots", "bench_fig2_panelH.jpg")
)

# --- canvas geometry ---
DPI = 300
FIGSIZE = (10, 5)                                   # -> 3000 x 1500 px at dpi=300
# margins kept as fractions measured from the original 2371x1168 raster
MARGINS = dict(left=206 / 2371, right=2342 / 2371,
               top=1 - 39 / 1168, bottom=1 - 1012 / 1168)

# --- font sizes: all 17 pt per request ---
FS_AXLABEL = 17
FS_TICK = 17
FS_LEGEND = 17

DISPLAY = {
    "GLOSS": "GLOSS",
    "UCB-BO": "BO-UCB",
    "BO(EI)": "BO-EI",
    "GA": "GA",
    "Random": "Random sampling",
}


def main():
    df = pd.read_csv(os.path.join(RESULTS, "bench_main_summary.csv"))
    q = df[df["dataset"] == "QM9-gap"]
    rounds = np.sort(q["round"].unique())          # -1 (initial) .. 19
    x = rounds + 1                                  # 0 (initial) .. 20

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(**MARGINS)

    for algo in pb.ALGO_ORDER:
        sub = q[q["algorithm"] == algo]
        means = np.array([sub[sub["round"] == r]["pct_opt"].mean() for r in rounds])
        stds = np.array([sub[sub["round"] == r]["pct_opt"].std(ddof=0) for r in rounds])
        c = pb.ALGO_COLORS[algo]
        ls = pb.ALGO_LINES[algo]
        lw = 2.8 if algo == "GLOSS" else 2.2
        ax.plot(x, means, color=c, linestyle=ls, linewidth=lw, label=DISPLAY[algo])
        ax.fill_between(x, means - stds, means + stds, color=c, alpha=0.15, linewidth=0)

    ax.axhline(95, color="0.4", linestyle="--", linewidth=1.6, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(["initial"] + [str(i) for i in range(1, len(x))])
    ax.set_xlabel("Round", fontsize=FS_AXLABEL)
    ax.set_ylabel("% of optimum", fontsize=FS_AXLABEL)
    ax.set_xlim(x.min(), x.max())
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(loc="lower right", fontsize=FS_LEGEND, frameon=True)

    # NO bbox_inches='tight' -> exact 3000 x 1500 px
    fig.savefig(OUT, dpi=DPI, format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
