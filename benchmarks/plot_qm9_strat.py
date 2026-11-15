"""Verification plots for the quantile-stratified QM9-gap subsets.

Builds two figures:

  1. benchmarks/plots/qm9_strat_grid.jpg
       4 x 5 grid (one panel per n in 5k..100k). Each panel:
       - histogram of the subset's y-values, coloured by bin value;
       - red dashed line at the full-pool top-95 % threshold;
       - faded outline of the full-pool histogram (rescaled to the subset's
         y-axis) for direct shape comparison;
       - top-95 % count and percentage annotation.

  2. benchmarks/plots/qm9_strat_summary.jpg
       Three sanity-check panels:
       - top-95 % percentage vs n (log-log; the 1/n reference);
       - empirical CDF of each subset, all overlaid against the full pool;
       - KS-statistic (max |CDF_subset - CDF_full|) vs n.
"""

import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Arial"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "qm9_strat")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CMAP = "magma"


def load_full() -> tuple[np.ndarray, np.ndarray, dict]:
    d = np.load(os.path.join(DATA_DIR, "qm9_full_pool.npz"), allow_pickle=False)
    y_full = d["y"]
    y_min, y_max = float(y_full.min()), float(y_full.max())
    thresh = y_min + 0.95 * (y_max - y_min)
    n_top_full = int((y_full >= thresh).sum())
    return d["X"], y_full, dict(
        y_min=y_min, y_max=y_max, thresh=thresh,
        n_top=n_top_full, n=len(y_full),
    )


def load_subsets() -> list[dict]:
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "qm9_strat_n*.npz")))
    out = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        out.append({
            "path": p,
            "n": int(os.path.basename(p).split("_n")[-1].split(".")[0]),
            "y": d["y"], "X": d["X"],
            "thresh_full": float(d["thresh_full"]),
            "n_top_full": int(d["n_top_full"]),
        })
    return out


def fig_grid(subsets, full_y, full_stats, fname):
    n_subsets = len(subsets)
    cols = 5
    rows = (n_subsets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), sharex=True)
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap(CMAP)

    # Common histogram edges so the faded full-pool outline aligns 1:1.
    edges = np.linspace(full_stats["y_min"], full_stats["y_max"], 51)
    full_counts, _ = np.histogram(full_y, bins=edges)

    for i, sub in enumerate(subsets):
        ax = axes[i // cols, i % cols]
        y = sub["y"]
        counts, _, patches = ax.hist(y, bins=edges, edgecolor="black",
                                      linewidth=0.3)
        centers = (edges[:-1] + edges[1:]) / 2
        rng_y = full_stats["y_max"] - full_stats["y_min"]
        for c_val, p in zip(centers, patches):
            p.set_facecolor(cmap((c_val - full_stats["y_min"]) / rng_y))

        # Full-pool outline, rescaled to subset's y-axis for shape comparison.
        scale = (counts.max() / full_counts.max()) if full_counts.max() else 1.0
        ax.step(edges[:-1], full_counts * scale, where="post",
                color="0.35", linewidth=0.9, linestyle="--",
                alpha=0.85, label="full pool (rescaled)")

        ax.axvline(sub["thresh_full"], color="red", linestyle="--", lw=1.4,
                   label="top-95 % threshold")
        ax.set_xlim(full_stats["y_min"] - 0.01,
                     full_stats["y_max"] + 0.01)
        ax.set_title(f"n = {sub['n']:,}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
        if i % cols == 0:
            ax.set_ylabel("Count", fontsize=10)
        if i // cols == rows - 1:
            ax.set_xlabel("gap (eV)", fontsize=10)

        n_top = int((y >= sub["thresh_full"]).sum())
        pct = 100 * n_top / len(y)
        ax.text(0.97, 0.97,
                f"top-95 %: {n_top} ({pct:.4f} %)\n"
                f"y range: [{y.min():.3f}, {y.max():.3f}]",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="0.7",
                          alpha=0.9, pad=3))

    # Hide any extra empty axes.
    for j in range(n_subsets, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(
        f"QM9-gap stratified subsets vs full pool "
        f"(N = {full_stats['n']:,}, "
        f"top-95 % threshold = {full_stats['thresh']:.4f}, "
        f"full pool top-95 % count = {full_stats['n_top']})",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)
    print(f"Saved: {out}")


def fig_summary(subsets, full_y, full_stats, fname):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ns = np.array([s["n"] for s in subsets])
    cmap_curve = plt.get_cmap("viridis")
    colors = [cmap_curve(i / max(1, len(subsets) - 1)) for i in range(len(subsets))]

    # Panel A: top-95% percentage vs n (with 1/n reference and full pool rate).
    ax = axes[0]
    pcts = np.array([
        100 * int((s["y"] >= s["thresh_full"]).sum()) / s["n"] for s in subsets
    ])
    ax.loglog(ns, pcts, "o-", color="#2166ac",
              label="stratified subsets", markersize=6)
    ax.axhline(100 * full_stats["n_top"] / full_stats["n"],
               linestyle=":", color="gray",
               label=f"full pool ({full_stats['n_top']}/{full_stats['n']:,})")
    ax.set_xlabel("n", fontsize=11)
    ax.set_ylabel("top-95 % share (%)", fontsize=11)
    ax.set_title("Top-95 % percentage vs n", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    # Panel B: empirical CDFs (subsets) overlaid on the full pool CDF.
    ax = axes[1]
    y_sorted_full = np.sort(full_y)
    cdf_full = np.arange(1, len(y_sorted_full) + 1) / len(y_sorted_full)
    ax.plot(y_sorted_full, cdf_full, color="black", lw=2.4,
            label=f"full pool (N={full_stats['n']:,})", zorder=10)
    for sub, c in zip(subsets, colors):
        y_sorted = np.sort(sub["y"])
        cdf = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        ax.plot(y_sorted, cdf, color=c, lw=0.9, alpha=0.85)
    ax.set_xlabel("gap (eV)", fontsize=11)
    ax.set_ylabel("empirical CDF", fontsize=11)
    ax.set_title("CDFs: subsets vs full pool", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # Panel C: KS statistic (max |CDF_sub - CDF_full|) vs n.
    ax = axes[2]
    # Evaluate both CDFs on a common grid.
    grid = np.linspace(full_stats["y_min"], full_stats["y_max"], 2000)
    cdf_full_grid = np.searchsorted(y_sorted_full, grid, side="right") / len(y_sorted_full)
    ks = []
    for s in subsets:
        y_s = np.sort(s["y"])
        cdf_s = np.searchsorted(y_s, grid, side="right") / len(y_s)
        ks.append(float(np.max(np.abs(cdf_s - cdf_full_grid))))
    ax.plot(ns, ks, "o-", color="#b2182b", markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("n", fontsize=11)
    ax.set_ylabel("KS statistic", fontsize=11)
    ax.set_title("Kolmogorov–Smirnov distance to full pool",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.97, f"max = {max(ks):.4f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9))

    fig.suptitle("Sanity check: stratified subsets vs full QM9-gap pool",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    _, full_y, full_stats = load_full()
    print(f"Full pool: N={full_stats['n']:,}, "
          f"y range=[{full_stats['y_min']:.4f}, {full_stats['y_max']:.4f}], "
          f"thresh={full_stats['thresh']:.4f}, "
          f"top-95 %={full_stats['n_top']}")
    subsets = load_subsets()
    print(f"Loaded {len(subsets)} subsets")
    fig_grid(subsets, full_y, full_stats, "qm9_strat_grid.jpg")
    fig_summary(subsets, full_y, full_stats, "qm9_strat_summary.jpg")


if __name__ == "__main__":
    main()
