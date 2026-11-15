"""Analysis figure: sampling space characteristics for Alchemy-gap, QM9-alpha, Medium-2D.

Layout (2 rows × 3 columns, square subplots):
  Row 1 — A: Alchemy-gap PCA scatter
           B: QM9-alpha PCA scatter
           C: Medium-2D 2D heatmap
  Row 2 — D: Alchemy-gap y-distribution histogram
           E: QM9-alpha y-distribution histogram
           F: Medium-2D y-distribution histogram

Usage:
    python -m benchmarks.analysis.plot_distribution_analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

matplotlib.rcParams["font.family"] = "Arial"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR  = os.path.join(REPO_ROOT, "benchmarks", "data")
OUT_DIR   = os.path.dirname(__file__)
OUT_PATH  = os.path.join(OUT_DIR, "distribution_analysis.png")

sys.path.insert(0, REPO_ROOT)


# ─── Row 1, Panel A/B: PCA scatter ───────────────────────────────────────────
def _pca_scatter(ax, X_norm, y, panel_label, title, y_label, seed=42):
    """PCA scatter of descriptor space, points coloured by y value."""
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=2, random_state=seed)
    Z   = pca.fit_transform(X_norm)

    # Subsample to 4000 for display speed while keeping density representative
    n_show = min(4000, len(Z))
    idx    = rng.choice(len(Z), size=n_show, replace=False)

    norm = mcolors.Normalize(vmin=np.percentile(y, 2), vmax=np.percentile(y, 98))
    sc = ax.scatter(Z[idx, 0], Z[idx, 1],
                    c=y[idx], cmap="viridis", norm=norm,
                    alpha=0.5, s=9, rasterized=True, linewidths=0)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.03, aspect=20)
    cbar.set_label(y_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    var_pct = pca.explained_variance_ratio_[:2] * 100
    ax.set_xlabel(f"PC1  ({var_pct[0]:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC2  ({var_pct[1]:.1f}% var)", fontsize=9)
    ax.set_title(f"{panel_label}   {title}", fontsize=10, fontweight="bold", pad=5)
    ax.tick_params(labelsize=8)


# ─── Row 1, Panel C: 2D heatmap ──────────────────────────────────────────────
def _heatmap(ax, func, bounds, panel_label, title):
    """2D filled contour of the landscape with peak centres marked."""
    from benchmarks.virtual_2d import _M2_CENTERS, _M2_HEIGHTS

    res = 300
    xs  = np.linspace(bounds[0][0], bounds[0][1], res)
    ys  = np.linspace(bounds[1][0], bounds[1][1], res)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = func(np.column_stack([XX.ravel(), YY.ravel()])).reshape(res, res)

    cf = ax.contourf(XX, YY, ZZ, levels=40, cmap="YlOrRd")
    ax.contour(XX, YY, ZZ, levels=8, colors="white", alpha=0.25, linewidths=0.4)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.03, aspect=20)
    cbar.set_label("f(x1, x2)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    peak_labels = ["Peak 1\n(h=10)", "Peak 2\n(h=7)", "Peak 3\n(h=5)"]
    offsets     = [(0.04, 0.04), (-0.18, 0.04), (0.04, -0.10)]
    for c, lbl, off in zip(_M2_CENTERS, peak_labels, offsets):
        ax.plot(c[0], c[1], "w*", ms=12, zorder=6,
                markeredgecolor="#333333", markeredgewidth=0.6)
        ax.text(c[0] + off[0], c[1] + off[1], lbl,
                fontsize=7.5, color="white", fontweight="bold",
                bbox=dict(fc=(0, 0, 0, 0.30), ec="none", pad=1.5))

    ax.set_xlabel("x1", fontsize=9)
    ax.set_ylabel("x2", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{panel_label}   {title}", fontsize=10, fontweight="bold", pad=5)
    ax.tick_params(labelsize=8)


# ─── Row 2, Panels D/E/F: y-distribution histogram ───────────────────────────
def _histogram(ax, y, panel_label, title, x_label, bins=50):
    """Histogram of y values with 95%-of-max threshold marked."""
    y_opt   = y.max()
    thresh  = y_opt * 0.95
    n_total = len(y)
    n_top   = int((y >= thresh).sum())
    p8      = (1 - (1 - n_top / n_total) ** 8) * 100
    mean_max = y.mean() / y_opt

    counts, edges = np.histogram(y, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width   = edges[1] - edges[0]

    # Colour bars: grey below threshold, blue above
    colors = ["#5aade0" if c >= thresh else "#b0c4d8" for c in centers]
    ax.bar(centers, counts, width=width * 0.92,
           color=colors, edgecolor="none", alpha=0.9)

    # 95%-of-max threshold line
    ax.axvline(thresh, color="#cc3300", ls="--", lw=1.5,
               label=f"95% of max\n({thresh:.2f})")

    # Stats annotation
    stats_txt = (
        f"n = {n_total:,}\n"
        f"n$_{{top}}$ = {n_top}\n"
        f"P(8 rnd) = {p8:.2f}%\n"
        f"mean/max = {mean_max:.2f}"
    )
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.90,
                      ec="#bbbbbb", lw=0.7))

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(f"{panel_label}   {title}", fontsize=10, fontweight="bold", pad=5)
    ax.legend(fontsize=8, loc="upper left" if y[np.argmax(counts)] < thresh else "upper right",
              framealpha=0.85)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load datasets ──────────────────────────────────────────────────────────
    print("Loading Alchemy-gap...")
    df_alc = pd.read_csv(os.path.join(DATA_DIR, "alchemy_features.csv"))
    desc_cols = [
        'mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'rings', 'arom_rings',
        'n_hetero', 'fsp3', 'heavy_atoms', 'n_val_elec', 'bertz', 'chi0',
        'chi1', 'kappa1', 'kappa2', 'labute_asa', 'balaban_j', 'hall_kier_alpha',
    ]
    df_alc  = df_alc.sample(n=20000, random_state=42).reset_index(drop=True)
    X_alc   = df_alc[desc_cols].values.astype(float)
    y_alc   = df_alc['gap_eV'].values.astype(float)
    mask    = np.all(np.isfinite(X_alc), axis=1) & np.isfinite(y_alc)
    X_alc, y_alc = X_alc[mask], y_alc[mask]
    mn, mx  = X_alc.min(0), X_alc.max(0); r_ = mx - mn; r_[r_ == 0] = 1
    X_alc_norm = (X_alc - mn) / r_

    print("Loading QM9-alpha...")
    from benchmarks.datasets import load_qm9
    ds_alpha   = load_qm9(n_samples=20000, target="alpha")
    X_alpha    = ds_alpha["candidates"]
    y_alpha    = ds_alpha["y_true"]

    print("Loading Medium-2D grid values...")
    from benchmarks.virtual_2d import medium_2d, BOUNDS_2D
    rng_m2 = np.random.default_rng(0)
    X_m2   = rng_m2.uniform(0, 1, size=(50000, 2))
    y_m2   = medium_2d(X_m2)

    # ── Figure: 2 rows × 3 cols, square subplots ───────────────────────────────
    subplot_size = 4.2
    fig, axes = plt.subplots(
        2, 3,
        figsize=(subplot_size * 3 + 1.2, subplot_size * 2 + 0.8),
        gridspec_kw={"wspace": 0.42, "hspace": 0.45},
    )
    # Make each axes box square
    for ax in axes.flat:
        ax.set_box_aspect(1)

    # ── Row 1: scatter / heatmap ───────────────────────────────────────────────
    _pca_scatter(axes[0, 0], X_alc_norm, y_alc,
                 "A", "Alchemy-gap  (DFT, n=20k)",
                 "HOMO-LUMO gap (eV)")

    _pca_scatter(axes[0, 1], X_alpha, y_alpha,
                 "B", "QM9-alpha  (DFT, n=20k)",
                 "Polarizability (Bohr\u00b3)")

    _heatmap(axes[0, 2], medium_2d, BOUNDS_2D,
             "C", "Medium-2D  (continuous, 3 peaks)")

    # ── Row 2: histograms ──────────────────────────────────────────────────────
    _histogram(axes[1, 0], y_alc,
               "D", "Alchemy-gap  distribution",
               "HOMO-LUMO gap (eV)")

    _histogram(axes[1, 1], y_alpha,
               "E", "QM9-alpha  distribution",
               "Polarizability (Bohr\u00b3)")

    _histogram(axes[1, 2], y_m2,
               "F", "Medium-2D  f(x) distribution",
               "f(x1, x2)")

    # ── title ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Search Space Characteristics: Structural and Value Distribution Analysis",
        fontsize=11, fontweight="bold", y=1.01,
    )

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
