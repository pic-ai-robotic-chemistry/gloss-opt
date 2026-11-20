"""Generate main comparison from the paper records."""
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from benchmarks import plot_benchmark as pb
from reproduction.paths import FIGURES

STRIP = FIGURES / "fig2_panels_G-I_v3.jpg"
OUT = FIGURES / "fig2_main.jpg"

# Same three columns as `plot_benchmark.fig1_main`, rows 1 and 2 only.
COLS = [
    {"key": "BH", "title": "Buchwald-Hartwig (3955, Exp.)",
     "xlabel": "Yield (%)", "structure": "tsne_precomputed",
     "axes_labels": ("t-SNE 1", "t-SNE 2"), "cmap": "viridis",
     "cbar_title": "Yield (%)", "dist_mode": "hist"},
    {"key": "QM9", "title": "QM9-gap (100k, DFT)",
     "xlabel": "HOMO-LUMO gap (Hartree)", "structure": "pca_global",
     "axes_labels": ("PC1", "PC2"), "cmap": "magma",
     "cbar_title": "Gap (Hartree)", "dist_mode": "hist"},
    {"key": "AR", "title": "Arrhenius-2D (Continuous data)",
     "xlabel": r"$f(x)$", "structure": "function_heatmap",
     "axes_labels": (r"$x_1$", r"$x_2$"), "cmap": "Spectral_r",
     "cbar_title": r"$f(x)$", "dist_mode": "kde"},
]


def render_top_rows(dpi=300):
    """Rows A-F at the same size as the original composite."""
    raw = pb._load_main_fig_data()
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.05))
    letters = "ABCDEF"
    for i, col in enumerate(COLS):
        pb._row1_structure(axes[0, i], raw[col["key"]], col["structure"],
                           letters[i], col["title"], col["axes_labels"],
                           col["cmap"], col["cbar_title"], fs=12)
    row2_ylabels = ["Count", "Count", "Density"]
    for i, col in enumerate(COLS):
        pb._row2_distribution(
            axes[1, i], raw[col["key"]]["y"], letters[3 + i], col["xlabel"],
            col["cmap"], show_ylabel=(row2_ylabels[i] is not None),
            mode=col["dist_mode"], fs=12, ylabel=row2_ylabels[i],
            annot_pct=(pb._arrhenius_area_pct() if col["key"] == "AR" else None))
    fig.tight_layout(w_pad=2.0, h_pad=2.5)
    path = FIGURES / "_fig2_top_rows.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    top_path = render_top_rows()
    top = Image.open(top_path).convert("RGB")
    strip = Image.open(STRIP).convert("RGB")

    width = top.size[0]
    strip_h = int(round(strip.size[1] * width / strip.size[0]))
    strip = strip.resize((width, strip_h), Image.LANCZOS)

    out = Image.new("RGB", (width, top.size[1] + strip_h), "white")
    out.paste(top, (0, 0))
    out.paste(strip, (0, top.size[1]))
    out.save(OUT, quality=92)
    print("wrote {} ({}x{})".format(OUT, out.size[0], out.size[1]))

    os.remove(top_path)


if __name__ == "__main__":
    main()
