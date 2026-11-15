"""Three minimal icons for the GLOSS architecture diagram.

Each strategy gets one small figure (3 x 2 in), saved as its own JPG so the
icons can be placed independently in a slide / pptx framework diagram.

  strategies_global.jpg      — peak-on-landscape + star pick
  strategies_local.jpg       — multi-peak landscape + triangles on each peak
  strategies_unexplored.jpg  — clustered observations + diamond in the gap

No legends, no axis labels, no tick text — purely visual essence.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Arial"

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

C_GLOBAL = "#d6604d"
C_LOCAL = "#1b7837"
C_UNEXPL = "#2166ac"
C_CURVE = "#222222"
C_OBS = "white"

FIGSIZE = (3.75, 1.5)   # 2.5 : 1 aspect ratio
DPI = 240


def _clean(ax):
    """Hide every axis decoration; only the curve + markers + title remain."""
    ax.set_axis_off()


def _landscape(x: np.ndarray) -> np.ndarray:
    return (1.00 * np.exp(-((x - 2.3) / 0.65) ** 2)
            + 0.62 * np.exp(-((x - 5.2) / 0.55) ** 2)
            + 0.85 * np.exp(-((x - 7.7) / 0.55) ** 2)
            + 0.04)


def _save(fig, name):
    out = os.path.join(PLOTS_DIR, name)
    fig.savefig(out, dpi=DPI, bbox_inches="tight",
                format="png", transparent=True)
    plt.close(fig)
    print(f"Saved: {out}")


def make_global():
    x = np.linspace(0, 10, 500)
    y = _landscape(x)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, y, color=C_CURVE, lw=2.4)
    # The single highest peak
    i = int(np.argmax(y))
    ax.scatter(x[i], y[i], marker="*", s=560,
               color=C_GLOBAL, edgecolors="black",
               linewidths=1.4, zorder=5)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 1.25)
    ax.set_title("Global", fontsize=15, fontweight="bold",
                 color=C_GLOBAL, pad=6)
    _clean(ax)
    _save(fig, "strategies_global.png")


def make_local():
    from scipy.signal import argrelmax
    x = np.linspace(0, 10, 500)
    y = _landscape(x)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, y, color=C_CURVE, lw=2.4)
    # Only the middle local maximum
    peaks = argrelmax(y, order=8)[0]
    p = peaks[len(peaks) // 2]
    ax.scatter(x[p], y[p], marker="^", s=340,
               color=C_LOCAL, edgecolors="black",
               linewidths=1.3, zorder=5)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 1.25)
    ax.set_title("Local", fontsize=15, fontweight="bold",
                 color=C_LOCAL, pad=6)
    _clean(ax)
    _save(fig, "strategies_local.png")


def make_unexplored():
    x = np.linspace(0, 10, 500)
    y = _landscape(x)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, y, color=C_CURVE, lw=2.4)
    # Observations clustered on the left
    x_obs = np.array([1.5, 2.4, 3.1, 3.9])
    y_obs = _landscape(x_obs)
    ax.scatter(x_obs, y_obs, color=C_OBS, edgecolors="black",
               s=120, linewidths=1.4, zorder=4)
    # Shade the "near observations" exclusion zone
    ax.axvspan(0.6, 4.8, color="0.35", alpha=0.14, zorder=1)
    # Pick a point far from cluster, near the rightmost peak
    pick_x = 7.7
    pick_y = float(_landscape(np.array([pick_x]))[0])
    ax.scatter(pick_x, pick_y, marker="D", s=330,
               color=C_UNEXPL, edgecolors="black",
               linewidths=1.3, zorder=5)
    # Arrow from cluster to pick to suggest "go far"
    ax.annotate("", xy=(pick_x - 0.6, pick_y + 0.05),
                xytext=(4.4, 0.55),
                arrowprops=dict(arrowstyle="-|>",
                                color=C_UNEXPL, lw=1.6,
                                shrinkA=0, shrinkB=8))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 1.25)
    ax.set_title("Unexplored", fontsize=15, fontweight="bold",
                 color=C_UNEXPL, pad=6)
    _clean(ax)
    _save(fig, "strategies_unexplored.png")


if __name__ == "__main__":
    make_global()
    make_local()
    make_unexplored()
