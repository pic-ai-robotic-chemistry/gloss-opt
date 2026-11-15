"""Schematic for GLOSS's three picking strategies on a shared 1D landscape.

Each panel uses the SAME observed points and SAME GP surrogate so the
three picking rules can be compared directly.

  Global      : argmax UCB = mu + kappa*sigma
  Local       : top-K local maxima of mu (BallTree window dominates)
  Unexplored  : argmax mu restricted to {x : dist(x, X_obs) > tau}

Output: benchmarks/plots/strategies_demo.jpg
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from scipy.signal import argrelmax
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

matplotlib.rcParams["font.family"] = "Arial"

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Strategy palette (warm exploit → cool explore).
C_GLOBAL = "#d6604d"   # red — UCB exploit
C_LOCAL = "#1b7837"    # green — local refinement
C_UNEXPL = "#2166ac"   # blue — exploration
C_TRUE = "#666666"
C_MU = "#222222"
C_BAND = "#cccccc"


def _true_function(x: np.ndarray) -> np.ndarray:
    """Multi-modal toy function with a clear global peak + two local peaks."""
    return (1.05 * np.exp(-((x - 2.3) / 0.65) ** 2)
            + 0.68 * np.exp(-((x - 4.9) / 0.55) ** 2)
            + 0.92 * np.exp(-((x - 7.7) / 0.55) ** 2)
            + 0.07 * np.sin(1.6 * x))


def _make_setup():
    x_grid = np.linspace(0, 10, 700).reshape(-1, 1)
    y_true = _true_function(x_grid.ravel())

    rng = np.random.RandomState(42)
    x_obs = np.array([0.5, 1.6, 3.0, 4.0, 7.0]).reshape(-1, 1)
    y_obs = _true_function(x_obs.ravel()) + rng.normal(0, 0.015, size=len(x_obs))

    kernel = (C(1.0, (0.1, 5.0))
              * RBF(length_scale=0.7, length_scale_bounds=(0.3, 3.0)))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8,
                                  alpha=1e-4, normalize_y=True,
                                  random_state=0)
    gp.fit(x_obs, y_obs)
    mu, sigma = gp.predict(x_grid, return_std=True)
    return dict(x_grid=x_grid, y_true=y_true, x_obs=x_obs, y_obs=y_obs,
                mu=mu, sigma=sigma)


def _draw_base(ax, S, *, show_band=True, show_legend_extras=True):
    ax.plot(S["x_grid"], S["y_true"], color=C_TRUE, ls=":", lw=1.3,
            alpha=0.8, label=r"true $f(x)$" if show_legend_extras else None)
    if show_band:
        ax.fill_between(S["x_grid"].ravel(), S["mu"] - S["sigma"],
                        S["mu"] + S["sigma"], color=C_BAND, alpha=0.55,
                        edgecolor="none",
                        label=r"$\mu \pm \sigma$" if show_legend_extras else None)
    ax.plot(S["x_grid"], S["mu"], color=C_MU, lw=2.0,
            label=r"surrogate $\mu(x)$" if show_legend_extras else None)
    ax.scatter(S["x_obs"].ravel(), S["y_obs"], color="white",
               edgecolors="black", s=78, zorder=5, linewidths=1.5,
               label="observed" if show_legend_extras else None)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.15, 1.55)
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.tick_params(labelsize=9)


def main():
    S = _make_setup()
    x_grid = S["x_grid"].ravel()
    mu = S["mu"]
    sigma = S["sigma"]
    x_obs = S["x_obs"].ravel()

    kappa = 2.0
    ucb = mu + kappa * sigma

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True,
                              gridspec_kw=dict(wspace=0.08))

    # ── Panel A: Global — argmax UCB ────────────────────────────────────────
    ax = axes[0]
    _draw_base(ax, S)
    ax.plot(x_grid, ucb, color=C_GLOBAL, ls="--", lw=1.5,
            label=r"UCB $=\mu+\kappa\sigma$ ($\kappa=2$)")
    idx = int(np.argmax(ucb))
    ax.axvline(x_grid[idx], color=C_GLOBAL, ls=":", lw=0.9, alpha=0.5)
    ax.scatter(x_grid[idx], mu[idx], marker="*", s=400,
               color=C_GLOBAL, edgecolors="black",
               zorder=7, linewidths=1.3, label="Global pick")
    ax.set_ylabel("value", fontsize=11)
    ax.set_title("Global · maximize UCB",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.text(0.02, 0.04, "A", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")

    # ── Panel B: Local — top-K local maxima with BallTree window ────────────
    ax = axes[1]
    _draw_base(ax, S)
    local_idx_all = argrelmax(mu, order=8)[0]
    # Rank candidate local maxima by mu and take the top 3.
    local_idx = sorted(local_idx_all.tolist(), key=lambda i: mu[i], reverse=True)[:3]
    window_half = 1.0  # half of the "BallTree" window in data units
    for k, idx in enumerate(local_idx):
        x_lm = x_grid[idx]
        ax.axvspan(x_lm - window_half, x_lm + window_half,
                   color=C_LOCAL, alpha=0.10,
                   label="window radius" if k == 0 else None)
        ax.scatter(x_lm, mu[idx], marker="^", s=230,
                   color=C_LOCAL, edgecolors="black",
                   zorder=7, linewidths=1.2,
                   label="Local pick" if k == 0 else None)
    ax.set_title(r"Local · top-$K$ local maxima of $\mu$",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.text(0.02, 0.04, "B", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")

    # ── Panel C: Unexplored — far from obs, then best mu ────────────────────
    ax = axes[2]
    _draw_base(ax, S)
    # Distance to nearest observation (1D)
    dist = np.min(np.abs(x_grid[:, None] - x_obs[None, :]), axis=1)
    tau = pdist(x_obs.reshape(-1, 1)).mean() * 0.5
    far_mask = dist > tau
    near_mask = ~far_mask
    # Shade "near" (excluded) regions; identify contiguous segments.
    in_seg = False
    s0 = 0
    first_excl = True
    for i, near in enumerate(near_mask):
        if near and not in_seg:
            s0 = i
            in_seg = True
        elif not near and in_seg:
            ax.axvspan(x_grid[s0], x_grid[i],
                       color="0.35", alpha=0.18,
                       label=(r"excluded ($d<\tau$)" if first_excl else None))
            first_excl = False
            in_seg = False
    if in_seg:
        ax.axvspan(x_grid[s0], x_grid[-1], color="0.35", alpha=0.18,
                   label=(r"excluded ($d<\tau$)" if first_excl else None))
    # Best mu among far candidates
    far_indices = np.where(far_mask)[0]
    best = far_indices[int(np.argmax(mu[far_indices]))]
    ax.scatter(x_grid[best], mu[best], marker="D", s=210,
               color=C_UNEXPL, edgecolors="black",
               zorder=7, linewidths=1.2, label="Unexplored pick")
    ax.set_title(r"Unexplored · $d(x,X_{obs})>\tau$, then $\max\mu$",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.text(0.02, 0.04, "C", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")

    fig.suptitle(
        "GLOSS three-stream sampling strategies on a shared GP surrogate",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "strategies_demo.jpg")
    fig.savefig(out, dpi=220, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 94})
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
