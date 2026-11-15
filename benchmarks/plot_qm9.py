"""Plot QM9 benchmark results across all candidate sizes.

Usage:
    python -m benchmarks.plot_qm9
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ── Style ─────────────────────────────────────────────────────────────────────
ALGO_STYLES = {
    "GLOSS":   {"color": "#2563EB", "lw": 2.5, "ls": "-",  "marker": "o", "zorder": 5},
    "BO (EI)": {"color": "#DC2626", "lw": 2.0, "ls": "--", "marker": "s", "zorder": 4},
    "GA":      {"color": "#16A34A", "lw": 1.8, "ls": "-.", "marker": "^", "zorder": 3},
    "Random":  {"color": "#9CA3AF", "lw": 1.5, "ls": ":",  "marker": "D", "zorder": 2},
}
ALGO_ORDER = ["GLOSS", "BO (EI)", "GA", "Random"]

SIZES = [3000, 5000, 10000, 20000]
SIZE_LABELS = ["3k", "5k", "10k", "20k"]


def load(n):
    path = os.path.join(RESULTS_DIR, f"qm9_{n}_results.csv")
    return pd.read_csv(path)


def mean_curve(df, algo):
    """Mean ± std best-so-far curve across seeds.

    round=-1 is the shared initial state (before any recommendations).
    """
    seeds = sorted(df["seed"].unique())
    rounds = sorted(df["round"].unique())
    curves = []
    for s in seeds:
        sub = df[(df["algorithm"] == algo) & (df["seed"] == s)].sort_values("round")
        curves.append(sub["best_so_far"].values)
    arr = np.array(curves)           # (n_seeds, n_rounds)
    return rounds, arr.mean(0), arr.std(0)


def pct_optimal(df, algo):
    """Final best / true optimum × 100 per seed."""
    y_max = df["best_so_far"].max()
    seeds = sorted(df["seed"].unique())
    finals = []
    for s in seeds:
        sub = df[(df["algorithm"] == algo) & (df["seed"] == s)]
        finals.append(sub["best_so_far"].max())
    return np.array(finals) / y_max * 100


def rounds_to_pct(df, algo, pct=0.95):
    """Mean rounds to reach pct of optimum (or NaN)."""
    y_max = df["best_so_far"].max()
    target = y_max * pct
    seeds = sorted(df["seed"].unique())
    hits = []
    for s in seeds:
        sub = df[(df["algorithm"] == algo) & (df["seed"] == s)].sort_values("round")
        idxs = sub[sub["best_so_far"] >= target]["round"].values
        hits.append(idxs[0] + 1 if len(idxs) > 0 else np.nan)  # 1-indexed
    return np.nanmean(hits) if any(~np.isnan(h) for h in hits) else np.nan


def make_figure():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#F8FAFC")

    # Title
    fig.text(
        0.5, 0.97,
        "GLOSS v2  ·  QM9-Gap Benchmark  (20 rounds, 8 pts/round, 5 seeds)",
        ha="center", va="top", fontsize=16, fontweight="bold", color="#1E293B",
    )
    fig.text(
        0.5, 0.945,
        "GLOSS: UCB acquisition (κ=2.0) · Strategy 6:1:1:0  |  "
        "BO: Expected Improvement  |  GA: UCB-fitness + niche selection",
        ha="center", va="top", fontsize=10, color="#64748B",
    )

    # ── Layout ─────────────────────────────────────────────────────────────
    # Row 0: convergence curves (4 panels)
    # Row 1: % optimal bar chart + rounds-to-95% bar chart
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        top=0.91, bottom=0.08,
        left=0.06, right=0.97,
        hspace=0.38, wspace=0.30,
    )

    dfs = {n: load(n) for n in SIZES}

    # ── Row 0: convergence curves ──────────────────────────────────────────
    # X-axis layout:
    #   x=0        → Initial (round=-1 in CSV): same for all algos
    #   gap x=1    → visual separator (no data)
    #   x=2..21    → Recommendation rounds 1..20 (round=0..19 in CSV)
    def to_plot_x(csv_round):
        """Map CSV round number to plot x coordinate."""
        if csv_round == -1:
            return 0
        return csv_round + 2   # round 0→x=2, round 19→x=21

    for col, (n, label) in enumerate(zip(SIZES, SIZE_LABELS)):
        ax = fig.add_subplot(gs[0, col])
        df = dfs[n]
        y_max = df["best_so_far"].max()

        for algo in ALGO_ORDER:
            if algo not in df["algorithm"].values:
                continue
            st = ALGO_STYLES[algo]
            rds, mu, sd = mean_curve(df, algo)
            xs = [to_plot_x(r) for r in rds]

            # Main line (skip the gap: don't connect x=0 to x=2 with a line)
            # Split into initial segment (x=0) and recommendation segment (x=2..21)
            init_idx = [i for i, r in enumerate(rds) if r == -1]
            rec_idx  = [i for i, r in enumerate(rds) if r >= 0]

            # Recommendation rounds: continuous line with markers every 5 rounds
            if rec_idx:
                xs_r  = [xs[i] for i in rec_idx]
                mu_r  = mu[rec_idx]
                sd_r  = sd[rec_idx]
                ax.plot(
                    xs_r, mu_r / y_max * 100,
                    color=st["color"], lw=st["lw"], ls=st["ls"],
                    marker=st["marker"], markersize=4,
                    markevery=[0, 3, 8, 13, 18],   # rounds 1,4,9,14,19 → display 1,5,10,15,20
                    zorder=st["zorder"], label=algo,
                )
                ax.fill_between(
                    xs_r,
                    (mu_r - sd_r) / y_max * 100,
                    (mu_r + sd_r) / y_max * 100,
                    color=st["color"], alpha=0.12, zorder=st["zorder"] - 1,
                )

            # Initial point: explicit large marker + dashed connector to round 1
            if init_idx:
                i0 = init_idx[0]
                y0 = mu[i0] / y_max * 100
                # Dashed connector from Initial to Round 1
                if rec_idx:
                    ax.plot(
                        [0, xs[rec_idx[0]]],
                        [y0, mu[rec_idx[0]] / y_max * 100],
                        color=st["color"], lw=1.0, ls=":", alpha=0.5, zorder=st["zorder"] - 1,
                    )
                # Large marker for initial point
                ax.plot(
                    0, y0,
                    marker=st["marker"], markersize=8,
                    color=st["color"], markeredgecolor="white", markeredgewidth=1.2,
                    zorder=st["zorder"] + 1, linestyle="none",
                )

        # 95% threshold line
        ax.axhline(95, color="#94A3B8", lw=1.0, ls="--", zorder=1)
        ax.text(21, 95.8, "95%", fontsize=7, color="#94A3B8", ha="right")

        # Vertical separator between Initial and recommendation rounds
        ax.axvspan(-0.6, 1.4, color="#F1F5F9", zorder=0)   # light grey gap region
        ax.axvline(1.4, color="#94A3B8", lw=1.2, ls="-", zorder=2, alpha=0.5)
        ax.text(0.4, 56.5, "Initial\n(all same)", fontsize=6.5,
                color="#64748B", ha="center", va="bottom", style="italic")

        ax.set_title(f"QM9  —  {label} candidates", fontsize=11, fontweight="bold",
                     color="#1E293B", pad=6)
        ax.set_xlabel("Recommendation Round", fontsize=9, color="#475569")
        ax.set_ylabel("% of Global Optimum" if col == 0 else "", fontsize=9, color="#475569")
        ax.set_xlim(-0.8, 21.5)
        # x=0→"Initial",  x=2→"1", x=6→"5", x=11→"10", x=16→"15", x=21→"20"
        ax.set_xticks([0, 2, 6, 11, 16, 21])
        ax.set_xticklabels(["Initial", "1", "5", "10", "15", "20"])
        ax.set_ylim(55, 103)
        ax.tick_params(labelsize=8, colors="#475569")
        ax.set_facecolor("#F8FAFC")
        ax.grid(True, color="#E2E8F0", lw=0.7, zorder=0)
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")

        if col == 3:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9,
                      edgecolor="#CBD5E1", facecolor="white")

    # ── Row 1 left (span 2 cols): % optimal by dataset size ───────────────
    ax_bar = fig.add_subplot(gs[1, :2])
    ax_bar.set_facecolor("#F8FAFC")

    n_algos = len(ALGO_ORDER)
    x = np.arange(len(SIZES))
    width = 0.18
    offsets = np.linspace(-(n_algos - 1) / 2, (n_algos - 1) / 2, n_algos) * width

    for i, algo in enumerate(ALGO_ORDER):
        st = ALGO_STYLES[algo]
        means, stds = [], []
        for n in SIZES:
            df = dfs[n]
            if algo not in df["algorithm"].values:
                means.append(0); stds.append(0)
                continue
            vals = pct_optimal(df, algo)
            means.append(vals.mean())
            stds.append(vals.std())
        bars = ax_bar.bar(
            x + offsets[i], means, width=width,
            color=st["color"], alpha=0.85, label=algo,
            yerr=stds, capsize=3, error_kw={"ecolor": st["color"], "alpha": 0.7, "lw": 1.2},
            zorder=3,
        )
        # Value labels on GLOSS bars
        if algo == "GLOSS":
            for bar, m in zip(bars, means):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2, m + 0.5,
                    f"{m:.1f}%", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=st["color"],
                )

    ax_bar.axhline(95, color="#94A3B8", lw=1.0, ls="--", zorder=1)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{l} candidates" for l in SIZE_LABELS], fontsize=9)
    ax_bar.set_ylabel("Final Best (% of Optimum)", fontsize=9, color="#475569")
    ax_bar.set_ylim(60, 108)
    ax_bar.set_title("Final Optimisation Quality by Dataset Size", fontsize=11,
                     fontweight="bold", color="#1E293B", pad=6)
    ax_bar.legend(fontsize=8.5, loc="lower right", framealpha=0.9,
                  edgecolor="#CBD5E1", facecolor="white")
    ax_bar.tick_params(labelsize=8.5, colors="#475569")
    ax_bar.grid(True, axis="y", color="#E2E8F0", lw=0.7, zorder=0)
    for sp in ax_bar.spines.values():
        sp.set_color("#CBD5E1")

    # ── Row 1 right (span 2 cols): rounds to 95% ──────────────────────────
    ax_spd = fig.add_subplot(gs[1, 2:])
    ax_spd.set_facecolor("#F8FAFC")

    for i, algo in enumerate(ALGO_ORDER):
        st = ALGO_STYLES[algo]
        r95s = []
        for n in SIZES:
            df = dfs[n]
            if algo not in df["algorithm"].values:
                r95s.append(np.nan)
                continue
            r95s.append(rounds_to_pct(df, algo, pct=0.95))

        # Replace nan with a sentinel for display
        r95_plot = [v if not np.isnan(v) else 22 for v in r95s]
        nan_mask = [np.isnan(v) for v in r95s]

        bars = ax_spd.bar(
            x + offsets[i], r95_plot, width=width,
            color=st["color"], alpha=0.85, label=algo,
            zorder=3,
        )
        # Mark bars that never reached 95% with a ✗
        for bar, is_nan in zip(bars, nan_mask):
            if is_nan:
                ax_spd.text(
                    bar.get_x() + bar.get_width() / 2, 21.3,
                    "✗", ha="center", va="bottom",
                    fontsize=9, color=st["color"], fontweight="bold",
                )
        # Label GLOSS bars
        if algo == "GLOSS":
            for bar, v, is_nan in zip(bars, r95s, nan_mask):
                if not is_nan:
                    ax_spd.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color=st["color"],
                    )

    ax_spd.set_xticks(x)
    ax_spd.set_xticklabels([f"{l} candidates" for l in SIZE_LABELS], fontsize=9)
    ax_spd.set_ylabel("Rounds to Reach 95% of Optimum", fontsize=9, color="#475569")
    ax_spd.set_ylim(0, 23)
    ax_spd.set_yticks([0, 5, 10, 15, 20])
    ax_spd.set_title("Convergence Speed (fewer = better)", fontsize=11,
                     fontweight="bold", color="#1E293B", pad=6)
    ax_spd.legend(fontsize=8.5, loc="upper left", framealpha=0.9,
                  edgecolor="#CBD5E1", facecolor="white")
    ax_spd.tick_params(labelsize=8.5, colors="#475569")
    ax_spd.grid(True, axis="y", color="#E2E8F0", lw=0.7, zorder=0)
    ax_spd.text(3.6, 21.5, "✗ = never reached 95%",
                fontsize=7.5, color="#94A3B8", ha="right")
    for sp in ax_spd.spines.values():
        sp.set_color("#CBD5E1")

    # ── Footer ────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.012,
        "Initial = shared starting point before any recommendations (identical across all algorithms within each seed)  ·  "
        "Round 1–20 = after each recommendation batch  ·  "
        "Shaded bands = ±1 std (5 seeds)  ·  GLOSS metric: global_best points only",
        ha="center", fontsize=8, color="#94A3B8",
    )

    out = os.path.join(PLOTS_DIR, "qm9_benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    out = make_figure()
    # Try to open
    import subprocess, sys
    if sys.platform == "darwin":
        subprocess.run(["open", out])
