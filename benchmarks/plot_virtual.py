"""Virtual landscape benchmark plot: 6 algorithms × 6 virtual spaces.

3-row × 6-column figure:
  Row 0: Convergence curves (best-so-far % optimum vs rounds)
  Row 1: Rounds-to-95% bar charts
  Row 2: Heatmap of %optimum reached across all algo × space pairs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")

ALGO_STYLES = {
    "GLOSS (6:1:1)":        {"color": "#2563EB", "lw": 2.5, "ls": "-",  "marker": "o", "ms": 6,  "zorder": 6},
    "GLOSS-global (8:0:0)": {"color": "#7C3AED", "lw": 2.0, "ls": "--", "marker": "s", "ms": 5,  "zorder": 5},
    "UCB-BO":               {"color": "#F59E0B", "lw": 1.8, "ls": "-.", "marker": "^", "ms": 5,  "zorder": 4},
    "BO (EI)":              {"color": "#EF4444", "lw": 1.8, "ls": ":",  "marker": "D", "ms": 4,  "zorder": 3},
    "GA":                   {"color": "#10B981", "lw": 1.5, "ls": "--", "marker": "v", "ms": 4,  "zorder": 2},
    "Random":               {"color": "#94A3B8", "lw": 1.2, "ls": ":",  "marker": "x", "ms": 4,  "zorder": 1},
}
ALGO_ORDER = ["GLOSS (6:1:1)", "GLOSS-global (8:0:0)", "UCB-BO", "BO (EI)", "GA", "Random"]

SPACES = [
    ("virtual_single_peak_6d_results.csv",   "Single-Peak-6D\n[Unimodal smooth]"),
    ("virtual_double_peak_6d_results.csv",   "Double-Peak-6D\n[Bimodal deceptive]"),
    ("virtual_multi_peak_8d_results.csv",    "Multi-Peak-8D\n[7 Gaussian peaks]"),
    ("virtual_rastrigin_8d_results.csv",     "Rastrigin-8D\n[Oscillatory traps]"),
    ("virtual_high_dim_15d_results.csv",     "High-Dim-15D\n[Smooth 15D]"),
    ("virtual_plateau_6d_results.csv",       "Plateau-6D\n[Flat + narrow peak]"),
]


def _mean_curve(df, algo):
    seeds  = sorted(df["seed"].unique())
    rounds = sorted(df["round"].unique())
    y_max  = df["best_so_far"].max()
    curves = [
        df[(df["algorithm"] == algo) & (df["seed"] == s)]
          .sort_values("round")["best_so_far"].values / y_max * 100
        for s in seeds
    ]
    arr = np.array(curves)
    return np.array(rounds), arr.mean(0), arr.std(0)


def _rounds_to_threshold(df, algo, y_max, pct):
    target = y_max * pct / 100
    seeds  = sorted(df["seed"].unique())
    hits   = []
    for s in seeds:
        sub  = df[(df["algorithm"] == algo) & (df["seed"] == s)].sort_values("round")
        idxs = sub[sub["best_so_far"] >= target]["round"].values
        hits.append(idxs[0] + 2 if len(idxs) > 0 else np.nan)
    valid = [h for h in hits if not np.isnan(h)]
    return (np.mean(valid) if valid else np.nan), len(valid), len(seeds)


def make_figure():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    loaded = []
    for fname, title in SPACES:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            loaded.append((pd.read_csv(path), title, fname))
        else:
            print(f"  Missing: {fname}")

    if not loaded:
        print("No result files found. Run virtual_benchmark.py first.")
        return

    n_cols = len(loaded)
    fig    = plt.figure(figsize=(3.8 * n_cols, 12))
    fig.patch.set_facecolor("#F8FAFC")

    fig.text(0.5, 0.988,
             "Virtual Landscape Benchmark: 6 Algorithms × 6 Spaces",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#1E293B")
    fig.text(0.5, 0.968,
             "Blue=GLOSS(6:1:1)  Purple=GLOSS-global  Orange=UCB-BO  Red=BO(EI)  Green=GA  Grey=Random  "
             "| 5 seeds, init=8, 8pts/round, 20 rounds",
             ha="center", va="top", fontsize=8.5, color="#64748B")

    gs = gridspec.GridSpec(2, n_cols, figure=fig,
                           top=0.94, bottom=0.20,
                           left=0.05, right=0.98,
                           hspace=0.52, wspace=0.28)

    # ── Rows 0-1: Per-space convergence + bar charts ──────────────────────────
    summary_pct  = {algo: [] for algo in ALGO_ORDER}
    summary_t95  = {algo: [] for algo in ALGO_ORDER}
    space_labels = []

    for col, (df, title, fname) in enumerate(loaded):
        y_max = df["best_so_far"].max()
        space_labels.append(title.split("\n")[0])

        # Row 0: Convergence curves
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_facecolor("#F8FAFC")

        for algo in ALGO_ORDER:
            if algo not in df["algorithm"].values:
                continue
            st = ALGO_STYLES[algo]
            rounds, mu, sd = _mean_curve(df, algo)
            xs = np.where(rounds == -1, 0, rounds + 2)

            ax0.plot(xs, mu, color=st["color"], lw=st["lw"],
                     ls=st["ls"], label=algo.replace(" (", "\n("), zorder=st["zorder"])
            ax0.fill_between(xs, mu - sd, mu + sd,
                             color=st["color"], alpha=0.08, zorder=1)

        ax0.axhline(95, color="#94A3B8", lw=0.7, ls="--", zorder=0)
        ax0.text(21.5, 96, "95%", fontsize=6.5, color="#94A3B8", ha="right")

        # Winner badge: best final %optimum
        finals = {}
        for algo in ALGO_ORDER:
            if algo not in df["algorithm"].values:
                continue
            seeds_ = sorted(df["seed"].unique())
            y_m    = df["best_so_far"].max()
            vals   = [df[(df["algorithm"]==algo) & (df["seed"]==s)]["best_so_far"].max()
                      for s in seeds_]
            finals[algo] = np.mean(vals) / y_m * 100

        best_algo = max(finals, key=finals.get)
        best_pct  = finals[best_algo]
        badge_color = ALGO_STYLES[best_algo]["color"]
        short = best_algo.split(" ")[0]
        badge_text = f"★ {short}\n{best_pct:.1f}%"

        ax0.text(0.98, 0.05, badge_text, transform=ax0.transAxes,
                 fontsize=7, ha="right", va="bottom", color=badge_color, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor=badge_color, alpha=0.9))

        ax0.set_title(title, fontsize=9, fontweight="bold", color="#1E293B", pad=4)
        ax0.set_xlabel("Round", fontsize=7.5, color="#475569")
        if col == 0:
            ax0.set_ylabel("Best Found (% optimum)", fontsize=8, color="#475569")
        ax0.set_xlim(-0.5, 22)
        ax0.set_xticks([0, 2, 6, 11, 16, 21])
        ax0.set_xticklabels(["Init", "1", "5", "10", "15", "20"], fontsize=6.5)
        ax0.tick_params(labelsize=7, colors="#475569")
        ax0.set_ylim(bottom=max(0, 40))
        ax0.grid(True, color="#E2E8F0", lw=0.5, zorder=0)
        for sp in ax0.spines.values():
            sp.set_color("#CBD5E1")
        if col == 0:
            ax0.legend(fontsize=5.5, loc="lower right", framealpha=0.9,
                       edgecolor="#CBD5E1", facecolor="white", ncol=2)

        # Row 1: Rounds-to-95% bar chart
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_facecolor("#F8FAFC")

        r95_vals = []
        r95_hits = []
        for algo in ALGO_ORDER:
            if algo not in df["algorithm"].values:
                r95_vals.append(np.nan)
                r95_hits.append((0, 5))
                continue
            t95, n_hit, n_tot = _rounds_to_threshold(df, algo, y_max, 95)
            r95_vals.append(t95)
            r95_hits.append((n_hit, n_tot))
            # Summary
            summary_t95[algo].append(t95 if not np.isnan(t95) else np.nan)
            seeds_ = sorted(df["seed"].unique())
            vals   = [df[(df["algorithm"]==algo) & (df["seed"]==s)]["best_so_far"].max()
                      for s in seeds_]
            summary_pct[algo].append(np.mean(vals) / y_max * 100)

        plot_vals = [v if not np.isnan(v) else 22 for v in r95_vals]
        colors = [ALGO_STYLES[a]["color"] for a in ALGO_ORDER]
        x_pos  = np.arange(len(ALGO_ORDER))
        bars   = ax1.bar(x_pos, plot_vals, color=colors, alpha=0.82, width=0.65, zorder=3)

        for bar, algo, orig_val, (n_hit, n_tot) in zip(bars, ALGO_ORDER, r95_vals, r95_hits):
            if np.isnan(orig_val):
                ax1.text(bar.get_x() + bar.get_width()/2, 22.8, "✗",
                         ha="center", va="bottom", fontsize=8,
                         color=ALGO_STYLES[algo]["color"])
            else:
                lbl = f"{orig_val:.1f}\n({n_hit}/{n_tot})"
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, lbl,
                         ha="center", va="bottom", fontsize=5.5, fontweight="bold",
                         color=ALGO_STYLES[algo]["color"])

        ax1.set_ylim(0, 25)
        ax1.set_xticks(x_pos)
        short_labels = ["GLOSS\n6:1:1", "GLOSS\n8:0:0", "UCB-BO", "BO(EI)", "GA", "Rand"]
        ax1.set_xticklabels(short_labels, fontsize=6)
        ax1.set_xlabel("Algorithm", fontsize=7.5, color="#475569")
        if col == 0:
            ax1.set_ylabel("Rounds to 95% Opt.\n(fewer=better; n/5 seeds)", fontsize=7.5, color="#475569")
        ax1.tick_params(labelsize=7, colors="#475569")
        ax1.grid(True, axis="y", color="#E2E8F0", lw=0.5, zorder=0)
        ax1.text(5.4, 23.2, "✗=never", fontsize=6, color="#94A3B8", ha="right")
        for sp in ax1.spines.values():
            sp.set_color("#CBD5E1")

    # ── Row 2: Heatmap ─────────────────────────────────────────────────────────
    ax_heat = fig.add_axes([0.05, 0.04, 0.92, 0.13])
    ax_heat.set_facecolor("#F8FAFC")

    matrix = np.array([summary_pct[algo] for algo in ALGO_ORDER])  # (6_algos, 6_spaces)
    cmap   = plt.cm.RdYlGn
    norm   = mcolors.Normalize(vmin=40, vmax=100)
    im     = ax_heat.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for i, algo in enumerate(ALGO_ORDER):
        for j in range(len(space_labels)):
            val = matrix[i, j]
            c   = "white" if val < 65 or val > 90 else "#1E293B"
            ax_heat.text(j, i, f"{val:.0f}%", ha="center", va="center",
                         fontsize=7.5, color=c, fontweight="bold")

    ax_heat.set_xticks(range(len(space_labels)))
    ax_heat.set_xticklabels(space_labels, fontsize=7.5, color="#475569")
    ax_heat.set_yticks(range(len(ALGO_ORDER)))
    short_algo = ["GLOSS(6:1:1)", "GLOSS-global", "UCB-BO", "BO(EI)", "GA", "Random"]
    ax_heat.set_yticklabels(short_algo, fontsize=7.5, color="#475569")
    ax_heat.set_title("Final % Optimum Reached (mean over 5 seeds)",
                       fontsize=9, fontweight="bold", color="#1E293B", pad=4)
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("%Opt", fontsize=7)

    # Insight text
    insight = (
        "Key observations:  "
        "GLOSS(6:1:1) leads on unimodal and high-D smooth landscapes  |  "
        "UCB methods outperform EI on oscillatory/deceptive landscapes  |  "
        "Plateau-6D reveals BO(EI) struggle when improvement signal is tiny  |  "
        "Random sampling fails on all non-trivial landscapes"
    )
    fig.text(0.5, 0.007, insight, ha="center", fontsize=8, color="#475569", style="italic")

    out = os.path.join(PLOTS_DIR, "virtual_benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    out = make_figure()
    if out:
        import subprocess, sys
        if sys.platform == "darwin":
            subprocess.run(["open", out])
