"""Strategy ratio comparison plot: GLOSS(4:2:2) vs GLOSS(6:1:1).

2-row × 4-column figure:
  Row 0: Convergence curves (best-so-far % optimum vs rounds)
  Row 1: Rounds-to-95% bar charts
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")

ALGO_STYLES = {
    "GLOSS (4:2:2)":        {"color": "#059669", "lw": 2.5, "ls": "-",  "marker": "o", "ms": 7,  "zorder": 6},
    "GLOSS (6:1:1)":        {"color": "#2563EB", "lw": 2.5, "ls": "--", "marker": "s", "ms": 6,  "zorder": 5},
    "GLOSS-global (8:0:0)": {"color": "#7C3AED", "lw": 1.8, "ls": ":",  "marker": "D", "ms": 5,  "zorder": 4},
    "UCB-BO":               {"color": "#F59E0B", "lw": 1.5, "ls": "-.", "marker": "^", "ms": 5,  "zorder": 3},
}
ALGO_ORDER = ["GLOSS (4:2:2)", "GLOSS (6:1:1)", "GLOSS-global (8:0:0)", "UCB-BO"]

DATASETS = [
    ("ratio_qm9_gap_results.csv",              "QM9-gap (20k)\n[Smooth, discrete]"),
    ("ratio_qm9_alpha_results.csv",            "QM9-alpha (20k)\n[Moderate, discrete]"),
    ("ratio_multicluster_7c_8d_results.csv",   "MultiCluster-7C-8D (50k)\n[Multi-modal, discrete]"),
    ("ratio_virtual_materials_10d_results.csv","Virtual Materials 10D\n[Continuous, high-D]"),
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
    # Return both mean-of-hits and count of seeds that hit
    valid = [h for h in hits if not np.isnan(h)]
    return (np.mean(valid) if valid else np.nan), len(valid), len(seeds)


def make_figure():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    loaded = []
    for fname, title in DATASETS:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            loaded.append((pd.read_csv(path), title))
        else:
            print(f"  Missing: {fname}")

    n_ds = len(loaded)
    fig  = plt.figure(figsize=(4 * n_ds, 10))
    fig.patch.set_facecolor("#F8FAFC")

    fig.text(0.5, 0.985,
             "Strategy Ratio Study: GLOSS(4:2:2) vs GLOSS(6:1:1)",
             ha="center", va="top", fontsize=13, fontweight="bold", color="#1E293B")
    fig.text(0.5, 0.963,
             "Green (4:2:2)  vs  Blue (6:1:1)  vs  Purple (8:0:0, global-only)  vs  Orange (UCB-BO)  |  "
             "5 seeds, init=8, 8pts/round, 20 rounds",
             ha="center", va="top", fontsize=9, color="#64748B")

    gs = gridspec.GridSpec(2, n_ds, figure=fig,
                           top=0.93, bottom=0.08,
                           left=0.06, right=0.98,
                           hspace=0.48, wspace=0.3)

    for col, (df, title) in enumerate(loaded):
        y_max = df["best_so_far"].max()

        # ── Row 0: Convergence curves ──────────────────────────────────────────
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
                             color=st["color"], alpha=0.09, zorder=1)

        ax0.axhline(95, color="#94A3B8", lw=0.7, ls="--", zorder=0)
        ax0.text(21.5, 95.5, "95%", fontsize=7, color="#94A3B8", ha="right")

        # Winner badge
        t422, n422, ntot = _rounds_to_threshold(df, "GLOSS (4:2:2)", y_max, 95)
        t611, n611, _    = _rounds_to_threshold(df, "GLOSS (6:1:1)", y_max, 95)
        f422 = df[df["algorithm"]=="GLOSS (4:2:2)"]["best_so_far"].max()
        f611 = df[df["algorithm"]=="GLOSS (6:1:1)"]["best_so_far"].max()

        if not np.isnan(t422) and not np.isnan(t611):
            if t422 < t611 * 0.95:
                outcome, bkgcol, edgcol = "✓ 4:2:2 wins (speed)", "#DCFCE7", "#16A34A"
            elif t611 < t422 * 0.95:
                outcome, bkgcol, edgcol = "✓ 6:1:1 wins (speed)", "#DBEAFE", "#2563EB"
            else:
                outcome, bkgcol, edgcol = "≈ Comparable", "#F1F5F9", "#94A3B8"
        elif n422 > n611:
            outcome, bkgcol, edgcol = "✓ 4:2:2 wins (reach)", "#DCFCE7", "#16A34A"
        elif n611 > n422:
            outcome, bkgcol, edgcol = "✓ 6:1:1 wins (reach)", "#DBEAFE", "#2563EB"
        else:
            outcome, bkgcol, edgcol = "≈ Comparable", "#F1F5F9", "#94A3B8"

        ax0.text(0.98, 0.05, outcome, transform=ax0.transAxes,
                 fontsize=7.5, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=bkgcol, edgecolor=edgcol, alpha=0.9),
                 color=edgcol, fontweight="bold")

        ax0.set_title(title, fontsize=9.5, fontweight="bold", color="#1E293B", pad=5)
        ax0.set_xlabel("Round", fontsize=8, color="#475569")
        if col == 0:
            ax0.set_ylabel("Best Found (% optimum)", fontsize=8.5, color="#475569")
        ax0.set_xlim(-0.5, 22)
        ax0.set_xticks([0, 2, 6, 11, 16, 21])
        ax0.set_xticklabels(["Init", "1", "5", "10", "15", "20"], fontsize=7)
        ax0.tick_params(labelsize=7.5, colors="#475569")
        ax0.set_ylim(bottom=max(0, 50))
        ax0.grid(True, color="#E2E8F0", lw=0.6, zorder=0)
        for sp in ax0.spines.values():
            sp.set_color("#CBD5E1")
        if col == 0:
            ax0.legend(fontsize=6.5, loc="lower right", framealpha=0.9,
                       edgecolor="#CBD5E1", facecolor="white", ncol=2)

        # ── Row 1: Rounds-to-95% bar chart ─────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_facecolor("#F8FAFC")

        r95_vals  = []
        r95_hits  = []
        for algo in ALGO_ORDER:
            t95, n_hit, n_tot = _rounds_to_threshold(df, algo, y_max, 95)
            r95_vals.append(t95 if not np.isnan(t95) else 22)
            r95_hits.append((n_hit, n_tot))

        colors = [ALGO_STYLES[a]["color"] for a in ALGO_ORDER]
        x_pos  = np.arange(len(ALGO_ORDER))
        bars   = ax1.bar(x_pos, r95_vals, color=colors, alpha=0.82, width=0.65, zorder=3)

        for bar, algo, val, (n_hit, n_tot) in zip(bars, ALGO_ORDER, r95_vals, r95_hits):
            orig_t95, _, _ = _rounds_to_threshold(df, algo, y_max, 95)
            if np.isnan(orig_t95):
                ax1.text(bar.get_x() + bar.get_width()/2, 22.8, "✗",
                         ha="center", va="bottom", fontsize=9,
                         color=ALGO_STYLES[algo]["color"])
            else:
                lbl = f"{orig_t95:.1f}\n({n_hit}/{n_tot})"
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, lbl,
                         ha="center", va="bottom", fontsize=6.8, fontweight="bold",
                         color=ALGO_STYLES[algo]["color"])

        ax1.set_ylim(0, 25)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(["GLOSS\n4:2:2", "GLOSS\n6:1:1", "GLOSS\n8:0:0", "UCB-BO"],
                            fontsize=7)
        ax1.set_xlabel("Algorithm", fontsize=8, color="#475569")
        if col == 0:
            ax1.set_ylabel("Rounds to 95% Optimum\n(fewer = better; n/5 seeds reached)",
                           fontsize=8.0, color="#475569")
        ax1.tick_params(labelsize=7.5, colors="#475569")
        ax1.grid(True, axis="y", color="#E2E8F0", lw=0.6, zorder=0)
        ax1.text(3.5, 22.8, "✗=never", fontsize=6.5, color="#94A3B8", ha="right")
        for sp in ax1.spines.values():
            sp.set_color("#CBD5E1")

    # ── Insight box ──────────────────────────────────────────────────────────
    insight = (
        "Key finding:  "
        "6:1:1 wins on smooth landscapes (QM9-gap: 2.6 vs 3.2 rounds) and continuous spaces (65.8% vs 53.8%)  |  "
        "4:2:2 wins on reliability-critical tasks (QM9-alpha: 5/5 vs 4/5 seeds succeed)  |  "
        "4:2:2 reaches 95% in more seeds on multi-cluster (4/5 vs 2/5 seeds) though slower per-seed"
    )
    fig.text(0.5, 0.018, insight, ha="center", fontsize=8, color="#475569", style="italic")

    out = os.path.join(PLOTS_DIR, "ratio_study.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    out = make_figure()
    import subprocess, sys
    if sys.platform == "darwin":
        subprocess.run(["open", out])
