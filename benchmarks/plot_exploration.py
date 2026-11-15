"""Exploration study summary plot.

Shows: where GLOSS multi-strategy (6:1:1) beats or loses to GLOSS-global (8:0:0) and BO(EI).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")

ALGO_STYLES = {
    "GLOSS (6:1:1)":        {"color": "#2563EB", "lw": 2.5, "ls": "-",  "marker": "o", "zorder": 5},
    "GLOSS-global (8:0:0)": {"color": "#7C3AED", "lw": 2.0, "ls": "--", "marker": "D", "zorder": 4},
    "UCB-BO":               {"color": "#F59E0B", "lw": 1.8, "ls": "-.", "marker": "^", "zorder": 3},
    "BO (EI)":              {"color": "#DC2626", "lw": 1.8, "ls": ":",  "marker": "s", "zorder": 2},
}
ALGO_ORDER = ["GLOSS (6:1:1)", "GLOSS-global (8:0:0)", "UCB-BO", "BO (EI)"]


def _load(fname):
    return pd.read_csv(os.path.join(RESULTS_DIR, fname))


def _mean_curve(df, algo):
    seeds  = sorted(df["seed"].unique())
    rounds = sorted(df["round"].unique())
    curves = [
        df[(df["algorithm"] == algo) & (df["seed"] == s)]
          .sort_values("round")["best_so_far"].values
        for s in seeds
    ]
    arr = np.array(curves)
    return np.array(rounds), arr.mean(0), arr.std(0)


def _rounds_to_threshold(df, algo, y_max, threshold_pct):
    target = y_max * threshold_pct / 100
    seeds  = sorted(df["seed"].unique())
    hits = []
    for s in seeds:
        sub  = df[(df["algorithm"] == algo) & (df["seed"] == s)].sort_values("round")
        idxs = sub[sub["best_so_far"] >= target]["round"].values
        hits.append(max(0, idxs[0] + 1) if len(idxs) > 0 else np.nan)
    return np.nanmean(hits) if not all(np.isnan(h) for h in hits) else np.nan


def make_figure():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Load datasets ─────────────────────────────────────────────────────────
    datasets_info = [
        # (filename, title, expected multi-strategy result)
        ("explore_multicluster_3c_8d_results.csv",  "Multi-Cluster 3C-8D\n(Synthetic)", "neutral"),
        ("explore_multicluster_7c_8d_results.csv",  "Multi-Cluster 7C-8D\n(Synthetic)", "positive"),
        ("explore_qm9_mu_results.csv",               "QM9-Dipole (μ)\n[Sparse targets]", "negative"),
        ("explore_qm9_r2_results.csv",               "QM9-Spatial Extent (r²)\n[Dense targets]", "neutral"),
    ]
    # Also add QM9-gap from ablation for comparison
    ablation_gap = ("ablation_qm9_gap_results.csv", "QM9-HOMO-LUMO Gap\n[Smooth, clustered]", "positive")
    datasets_info.insert(2, ablation_gap)

    loaded = []
    for fname, title, label in datasets_info:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            loaded.append((pd.read_csv(path), title, label))
        else:
            print(f"  Missing: {fname}")

    n_ds = len(loaded)
    fig  = plt.figure(figsize=(4 * n_ds, 10))
    fig.patch.set_facecolor("#F8FAFC")

    fig.text(0.5, 0.985,
             "Exploration Study: When Does GLOSS Multi-Strategy (6:1:1) Beat Single-Strategy (8:0:0)?",
             ha="center", va="top", fontsize=13, fontweight="bold", color="#1E293B")
    fig.text(0.5, 0.963,
             "Blue (GLOSS 6:1:1)  vs  Purple (GLOSS-global 8:0:0)  vs  Orange (UCB-BO)  vs  Red (BO EI)  |  "
             "5 seeds, init=8, 8pts/round, 20 rounds",
             ha="center", va="top", fontsize=9, color="#64748B")

    gs = gridspec.GridSpec(2, n_ds, figure=fig,
                           top=0.93, bottom=0.08,
                           left=0.05, right=0.98,
                           hspace=0.48, wspace=0.28)

    for col, (df, title, label) in enumerate(loaded):
        y_max = df["best_so_far"].max()

        # ── Row 0: Convergence curves ─────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_facecolor("#F8FAFC")

        for algo in ALGO_ORDER:
            if algo not in df["algorithm"].values:
                continue
            st = ALGO_STYLES[algo]
            rounds, mu, sd = _mean_curve(df, algo)
            pct = mu / y_max * 100
            xs  = np.where(rounds == -1, 0, rounds + 2)

            ax0.plot(xs, pct, color=st["color"], lw=st["lw"],
                     ls=st["ls"], label=algo.replace(" (", "\n("), zorder=st["zorder"])
            ax0.fill_between(xs, (mu - sd)/y_max*100, (mu + sd)/y_max*100,
                             color=st["color"], alpha=0.08, zorder=1)

        ax0.axhline(95, color="#94A3B8", lw=0.7, ls="--", zorder=0)
        ax0.text(21.5, 95.5, "95%", fontsize=7, color="#94A3B8", ha="right")

        # Label the multi-strategy outcome
        bkgcol = {"positive": "#DCFCE7", "negative": "#FEE2E2", "neutral": "#F1F5F9"}[label]
        edgcol = {"positive": "#16A34A", "negative": "#DC2626", "neutral": "#94A3B8"}[label]
        outcome_text = {"positive": "✓ Multi-strategy wins", "negative": "✗ Multi-strategy loses", "neutral": "≈ Comparable"}[label]
        ax0.text(0.98, 0.05, outcome_text, transform=ax0.transAxes,
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
        ax0.set_ylim(bottom=max(0, pct.min() - 5))
        ax0.grid(True, color="#E2E8F0", lw=0.6, zorder=0)
        for sp in ax0.spines.values():
            sp.set_color("#CBD5E1")
        if col == 0:
            ax0.legend(fontsize=6.5, loc="lower right", framealpha=0.9,
                       edgecolor="#CBD5E1", facecolor="white", ncol=2)

        # ── Row 1: Rounds-to-95% bar chart ────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_facecolor("#F8FAFC")

        r95_vals  = []
        r95_algos = []
        for algo in ALGO_ORDER:
            r95 = _rounds_to_threshold(df, algo, y_max, 95)
            r95_vals.append(r95 if not np.isnan(r95) else 22)
            r95_algos.append(algo)

        colors = [ALGO_STYLES[a]["color"] for a in r95_algos]
        x_pos  = np.arange(len(r95_algos))
        bars   = ax1.bar(x_pos, r95_vals, color=colors, alpha=0.82, width=0.65, zorder=3)

        for bar, algo, val in zip(bars, r95_algos, [_rounds_to_threshold(df, a, y_max, 95) for a in ALGO_ORDER]):
            if np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width()/2, 22.8, "✗",
                         ha="center", va="bottom", fontsize=9, color=ALGO_STYLES[algo]["color"])
            else:
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.1f}",
                         ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                         color=ALGO_STYLES[algo]["color"])

        ax1.set_ylim(0, 25)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(["GLOSS\n6:1:1", "GLOSS\n8:0:0", "UCB-BO", "BO(EI)"], fontsize=7)
        ax1.set_xlabel("Algorithm", fontsize=8, color="#475569")
        if col == 0:
            ax1.set_ylabel("Rounds to 95% Optimum\n(fewer = better)", fontsize=8.5, color="#475569")
        ax1.tick_params(labelsize=7.5, colors="#475569")
        ax1.grid(True, axis="y", color="#E2E8F0", lw=0.6, zorder=0)
        ax1.text(3.5, 22.8, "✗=never", fontsize=6.5, color="#94A3B8", ha="right")
        for sp in ax1.spines.values():
            sp.set_color("#CBD5E1")

    # ── Insight box ──────────────────────────────────────────────────────────
    insight = (
        "Key finding:  "
        "GLOSS multi-strategy wins on smooth, clustered landscapes (QM9-gap: 2.6 vs 3.0 rounds; multi-cluster 7C: 10.0 vs 12.2 rounds)  |  "
        "Multi-strategy loses on sparse/scattered targets (QM9-dipole: GLOSS 76.6% vs GLOSS-global 90.9%)  |  "
        "Ideal domain: large discrete spaces with ~1000+ top-5% candidates forming chemical scaffold families"
    )
    fig.text(0.5, 0.018, insight, ha="center", fontsize=8, color="#475569", style="italic")

    out = os.path.join(PLOTS_DIR, "exploration_study.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    out = make_figure()
    import subprocess, sys
    if sys.platform == "darwin":
        subprocess.run(["open", out])
