"""Ablation study plot: UCB-BO baseline + GLOSS strategy ratio study."""

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
    "UCB-BO":               {"color": "#F59E0B", "lw": 2.0, "ls": "-.", "marker": "^", "zorder": 3},
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
    seeds = sorted(df["seed"].unique())
    hits = []
    for s in seeds:
        sub = df[(df["algorithm"] == algo) & (df["seed"] == s)].sort_values("round")
        idxs = sub[sub["best_so_far"] >= target]["round"].values
        hits.append(max(0, idxs[0] + 1) if len(idxs) > 0 else np.nan)
    return np.nanmean(hits) if not all(np.isnan(h) for h in hits) else np.nan


def make_figure():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df_gap   = _load("ablation_qm9_gap_results.csv")
    df_alpha = _load("ablation_qm9_alpha_results.csv")
    df_virt  = _load("ablation_virtual_materials_10d_results.csv")

    y_gap   = df_gap["best_so_far"].max()
    y_alpha = df_alpha["best_so_far"].max()
    y_virt  = df_virt["best_so_far"].max()

    datasets = [
        (df_gap,   y_gap,   "QM9-gap (20k)"),
        (df_alpha, y_alpha, "QM9-alpha (20k)"),
        (df_virt,  y_virt,  "Virtual 10D"),
    ]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#F8FAFC")

    fig.text(0.5, 0.985,
             "Ablation Study: UCB Acquisition vs EI  |  Multi-Strategy (6:1:1) vs Global-only (8:0:0)",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#1E293B")
    fig.text(0.5, 0.962,
             "All algorithms use identical GP surrogate (fixed Matérn-2.5, κ=2.0)  |  "
             "init=8, 8 pts/round, 20 rounds, 5 seeds",
             ha="center", va="top", fontsize=9.5, color="#64748B")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           top=0.93, bottom=0.08,
                           left=0.07, right=0.97,
                           hspace=0.48, wspace=0.30)

    thresholds = [75, 85, 95]

    # ── Row 0: Convergence curves ─────────────────────────────────────────────
    for col, (df, y_max, title) in enumerate(datasets):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#F8FAFC")

        for algo in ALGO_ORDER:
            if algo not in df["algorithm"].values:
                continue
            st = ALGO_STYLES[algo]
            rounds, mu, sd = _mean_curve(df, algo)

            pct = mu / y_max * 100
            xs = np.where(rounds == -1, 0, rounds + 2)

            ax.plot(xs, pct, color=st["color"], lw=st["lw"],
                    ls=st["ls"], label=algo, zorder=st["zorder"])
            ax.fill_between(xs,
                            (mu - sd) / y_max * 100,
                            (mu + sd) / y_max * 100,
                            color=st["color"], alpha=0.08, zorder=1)

        ax.axhline(95, color="#94A3B8", lw=0.8, ls="--", zorder=0)
        ax.text(21.5, 95.5, "95%", fontsize=7.5, color="#94A3B8", ha="right")

        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="#1E293B", pad=6)
        ax.set_xlabel("Round (0=Initial)", fontsize=9, color="#475569")
        ax.set_ylabel("Best Found (% of Optimum)" if col == 0 else "",
                      fontsize=9, color="#475569")
        ax.set_xlim(-0.5, 22)
        ax.set_xticks([0, 2, 6, 11, 16, 21])
        ax.set_xticklabels(["Init", "1", "5", "10", "15", "20"])
        ax.tick_params(labelsize=8, colors="#475569")
        ax.set_ylim(bottom=0)
        ax.grid(True, color="#E2E8F0", lw=0.7, zorder=0)
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")
        if col == 0:
            ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9,
                      edgecolor="#CBD5E1", facecolor="white")

    # ── Row 1: Rounds to threshold ────────────────────────────────────────────
    for col, (df, y_max, title) in enumerate(datasets):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("#F8FAFC")

        n_algos = len(ALGO_ORDER)
        x = np.arange(len(thresholds))
        width = 0.18
        offsets = np.linspace(-(n_algos - 1) / 2, (n_algos - 1) / 2, n_algos) * width

        for i, algo in enumerate(ALGO_ORDER):
            st = ALGO_STYLES[algo]
            r_vals = [_rounds_to_threshold(df, algo, y_max, t) for t in thresholds]
            r_plot = [v if not np.isnan(v) else 22 for v in r_vals]
            nan_mask = [np.isnan(v) for v in r_vals]

            bars = ax.bar(x + offsets[i], r_plot, width=width,
                          color=st["color"], alpha=0.82, label=algo, zorder=3)
            for bar, is_nan in zip(bars, nan_mask):
                if is_nan:
                    ax.text(bar.get_x() + bar.get_width() / 2, 22.5, "✗",
                            ha="center", va="bottom", fontsize=8,
                            color=st["color"], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}%" for t in thresholds], fontsize=9)
        ax.set_ylim(0, 25)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1E293B", pad=6)
        ax.set_xlabel("Target Quality Threshold", fontsize=9, color="#475569")
        ax.set_ylabel("Rounds Required (fewer = better)" if col == 0 else "",
                      fontsize=9, color="#475569")
        ax.tick_params(labelsize=8, colors="#475569")
        ax.grid(True, axis="y", color="#E2E8F0", lw=0.7, zorder=0)
        ax.text(2.8, 22.8, "✗ = never reached",
                fontsize=7, color="#94A3B8", ha="right")
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")
        if col == 0:
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9,
                      edgecolor="#CBD5E1", facecolor="white")

    # ── Annotation box ────────────────────────────────────────────────────────
    findings = (
        "Key findings:  "
        "① UCB acquisition (UCB-BO) is 2.6× faster than EI (BO) on discrete datasets (3.0 vs 7.8 rounds to 95%)  "
        "② Multi-strategy coordination (6:1:1) adds ~15% further speedup over UCB-BO on discrete (2.6 vs 3.0 rounds)  "
        "③ GLOSS framework achieves 61.6% vs UCB-BO 43.4% on 10D continuous (+18pp), showing diversity-aware selection matters in high-D"
    )
    fig.text(0.5, 0.015, findings,
             ha="center", fontsize=8, color="#475569",
             wrap=True, style="italic")

    out = os.path.join(PLOTS_DIR, "ablation_study.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    out = make_figure()
    import subprocess, sys
    if sys.platform == "darwin":
        subprocess.run(["open", out])
