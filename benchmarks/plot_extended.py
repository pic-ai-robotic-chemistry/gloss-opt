"""Plot extended benchmark results: QM8-f1-CC2 + QM9-alpha + 10D Virtual Materials.

Usage:
    python -m benchmarks.plot_extended
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")

ALGO_STYLES = {
    "GLOSS":   {"color": "#2563EB", "lw": 2.5, "ls": "-",  "marker": "o", "zorder": 5},
    "BO (EI)": {"color": "#DC2626", "lw": 2.0, "ls": "--", "marker": "s", "zorder": 4},
    "GA":      {"color": "#16A34A", "lw": 1.8, "ls": "-.", "marker": "^", "zorder": 3},
    "Random":  {"color": "#9CA3AF", "lw": 1.5, "ls": ":",  "marker": "D", "zorder": 2},
}
ALGO_ORDER = ["GLOSS", "BO (EI)", "GA", "Random"]

DATASETS = [
    ("ext_qm9_gap_results.csv",              "QM9-gap\n(20k, 20D)"),
    ("ext_qm9_alpha_results.csv",            "QM9-alpha\n(20k, 20D)"),
    ("ext_virtual_materials_10d_results.csv","Virtual Materials 10D\n(continuous)"),
]


def _load(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _mean_curve(df, algo):
    seeds  = sorted(df["seed"].unique())
    rounds = sorted(df["round"].unique())
    curves = [
        df[(df["algorithm"] == algo) & (df["seed"] == s)]
          .sort_values("round")["best_so_far"].values
        for s in seeds
    ]
    arr = np.array(curves)
    return rounds, arr.mean(0), arr.std(0)


def _to_plot_x(csv_round):
    """round=-1 → x=0 (Initial), round 0..19 → x=2..21."""
    return 0 if csv_round == -1 else csv_round + 2


def _pct_optimal(df, algo):
    y_max  = df["best_so_far"].max()
    seeds  = sorted(df["seed"].unique())
    finals = [df[(df["algorithm"] == algo) & (df["seed"] == s)]["best_so_far"].max()
              for s in seeds]
    return np.array(finals) / y_max * 100


def _rounds_to_95(df, algo):
    y_max  = df["best_so_far"].max()
    target = y_max * 0.95
    seeds  = sorted(df["seed"].unique())
    hits   = []
    for s in seeds:
        sub  = df[(df["algorithm"] == algo) & (df["seed"] == s)].sort_values("round")
        idxs = sub[sub["best_so_far"] >= target]["round"].values
        # round=-1 counts as 0 recommendation rounds
        hits.append(max(0, idxs[0] + 1) if len(idxs) > 0 else np.nan)
    return np.nanmean(hits) if not all(np.isnan(h) for h in hits) else np.nan


def make_figure():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data
    dfs = {}
    for fname, panel_label in DATASETS:
        try:
            dfs[fname] = _load(fname)
        except FileNotFoundError as e:
            print(f"  Missing: {e}")
            return

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#F8FAFC")

    fig.text(0.5, 0.975,
             "GLOSS v2  ·  Extended Benchmark: QM9-gap + QM9-α + 10D Virtual  (20 rounds, 8 pts/round, 5 seeds)",
             ha="center", va="top", fontsize=15, fontweight="bold", color="#1E293B")
    fig.text(0.5, 0.950,
             "GLOSS: UCB (κ=2.0) · Strategy 6:1:1:0  |  "
             "BO: Expected Improvement  |  GA: UCB-fitness + niche selection",
             ha="center", va="top", fontsize=9.5, color="#64748B")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           top=0.92, bottom=0.09,
                           left=0.06, right=0.97,
                           hspace=0.42, wspace=0.28)

    # ── Row 0: convergence curves ─────────────────────────────────────────
    for col, (fname, panel_label) in enumerate(DATASETS):
        ax  = fig.add_subplot(gs[0, col])
        df  = dfs[fname]
        y_max = df["best_so_far"].max()

        for algo in ALGO_ORDER:
            if algo not in df["algorithm"].values:
                continue
            st = ALGO_STYLES[algo]
            rds, mu, sd = _mean_curve(df, algo)
            xs = [_to_plot_x(r) for r in rds]

            init_idx = [i for i, r in enumerate(rds) if r == -1]
            rec_idx  = [i for i, r in enumerate(rds) if r >= 0]

            # Recommendation curve
            if rec_idx:
                xs_r = [xs[i] for i in rec_idx]
                mu_r = mu[rec_idx]; sd_r = sd[rec_idx]
                ax.plot(xs_r, mu_r / y_max * 100,
                        color=st["color"], lw=st["lw"], ls=st["ls"],
                        marker=st["marker"], markersize=4,
                        markevery=[0, 3, 8, 13, 18],
                        zorder=st["zorder"], label=algo)
                ax.fill_between(xs_r,
                                (mu_r - sd_r) / y_max * 100,
                                (mu_r + sd_r) / y_max * 100,
                                color=st["color"], alpha=0.12, zorder=st["zorder"] - 1)

            # Initial point
            if init_idx:
                i0 = init_idx[0]
                y0 = mu[i0] / y_max * 100
                if rec_idx:
                    ax.plot([0, xs[rec_idx[0]]],
                            [y0, mu[rec_idx[0]] / y_max * 100],
                            color=st["color"], lw=1.0, ls=":", alpha=0.5,
                            zorder=st["zorder"] - 1)
                ax.plot(0, y0,
                        marker=st["marker"], markersize=8,
                        color=st["color"], markeredgecolor="white", markeredgewidth=1.2,
                        zorder=st["zorder"] + 1, linestyle="none")

        # 95% line
        ax.axhline(95, color="#94A3B8", lw=1.0, ls="--", zorder=1)
        ax.text(21, 95.8, "95%", fontsize=7, color="#94A3B8", ha="right")

        # Initial separator
        ax.axvspan(-0.6, 1.4, color="#F1F5F9", zorder=0)
        ax.axvline(1.4, color="#94A3B8", lw=1.2, ls="-", zorder=2, alpha=0.5)
        ax.text(0.4, ax.get_ylim()[0] + 1 if ax.get_ylim()[0] < 50 else 56.5,
                "Initial\n(all same)", fontsize=6.5,
                color="#64748B", ha="center", va="bottom", style="italic")

        ax.set_title(panel_label, fontsize=10.5, fontweight="bold",
                     color="#1E293B", pad=6)
        ax.set_xlabel("Recommendation Round", fontsize=9, color="#475569")
        ax.set_ylabel("% of Global Optimum" if col == 0 else "",
                      fontsize=9, color="#475569")
        ax.set_xlim(-0.8, 21.5)
        ax.set_xticks([0, 2, 6, 11, 16, 21])
        ax.set_xticklabels(["Initial", "1", "5", "10", "15", "20"])
        ax.set_ylim(None, 103)
        ax.tick_params(labelsize=8, colors="#475569")
        ax.set_facecolor("#F8FAFC")
        ax.grid(True, color="#E2E8F0", lw=0.7, zorder=0)
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")

        if col == 2:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9,
                      edgecolor="#CBD5E1", facecolor="white")

    # ── Row 1 left: final quality bar chart ──────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :2])
    ax_bar.set_facecolor("#F8FAFC")

    ds_labels   = [lbl.replace("\n", " ") for _, lbl in DATASETS]
    n_algos = len(ALGO_ORDER)
    x = np.arange(len(DATASETS))
    width   = 0.18
    offsets = np.linspace(-(n_algos-1)/2, (n_algos-1)/2, n_algos) * width

    for i, algo in enumerate(ALGO_ORDER):
        st = ALGO_STYLES[algo]
        means, stds = [], []
        for fname, _ in DATASETS:
            df = dfs[fname]
            vals = _pct_optimal(df, algo)
            means.append(vals.mean()); stds.append(vals.std())
        bars = ax_bar.bar(x + offsets[i], means, width=width,
                          color=st["color"], alpha=0.85, label=algo,
                          yerr=stds, capsize=3,
                          error_kw={"ecolor": st["color"], "alpha": 0.7, "lw": 1.2},
                          zorder=3)
        if algo == "GLOSS":
            for bar, m in zip(bars, means):
                ax_bar.text(bar.get_x() + bar.get_width()/2, m + 0.6,
                            f"{m:.1f}%", ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold", color=st["color"])

    ax_bar.axhline(95, color="#94A3B8", lw=1.0, ls="--", zorder=1)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ds_labels, fontsize=9)
    ax_bar.set_ylabel("Final Best (% of Optimum)", fontsize=9, color="#475569")
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title("Final Optimisation Quality", fontsize=11,
                     fontweight="bold", color="#1E293B", pad=6)
    ax_bar.legend(fontsize=8.5, loc="lower right", framealpha=0.9,
                  edgecolor="#CBD5E1", facecolor="white")
    ax_bar.tick_params(labelsize=8.5, colors="#475569")
    ax_bar.grid(True, axis="y", color="#E2E8F0", lw=0.7, zorder=0)
    for sp in ax_bar.spines.values():
        sp.set_color("#CBD5E1")

    # ── Row 1 right: rounds-to-95% bar chart ─────────────────────────────
    ax_spd = fig.add_subplot(gs[1, 2])
    ax_spd.set_facecolor("#F8FAFC")

    x1 = np.arange(len(DATASETS))

    for i, algo in enumerate(ALGO_ORDER):
        st  = ALGO_STYLES[algo]
        r95 = []
        for fname, _ in DATASETS:
            df = dfs[fname]
            r95.append(_rounds_to_95(df, algo))

        r95_plot = [v if not np.isnan(v) else 22 for v in r95]
        nan_mask = [np.isnan(v) for v in r95]

        bars = ax_spd.bar(x1 + offsets[i], r95_plot, width=width,
                          color=st["color"], alpha=0.85, label=algo, zorder=3)
        for bar, is_nan in zip(bars, nan_mask):
            if is_nan:
                ax_spd.text(bar.get_x() + bar.get_width()/2, 22.3, "✗",
                            ha="center", va="bottom", fontsize=9,
                            color=st["color"], fontweight="bold")
        if algo == "GLOSS":
            for bar, v, is_nan in zip(bars, r95, nan_mask):
                if not is_nan:
                    ax_spd.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.3,
                                f"{v:.1f}", ha="center", va="bottom",
                                fontsize=7.5, fontweight="bold", color=st["color"])

    ax_spd.set_xticks(x1)
    ax_spd.set_xticklabels(ds_labels, fontsize=8.5)
    ax_spd.set_ylabel("Rounds to Reach 95% of Optimum", fontsize=9, color="#475569")
    ax_spd.set_ylim(0, 25)
    ax_spd.set_title("Convergence Speed (fewer = better)", fontsize=11,
                     fontweight="bold", color="#1E293B", pad=6)
    ax_spd.legend(fontsize=8, loc="upper left", framealpha=0.9,
                  edgecolor="#CBD5E1", facecolor="white")
    ax_spd.tick_params(labelsize=8.5, colors="#475569")
    ax_spd.grid(True, axis="y", color="#E2E8F0", lw=0.7, zorder=0)
    ax_spd.text(2.6, 22.8, "✗ = never reached 95%",
                fontsize=7.5, color="#94A3B8", ha="right")
    for sp in ax_spd.spines.values():
        sp.set_color("#CBD5E1")

    # ── Footer ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.012,
             "Initial = shared starting point (8 random observations, identical across all algorithms within each seed)  ·  "
             "Round 1–20 = after each recommendation batch  ·  "
             "Shaded bands = ±1 std (5 seeds)  ·  GLOSS: global_best points only",
             ha="center", fontsize=7.5, color="#94A3B8")

    out = os.path.join(PLOTS_DIR, "extended_benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    make_figure()
    import subprocess, sys
    if sys.platform == "darwin":
        out = os.path.join(PLOTS_DIR, "extended_benchmark.png")
        subprocess.run(["open", out])
