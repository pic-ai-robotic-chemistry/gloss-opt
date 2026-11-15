"""Comprehensive benchmark visualization."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

COLORS = {
    "GLOSS": "#1565C0",
    "BO (EI)": "#E65100",
    "Random (GP)": "#7B1FA2",
    "Random": "#2E7D32",
}
MARKERS = {"GLOSS": "o", "BO (EI)": "s", "Random (GP)": "D", "Random": "^"}
LINESTYLES = {"GLOSS": "-", "BO (EI)": "-", "Random (GP)": "--", "Random": ":"}
ALGO_ORDER = ["GLOSS", "BO (EI)", "Random (GP)", "Random"]
N_INIT = 8
PTS_PER_ROUND = 8


def _style(name):
    return dict(
        color=COLORS.get(name, "#666"),
        marker=MARKERS.get(name, "."),
        linestyle=LINESTYLES.get(name, "-"),
        linewidth=2.2,
        markersize=5,
    )


# ──────────────────────────────────────────────────────────────
# 1. Per-dataset best-so-far curves
# ──────────────────────────────────────────────────────────────
def _plot_curve(ax, ds_name, all_results, n_rounds):
    for algo_name in ALGO_ORDER:
        if algo_name not in all_results:
            continue
        runs = all_results[algo_name]
        curves = [r["round_bests"][:n_rounds] for r in runs if len(r["round_bests"]) >= n_rounds]
        if not curves:
            continue
        curves = np.array(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        x = N_INIT + (np.arange(n_rounds) + 1) * PTS_PER_ROUND

        st = _style(algo_name)
        ax.plot(x, mean, label=algo_name, markevery=max(1, n_rounds // 8), **st)
        ax.fill_between(x, mean - std, mean + std, alpha=0.10, color=st["color"])

    ax.set_xlabel("Total Samples", fontsize=11)
    ax.set_ylabel("Best Observed Value", fontsize=11)
    ax.set_title(ds_name, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.annotate("GLOSS: global_best only", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=7, color="#888", style="italic")


def save_individual_curves(all_dataset_results, n_rounds=20):
    for ds_name, results in all_dataset_results.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _plot_curve(ax, ds_name, results, n_rounds)
        plt.tight_layout()
        fname = ds_name.lower().replace(" ", "_").replace("-", "_")
        fig.savefig(os.path.join(PLOT_DIR, f"{fname}_best_so_far.png"), dpi=180)
        plt.close(fig)
    print("  Saved individual curves")


# ──────────────────────────────────────────────────────────────
# 2. Combined multi-panel iteration curves
# ──────────────────────────────────────────────────────────────
def save_combined_figure(all_dataset_results, n_rounds=20):
    ds_names = list(all_dataset_results.keys())
    n = len(ds_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for i, ds_name in enumerate(ds_names):
        _plot_curve(axes[i], ds_name, all_dataset_results[ds_name], n_rounds)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Best-so-far Iteration Curves — GLOSS vs BO(EI) vs Random(GP) vs Random\n"
                 f"init={N_INIT}, {PTS_PER_ROUND} pts/round, GLOSS(4:2:1:1), GP fixed params",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(PLOT_DIR, "combined_iteration_curves.png"), dpi=180)
    plt.close(fig)
    print("  Saved combined iteration curves")


# ──────────────────────────────────────────────────────────────
# 3. Summary bar charts
# ──────────────────────────────────────────────────────────────
def save_summary_bars(all_dataset_results):
    ds_names = list(all_dataset_results.keys())
    short_names = []
    for n in ds_names:
        sn = n.replace("Buchwald-Hartwig", "Buchwald").replace("Synthetic-30D", "Synth-30D")
        short_names.append(sn)

    final_vals, round95_vals, time_vals = [], [], []
    for ds_name in ds_names:
        results = all_dataset_results[ds_name]
        fv, rv, tv = {}, {}, {}
        for algo in ALGO_ORDER:
            runs = results.get(algo, [])
            finals = [r["final_best"] for r in runs if r["final_best"] is not None]
            targets = [r["target_95_round"] for r in runs if r["target_95_round"] is not None]
            times = [r["avg_time"] for r in runs]
            fv[algo] = {"m": np.mean(finals) if finals else 0, "s": np.std(finals) if finals else 0}
            rv[algo] = {"m": np.mean(targets) if targets else 20, "s": np.std(targets) if targets else 0}
            tv[algo] = {"m": np.mean(times) if times else 0, "s": np.std(times) if times else 0}
        final_vals.append(fv)
        round95_vals.append(rv)
        time_vals.append(tv)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    x = np.arange(len(ds_names))
    n_algo = len(ALGO_ORDER)
    width = 0.8 / n_algo

    for panel, (ax, vals, title, ylabel, fmt) in enumerate(zip(
        axes,
        [final_vals, round95_vals, time_vals],
        ["Final Best Value", "Rounds to 95% Optimum\n(fewer = better)", "Avg Time per Round"],
        ["Best Observed", "Rounds", "Seconds"],
        ["{:.2f}", "{:.1f}", "{:.3f}"],
    )):
        for i, algo in enumerate(ALGO_ORDER):
            means = [v[algo]["m"] for v in vals]
            stds = [v[algo]["s"] for v in vals]
            offset = (i - n_algo / 2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds,
                          label=algo, color=COLORS[algo], alpha=0.85,
                          capsize=2, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("GLOSS Benchmark Summary (GLOSS metric: global_best only)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "summary_bars.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary bars")


# ──────────────────────────────────────────────────────────────
# 4. Summary table
# ──────────────────────────────────────────────────────────────
def save_summary_table(summary_df):
    fig, ax = plt.subplots(figsize=(15, len(summary_df) * 0.4 + 2))
    ax.axis("off")

    col_labels = summary_df.columns.tolist()
    cell_text = [[str(v) for v in row.values] for _, row in summary_df.iterrows()]

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", weight="bold")

    algo_col = col_labels.index("Algorithm")
    for i in range(1, len(cell_text) + 1):
        bg = "#F5F5F5" if i % 2 == 0 else "white"
        algo = cell_text[i - 1][algo_col]
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
        if algo in COLORS:
            table[i, algo_col].set_text_props(color=COLORS[algo], weight="bold")

    ax.set_title("Benchmark Summary\n(GLOSS metric: global_best only; 8 pts/round)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "summary_table.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary table")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def plot_all(all_dataset_results, summary_df):
    print("\nGenerating plots...")
    save_individual_curves(all_dataset_results)
    save_combined_figure(all_dataset_results)
    save_summary_bars(all_dataset_results)
    save_summary_table(summary_df)
    print("All plots saved to benchmarks/plots/")
