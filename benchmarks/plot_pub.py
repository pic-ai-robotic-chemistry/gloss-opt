"""Publication-quality plots for pub_benchmark results.

Usage:
    python -m benchmarks.plot_pub

Outputs (in benchmarks/plots/):
  pub_fig1_convergence.png  — convergence curves (main study)
  pub_fig2_diversity.png    — diversity comparison
  pub_fig3_scaling.png      — GLOSS advantage vs space size
  pub_fig4_complexity.png   — GLOSS advantage vs complexity
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")

ALGO_COLORS = {
    "GLOSS":  "#2166ac",
    "BO":     "#d6604d",
    "GA":     "#4dac26",
    "Random": "#969696",
}
ALGO_LINES = {
    "GLOSS":  "-",
    "BO":     "--",
    "GA":     "-.",
    "Random": ":",
}
ALGO_ORDER = ["GLOSS", "BO", "GA", "Random"]


def load_main():
    path = os.path.join(RESULTS_DIR, "pub_main_results.csv")
    return pd.read_csv(path)


def _t95(group, y_opt_pct=95.0):
    """Round where pct_opt first reaches 95% of optimum; 20 if never."""
    above = group[group["pct_opt"] >= y_opt_pct]
    if len(above) == 0:
        return 20
    return int(above["round"].min())


def fig1_convergence(df):
    """Figure 1: mean ± std convergence curves for each main dataset."""
    datasets = df["dataset"].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, dsname in zip(axes, datasets):
        sub = df[df["dataset"] == dsname]
        rounds = sorted(sub["round"].unique())

        for algo in ALGO_ORDER:
            asub = sub[sub["algorithm"] == algo]
            means, stds = [], []
            for r in rounds:
                vals = asub[asub["round"] == r]["pct_opt"].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means, stds = np.array(means), np.array(stds)
            ax.plot(rounds, means,
                    color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                    lw=2, label=algo)
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.12, color=ALGO_COLORS[algo])

        ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% target")
        ax.set_xlabel("Optimization round")
        ax.set_ylabel("% of optimum")
        ax.set_title(dsname)
        ax.legend(fontsize=8)
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence: GLOSS(4:2:2) vs Baselines (RF surrogate)", fontsize=12)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "pub_fig1_convergence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig2_diversity(df):
    """Figure 2: mean diversity (mean pairwise dist) per algorithm per dataset."""
    final_round = df["round"].max()
    final = df[df["round"] == final_round]
    datasets = df["dataset"].unique()
    n = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, dsname in zip(axes, datasets):
        sub = final[final["dataset"] == dsname]
        means, errs, labels = [], [], []
        for algo in ALGO_ORDER:
            vals = sub[sub["algorithm"] == algo]["diversity"].values
            if len(vals) == 0:
                continue
            means.append(np.mean(vals))
            errs.append(np.std(vals))
            labels.append(algo)

        colors = [ALGO_COLORS[a] for a in labels]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=errs, color=colors,
               capsize=4, width=0.6, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel("Mean pairwise distance")
        ax.set_title(dsname)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Sampling Diversity: Mean Pairwise Distance of Sampled Points", fontsize=11)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pub_fig2_diversity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig3_scaling():
    """Figure 3: GLOSS speed advantage (rounds-to-95%) vs space size."""
    sizes = [2000, 5000, 10000, 20000]
    data = {}
    for n in sizes:
        path = os.path.join(RESULTS_DIR, f"pub_scaling_{n}_results.csv")
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        data[n] = pd.read_csv(path)

    if not data:
        print("No scaling data found, skipping Figure 3")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo in ALGO_ORDER:
        t95s, ns_available = [], []
        for n, df in sorted(data.items()):
            grp = df[df["algorithm"] == algo]
            seed_t95s = []
            for seed in grp["seed"].unique():
                sg = grp[grp["seed"] == seed].sort_values("round")
                seed_t95s.append(_t95(sg))
            t95s.append(np.mean(seed_t95s))
            ns_available.append(n)

        ax.plot(ns_available, t95s,
                color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                marker="o", lw=2, label=algo)

    ax.set_xlabel("Chemical space size (# candidates)")
    ax.set_ylabel("Rounds to 95% optimum (lower=better)")
    ax.set_title("GLOSS Advantage Grows with Chemical Space Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pub_fig3_scaling.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig4_complexity():
    """Figure 4: GLOSS advantage vs landscape complexity (nD functions)."""
    path = os.path.join(RESULTS_DIR, "pub_complexity_results.csv")
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return

    df = pd.read_csv(path)
    complexity_names = ["Simple-3D", "Medium-5D", "Complex-8D"]
    complexity_levels = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo in ALGO_ORDER:
        t95s = []
        for dsname in complexity_names:
            sub = df[(df["dataset"] == dsname) & (df["algorithm"] == algo)]
            seed_t95s = []
            for seed in sub["seed"].unique():
                sg = sub[sub["seed"] == seed].sort_values("round")
                seed_t95s.append(_t95(sg))
            t95s.append(np.mean(seed_t95s) if seed_t95s else 20)

        ax.plot(complexity_levels, t95s,
                color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                marker="s", lw=2, label=algo)

    ax.set_xticks(complexity_levels)
    ax.set_xticklabels(["Simple\n(3D, 1 peak)", "Medium\n(5D, 3 peaks)", "Complex\n(8D, 5 peaks)"])
    ax.set_xlabel("Landscape complexity")
    ax.set_ylabel("Rounds to 95% optimum (lower=better)")
    ax.set_title("GLOSS Advantage Grows with Landscape Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pub_fig4_complexity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig5_scaling_convergence():
    """Figure 5: convergence curves (fig1-style) for scaling study (n=2000/10000/20000)."""
    sizes = [2000, 10000, 20000]
    dfs = {}
    for n in sizes:
        path = os.path.join(RESULTS_DIR, f"pub_scaling_{n}_results.csv")
        if os.path.exists(path):
            dfs[n] = pd.read_csv(path)
    if not dfs:
        print("No scaling data found, skipping Figure 5")
        return

    n_panels = len(dfs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (n, df) in zip(axes, sorted(dfs.items())):
        rounds = sorted(df["round"].unique())
        for algo in ALGO_ORDER:
            asub = df[df["algorithm"] == algo]
            means, stds = [], []
            for r in rounds:
                vals = asub[asub["round"] == r]["pct_opt"].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means, stds = np.array(means), np.array(stds)
            ax.plot(rounds, means,
                    color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                    lw=2, label=algo)
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.12, color=ALGO_COLORS[algo])

        ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% target")
        ax.set_xlabel("Optimization round")
        ax.set_ylabel("% of optimum")
        ax.set_title(f"QM9-gap  (n = {n:,})")
        ax.legend(fontsize=8)
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Scaling Study: Convergence at Different Chemical Space Sizes (RF surrogate)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pub_fig5_scaling_convergence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig6_complexity_convergence():
    """Figure 6: convergence curves (fig1-style) for complexity study."""
    path = os.path.join(RESULTS_DIR, "pub_complexity_results.csv")
    if not os.path.exists(path):
        print(f"Missing: {path}, skipping Figure 6")
        return

    df = pd.read_csv(path)
    datasets = ["Simple-3D", "Medium-5D", "Complex-8D"]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)

    for ax, dsname in zip(axes, datasets):
        sub = df[df["dataset"] == dsname]
        rounds = sorted(sub["round"].unique())
        for algo in ALGO_ORDER:
            asub = sub[sub["algorithm"] == algo]
            means, stds = [], []
            for r in rounds:
                vals = asub[asub["round"] == r]["pct_opt"].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means, stds = np.array(means), np.array(stds)
            ax.plot(rounds, means,
                    color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                    lw=2, label=algo)
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.12, color=ALGO_COLORS[algo])

        ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% target")
        ax.set_xlabel("Optimization round")
        ax.set_ylabel("% of optimum")
        ax.set_title(dsname)
        ax.legend(fontsize=8)
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Complexity Study: Convergence on Virtual 2D Functions (RF surrogate)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pub_fig6_complexity_convergence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig7_scaling_convergence4():
    """Figure 7: convergence curves for n=5000/10000/15000/20000 (4-panel scaling)."""
    sizes = [5000, 10000, 15000, 20000]
    dfs = {}
    for n in sizes:
        path = os.path.join(RESULTS_DIR, f"pub_scaling_{n}_results.csv")
        if os.path.exists(path):
            dfs[n] = pd.read_csv(path)
        else:
            print(f"  Missing: {path}")
    if not dfs:
        print("No scaling data found, skipping Figure 7")
        return

    n_panels = len(dfs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (n, df) in zip(axes, sorted(dfs.items())):
        rounds = sorted(df["round"].unique())
        for algo in ALGO_ORDER:
            asub = df[df["algorithm"] == algo]
            means, stds = [], []
            for r in rounds:
                vals = asub[asub["round"] == r]["pct_opt"].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means, stds = np.array(means), np.array(stds)
            ax.plot(rounds, means,
                    color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                    lw=2, label=algo)
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.12, color=ALGO_COLORS[algo])

        ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% target")
        ax.set_xlabel("Optimization round")
        ax.set_ylabel("% of optimum")
        ax.set_title(f"QM9-gap  (n = {n:,})")
        ax.legend(fontsize=8)
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("GLOSS Advantage Grows with Chemical Space Size (n = 5k → 20k)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pub_fig7_scaling4.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(df):
    """Print t95 summary table."""
    print("\n=== t95 (rounds to 95% optimum) — lower is better ===")
    rows = []
    for dsname in df["dataset"].unique():
        for algo in ALGO_ORDER:
            sub = df[(df["dataset"] == dsname) & (df["algorithm"] == algo)]
            seed_t95s = []
            for seed in sub["seed"].unique():
                sg = sub[sub["seed"] == seed].sort_values("round")
                seed_t95s.append(_t95(sg))
            rows.append({
                "dataset": dsname, "algorithm": algo,
                "t95_mean": np.mean(seed_t95s),
                "t95_std":  np.std(seed_t95s),
                "n_success": sum(1 for t in seed_t95s if t < 20),
            })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    print("\n=== Final diversity (mean pairwise dist) ===")
    final_round = df["round"].max()
    final = df[df["round"] == final_round]
    div = final.groupby(["dataset", "algorithm"])["diversity"].mean().round(4)
    print(div.to_string())


def fig8_chembl_egfr():
    """Figure 8: convergence curves for ChEMBL EGFR benchmark (init from bottom 50%)."""
    path = os.path.join(RESULTS_DIR, "pub_chembl_egfr_results.csv")
    if not os.path.exists(path):
        print(f"Skipping fig8_egfr: {path} not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    rounds = sorted(df["round"].unique())

    for algo in ALGO_ORDER:
        asub = df[df["algorithm"] == algo]
        means, stds = [], []
        for r in rounds:
            vals = asub[asub["round"] == r]["pct_opt"].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        ax.plot(rounds, means,
                color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                lw=2, label=algo)
        ax.fill_between(rounds, means - stds, means + stds,
                        alpha=0.12, color=ALGO_COLORS[algo])

    ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% target")
    ax.set_xlabel("Optimization round")
    ax.set_ylabel("% of optimum (pIC50)")
    ax.set_title("ChEMBL-EGFR (10.8k compounds, init from bottom 50%)")
    ax.legend(fontsize=9)
    ax.set_xlim(left=-1)
    ax.set_ylim(55, 102)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "pub_fig8_chembl_egfr.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig8_zinc_sas():
    """Figure 8: convergence curves for ZINC-SAS benchmark (init from bottom 50%)."""
    path = os.path.join(RESULTS_DIR, "pub_zinc_sas_results.csv")
    if not os.path.exists(path):
        print(f"Skipping fig8: {path} not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    rounds = sorted(df["round"].unique())

    for algo in ALGO_ORDER:
        asub = df[df["algorithm"] == algo]
        means, stds = [], []
        for r in rounds:
            vals = asub[asub["round"] == r]["pct_opt"].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        ax.plot(rounds, means,
                color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                lw=2, label=algo)
        ax.fill_between(rounds, means - stds, means + stds,
                        alpha=0.12, color=ALGO_COLORS[algo])

    ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% target")
    ax.set_xlabel("Optimization round")
    ax.set_ylabel("% of optimum (10 − SAS)")
    ax.set_title("ZINC-SAS (20k compounds, init from bottom 50%)")
    ax.legend(fontsize=9)
    ax.set_xlim(left=-1)
    ax.set_ylim(70, 102)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "pub_fig8_zinc_sas.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig8_chembl_drd2():
    """Figure 8: convergence curves for ChEMBL DRD2 benchmark (init from bottom 50%)."""
    path = os.path.join(RESULTS_DIR, "pub_chembl_drd2_results.csv")
    if not os.path.exists(path):
        print(f"Skipping fig8: {path} not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    rounds = sorted(df["round"].unique())

    for algo in ALGO_ORDER:
        asub = df[df["algorithm"] == algo]
        means, stds = [], []
        for r in rounds:
            vals = asub[asub["round"] == r]["pct_opt"].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        ax.plot(rounds, means,
                color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                lw=2, label=algo)
        ax.fill_between(rounds, means - stds, means + stds,
                        alpha=0.12, color=ALGO_COLORS[algo])

    ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% target")
    ax.set_xlabel("Optimization round")
    ax.set_ylabel("% of optimum (pIC50)")
    ax.set_title("ChEMBL-DRD2 (9k compounds, init from bottom 50%)")
    ax.legend(fontsize=9)
    ax.set_xlim(left=-1)
    ax.set_ylim(60, 102)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "pub_fig8_chembl_drd2.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig9_alchemy():
    """Figure 9: convergence curves for Alchemy-gap benchmark (DFT-computed, non-QM9)."""
    path = os.path.join(RESULTS_DIR, "pub_alchemy_results.csv")
    if not os.path.exists(path):
        print(f"Skipping fig9_alchemy: {path} not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    rounds = sorted(df["round"].unique())

    for algo in ALGO_ORDER:
        asub = df[df["algorithm"] == algo]
        means, stds = [], []
        for r in rounds:
            vals = asub[asub["round"] == r]["pct_opt"].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        ax.plot(rounds, means,
                color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                lw=2, label=algo)
        ax.fill_between(rounds, means - stds, means + stds,
                        alpha=0.12, color=ALGO_COLORS[algo])

    ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% target")
    ax.set_xlabel("Optimization round")
    ax.set_ylabel("% of optimum (HOMO-LUMO gap)")
    ax.set_title("Alchemy-gap (20k molecules, DFT B3LYP/cc-pVDZ)")
    ax.legend(fontsize=9)
    ax.set_xlim(left=-1)
    ax.set_ylim(55, 102)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "pub_fig9_alchemy.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig10_alchemy_scaling():
    """Figure 10: GLOSS advantage vs Alchemy chemical space size (t95 summary)."""
    sizes = [2000, 5000, 10000, 20000]
    data = {}
    for n in sizes:
        path = os.path.join(RESULTS_DIR, f"pub_alchemy_scaling_{n}_results.csv")
        if os.path.exists(path):
            data[n] = pd.read_csv(path)
        else:
            print(f"  Missing: {path}")

    if not data:
        print("No Alchemy scaling data found, skipping Figure 10")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo in ALGO_ORDER:
        t95s, ns_available = [], []
        for n, df in sorted(data.items()):
            grp = df[df["algorithm"] == algo]
            seed_t95s = []
            for seed in grp["seed"].unique():
                sg = grp[grp["seed"] == seed].sort_values("round")
                seed_t95s.append(_t95(sg))
            t95s.append(np.mean(seed_t95s))
            ns_available.append(n)

        ax.plot(ns_available, t95s,
                color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                marker="o", lw=2, label=algo)

    ax.set_xlabel("Chemical space size (# candidates)")
    ax.set_ylabel("Rounds to 95% optimum (lower=better)")
    ax.set_title("GLOSS Scaling on Alchemy-gap (DFT B3LYP/cc-pVDZ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "pub_fig10_alchemy_scaling.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig11_alchemy_scaling_convergence():
    """Figure 11: convergence curves for Alchemy scaling (n=2000/10000/20000)."""
    sizes = [2000, 10000, 20000]
    dfs = {}
    for n in sizes:
        path = os.path.join(RESULTS_DIR, f"pub_alchemy_scaling_{n}_results.csv")
        if os.path.exists(path):
            dfs[n] = pd.read_csv(path)
        else:
            print(f"  Missing: {path}")

    if not dfs:
        print("No Alchemy scaling data found, skipping Figure 11")
        return

    n_panels = len(dfs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (n, df) in zip(axes, sorted(dfs.items())):
        rounds = sorted(df["round"].unique())
        for algo in ALGO_ORDER:
            asub = df[df["algorithm"] == algo]
            means, stds = [], []
            for r in rounds:
                vals = asub[asub["round"] == r]["pct_opt"].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means, stds = np.array(means), np.array(stds)
            ax.plot(rounds, means,
                    color=ALGO_COLORS[algo], ls=ALGO_LINES[algo],
                    lw=2, label=algo)
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.12, color=ALGO_COLORS[algo])

        ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% target")
        ax.set_xlabel("Optimization round")
        ax.set_ylabel("% of optimum")
        ax.set_title(f"Alchemy-gap  (n = {n:,})")
        ax.legend(fontsize=8)
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Alchemy-gap Scaling: GLOSS Advantage at Different Chemical Space Sizes", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pub_fig11_alchemy_scaling_convergence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


RATIO_COLORS = {
    "GLOSS(2:3:3)": "#d6604d",
    "GLOSS(4:2:2)": "#2166ac",
    "GLOSS(6:1:1)": "#4dac26",
}
RATIO_LINES = {
    "GLOSS(2:3:3)": "-.",
    "GLOSS(4:2:2)": "-",
    "GLOSS(6:1:1)": "--",
}
RATIO_ORDER = ["GLOSS(2:3:3)", "GLOSS(4:2:2)", "GLOSS(6:1:1)"]


def fig12_ratio_scaling():
    """Figure 12: GLOSS strategy ratio comparison across chemical space sizes."""
    sizes = [2000, 5000, 10000, 20000]
    data = {}
    for n in sizes:
        path = os.path.join(RESULTS_DIR, f"pub_ratio_scaling_{n}_results.csv")
        if os.path.exists(path):
            data[n] = pd.read_csv(path)
        else:
            print(f"  Missing: {path}")
    if not data:
        print("No ratio-scaling data found, skipping Figure 12")
        return

    # Panel 1: t95 vs space size
    # Panel 2-5: convergence curves at each size
    n_panels = 1 + len(data)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))

    # Left panel: t95 summary
    ax = axes[0]
    for algo in RATIO_ORDER:
        t95s, ns_available = [], []
        for n, df in sorted(data.items()):
            grp = df[df["algorithm"] == algo]
            seed_t95s = []
            for seed in grp["seed"].unique():
                sg = grp[grp["seed"] == seed].sort_values("round")
                seed_t95s.append(_t95(sg))
            t95s.append(np.mean(seed_t95s))
            ns_available.append(n)
        ax.plot(ns_available, t95s,
                color=RATIO_COLORS[algo], ls=RATIO_LINES[algo],
                marker="o", lw=2, label=algo)
    ax.set_xlabel("Chemical space size")
    ax.set_ylabel("Rounds to 95% (lower=better)")
    ax.set_title("t95 vs Space Size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Convergence panels
    for i, (n, df) in enumerate(sorted(data.items())):
        ax = axes[i + 1]
        rounds = sorted(df["round"].unique())
        for algo in RATIO_ORDER:
            asub = df[df["algorithm"] == algo]
            means, stds = [], []
            for r in rounds:
                vals = asub[asub["round"] == r]["pct_opt"].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means, stds = np.array(means), np.array(stds)
            ax.plot(rounds, means,
                    color=RATIO_COLORS[algo], ls=RATIO_LINES[algo],
                    lw=2, label=algo)
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.12, color=RATIO_COLORS[algo])
        ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("Round")
        ax.set_ylabel("% of optimum")
        ax.set_title(f"QM9-gap (n={n:,})")
        ax.legend(fontsize=7)
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Strategy Ratio Comparison: Scaling with Chemical Space Size", fontsize=12)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "pub_fig12_ratio_scaling.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig13_ratio_complexity():
    """Figure 13: GLOSS strategy ratio comparison across landscape complexity."""
    path = os.path.join(RESULTS_DIR, "pub_ratio_complexity_results.csv")
    if not os.path.exists(path):
        print(f"Missing: {path}, skipping Figure 13")
        return
    df = pd.read_csv(path)

    complexity_names = ["MultiCluster-3C-8D", "MultiCluster-5C-8D", "MultiCluster-7C-8D"]

    # Panel 1: t95 summary; Panels 2-4: convergence curves
    n_panels = 1 + len(complexity_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))

    # Left panel: t95 vs complexity
    ax = axes[0]
    for algo in RATIO_ORDER:
        t95s = []
        for dsname in complexity_names:
            sub = df[(df["dataset"] == dsname) & (df["algorithm"] == algo)]
            seed_t95s = []
            for seed in sub["seed"].unique():
                sg = sub[sub["seed"] == seed].sort_values("round")
                seed_t95s.append(_t95(sg))
            t95s.append(np.mean(seed_t95s) if seed_t95s else 20)
        ax.plot([1, 2, 3], t95s,
                color=RATIO_COLORS[algo], ls=RATIO_LINES[algo],
                marker="s", lw=2, label=algo)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["3 clusters", "5 clusters", "7 clusters"])
    ax.set_xlabel("Landscape complexity")
    ax.set_ylabel("Rounds to 95% (lower=better)")
    ax.set_title("t95 vs Complexity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Convergence panels
    for i, dsname in enumerate(complexity_names):
        ax = axes[i + 1]
        sub = df[df["dataset"] == dsname]
        rounds = sorted(sub["round"].unique())
        for algo in RATIO_ORDER:
            asub = sub[sub["algorithm"] == algo]
            means, stds = [], []
            for r in rounds:
                vals = asub[asub["round"] == r]["pct_opt"].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means, stds = np.array(means), np.array(stds)
            ax.plot(rounds, means,
                    color=RATIO_COLORS[algo], ls=RATIO_LINES[algo],
                    lw=2, label=algo)
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.12, color=RATIO_COLORS[algo])
        ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("Round")
        ax.set_ylabel("% of optimum")
        ax.set_title(dsname)
        ax.legend(fontsize=7)
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Strategy Ratio Comparison: Landscape Complexity", fontsize=12)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "pub_fig13_ratio_complexity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Loading main study results...")
    df = load_main()
    print_summary(df)
    fig1_convergence(df)
    fig2_diversity(df)
    fig3_scaling()
    fig4_complexity()
    fig5_scaling_convergence()
    fig6_complexity_convergence()
    fig7_scaling_convergence4()
    fig8_chembl_drd2()
    fig9_alchemy()
    fig10_alchemy_scaling()
    fig11_alchemy_scaling_convergence()
    fig12_ratio_scaling()
    fig13_ratio_complexity()
    print("\nAll plots saved to benchmarks/plots/")
