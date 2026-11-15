"""Plot S5 ablation results: κ-schedule + bandit comparison."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Arial"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFIGS = ["fixed_static", "decay_static", "cosine_static",
           "fixed_bandit", "cosine_bandit"]
COLORS = {"fixed_static":  "#2166ac",
          "decay_static":  "#5aae61",
          "cosine_static": "#9b59b6",
          "fixed_bandit":  "#d6604d",
          "cosine_bandit": "#b2182b"}
LABELS = {"fixed_static":  "fixed κ (paper default)",
          "decay_static":  "geometric decay κ",
          "cosine_static": "cosine decay κ",
          "fixed_bandit":  "fixed κ + UCB-1 bandit",
          "cosine_bandit": "cosine κ + UCB-1 bandit"}
SEEDS = [47, 53, 59, 61, 67]


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_kappa_bandit_summary.csv"))
    datasets = ["QM9-gap", "Arrhenius-2D"]

    # Console summary table
    rows = []
    for cfg in CONFIGS:
        for ds in datasets:
            sub = df[(df["config"] == cfg) & (df["dataset"] == ds)]
            seeds_t95 = []
            for seed in SEEDS:
                ssub = sub[sub["seed"] == seed]
                r = ssub[ssub["pct_opt"] >= 95]
                seeds_t95.append(int(r["round"].min()) if len(r) > 0 else 20)
            rows.append(dict(config=cfg, dataset=ds,
                             mean_t95=np.mean(seeds_t95),
                             std_t95=np.std(seeds_t95),
                             reach=sum(1 for t in seeds_t95 if t < 20)))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "bench_kappa_bandit_table.csv"),
                   index=False)
    print(summary.to_string(index=False))

    # Convergence plots, one panel per dataset
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for col, ds in enumerate(datasets):
        ax = axes[col]
        sub = df[df["dataset"] == ds]
        rounds = sorted(sub["round"].unique())
        for cfg in CONFIGS:
            csub = sub[sub["config"] == cfg]
            means = [csub[csub["round"] == r]["pct_opt"].mean() for r in rounds]
            stds = [csub[csub["round"] == r]["pct_opt"].std() for r in rounds]
            means, stds = np.array(means), np.array(stds)
            ax.plot(rounds, means, color=COLORS[cfg], lw=1.6, label=LABELS[cfg])
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.10, color=COLORS[cfg])
        ax.axhline(95, color="black", ls=":", lw=0.7, alpha=0.4)
        ax.set_xlim(-1, 20)
        ax.set_xlabel("Round", fontsize=10)
        if col == 0:
            ax.set_ylabel("% of optimum", fontsize=10)
        else:
            ax.tick_params(left=False, labelleft=False)
        ax.set_title(f"{chr(65+col)}   {ds}", fontsize=11,
                     fontweight="bold", loc="left")
        ax.tick_params(labelsize=10)
        if col == 1:
            ax.legend(fontsize=10, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "si_fig_kappa_bandit.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
