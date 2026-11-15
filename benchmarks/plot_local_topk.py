"""Plot S4 ablation results: per-round runtime and quality vs top-K."""

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


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bench_local_topk_summary.csv"))

    SEEDS = [47, 53, 59, 61, 67]
    K_LABELS = {100: "100", 250: "250", 500: "500",
                1000: "1000", 2000: "2000", 0: "full O(n)"}
    KS = [100, 250, 500, 1000, 2000, 0]

    # Build summary table
    rows = []
    for k in KS:
        sub = df[df["top_k"] == k]
        per_round = sub[sub["round"] >= 0]
        mean_time = per_round["time_s"].mean()
        std_time = per_round["time_s"].std()
        seeds_t95 = []
        seeds_final = []
        for seed in SEEDS:
            ssub = per_round[per_round["seed"] == seed]
            if len(ssub) == 0:
                continue
            r = ssub[ssub["pct_opt"] >= 95]
            seeds_t95.append(int(r["round"].min()) + 1 if len(r) > 0 else 21)
            seeds_final.append(float(ssub[ssub["round"] == 19]["pct_opt"].iloc[0]))
        rows.append(dict(K=k, label=K_LABELS[k],
                         mean_time=mean_time, std_time=std_time,
                         t95_mean=np.mean(seeds_t95) if seeds_t95 else np.nan,
                         t95_std=np.std(seeds_t95) if seeds_t95 else np.nan,
                         reach_95=sum(1 for t in seeds_t95 if t < 21),
                         n_seeds=len(seeds_t95),
                         final_mean=np.mean(seeds_final) if seeds_final else np.nan))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "bench_local_topk_table.csv"), index=False)
    print(summary.to_string(index=False))

    # Two-panel figure: runtime vs K, t95 vs K
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # x-axis: K with full O(n) at the right end (use n_pool = 100k for plot)
    x_vals = []
    x_labels = []
    for r in rows:
        x_vals.append(r["K"] if r["K"] > 0 else 100000)
        x_labels.append(r["label"])

    # Panel A: runtime
    ax = axes[0]
    ax.errorbar(x_vals, summary["mean_time"], yerr=summary["std_time"],
                marker="o", color="#2166ac", lw=1.6, ms=6, capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Local-best top-$K$ candidates", fontsize=10)
    ax.set_ylabel("Mean per-round wall-clock time (s)", fontsize=10)
    ax.set_title("A   Runtime vs $K$  (QM9-gap, $n=100$k)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=10)
    for x, y, lbl in zip(x_vals, summary["mean_time"], x_labels):
        ax.annotate(f"{y:.1f}s", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", fontsize=10, ha="center")

    # Panel B: t95 vs K
    ax = axes[1]
    ax.errorbar(x_vals, summary["t95_mean"], yerr=summary["t95_std"],
                marker="s", color="#b2182b", lw=1.6, ms=6, capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Local-best top-$K$ candidates", fontsize=10)
    ax.set_ylabel("Mean $t_{95}$ (rounds)", fontsize=10)
    ax.set_title("B   Quality vs $K$  ($t_{95}$ over 5 seeds)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.axhline(8.2, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.text(1.5e4, 8.45, "main paper $t_{95}=8.2$", fontsize=10, color="gray")
    ax.tick_params(labelsize=10)
    ax.set_ylim(0, max(summary["t95_mean"].max() + 2, 10))

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "si_fig_local_topk.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
