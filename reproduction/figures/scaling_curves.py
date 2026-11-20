"""Generate scaling curves from the paper records."""
from pathlib import Path

from . import scaling

import matplotlib.pyplot as plt


def main():
    data = scaling.load_scaling_runs().copy()
    data["algorithm"] = data["algorithm"].replace(scaling.RENAME)
    data["size"] = data["dataset"].str.extract(r"-(\d+)k$")[0].astype(int)
    sizes = list(range(5, 101, 5))
    cells = ["size", "algorithm", "round"]
    counts = data.groupby(cells)["seed"].nunique()
    if (len(counts) != len(sizes) * len(scaling.ARMS) * 21
            or not counts.eq(10).all() or data.duplicated(cells + ["seed"]).any()):
        raise ValueError("Expected ten distinct seeds per size, algorithm and round")
    summary = data.groupby(cells)["pct_opt"].agg(["mean", "std"])
    fig, axes = plt.subplots(5, 4, figsize=(11, 13), sharex=True, sharey=True)
    for size, ax in zip(sizes, axes.flat):
        for arm, label, color, _ in scaling.ARMS:
            curve = summary.loc[(size, arm)]
            rounds = curve.index.to_numpy() + 1
            mean, sd = curve["mean"].to_numpy(), curve["std"].to_numpy()
            ax.fill_between(rounds, mean - sd, mean + sd, color=color,
                            alpha=0.09, linewidth=0)
            ax.plot(rounds, mean, color=color, linewidth=1.25, label=label)
        ax.axhline(95, ls="--", lw=0.7, color="black")
        ax.set_title(f"{size * 1000:,} candidates", fontsize=10)
        ax.set(xlim=(0, 20), ylim=(0, 110), xticks=[0, 5, 10, 15, 20])
        ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.1,
                        wspace=0.17, hspace=0.32)
    fig.supxlabel("Recommendation round (0 = initialization)", y=0.061, fontsize=11)
    fig.supylabel("Normalized best objective (%)", x=0.012, fontsize=11)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center",
               ncol=4, frameon=False, fontsize=9)
    output = Path(scaling.OUT) / "si_fig_scaling_convergence.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, pil_kwargs={"quality": 95})
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
