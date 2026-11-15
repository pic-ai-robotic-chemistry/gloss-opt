"""Benchmark visualization."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


def _ensure_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


COLORS = {"GLOSS": "#2196F3", "BO (EI)": "#FF9800", "Random": "#4CAF50"}
MARKERS = {"GLOSS": "o", "BO (EI)": "s", "Random": "^"}


def plot_best_so_far(dataset_name, all_results, n_rounds, n_init):
    """Plot best-so-far curves for all algorithms.

    Args:
        dataset_name: str, dataset name for title/filename.
        all_results: dict {algo_name: list of run dicts}, each run dict has "round_bests".
        n_rounds: number of optimization rounds.
        n_init: number of initial points.
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))

    rounds = np.arange(n_rounds)
    total_samples = n_init + (rounds + 1) * 4  # 4 points per round

    for algo_name, runs in all_results.items():
        # Collect round_bests across seeds
        curves = []
        for run in runs:
            rb = run["round_bests"]
            if len(rb) >= n_rounds:
                curves.append(rb[:n_rounds])
        if not curves:
            continue

        curves = np.array(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)

        color = COLORS.get(algo_name, "#999")
        marker = MARKERS.get(algo_name, ".")

        ax.plot(total_samples, mean, color=color, marker=marker,
                markevery=max(1, n_rounds // 10), label=algo_name, linewidth=2)
        ax.fill_between(total_samples, mean - std, mean + std,
                        alpha=0.15, color=color)

    ax.set_xlabel("Total Samples", fontsize=12)
    ax.set_ylabel("Best Observed Value", fontsize=12)
    ax.set_title(f"Best-so-far: {dataset_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, f"{dataset_name.lower().replace(' ', '_')}_best_so_far.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_table(summary_df):
    """Render summary metrics as a table figure."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(12, len(summary_df) * 0.5 + 2))
    ax.axis("off")

    col_labels = summary_df.columns.tolist()
    cell_text = []
    for _, row in summary_df.iterrows():
        cell_text.append([str(v) for v in row.values])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(weight="bold")

    plt.title("Benchmark Summary", fontsize=14, pad=20)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_virtual_function_surface():
    """Plot the virtual chemical yield function as a 3D surface."""
    _ensure_dir()
    from benchmarks.virtual_function import chemical_yield

    T = np.linspace(100, 300, 200)
    C = np.linspace(0.1, 1.0, 200)
    TT, CC = np.meshgrid(T, C)
    X = np.column_stack([TT.ravel(), CC.ravel()])
    Y = chemical_yield(X).reshape(TT.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(TT, CC, Y, cmap="viridis", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Catalyst Conc. (mol/L)")
    ax.set_zlabel("Yield (%)")
    ax.set_title("Virtual Chemical Reaction Yield Surface")

    path = os.path.join(PLOT_DIR, "virtual_function_surface.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
