"""Trajectory visualisation: optimization path on 2D virtual functions.

Workflow (same pattern as all other benchmarks):
  1. Run benchmark, save results to benchmarks/results/trajectory_results.csv
  2. Plot from CSV — pure style changes never touch the data

For each virtual function × seed, produces one figure with 4 subplots
(GLOSS / BO / GA / Random), each showing the 2D function heatmap and the
best-point-per-round trajectory overlaid as connected scatter points.

For GLOSS the marker shape encodes the strategy:
  ★ (star)     → global_best
  ▲ (triangle) → local_best
  ● (circle)   → unexplored

Usage:
    python -m benchmarks.plot_trajectory --run        # run benchmark, save CSV, then plot
    python -m benchmarks.plot_trajectory              # plot from existing CSV
    python -m benchmarks.plot_trajectory --func medium --seeds 0 1
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots", "trajectories")
STUDY_NAME  = "trajectory"

# ── colours / markers ────────────────────────────────────────────────────────
STRATEGY_MARKER = {
    "global_best": "*",
    "local_best":  "^",
    "unexplored":  "o",
    "unconverged": "D",
    "unknown":     "s",
    "init":        "P",
    "point":       "o",
}
ALGO_COLOR = {
    "GLOSS":  "#2166ac",
    "BO":     "#d6604d",
    "GA":     "#4dac26",
    "Random": "#969696",
}


# ── benchmark runners ─────────────────────────────────────────────────────────
def _points_csv_path(func_name, algo_name, seed):
    safe = func_name.replace(" ", "_")
    return os.path.join(RESULTS_DIR, f"{STUDY_NAME}-{algo_name}-{safe}-seed{seed}.csv")


def _run_one_algo(func, bounds, func_name, algo_name, algo, seed,
                  init_X, init_y, N_INIT, N_ROUNDS, N_POINTS, GLOSS_STRATEGY,
                  GLOSS_DIVERSITY_RADIUS):
    """Run one algorithm for one seed. Returns point_rows (all points, every round)."""
    from benchmarks.rf_surrogate import make_rf
    from gloss import GLOSS

    point_rows = []

    # init: save all N_INIT points
    for i in range(len(init_X)):
        point_rows.append({
            "round": -1, "point_idx": i,
            "y": float(init_y[i]), "strategy": "init",
            "candidate_idx": "",
            "x0": float(init_X[i, 0]), "x1": float(init_X[i, 1]),
            "is_best_in_round": bool(i == int(np.argmax(init_y))),
        })

    X_obs = init_X.copy()
    y_obs = init_y.copy()

    for r in range(N_ROUNDS):
        rf = make_rf(seed=seed * 100 + r)
        rf.fit(X_obs, y_obs)

        if algo_name == "GLOSS":
            gloss = GLOSS(
                space={"bounds": bounds}, mode="continuous",
                direction="maximize", n_random_samples=5000,
                n_top_for_optimization=3, ucb_kappa=2.0,
                diversity_radius=GLOSS_DIVERSITY_RADIUS,
                seed=seed * 1000 + r,
            )
            results = gloss.recommend(
                surrogate=rf, X_existing=X_obs,
                strategy_points=GLOSS_STRATEGY,
            )
            pts = []
            strategies = []
            for res in results:
                if "point" in res:
                    pts.append(np.array(res["point"]))
                    strategies.append(res.get("strategy", "unknown"))
            if not pts:
                continue
            pts = np.array(pts)
            new_y = func(pts)
        else:
            result = algo.recommend_continuous(X_obs, y_obs, bounds, N_POINTS, rf)
            pts = result[0] if isinstance(result, tuple) else result
            pts = np.atleast_2d(pts)[:N_POINTS]
            new_y = func(pts)
            strategies = ["recommended"] * len(pts)

        best_in_round = int(np.argmax(new_y))
        for i in range(len(pts)):
            point_rows.append({
                "round": r, "point_idx": i,
                "y": float(new_y[i]), "strategy": strategies[i],
                "candidate_idx": "",
                "x0": float(pts[i, 0]), "x1": float(pts[i, 1]),
                "is_best_in_round": bool(i == best_in_round),
            })

        X_obs = np.vstack([X_obs, pts])
        y_obs = np.concatenate([y_obs, new_y])

    return point_rows


def _run_all(func_specs, seeds):
    from benchmarks.pub_benchmark import (
        N_INIT, N_ROUNDS, N_POINTS, GLOSS_STRATEGY,
        GLOSS_DIVERSITY_RADIUS, _sample_init_continuous,
    )
    from benchmarks.algorithms import BOAlgorithm, GeneticAlgorithm, RandomAlgorithm

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for spec in func_specs:
        func      = spec["func"]
        bounds    = spec["bounds"]
        func_name = spec["name"]
        y_opt, _  = spec["optimum"]()
        print(f"\n=== {func_name} (y_opt={y_opt:.4f}) ===")

        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            t0 = time.time()

            init_X = _sample_init_continuous(bounds, N_INIT, seed,
                                             func=func, init_bottom_pct=0.2)
            init_y = func(init_X)

            all_algos = {
                "GLOSS":  None,
                "BO":     BOAlgorithm(seed=seed),
                "GA":     GeneticAlgorithm(kappa=2.0, seed=seed),
                "Random": RandomAlgorithm(seed=seed),
            }
            for algo_name, algo in all_algos.items():
                point_rows = _run_one_algo(
                    func, bounds, func_name, algo_name, algo, seed,
                    init_X, init_y, N_INIT, N_ROUNDS, N_POINTS,
                    GLOSS_STRATEGY, GLOSS_DIVERSITY_RADIUS,
                )
                pd.DataFrame(point_rows).to_csv(
                    _points_csv_path(func_name, algo_name, seed), index=False)

            print(f"done ({time.time()-t0:.1f}s)")


# ── heatmap helper ────────────────────────────────────────────────────────────
def _make_heatmap(func, bounds, resolution=200):
    x0 = np.linspace(bounds[0][0], bounds[0][1], resolution)
    x1 = np.linspace(bounds[1][0], bounds[1][1], resolution)
    XX, YY = np.meshgrid(x0, x1)
    Z = func(np.column_stack([XX.ravel(), YY.ravel()])).reshape(resolution, resolution)
    return x0, x1, Z


# ── plotting ──────────────────────────────────────────────────────────────────
def _draw_trajectory(ax, df_algo, algo_name, x0_grid, x1_grid, Z, y_opt):
    im = ax.pcolormesh(x0_grid, x1_grid, Z, cmap="Spectral_r", shading="auto",
                       vmin=Z.min(), vmax=Z.max(), zorder=0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    ax.get_figure().colorbar(im, cax=cax)

    init_rows = df_algo[df_algo["round"] == -1]
    all_rounds = df_algo[df_algo["round"] >= 0]
    # best point per round (for trajectory line)
    best_per_round = (all_rounds[all_rounds["is_best_in_round"] == True]
                      .sort_values("round"))

    # draw init best point
    init_best = init_rows[init_rows["is_best_in_round"] == True]
    if init_best.empty:
        init_best = init_rows.iloc[[0]] if not init_rows.empty else init_rows
    if not init_best.empty:
        ax.scatter(init_best["x0"].iloc[0], init_best["x1"].iloc[0],
                   marker="P", s=120, color="white",
                   edgecolors="black", linewidths=0.8, zorder=4)

    if best_per_round.empty:
        ax.set_title(algo_name)
        return

    # connecting line: init best → best of round 0 → … → best of round N-1
    lx = list(init_best["x0"]) + list(best_per_round["x0"])
    ly = list(init_best["x1"]) + list(best_per_round["x1"])
    ax.plot(lx, ly, color="black", lw=1.5, zorder=2)

    # scatter only best point per round
    if algo_name == "GLOSS":
        for _, rec in best_per_round.iterrows():
            mk = STRATEGY_MARKER.get(rec["strategy"], "s")
            ax.scatter(rec["x0"], rec["x1"],
                       marker=mk, s=70, color="white", alpha=0.9,
                       edgecolors="black", linewidths=0.6, zorder=3)
    else:
        ax.scatter(best_per_round["x0"], best_per_round["x1"],
                   marker="o", s=50, color="white", alpha=0.9,
                   edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    y_min_landscape = Z.min()
    final_pct = (all_rounds["y"].max() - y_min_landscape) / (y_opt - y_min_landscape) * 100 if y_opt > y_min_landscape else 0
    ax.set_title(f"{algo_name}  (best {final_pct:.0f}%)")

    if algo_name == "GLOSS":
        handles = [
            Line2D([0],[0], marker="*", color="w", markerfacecolor="white",
                   markersize=9,  markeredgecolor="black", label="global_best"),
            Line2D([0],[0], marker="^", color="w", markerfacecolor="white",
                   markersize=8,  markeredgecolor="black", label="local_best"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
                   markersize=7,  markeredgecolor="black", label="unexplored"),
            Line2D([0],[0], marker="P", color="w", markerfacecolor="white",
                   markersize=8,  markeredgecolor="black", label="init best"),
        ]
        ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.8)


def _plot_all(func_specs, seeds):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    algo_order = ["GLOSS", "Random", "GA", "BO"]

    for spec in func_specs:
        func_name = spec["name"]
        y_opt, _  = spec["optimum"]()
        func_tag  = func_name.lower().replace("-","_").replace(" ","_")

        for seed in seeds:
            # load per-file CSVs
            algo_dfs = {}
            for algo_name in algo_order:
                p = _points_csv_path(func_name, algo_name, seed)
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing: {p} — run with --run first")
                algo_dfs[algo_name] = pd.read_csv(p)

            fig, axes = plt.subplots(1, 4, figsize=(22, 5))
            x0g, x1g, Z = _make_heatmap(spec["func"], spec["bounds"])

            for ax, algo_name in zip(axes, algo_order):
                _draw_trajectory(ax, algo_dfs[algo_name], algo_name, x0g, x1g, Z, y_opt)

            fig.suptitle(f"{func_name}  —  seed {seed}", fontsize=12)
            plt.tight_layout()

            fname = f"trajectory_{func_tag}_seed{seed}.png"
            fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {fname}")


# ── main ──────────────────────────────────────────────────────────────────────
def run(func_tags=None, seeds=None, do_run=False):
    from benchmarks.virtual_2d import VIRTUAL_2D_SPACES
    from benchmarks.pub_benchmark import SEEDS

    if func_tags is None:
        func_specs = VIRTUAL_2D_SPACES
    else:
        func_specs = [s for s in VIRTUAL_2D_SPACES
                      if any(t in s["name"].lower() for t in func_tags)]

    if seeds is None:
        seeds = SEEDS

    if do_run:
        _run_all(func_specs, seeds)

    print("\nPlotting...")
    _plot_all(func_specs, seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="run benchmark and save results CSV, then plot")
    parser.add_argument("--func", nargs="+", default=None,
                        help="function name substrings, e.g. simple medium complex")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    run(func_tags=args.func, seeds=args.seeds, do_run=args.run)
