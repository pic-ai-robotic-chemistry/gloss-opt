"""Virtual landscape benchmark: 6 algorithms × 6 virtual spaces.

Algorithms:
  GLOSS (6:1:1)        — multi-strategy UCB
  GLOSS-global (8:0:0) — pure global UCB (GLOSS framework)
  UCB-BO               — flat batch UCB (no per-point optimization)
  BO (EI)              — Expected Improvement
  GA                   — Genetic Algorithm (UCB fitness + diversity)
  Random               — Uniform random sampling

Virtual spaces (see virtual_spaces.py):
  Single-Peak-6D       — Unimodal smooth
  Double-Peak-6D       — Bimodal deceptive
  Multi-Peak-8D        — 7 Gaussian peaks
  Rastrigin-8D         — Oscillatory local traps
  High-Dim-15D         — Smooth 15D
  Sparse-Needle-10D    — Single narrow peak in 10D flat landscape

Usage:
    python -m benchmarks.virtual_benchmark
    python -m benchmarks.virtual_benchmark --spaces 0 2 3   # specific indices
    python -m benchmarks.virtual_benchmark --seeds 3
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd

from benchmarks.algorithms import (
    GLOSSAlgorithm, UCBOAlgorithm, BOAlgorithm,
    GeneticAlgorithm, RandomAlgorithm, fit_gp,
)
from benchmarks.virtual_spaces import VIRTUAL_SPACES
from benchmarks.metrics import compute_metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

N_INIT           = 8
N_ROUNDS         = 20
POINTS_PER_ROUND = 8
N_SEEDS          = 5
UCB_KAPPA        = 2.0

GLOSS_STRATEGY        = {"global_best": 6, "local_best": 1, "unexplored": 1, "unconverged": 0}
GLOSS_GLOBAL_STRATEGY = {"global_best": 8, "local_best": 0, "unexplored": 0, "unconverged": 0}

ALGO_NAMES = [
    "GLOSS (6:1:1)",
    "GLOSS-global (8:0:0)",
    "UCB-BO",
    "BO (EI)",
    "GA",
    "Random",
]


def _make_algos():
    return {
        "GLOSS (6:1:1)":        GLOSSAlgorithm(strategy_points=GLOSS_STRATEGY,       ucb_kappa=UCB_KAPPA),
        "GLOSS-global (8:0:0)": GLOSSAlgorithm(strategy_points=GLOSS_GLOBAL_STRATEGY, ucb_kappa=UCB_KAPPA),
        "UCB-BO":               UCBOAlgorithm(kappa=UCB_KAPPA),
        "BO (EI)":              BOAlgorithm(),
        "GA":                   GeneticAlgorithm(kappa=UCB_KAPPA),
        "Random":               RandomAlgorithm(),
    }


def run_continuous(space, seeds=range(N_SEEDS)):
    func   = space["func"]
    bounds = space["bounds"]
    name   = space["name"]
    y_opt, x_opt = space["optimum"]()
    ndim   = space["dim"]

    print(f"\n{'='*64}")
    print(f"Space: {name}  ({ndim}D, true max={y_opt:.4f})")
    print(f"  [{space['tag']}]  — {space['expected']}")
    print(f"{'='*64}")

    all_results = {a: [] for a in ALGO_NAMES}
    lows  = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    for seed in seeds:
        rng    = np.random.default_rng(seed)
        X_init = rng.uniform(lows, highs, size=(N_INIT, ndim))
        y_init = func(X_init)
        init_best = float(np.max(y_init))

        algos = _make_algos()

        for algo_name in ALGO_NAMES:
            algo = algos[algo_name]
            algo.reset()

            X_obs = X_init.copy()
            y_obs = y_init.copy()
            effective_best = init_best

            round_bests = [init_best]
            round_times = [0.0]

            for _ in range(N_ROUNDS):
                gp = fit_gp(X_obs, y_obs)
                t0 = time.time()
                is_gloss = algo_name.startswith("GLOSS")

                if is_gloss:
                    all_pts, global_pts = algo.recommend_continuous(
                        X_obs, y_obs, bounds, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    new_y = func(all_pts)
                    X_obs = np.vstack([X_obs, all_pts])
                    y_obs = np.append(y_obs, new_y)
                    if len(global_pts) > 0:
                        global_y = func(global_pts)
                        effective_best = max(effective_best, float(np.max(global_y)))

                elif algo_name == "Random":
                    rec_pts = rng.uniform(lows, highs, size=(POINTS_PER_ROUND, ndim))
                    elapsed = time.time() - t0
                    new_y   = func(rec_pts)
                    X_obs   = np.vstack([X_obs, rec_pts])
                    y_obs   = np.append(y_obs, new_y)
                    effective_best = float(np.max(y_obs))

                else:
                    rec_pts = algo.recommend_continuous(
                        X_obs, y_obs, bounds, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    new_y   = func(rec_pts)
                    X_obs   = np.vstack([X_obs, rec_pts])
                    y_obs   = np.append(y_obs, new_y)
                    effective_best = float(np.max(y_obs))

                round_bests.append(effective_best)
                round_times.append(elapsed)

            metrics = compute_metrics(round_bests, round_times, y_opt, N_INIT)
            all_results[algo_name].append(metrics)
            pct = metrics["final_best"] / y_opt * 100
            t95 = metrics["target_95_round"]
            print(f"  [{algo_name:24s}] s{seed}: {pct:5.1f}%  "
                  f"t95={str(t95 or 'N/A'):>4}  t={metrics['avg_time']:.2f}s/r")

    return all_results


def print_summary(space_name, all_results, y_opt, ref="GLOSS (6:1:1)"):
    ref_finals = [r["final_best"] for r in all_results[ref]]
    print(f"\n{'='*72}")
    print(f"SUMMARY — {space_name}")
    print(f"{'Algorithm':<26} {'%Opt':>7} {'±':>5}  {'→95%':>6}  {'vs GLOSS(6:1:1)':>18}")
    print("-" * 72)
    for algo in ALGO_NAMES:
        runs   = all_results[algo]
        finals = [r["final_best"] for r in runs]
        t95s   = [r["target_95_round"] for r in runs if r["target_95_round"] is not None]
        pct    = np.mean(finals) / y_opt * 100
        std    = np.std(finals)  / y_opt * 100
        t95_s  = f"{np.mean(t95s):.1f}" if t95s else "N/A"
        if algo == ref:
            vs = "(reference)"
        else:
            diff = np.mean(finals) - np.mean(ref_finals)
            wins = sum(f >= r for f, r in zip(finals, ref_finals))
            vs   = f"{'+' if diff>=0 else ''}{diff/y_opt*100:.1f}% ({wins}/{N_SEEDS}W)"
        print(f"  {algo:<24} {pct:>6.1f}% {std:>4.1f}%  {t95_s:>6}  {vs:>18}")
    print("=" * 72)


def save_results(space_name, all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for algo_name, runs in all_results.items():
        for seed, run in enumerate(runs):
            for r, (best, t) in enumerate(zip(run["round_bests"], run["round_times"])):
                rows.append({
                    "space": space_name, "algorithm": algo_name,
                    "seed": seed, "round": r - 1,
                    "best_so_far": best, "time_seconds": t,
                })
    df    = pd.DataFrame(rows)
    fname = space_name.lower().replace("-", "_").replace(" ", "_")
    path  = os.path.join(RESULTS_DIR, f"virtual_{fname}_results.csv")
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", nargs="+", type=int,
                        default=list(range(len(VIRTUAL_SPACES))),
                        help="Indices of spaces to run (0-5)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args   = parser.parse_args()
    seeds  = range(args.seeds)

    print("=" * 64)
    print("Virtual Landscape Benchmark: 6 Algorithms × 6 Spaces")
    print(f"Config: init={N_INIT}, rounds={N_ROUNDS}, pts/round={POINTS_PER_ROUND}, "
          f"seeds={args.seeds}, κ={UCB_KAPPA}")
    print(f"Algorithms: {ALGO_NAMES}")
    print("=" * 64)

    # Print landscape overview
    print("\nLandscape overview:")
    for i, sp in enumerate(VIRTUAL_SPACES):
        print(f"  [{i}] {sp['name']:25s} {sp['dim']:2d}D  {sp['tag']}")

    summary_rows = []

    for idx in args.spaces:
        space = VIRTUAL_SPACES[idx]
        results = run_continuous(space, seeds=seeds)
        y_opt, _ = space["optimum"]()
        print_summary(space["name"], results, y_opt)
        save_results(space["name"], results)

        for algo in ALGO_NAMES:
            runs   = results[algo]
            finals = [r["final_best"] for r in runs]
            t95s   = [r["target_95_round"] for r in runs if r["target_95_round"] is not None]
            summary_rows.append({
                "space":        space["name"],
                "tag":          space["tag"],
                "dim":          space["dim"],
                "algorithm":    algo,
                "pct_optimum":  np.mean(finals) / y_opt * 100,
                "std_pct":      np.std(finals) / y_opt * 100,
                "rounds_to_95": np.mean(t95s) if t95s else np.nan,
                "n_seeds_95":   len(t95s),
            })

    # Global summary table
    if len(summary_rows) > 1:
        df = pd.DataFrame(summary_rows)
        print("\n" + "=" * 80)
        print("GLOBAL SUMMARY")
        print(f"{'Space':<26} {'Algorithm':<24} {'%Opt':>7} {'t95':>6} {'n/5':>5}")
        print("-" * 80)
        for space in VIRTUAL_SPACES:
            sub = df[df["space"] == space["name"]]
            if sub.empty:
                continue
            print(f"  {space['name']:<24}")
            for algo in ALGO_NAMES:
                row = sub[sub["algorithm"] == algo]
                if row.empty:
                    continue
                r = row.iloc[0]
                t95_s = f"{r['rounds_to_95']:.1f}" if not np.isnan(r["rounds_to_95"]) else "N/A"
                print(f"    {'':26} {algo:<24} {r['pct_optimum']:>6.1f}% {t95_s:>6} {int(r['n_seeds_95']):>5}/5")
        print("=" * 80)

        path = os.path.join(RESULTS_DIR, "virtual_summary.csv")
        df.to_csv(path, index=False)
        print(f"\nSummary saved → {path}")

    print("\nVirtual benchmark complete!")


if __name__ == "__main__":
    main()
