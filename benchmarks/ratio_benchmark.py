"""Strategy ratio benchmark: GLOSS(4:2:2) vs GLOSS(6:1:1).

Compares the effect of strategy allocation ratio on GLOSS performance.
Both variants use UCB(κ=2.0) acquisition and the same fixed GP surrogate.

Algorithms:
  GLOSS (4:2:2) — global_best=4, local_best=2, unexplored=2 (more balanced)
  GLOSS (6:1:1) — global_best=6, local_best=1, unexplored=1 (current default)
  GLOSS-global (8:0:0) — pure global UCB (ablation reference)
  UCB-BO         — flat batch UCB (non-GLOSS reference)

Datasets:
  QM9-gap (20k)        — discrete chemistry, smooth landscape
  QM9-alpha (20k)      — discrete chemistry, moderate landscape
  Virtual-10D          — continuous 10D materials function
  MultiCluster-7C-8D   — synthetic discrete, 7 well-separated clusters

Usage:
    python -m benchmarks.ratio_benchmark
    python -m benchmarks.ratio_benchmark --skip-continuous
    python -m benchmarks.ratio_benchmark --skip-synthetic
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd

from benchmarks.algorithms import GLOSSAlgorithm, UCBOAlgorithm, fit_gp
from benchmarks.datasets import load_qm9, load_multicluster_chemical
from benchmarks.virtual_materials_10d import virtual_materials_10d, get_true_optimum, BOUNDS
from benchmarks.metrics import compute_metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

N_INIT           = 8
N_ROUNDS         = 20
POINTS_PER_ROUND = 8
N_SEEDS          = 5
UCB_KAPPA        = 2.0

STRATEGIES = {
    "GLOSS (4:2:2)":        {"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0},
    "GLOSS (6:1:1)":        {"global_best": 6, "local_best": 1, "unexplored": 1, "unconverged": 0},
    "GLOSS-global (8:0:0)": {"global_best": 8, "local_best": 0, "unexplored": 0, "unconverged": 0},
}
ALGO_NAMES = ["GLOSS (4:2:2)", "GLOSS (6:1:1)", "GLOSS-global (8:0:0)", "UCB-BO"]


def _make_algos():
    algos = {
        name: GLOSSAlgorithm(strategy_points=strat, ucb_kappa=UCB_KAPPA)
        for name, strat in STRATEGIES.items()
    }
    algos["UCB-BO"] = UCBOAlgorithm(kappa=UCB_KAPPA)
    return algos


# ── Discrete benchmark ────────────────────────────────────────────────────────
def run_discrete(dataset, seeds=range(N_SEEDS)):
    candidates = dataset["candidates"]
    y_true     = dataset["y_true"]
    name       = dataset["name"]
    y_optimum  = np.max(y_true)
    n_cand     = len(candidates)

    print(f"\n{'='*64}")
    print(f"Dataset: {name}  ({n_cand} candidates, {candidates.shape[1]}D, "
          f"optimum={y_optimum:.4f})")
    print(f"{'='*64}")

    all_results = {a: [] for a in ALGO_NAMES}

    for seed in seeds:
        rng      = np.random.default_rng(seed)
        init_idx = rng.choice(n_cand, size=N_INIT, replace=False)
        X_init   = candidates[init_idx]
        y_init   = y_true[init_idx]
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
                    all_idx, global_idx = algo.recommend_discrete(
                        X_obs, y_obs, candidates, y_true, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    for idx in all_idx:
                        X_obs = np.vstack([X_obs, candidates[idx].reshape(1, -1)])
                        y_obs = np.append(y_obs, y_true[idx])
                    for idx in global_idx:
                        effective_best = max(effective_best, y_true[idx])
                else:
                    rec = algo.recommend_discrete(
                        X_obs, y_obs, candidates, y_true, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    for idx in rec:
                        X_obs = np.vstack([X_obs, candidates[idx].reshape(1, -1)])
                        y_obs = np.append(y_obs, y_true[idx])
                    effective_best = float(np.max(y_obs))

                round_bests.append(effective_best)
                round_times.append(elapsed)

            metrics = compute_metrics(round_bests, round_times, y_optimum, N_INIT)
            all_results[algo_name].append(metrics)
            print(f"  [{algo_name:24s}] s{seed}: "
                  f"{metrics['final_best']/y_optimum*100:5.1f}%  "
                  f"t95={str(metrics['target_95_round'] or 'N/A'):>4}  "
                  f"t={metrics['avg_time']:.2f}s/r")

    return all_results


# ── Continuous benchmark ──────────────────────────────────────────────────────
def run_continuous(seeds=range(N_SEEDS)):
    x_opt, y_optimum = get_true_optimum()

    print(f"\n{'='*64}")
    print(f"Virtual Materials 10D  (optimum={y_optimum:.4f})")
    print(f"{'='*64}")

    all_results = {a: [] for a in ALGO_NAMES}

    for seed in seeds:
        rng   = np.random.default_rng(seed)
        lows  = np.array([b[0] for b in BOUNDS])
        highs = np.array([b[1] for b in BOUNDS])
        X_init = rng.uniform(lows, highs, size=(N_INIT, len(BOUNDS)))
        y_init = virtual_materials_10d(X_init)
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
                        X_obs, y_obs, BOUNDS, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    new_y = virtual_materials_10d(all_pts)
                    X_obs = np.vstack([X_obs, all_pts])
                    y_obs = np.append(y_obs, new_y)
                    if len(global_pts) > 0:
                        global_y = virtual_materials_10d(global_pts)
                        effective_best = max(effective_best, float(np.max(global_y)))
                else:
                    rec_pts = algo.recommend_continuous(
                        X_obs, y_obs, BOUNDS, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    new_y = virtual_materials_10d(rec_pts)
                    X_obs = np.vstack([X_obs, rec_pts])
                    y_obs = np.append(y_obs, new_y)
                    effective_best = float(np.max(y_obs))

                round_bests.append(effective_best)
                round_times.append(elapsed)

            metrics = compute_metrics(round_bests, round_times, y_optimum, N_INIT)
            all_results[algo_name].append(metrics)
            print(f"  [{algo_name:24s}] s{seed}: "
                  f"{metrics['final_best']/y_optimum*100:5.1f}%  "
                  f"t={metrics['avg_time']:.2f}s/r")

    return all_results


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(name, all_results, y_optimum):
    print(f"\n{'='*72}")
    print(f"SUMMARY — {name}")
    print(f"{'Algorithm':<26} {'%Opt':>7} {'±':>5}  {'→95%':>6}  {'vs GLOSS(6:1:1)':>18}")
    print("-" * 72)

    ref_finals = [r["final_best"] for r in all_results["GLOSS (6:1:1)"]]
    ref_t95    = [r["target_95_round"] for r in all_results["GLOSS (6:1:1)"]
                  if r["target_95_round"] is not None]

    for algo in ALGO_NAMES:
        runs   = all_results[algo]
        finals = [r["final_best"] for r in runs]
        t95s   = [r["target_95_round"] for r in runs if r["target_95_round"] is not None]
        pct    = np.mean(finals) / y_optimum * 100
        std    = np.std(finals) / y_optimum * 100
        t95_str = f"{np.mean(t95s):.1f}" if t95s else "N/A"
        if algo == "GLOSS (6:1:1)":
            vs_str = "(reference)"
        else:
            diff = np.mean(finals) - np.mean(ref_finals)
            wins = sum(f >= r for f, r in zip(finals, ref_finals))
            vs_str = f"{'+' if diff>=0 else ''}{diff/y_optimum*100:.1f}% ({wins}/{N_SEEDS}W)"
        print(f"  {algo:<24} {pct:>6.1f}% {std:>4.1f}%  {t95_str:>6}  {vs_str:>18}")

    # Head-to-head: 4:2:2 vs 6:1:1 on t95
    t95_422  = [r["target_95_round"] for r in all_results["GLOSS (4:2:2)"]
                if r["target_95_round"] is not None]
    if t95_422 and ref_t95:
        speedup = (np.mean(ref_t95) - np.mean(t95_422)) / np.mean(ref_t95) * 100
        print(f"\n  4:2:2 vs 6:1:1 t95: {np.mean(t95_422):.1f} vs {np.mean(ref_t95):.1f} rounds "
              f"({speedup:+.1f}% speedup for 4:2:2)")
    print("=" * 72)


def save_results(name, all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for algo_name, runs in all_results.items():
        for seed, run in enumerate(runs):
            for r, (best, t) in enumerate(zip(run["round_bests"], run["round_times"])):
                rows.append({
                    "dataset": name, "algorithm": algo_name,
                    "seed": seed, "round": r - 1,
                    "best_so_far": best, "time_seconds": t,
                })
    df = pd.DataFrame(rows)
    fname = name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    path = os.path.join(RESULTS_DIR, f"ratio_{fname}_results.csv")
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-continuous", action="store_true")
    parser.add_argument("--skip-synthetic",  action="store_true")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    seeds = range(args.seeds)

    print("=" * 64)
    print("Strategy Ratio Benchmark: GLOSS(4:2:2) vs GLOSS(6:1:1)")
    print(f"Config: init={N_INIT}, rounds={N_ROUNDS}, pts/round={POINTS_PER_ROUND}, "
          f"seeds={args.seeds}, κ={UCB_KAPPA}")
    print(f"Algorithms: {ALGO_NAMES}")
    print("=" * 64)

    # QM9-gap
    ds_gap = load_qm9(n_samples=20000, target="gap")
    print(f"\nLoaded {ds_gap['name']}: n={len(ds_gap['y_true'])}, "
          f"y=[{ds_gap['y_true'].min():.4f},{ds_gap['y_true'].max():.4f}]")
    res_gap = run_discrete(ds_gap, seeds=seeds)
    print_summary(ds_gap["name"], res_gap, np.max(ds_gap["y_true"]))
    save_results(ds_gap["name"], res_gap)

    # QM9-alpha
    ds_alpha = load_qm9(n_samples=20000, target="alpha")
    print(f"\nLoaded {ds_alpha['name']}: n={len(ds_alpha['y_true'])}, "
          f"y=[{ds_alpha['y_true'].min():.4f},{ds_alpha['y_true'].max():.4f}]")
    res_alpha = run_discrete(ds_alpha, seeds=seeds)
    print_summary(ds_alpha["name"], res_alpha, np.max(ds_alpha["y_true"]))
    save_results(ds_alpha["name"], res_alpha)

    # MultiCluster-7C-8D
    if not args.skip_synthetic:
        ds_mc = load_multicluster_chemical(n_candidates=50000, ndim=8, n_clusters=7, seed=42)
        print(f"\nLoaded {ds_mc['name']}: n={len(ds_mc['y_true'])}, "
              f"y=[{ds_mc['y_true'].min():.4f},{ds_mc['y_true'].max():.4f}]")
        res_mc = run_discrete(ds_mc, seeds=seeds)
        print_summary(ds_mc["name"], res_mc, np.max(ds_mc["y_true"]))
        save_results(ds_mc["name"], res_mc)

    # Virtual 10D
    if not args.skip_continuous:
        res_cont = run_continuous(seeds=seeds)
        _, y_opt = get_true_optimum()
        print_summary("Virtual-Materials-10D", res_cont, y_opt)
        save_results("Virtual-Materials-10D", res_cont)

    print("\nRatio benchmark complete!")


if __name__ == "__main__":
    main()
