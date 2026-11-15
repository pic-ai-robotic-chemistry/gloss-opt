"""Quick QM9 benchmark: GLOSS (optimized) vs BO(EI) vs GA vs Random.

Usage:
    python -m benchmarks.qm9_benchmark            # n_candidates=5000 (default)
    python -m benchmarks.qm9_benchmark --n 3000   # faster
    python -m benchmarks.qm9_benchmark --n 10000  # more thorough
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd

from benchmarks.algorithms import GLOSSAlgorithm, BOAlgorithm, GeneticAlgorithm, RandomAlgorithm, fit_gp
from benchmarks.datasets import load_qm9
from benchmarks.metrics import compute_metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Benchmark config ──────────────────────────────────────────────────────────
N_INIT = 8
N_ROUNDS = 20
POINTS_PER_ROUND = 8
N_SEEDS = 5

# GLOSS optimised: more global_best (all counted), skip unconverged (needs 2x predict)
GLOSS_STRATEGY = {
    "global_best": 6,
    "local_best": 1,
    "unexplored": 1,
    "unconverged": 0,
}
UCB_KAPPA = 2.0

ALGO_NAMES = ["GLOSS", "BO (EI)", "GA", "Random"]


def _make_algos(seed):
    return {
        "GLOSS": GLOSSAlgorithm(strategy_points=GLOSS_STRATEGY, ucb_kappa=UCB_KAPPA),
        "BO (EI)": BOAlgorithm(),
        "GA": GeneticAlgorithm(kappa=UCB_KAPPA, seed=seed),
        "Random": RandomAlgorithm(seed=seed),
    }


def run_qm9(n_candidates: int, seeds=range(N_SEEDS)):
    print(f"\nLoading QM9 ({n_candidates} candidates)...")
    dataset = load_qm9(n_samples=n_candidates)
    candidates = dataset["candidates"]
    y_true = dataset["y_true"]
    y_optimum = np.max(y_true)
    n_cand = len(candidates)
    ndim = candidates.shape[1]

    print(f"QM9-Gap: {n_cand} candidates, {ndim}D, "
          f"y=[{y_true.min():.4f}, {y_true.max():.4f}], optimum={y_optimum:.4f}")
    print(f"Config: init={N_INIT}, rounds={N_ROUNDS}, pts/round={POINTS_PER_ROUND}, "
          f"seeds={N_SEEDS}")
    print(f"GLOSS: {GLOSS_STRATEGY}, UCB kappa={UCB_KAPPA}")
    print("=" * 60)

    all_results = {a: [] for a in ALGO_NAMES}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(n_cand, size=N_INIT, replace=False)
        X_init = candidates[init_idx]
        y_init = y_true[init_idx]
        init_best = float(np.max(y_init))   # shared starting point for all algos

        algos = _make_algos(seed)

        for algo_name in ALGO_NAMES:
            algo = algos[algo_name]
            algo.reset()

            # All algorithms start from identical observations
            X_obs = X_init.copy()
            y_obs = y_init.copy()
            effective_best = init_best

            # round 0 = initial state (before any recommendations); same for all algos
            round_bests = [init_best]
            round_times = [0.0]

            for r in range(N_ROUNDS):
                gp = fit_gp(X_obs, y_obs)
                t0 = time.time()

                if algo_name == "GLOSS":
                    all_idx, global_idx = algo.recommend_discrete(
                        X_obs, y_obs, candidates, y_true, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    for idx in all_idx:
                        X_obs = np.vstack([X_obs, candidates[idx].reshape(1, -1)])
                        y_obs = np.append(y_obs, y_true[idx])
                    for idx in global_idx:
                        effective_best = max(effective_best, y_true[idx])

                elif algo_name == "Random":
                    rec = algo.recommend_discrete(X_obs, y_obs, candidates, y_true, POINTS_PER_ROUND, gp)
                    elapsed = time.time() - t0
                    for idx in rec:
                        X_obs = np.vstack([X_obs, candidates[idx].reshape(1, -1)])
                        y_obs = np.append(y_obs, y_true[idx])
                    effective_best = np.max(y_obs)

                else:  # BO, GA
                    rec = algo.recommend_discrete(X_obs, y_obs, candidates, y_true, POINTS_PER_ROUND, gp)
                    elapsed = time.time() - t0
                    for idx in rec:
                        X_obs = np.vstack([X_obs, candidates[idx].reshape(1, -1)])
                        y_obs = np.append(y_obs, y_true[idx])
                    effective_best = np.max(y_obs)

                round_bests.append(effective_best)
                round_times.append(elapsed)

            # round_bests[0] = init, round_bests[1..N_ROUNDS] = after each recommendation round
            metrics = compute_metrics(round_bests, round_times, y_optimum, N_INIT)
            all_results[algo_name].append(metrics)
            print(f"  [{algo_name:12s}] seed={seed}: "
                  f"best={metrics['final_best']:.4f} ({metrics['final_best']/y_optimum*100:.1f}%), "
                  f"t={metrics['avg_time']:.2f}s/round")

    return all_results


def print_summary(all_results, y_optimum):
    print("\n" + "=" * 65)
    print(f"SUMMARY — QM9-Gap (optimum={y_optimum:.4f})")
    print(f"{'Algorithm':<14} {'Final Best':>12} {'  ±Std':>8} {'%Optimum':>10} {'Rounds→95%':>12}")
    print("-" * 65)
    for algo in ALGO_NAMES:
        runs = all_results[algo]
        finals = [r["final_best"] for r in runs]
        targets = [r["target_95_round"] for r in runs if r["target_95_round"] is not None]
        pct = np.mean(finals) / y_optimum * 100
        t95 = f"{np.mean(targets):.1f}" if targets else "N/A"
        print(f"  {algo:<12} {np.mean(finals):>12.4f} {np.std(finals):>8.4f} "
              f"{pct:>9.1f}% {t95:>12}")
    print("=" * 65)

    # Win check
    gloss_finals = [r["final_best"] for r in all_results["GLOSS"]]
    print("\nPairwise comparison (GLOSS vs others):")
    for algo in ["BO (EI)", "GA", "Random"]:
        other_finals = [r["final_best"] for r in all_results[algo]]
        wins = sum(g >= o for g, o in zip(gloss_finals, other_finals))
        mean_diff = np.mean(gloss_finals) - np.mean(other_finals)
        sign = "+" if mean_diff >= 0 else ""
        print(f"  GLOSS vs {algo:<12}: wins {wins}/{N_SEEDS} seeds, "
              f"Δmean={sign}{mean_diff:.4f}")


def save_csv(all_results, n_candidates):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for algo_name, runs in all_results.items():
        for seed, run in enumerate(runs):
            # round_bests[0] = init (before recommendations), [1..] = after each round
            for r, (best, t) in enumerate(zip(run["round_bests"], run["round_times"])):
                rows.append({
                    "algorithm": algo_name,
                    "seed": seed,
                    "round": r - 1,   # -1 = init, 0..N-1 = recommendation rounds
                    "best_so_far": best,
                    "time_seconds": t,
                })
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f"qm9_{n_candidates}_results.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000,
                        help="Number of QM9 candidates (default: 5000)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="Number of random seeds (default: 5)")
    args = parser.parse_args()

    seeds = range(args.seeds)
    all_results = run_qm9(args.n, seeds=seeds)

    dataset = load_qm9(n_samples=args.n)
    y_optimum = np.max(dataset["y_true"])

    print_summary(all_results, y_optimum)
    save_csv(all_results, args.n)


if __name__ == "__main__":
    main()
