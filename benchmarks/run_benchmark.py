"""Main benchmark runner.

Usage:
    python -m benchmarks.run_benchmark
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from benchmarks.algorithms import GLOSSAlgorithm, BOAlgorithm, RandomAlgorithm, fit_gp
from benchmarks.datasets import load_all_discrete
from benchmarks.virtual_function import chemical_yield, get_true_optimum, BOUNDS
from benchmarks.metrics import compute_metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_INIT = 8
N_ROUNDS = 20
POINTS_PER_ROUND = 8
GLOSS_STRATEGY_POINTS = {
    "global_best": 4,
    "local_best": 2,
    "unexplored": 1,
    "unconverged": 1,
}
N_SEEDS = 5

ALGO_NAMES = ["GLOSS", "BO (EI)", "Random (GP)", "Random"]


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _make_algos(seed):
    """Create algorithm instances. Random (GP) uses GP to sample, Random is pure random."""
    return {
        "GLOSS": GLOSSAlgorithm(strategy_points=GLOSS_STRATEGY_POINTS),
        "BO (EI)": BOAlgorithm(),
        "Random (GP)": RandomAlgorithm(seed=seed),  # random sampling, but GP is still fitted (for fair time comparison)
        "Random": RandomAlgorithm(seed=seed + 1000),  # pure random, no GP
    }


def run_discrete_benchmark(dataset, seeds=range(N_SEEDS)):
    """Run benchmark on a discrete dataset."""
    candidates = dataset["candidates"]
    y_true = dataset["y_true"]
    name = dataset["name"]
    y_optimum = np.max(y_true)
    n_candidates = len(candidates)

    print(f"\n{'='*60}")
    print(f"Dataset: {name} ({n_candidates} candidates, {candidates.shape[1]}D, "
          f"optimum={y_optimum:.4f})")
    print(f"{'='*60}")

    all_results = {a: [] for a in ALGO_NAMES}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        init_indices = rng.choice(n_candidates, size=N_INIT, replace=False)
        X_init = candidates[init_indices]
        y_init = y_true[init_indices]

        algos = _make_algos(seed)

        for algo_name in ALGO_NAMES:
            algo = algos[algo_name]
            algo.reset()

            X_obs = X_init.copy()
            y_obs = y_init.copy()
            effective_best = np.max(y_init)
            round_bests = []
            round_times = []

            for r in range(N_ROUNDS):
                # Pure Random: no GP fitting
                if algo_name == "Random":
                    t0 = time.time()
                    rec_indices = algo.recommend_discrete(
                        X_obs, y_obs, candidates, y_true, POINTS_PER_ROUND, None
                    )
                    elapsed = time.time() - t0
                    for idx in rec_indices:
                        X_obs = np.vstack([X_obs, candidates[idx].reshape(1, -1)])
                        y_obs = np.append(y_obs, y_true[idx])
                    effective_best = np.max(y_obs)
                    round_bests.append(effective_best)

                elif algo_name == "GLOSS":
                    gp = fit_gp(X_obs, y_obs)
                    t0 = time.time()
                    all_idx, global_idx = algo.recommend_discrete(
                        X_obs, y_obs, candidates, y_true, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    for idx in all_idx:
                        X_obs = np.vstack([X_obs, candidates[idx].reshape(1, -1)])
                        y_obs = np.append(y_obs, y_true[idx])
                    for idx in global_idx:
                        effective_best = max(effective_best, y_true[idx])
                    round_bests.append(effective_best)

                else:  # BO (EI) or Random (GP)
                    gp = fit_gp(X_obs, y_obs)
                    t0 = time.time()
                    rec_indices = algo.recommend_discrete(
                        X_obs, y_obs, candidates, y_true, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    for idx in rec_indices:
                        X_obs = np.vstack([X_obs, candidates[idx].reshape(1, -1)])
                        y_obs = np.append(y_obs, y_true[idx])
                    effective_best = np.max(y_obs)
                    round_bests.append(effective_best)

                round_times.append(elapsed)

            metrics = compute_metrics(round_bests, round_times, y_optimum, N_INIT)
            all_results[algo_name].append(metrics)

            print(f"  [{algo_name:12s}] seed={seed}: best={metrics['final_best']:.4f}, "
                  f"t={metrics['avg_time']:.3f}s")

    return all_results


def run_continuous_benchmark(seeds=range(N_SEEDS)):
    """Run benchmark on the 6D virtual chemical function."""
    opt_x, y_optimum = get_true_optimum()

    print(f"\n{'='*60}")
    print(f"Virtual Function 6D (optimum={y_optimum:.4f})")
    print(f"{'='*60}")

    all_results = {a: [] for a in ALGO_NAMES}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        lows = np.array([b[0] for b in BOUNDS])
        highs = np.array([b[1] for b in BOUNDS])
        X_init = rng.uniform(lows, highs, size=(N_INIT, 6))
        y_init = chemical_yield(X_init)

        algos = _make_algos(seed)

        for algo_name in ALGO_NAMES:
            algo = algos[algo_name]
            algo.reset()

            X_obs = X_init.copy()
            y_obs = y_init.copy()
            effective_best = np.max(y_init)
            round_bests = []
            round_times = []

            for r in range(N_ROUNDS):
                if algo_name == "Random":
                    t0 = time.time()
                    rec_points = algo.recommend_continuous(
                        X_obs, y_obs, BOUNDS, POINTS_PER_ROUND, None
                    )
                    elapsed = time.time() - t0
                    new_y = chemical_yield(rec_points)
                    X_obs = np.vstack([X_obs, rec_points])
                    y_obs = np.append(y_obs, new_y)
                    effective_best = np.max(y_obs)
                    round_bests.append(effective_best)

                elif algo_name == "GLOSS":
                    gp = fit_gp(X_obs, y_obs)
                    t0 = time.time()
                    all_pts, global_pts = algo.recommend_continuous(
                        X_obs, y_obs, BOUNDS, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    all_y = chemical_yield(all_pts)
                    X_obs = np.vstack([X_obs, all_pts])
                    y_obs = np.append(y_obs, all_y)
                    if len(global_pts) > 0:
                        global_y = chemical_yield(global_pts)
                        effective_best = max(effective_best, np.max(global_y))
                    round_bests.append(effective_best)

                else:  # BO (EI) or Random (GP)
                    gp = fit_gp(X_obs, y_obs)
                    t0 = time.time()
                    rec_points = algo.recommend_continuous(
                        X_obs, y_obs, BOUNDS, POINTS_PER_ROUND, gp
                    )
                    elapsed = time.time() - t0
                    new_y = chemical_yield(rec_points)
                    X_obs = np.vstack([X_obs, rec_points])
                    y_obs = np.append(y_obs, new_y)
                    effective_best = np.max(y_obs)
                    round_bests.append(effective_best)

                round_times.append(elapsed)

            metrics = compute_metrics(round_bests, round_times, y_optimum, N_INIT)
            all_results[algo_name].append(metrics)

            print(f"  [{algo_name:12s}] seed={seed}: best={metrics['final_best']:.4f}, "
                  f"t={metrics['avg_time']:.3f}s")

    return all_results


def save_results(dataset_name, all_results):
    _ensure_dir()
    rows = []
    for algo_name, runs in all_results.items():
        for seed, run in enumerate(runs):
            for r, (best, t) in enumerate(zip(run["round_bests"], run["round_times"])):
                rows.append({
                    "dataset": dataset_name, "algorithm": algo_name,
                    "seed": seed, "round": r,
                    "best_so_far": best, "time_seconds": t,
                })
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f"{dataset_name.lower().replace(' ', '_').replace('-', '_')}_results.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def build_summary(all_dataset_results):
    rows = []
    for ds_name, all_results in all_dataset_results.items():
        for algo_name in ALGO_NAMES:
            runs = all_results.get(algo_name, [])
            finals = [r["final_best"] for r in runs if r["final_best"] is not None]
            targets = [r["target_95_round"] for r in runs if r["target_95_round"] is not None]
            times = [r["avg_time"] for r in runs]
            rows.append({
                "Dataset": ds_name,
                "Algorithm": algo_name,
                "Final Best (mean)": f"{np.mean(finals):.4f}" if finals else "N/A",
                "Final Best (std)": f"{np.std(finals):.4f}" if finals else "N/A",
                "Rounds to 95% (mean)": f"{np.mean(targets):.1f}" if targets else "N/A",
                "Avg Time/Round (s)": f"{np.mean(times):.3f}",
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("GLOSS Benchmark v2: GLOSS vs BO(EI) vs Random(GP) vs Random")
    print(f"Config: init={N_INIT}, rounds={N_ROUNDS}, pts/round={POINTS_PER_ROUND}, seeds={N_SEEDS}")
    print(f"GLOSS: {GLOSS_STRATEGY_POINTS} (metric: global_best only)")
    print("=" * 60)

    print("\nLoading datasets...")
    discrete_datasets = load_all_discrete()

    all_dataset_results = {}

    for ds in discrete_datasets:
        results = run_discrete_benchmark(ds)
        save_results(ds["name"], results)
        all_dataset_results[ds["name"]] = results

    continuous_results = run_continuous_benchmark()
    save_results("Virtual-6D", continuous_results)
    all_dataset_results["Virtual-6D"] = continuous_results

    summary = build_summary(all_dataset_results)

    print("\n" + "=" * 60)
    print("SUMMARY (GLOSS: global_best only)")
    print("=" * 60)
    print(summary.to_string(index=False))

    _ensure_dir()
    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)

    from benchmarks.plot_results import plot_all
    plot_all(all_dataset_results, summary)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
