"""Ablation benchmark: UCB-BO baseline + GLOSS strategy ratio study.

Two experiments:
  1. UCB-BO as 5th baseline — same GP as BO(EI), but uses UCB acquisition.
     Isolates whether GLOSS wins due to UCB vs EI or due to multi-strategy design.
  2. GLOSS-global-only (8:0:0) vs GLOSS (6:1:1) ablation.
     Proves the three-strategy coordination adds value beyond pure UCB exploration.

Datasets: QM9-gap (20k), QM9-alpha (20k), Virtual Materials 10D
Config: init=8, 20 rounds, 8 pts/round, 5 seeds (same as extended_benchmark.py)

Usage:
    python -m benchmarks.ablation_benchmark
    python -m benchmarks.ablation_benchmark --skip-continuous
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd

from benchmarks.algorithms import (
    GLOSSAlgorithm, BOAlgorithm, UCBOAlgorithm, fit_gp,
)
from benchmarks.datasets import load_qm9
from benchmarks.virtual_materials_10d import virtual_materials_10d, get_true_optimum, BOUNDS
from benchmarks.metrics import compute_metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Config (identical to extended_benchmark.py) ───────────────────────────────
N_INIT           = 8
N_ROUNDS         = 20
POINTS_PER_ROUND = 8
N_SEEDS          = 5
UCB_KAPPA        = 2.0

GLOSS_STRATEGY       = {"global_best": 6, "local_best": 1, "unexplored": 1, "unconverged": 0}
GLOSS_GLOBAL_STRATEGY = {"global_best": 8, "local_best": 0, "unexplored": 0, "unconverged": 0}

ALGO_NAMES = ["GLOSS (6:1:1)", "GLOSS-global (8:0:0)", "UCB-BO", "BO (EI)"]


def _make_algos(seed):
    return {
        "GLOSS (6:1:1)":       GLOSSAlgorithm(strategy_points=GLOSS_STRATEGY,       ucb_kappa=UCB_KAPPA),
        "GLOSS-global (8:0:0)": GLOSSAlgorithm(strategy_points=GLOSS_GLOBAL_STRATEGY, ucb_kappa=UCB_KAPPA),
        "UCB-BO":              UCBOAlgorithm(kappa=UCB_KAPPA),
        "BO (EI)":             BOAlgorithm(),
    }


# ── Discrete benchmark ───────────────────────────────────────────────────────
def run_discrete(dataset, seeds=range(N_SEEDS)):
    candidates = dataset["candidates"]
    y_true     = dataset["y_true"]
    name       = dataset["name"]
    y_optimum  = np.max(y_true)
    n_cand     = len(candidates)

    print(f"\n{'='*60}")
    print(f"Dataset: {name}  ({n_cand} candidates, {candidates.shape[1]}D, "
          f"optimum={y_optimum:.4f})")
    print(f"{'='*60}")

    all_results = {a: [] for a in ALGO_NAMES}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(n_cand, size=N_INIT, replace=False)
        X_init   = candidates[init_idx]
        y_init   = y_true[init_idx]
        init_best = float(np.max(y_init))

        algos = _make_algos(seed)

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
            print(f"  [{algo_name:22s}] seed={seed}: "
                  f"best={metrics['final_best']:.4f} "
                  f"({metrics['final_best']/y_optimum*100:.1f}%), "
                  f"t={metrics['avg_time']:.2f}s/round")

    return all_results


# ── Continuous benchmark ─────────────────────────────────────────────────────
def run_continuous(seeds=range(N_SEEDS)):
    x_opt, y_optimum = get_true_optimum()

    print(f"\n{'='*60}")
    print(f"Virtual Materials 10D  (optimum={y_optimum:.4f})")
    print(f"{'='*60}")

    all_results = {a: [] for a in ALGO_NAMES}

    for seed in seeds:
        rng   = np.random.default_rng(seed)
        lows  = np.array([b[0] for b in BOUNDS])
        highs = np.array([b[1] for b in BOUNDS])
        X_init = rng.uniform(lows, highs, size=(N_INIT, len(BOUNDS)))
        y_init = virtual_materials_10d(X_init)
        init_best = float(np.max(y_init))

        algos = _make_algos(seed)

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
            print(f"  [{algo_name:22s}] seed={seed}: "
                  f"best={metrics['final_best']:.4f} "
                  f"({metrics['final_best']/y_optimum*100:.1f}%), "
                  f"t={metrics['avg_time']:.2f}s/round")

    return all_results


# ── Summary ──────────────────────────────────────────────────────────────────
def print_summary(name, all_results, y_optimum):
    print(f"\n{'='*70}")
    print(f"SUMMARY — {name}  (optimum={y_optimum:.4f})")
    print(f"{'Algorithm':<24} {'Final Best':>12} {'  ±Std':>8} {'%Optimum':>10} {'Rounds→95%':>12}")
    print("-" * 70)
    for algo in ALGO_NAMES:
        runs = all_results[algo]
        finals  = [r["final_best"] for r in runs]
        targets = [r["target_95_round"] for r in runs if r["target_95_round"] is not None]
        pct  = np.mean(finals) / y_optimum * 100
        t95  = f"{np.mean(targets):.1f}" if targets else "N/A"
        print(f"  {algo:<22} {np.mean(finals):>12.4f} {np.std(finals):>8.4f} "
              f"{pct:>9.1f}% {t95:>12}")
    print("=" * 70)

    gloss_finals = [r["final_best"] for r in all_results["GLOSS (6:1:1)"]]
    for algo in ["GLOSS-global (8:0:0)", "UCB-BO", "BO (EI)"]:
        other = [r["final_best"] for r in all_results[algo]]
        wins = sum(g >= o for g, o in zip(gloss_finals, other))
        diff = np.mean(gloss_finals) - np.mean(other)
        sign = "+" if diff >= 0 else ""
        print(f"  GLOSS(6:1:1) vs {algo:<22}: {wins}/{N_SEEDS} wins, Δmean={sign}{diff:.4f}")


def save_results(name, all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for algo_name, runs in all_results.items():
        for seed, run in enumerate(runs):
            for r, (best, t) in enumerate(zip(run["round_bests"], run["round_times"])):
                rows.append({
                    "dataset": name, "algorithm": algo_name,
                    "seed": seed,
                    "round": r - 1,  # -1=init, 0..N-1=recommendation rounds
                    "best_so_far": best, "time_seconds": t,
                })
    df = pd.DataFrame(rows)
    fname = name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    path = os.path.join(RESULTS_DIR, f"ablation_{fname}_results.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-discrete",   action="store_true")
    parser.add_argument("--skip-continuous", action="store_true")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    seeds = range(args.seeds)

    print("=" * 60)
    print("Ablation Benchmark: UCB-BO baseline + GLOSS strategy ratio study")
    print(f"Config: init={N_INIT}, rounds={N_ROUNDS}, pts/round={POINTS_PER_ROUND}, "
          f"seeds={args.seeds}, kappa={UCB_KAPPA}")
    print(f"Algorithms: {ALGO_NAMES}")
    print("=" * 60)

    if not args.skip_discrete:
        print("\nLoading datasets...")
        for loader, kwargs in [
            (load_qm9, {"n_samples": 20000, "target": "gap"}),
            (load_qm9, {"n_samples": 20000, "target": "alpha"}),
        ]:
            try:
                ds = loader(**kwargs)
                print(f"  {ds['name']}: {len(ds['y_true'])} candidates, "
                      f"{ds['candidates'].shape[1]}D, "
                      f"y=[{ds['y_true'].min():.4f}, {ds['y_true'].max():.4f}]")
                results = run_discrete(ds, seeds=seeds)
                print_summary(ds["name"], results, np.max(ds["y_true"]))
                save_results(ds["name"], results)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

    if not args.skip_continuous:
        results_cont = run_continuous(seeds=seeds)
        _, y_opt = get_true_optimum()
        print_summary("Virtual-Materials-10D", results_cont, y_opt)
        save_results("Virtual-Materials-10D", results_cont)

    print("\nAblation benchmark complete!")


if __name__ == "__main__":
    main()
