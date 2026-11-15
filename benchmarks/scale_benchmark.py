"""Scale benchmark: does GLOSS multi-strategy advantage grow with dataset size?

Tests QM9-gap at multiple sizes (5k, 10k, 20k, 50k, 100k) and compares
GLOSS (6:1:1) vs GLOSS-global (8:0:0) vs UCB-BO vs BO(EI).

Hypothesis: As the candidate pool grows, pure global UCB searches become
less efficient at covering the space, making local_best and unexplored
strategies progressively more valuable.

Usage:
    python -m benchmarks.scale_benchmark
    python -m benchmarks.scale_benchmark --target gap   # default
    python -m benchmarks.scale_benchmark --target mu    # sparse target
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd

from benchmarks.algorithms import GLOSSAlgorithm, BOAlgorithm, UCBOAlgorithm, fit_gp
from benchmarks.datasets import load_qm9
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

ALGO_NAMES = ["GLOSS (6:1:1)", "GLOSS-global (8:0:0)", "UCB-BO", "BO (EI)"]


def _make_algos(seed):
    return {
        "GLOSS (6:1:1)":        GLOSSAlgorithm(strategy_points=GLOSS_STRATEGY,       ucb_kappa=UCB_KAPPA),
        "GLOSS-global (8:0:0)": GLOSSAlgorithm(strategy_points=GLOSS_GLOBAL_STRATEGY, ucb_kappa=UCB_KAPPA),
        "UCB-BO":               UCBOAlgorithm(kappa=UCB_KAPPA),
        "BO (EI)":              BOAlgorithm(),
    }


def run_discrete(dataset, seeds=range(N_SEEDS)):
    candidates = dataset["candidates"]
    y_true     = dataset["y_true"]
    name       = dataset["name"]
    y_optimum  = np.max(y_true)
    n_cand     = len(candidates)

    print(f"\n{'='*60}")
    print(f"{name}  ({n_cand} candidates, {candidates.shape[1]}D)")
    print(f"{'='*60}")

    all_results = {a: [] for a in ALGO_NAMES}

    for seed in seeds:
        rng      = np.random.default_rng(seed)
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
            print(f"  [{algo_name:22s}] s{seed}: "
                  f"{metrics['final_best']/y_optimum*100:5.1f}% "
                  f"t95={metrics['target_95_round'] or 'N/A':>4}  "
                  f"t={metrics['avg_time']:.2f}s/r")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="gap", choices=["gap","alpha","mu","r2","cv","homo","lumo"])
    parser.add_argument("--seeds",  type=int, default=N_SEEDS)
    parser.add_argument("--sizes",  nargs="+", type=int, default=[5000, 10000, 20000, 50000, 100000])
    args = parser.parse_args()
    seeds = range(args.seeds)

    print("="*60)
    print(f"Scale Benchmark: QM9-{args.target} at {args.sizes}")
    print(f"Config: init={N_INIT}, rounds={N_ROUNDS}, pts/round={POINTS_PER_ROUND}, seeds={args.seeds}")
    print("="*60)

    scale_rows = []

    for n in args.sizes:
        try:
            ds = load_qm9(n_samples=n, target=args.target)
            print(f"\nLoaded {ds['name']}: n={len(ds['y_true'])}, y=[{ds['y_true'].min():.4f},{ds['y_true'].max():.4f}]")
            results = run_discrete(ds, seeds=seeds)
            y_opt = np.max(ds["y_true"])

            for algo in ALGO_NAMES:
                runs = results[algo]
                finals = [r["final_best"] for r in runs]
                t95s   = [r["target_95_round"] for r in runs if r["target_95_round"] is not None]
                scale_rows.append({
                    "n_candidates": len(ds["y_true"]),
                    "algorithm": algo,
                    "pct_optimum": np.mean(finals) / y_opt * 100,
                    "rounds_to_95": np.mean(t95s) if t95s else np.nan,
                    "n_seeds_95": len(t95s),
                })

            # Save individual results
            os.makedirs(RESULTS_DIR, exist_ok=True)
            rows = []
            for algo_name, runs in results.items():
                for seed, run in enumerate(runs):
                    for r, (best, t) in enumerate(zip(run["round_bests"], run["round_times"])):
                        rows.append({"n_candidates": n, "dataset": ds["name"],
                                     "algorithm": algo_name, "seed": seed,
                                     "round": r - 1, "best_so_far": best})
            pd.DataFrame(rows).to_csv(
                os.path.join(RESULTS_DIR, f"scale_{args.target}_{n}_results.csv"), index=False
            )
        except Exception as e:
            print(f"  FAILED n={n}: {e}")

    # Print scaling table
    if scale_rows:
        df = pd.DataFrame(scale_rows)
        print("\n" + "="*70)
        print(f"SCALING SUMMARY — QM9-{args.target}")
        print(f"{'N':>8} {'Algorithm':<24} {'%Opt':>7} {'→95%':>8} {'Seeds95':>8}")
        print("-"*70)
        for n in args.sizes:
            sub = df[df["n_candidates"] == n]
            if sub.empty: continue
            for algo in ALGO_NAMES:
                row = sub[sub["algorithm"] == algo]
                if row.empty: continue
                r = row.iloc[0]
                print(f"  {n:>6}  {algo:<24} {r['pct_optimum']:>6.1f}% "
                      f"{r['rounds_to_95']:>7.1f}  {int(r['n_seeds_95']):>8}/5")
            # Multi-strategy speedup
            gloss = sub[sub["algorithm"]=="GLOSS (6:1:1)"].iloc[0]
            global_ = sub[sub["algorithm"]=="GLOSS-global (8:0:0)"].iloc[0]
            if not np.isnan(gloss["rounds_to_95"]) and not np.isnan(global_["rounds_to_95"]):
                spdup = (global_["rounds_to_95"] - gloss["rounds_to_95"]) / global_["rounds_to_95"] * 100
                print(f"  {'':>6}  {'→ Multi-strategy speedup:':<24} {spdup:>+6.1f}%")
            print()
        print("="*70)

        path = os.path.join(RESULTS_DIR, f"scale_{args.target}_summary.csv")
        df.to_csv(path, index=False)
        print(f"Saved → {path}")


if __name__ == "__main__":
    main()
