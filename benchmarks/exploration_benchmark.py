"""Systematic dataset exploration benchmark.

Goal: Find domains where GLOSS multi-strategy coordination (6:1:1) shows
      clear advantage over single-strategy UCB (GLOSS-global 8:0:0).

Datasets tested:
  Synthetic:
    - MultiCluster-3C-8D  : 50k candidates, 3 well-separated clusters
    - MultiCluster-5C-8D  : 50k candidates, 5 clusters
    - MultiCluster-7C-8D  : 50k candidates, 7 clusters
  QM9 properties (20k, 20 RDKit descriptors):
    - QM9-homo : HOMO energy (maximize → electron donors)
    - QM9-lumo : LUMO energy (maximize → low electron affinity)
    - QM9-mu   : Dipole moment (maximize → polar molecules)
    - QM9-r2   : Spatial extent (maximize → large molecules)
    - QM9-cv   : Heat capacity (maximize → thermally complex molecules)

Key metric: GLOSS (6:1:1) vs GLOSS-global (8:0:0) → isolates multi-strategy benefit.

Usage:
    python -m benchmarks.exploration_benchmark
    python -m benchmarks.exploration_benchmark --only-synthetic
    python -m benchmarks.exploration_benchmark --only-qm9
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
from benchmarks.datasets import load_qm9, load_multicluster_chemical
from benchmarks.metrics import compute_metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Config ───────────────────────────────────────────────────────────────────
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
    print(f"Dataset: {name}  ({n_cand} candidates, {candidates.shape[1]}D, "
          f"optimum={y_optimum:.4f})")
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
            print(f"  [{algo_name:22s}] seed={seed}: "
                  f"{metrics['final_best']/y_optimum*100:5.1f}% "
                  f"t95={metrics['target_95_round'] or 'N/A':>4}, "
                  f"t={metrics['avg_time']:.2f}s/r")

    return all_results


def print_summary(name, all_results, y_optimum):
    print(f"\n{'='*72}")
    print(f"SUMMARY — {name}")
    print(f"{'Algorithm':<24} {'%Opt':>7} {'±':>6} {'→95%':>8} {'vs GLOSS-global':>18}")
    print("-" * 72)

    gloss_finals    = [r["final_best"] for r in all_results["GLOSS (6:1:1)"]]
    gloss_t95       = [r["target_95_round"] for r in all_results["GLOSS (6:1:1)"] if r["target_95_round"]]
    global_finals   = [r["final_best"] for r in all_results["GLOSS-global (8:0:0)"]]

    for algo in ALGO_NAMES:
        runs  = all_results[algo]
        finals = [r["final_best"] for r in runs]
        t95s   = [r["target_95_round"] for r in runs if r["target_95_round"] is not None]
        pct    = np.mean(finals) / y_optimum * 100
        std    = np.std(finals) / y_optimum * 100
        t95_str = f"{np.mean(t95s):.1f}" if t95s else "N/A"
        if algo == "GLOSS (6:1:1)":
            vs_str = "(reference)"
        else:
            diff = np.mean(gloss_finals) - np.mean(finals)
            wins = sum(g >= o for g, o in zip(gloss_finals, finals))
            vs_str = f"{'+' if diff>=0 else ''}{diff/y_optimum*100:.1f}% ({wins}/{N_SEEDS}W)"
        print(f"  {algo:<22} {pct:>6.1f}% {std:>5.1f}% {t95_str:>8}  {vs_str:>18}")

    # Key metric: GLOSS vs GLOSS-global
    gloss_t95_mean   = np.mean(gloss_t95)  if gloss_t95   else None
    global_t95       = [r["target_95_round"] for r in all_results["GLOSS-global (8:0:0)"] if r["target_95_round"]]
    global_t95_mean  = np.mean(global_t95) if global_t95  else None
    if gloss_t95_mean and global_t95_mean:
        speedup = (global_t95_mean - gloss_t95_mean) / global_t95_mean * 100
        print(f"\n  Multi-strategy speedup (→95%): {global_t95_mean:.1f}→{gloss_t95_mean:.1f} rounds ({speedup:+.1f}%)")
    elif gloss_t95_mean and not global_t95_mean:
        print(f"\n  Multi-strategy: GLOSS reaches 95% in {gloss_t95_mean:.1f} rounds; GLOSS-global never reaches 95%!")
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
    path = os.path.join(RESULTS_DIR, f"explore_{fname}_results.csv")
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-synthetic", action="store_true")
    parser.add_argument("--only-qm9",       action="store_true")
    parser.add_argument("--seeds",  type=int, default=N_SEEDS)
    parser.add_argument("--n-qm9",  type=int, default=20000)
    args = parser.parse_args()
    seeds = range(args.seeds)

    print("=" * 60)
    print("Dataset Exploration Benchmark")
    print(f"Config: init={N_INIT}, rounds={N_ROUNDS}, pts/round={POINTS_PER_ROUND}, "
          f"seeds={args.seeds}, kappa={UCB_KAPPA}")
    print(f"Key comparison: GLOSS(6:1:1) vs GLOSS-global(8:0:0) → multi-strategy benefit")
    print("=" * 60)

    summary_rows = []

    # ── Synthetic multi-cluster datasets ─────────────────────────────────────
    if not args.only_qm9:
        print("\n[SYNTHETIC MULTI-CLUSTER DATASETS]")
        for n_clusters in [3, 5, 7]:
            ds = load_multicluster_chemical(n_candidates=50000, ndim=8,
                                            n_clusters=n_clusters, seed=42)
            results = run_discrete(ds, seeds=seeds)
            print_summary(ds["name"], results, np.max(ds["y_true"]))
            save_results(ds["name"], results)

            # Record summary
            for algo in ALGO_NAMES:
                runs = results[algo]
                finals = [r["final_best"] for r in runs]
                t95s   = [r["target_95_round"] for r in runs if r["target_95_round"]]
                summary_rows.append({
                    "dataset": ds["name"], "algorithm": algo,
                    "pct_optimum": np.mean(finals) / np.max(ds["y_true"]) * 100,
                    "rounds_to_95": np.mean(t95s) if t95s else np.nan,
                    "n_seeds_95": len(t95s),
                })

    # ── QM9 properties ───────────────────────────────────────────────────────
    if not args.only_synthetic:
        print("\n[QM9 PROPERTIES SCAN]")
        for target in ["homo", "lumo", "mu", "r2", "cv"]:
            try:
                ds = load_qm9(n_samples=args.n_qm9, target=target)
                print(f"  Loaded {ds['name']}: {len(ds['y_true'])} candidates, "
                      f"y=[{ds['y_true'].min():.3f}, {ds['y_true'].max():.3f}]")
                results = run_discrete(ds, seeds=seeds)
                print_summary(ds["name"], results, np.max(ds["y_true"]))
                save_results(ds["name"], results)

                for algo in ALGO_NAMES:
                    runs = results[algo]
                    finals = [r["final_best"] for r in runs]
                    t95s   = [r["target_95_round"] for r in runs if r["target_95_round"]]
                    summary_rows.append({
                        "dataset": ds["name"], "algorithm": algo,
                        "pct_optimum": np.mean(finals) / np.max(ds["y_true"]) * 100,
                        "rounds_to_95": np.mean(t95s) if t95s else np.nan,
                        "n_seeds_95": len(t95s),
                    })
            except Exception as e:
                print(f"  FAILED {target}: {e}")

    # ── Global summary ────────────────────────────────────────────────────────
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        # Multi-strategy advantage: GLOSS(6:1:1) vs GLOSS-global(8:0:0)
        gloss_df  = df_sum[df_sum["algorithm"] == "GLOSS (6:1:1)"].set_index("dataset")
        global_df = df_sum[df_sum["algorithm"] == "GLOSS-global (8:0:0)"].set_index("dataset")
        merge = gloss_df[["pct_optimum","rounds_to_95"]].join(
            global_df[["pct_optimum","rounds_to_95"]],
            lsuffix="_gloss", rsuffix="_global"
        )
        merge["multi_strategy_pct_gain"] = merge["pct_optimum_gloss"] - merge["pct_optimum_global"]
        merge["rounds_speedup"] = merge["rounds_to_95_global"] - merge["rounds_to_95_gloss"]

        print("\n" + "="*75)
        print("GLOBAL SUMMARY — Multi-Strategy Advantage (GLOSS 6:1:1 vs GLOSS-global 8:0:0)")
        print(f"{'Dataset':<28} {'GLOSS%':>7} {'Global%':>8} {'Δ%':>7} {'GLOSS→95':>10} {'Global→95':>10} {'Speedup(r)':>11}")
        print("-" * 75)
        for ds_name, row in merge.iterrows():
            print(f"  {ds_name:<26} {row['pct_optimum_gloss']:>6.1f}% {row['pct_optimum_global']:>7.1f}% "
                  f"{row['multi_strategy_pct_gain']:>+6.1f}% "
                  f"{row['rounds_to_95_gloss']:>9.1f}  {row['rounds_to_95_global']:>9.1f}  "
                  f"{row['rounds_speedup']:>+9.1f}")
        print("="*75)

        path = os.path.join(RESULTS_DIR, "exploration_summary.csv")
        df_sum.to_csv(path, index=False)
        print(f"\nSummary saved → {path}")

    print("\nExploration benchmark complete!")


if __name__ == "__main__":
    main()
