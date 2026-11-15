"""Publication benchmark: GLOSS vs BO vs GA vs Random with RF surrogate.

Four studies:
  1. main_study    — QM9-gap-20k, QM9-alpha-20k, Medium-2D
  2. scaling_study — QM9-gap at n=2000/5000/10000/20000
  3. complexity_study — Simple-2D, Medium-2D, Complex-2D
  4. zinc_study    — ZINC-QED at n=10000/50000/100000

Usage:
    python -m benchmarks.pub_benchmark               # all studies
    python -m benchmarks.pub_benchmark --study main
    python -m benchmarks.pub_benchmark --study scaling
    python -m benchmarks.pub_benchmark --study complexity
    python -m benchmarks.pub_benchmark --study zinc
    python -m benchmarks.pub_benchmark --study zinc --zinc-n 10000
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

from benchmarks.rf_surrogate import make_rf
from benchmarks.algorithms import GLOSSAlgorithm, UCBOAlgorithm, GeneticAlgorithm, RandomAlgorithm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_INIT = 8
N_ROUNDS = 20
N_POINTS = 8
SEEDS = [47, 53, 59, 61, 67]

GLOSS_STRATEGY = {"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0}
GLOSS_DIVERSITY_RADIUS = 0.02

GLOSS_RATIOS = {
    "GLOSS(2:3:3)": {"global_best": 2, "local_best": 3, "unexplored": 3, "unconverged": 0},
    "GLOSS(4:2:2)": {"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0},
    "GLOSS(6:1:1)": {"global_best": 6, "local_best": 1, "unexplored": 1, "unconverged": 0},
}


def _make_algorithms(seed):
    """Return dict of algorithm instances for one run."""
    return {
        "GLOSS": GLOSSAlgorithm(
            strategy_points=GLOSS_STRATEGY,
            ucb_kappa=2.0,
            diversity_radius=GLOSS_DIVERSITY_RADIUS,
            seed=seed,
        ),
        "BO": UCBOAlgorithm(kappa=2.0, seed=seed),
        "GA": GeneticAlgorithm(kappa=2.0, seed=seed),
        "Random": RandomAlgorithm(seed=seed),
    }


def _make_ratio_algorithms(seed):
    """Return dict of GLOSS variants with different strategy ratios."""
    return {
        name: GLOSSAlgorithm(
            strategy_points=sp,
            ucb_kappa=2.0,
            diversity_radius=GLOSS_DIVERSITY_RADIUS,
            seed=seed,
        )
        for name, sp in GLOSS_RATIOS.items()
    }


def _sample_init_discrete(candidates, n_init, seed, y_true=None, init_bottom_pct=None):
    """Sample n_init indices from candidates (same across algorithms for a seed).

    If init_bottom_pct is set (e.g. 0.5), only sample from the bottom X% of y_true.
    This avoids lucky initialization in high-value regions and ensures all seeds
    start from a challenging position, making the benchmark more consistent.
    """
    rng = np.random.default_rng(seed)
    if init_bottom_pct is not None and y_true is not None:
        threshold = np.quantile(y_true, init_bottom_pct)
        pool = np.where(y_true <= threshold)[0]
    else:
        pool = np.arange(len(candidates))
    return rng.choice(pool, size=n_init, replace=False).tolist()


def _sample_init_continuous(bounds, n_init, seed, func=None, init_bottom_pct=None):
    """Sample n_init points in continuous bounds.

    If func and init_bottom_pct are provided, sample 50000 candidates, compute y,
    restrict to the bottom X% by y value, then pick n_init from that pool.
    """
    rng = np.random.default_rng(seed)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    if func is not None and init_bottom_pct is not None:
        candidates = rng.uniform(lows, highs, size=(50000, len(bounds)))
        y_cand = func(candidates)
        threshold = np.quantile(y_cand, init_bottom_pct)
        pool = candidates[y_cand <= threshold]
        idx = rng.choice(len(pool), size=n_init, replace=False)
        return pool[idx]
    return rng.uniform(lows, highs, size=(n_init, len(bounds)))


def _mean_pairwise_dist(X):
    """Mean pairwise Euclidean distance across rows of X."""
    if len(X) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    return float(pdist(X).mean())


def _points_csv_path(study_name, algo_name, dataset_name, seed):
    safe_ds = dataset_name.replace(" ", "_")
    fname = f"{study_name}-{algo_name}-{safe_ds}-seed{seed}.csv"
    return os.path.join(RESULTS_DIR, fname)


def _save_points(study_name, algo_name, dataset_name, seed, point_rows):
    df = pd.DataFrame(point_rows)
    df.to_csv(_points_csv_path(study_name, algo_name, dataset_name, seed), index=False)


def run_discrete_seed(dataset, algo_name, algo, seed, study_name,
                      n_rounds=N_ROUNDS, n_points=N_POINTS, init_bottom_pct=None):
    """Run one algorithm on one discrete dataset for one seed.
    Returns list of round-level summary dicts, and saves point-level CSV.
    """
    candidates = dataset["candidates"]
    y_true = dataset["y_true"]
    y_opt = float(y_true.max())
    y_min = float(y_true.min())
    dataset_name = dataset["name"]

    init_indices = _sample_init_discrete(candidates, N_INIT, seed,
                                         y_true=y_true, init_bottom_pct=init_bottom_pct)
    X_obs = candidates[init_indices].copy()
    y_obs = y_true[init_indices].copy()

    summary_rows = []
    point_rows = []
    all_X_sampled = X_obs.copy()

    # init round
    best_so_far = float(y_obs.max())
    summary_rows.append({
        "dataset": dataset_name, "algorithm": algo_name, "seed": seed,
        "round": -1, "best_so_far": best_so_far,
        "pct_opt": (best_so_far - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0,
        "time_s": 0.0, "diversity": 0.0,
    })
    for i, idx in enumerate(init_indices):
        point_rows.append({
            "round": -1, "point_idx": i, "y": float(y_true[idx]),
            "strategy": "init", "candidate_idx": int(idx), "x0": "", "x1": "",
        })

    for r in range(n_rounds):
        t0 = time.time()
        rf = make_rf(seed=seed * 100 + r)
        rf.fit(X_obs, y_obs)

        if isinstance(algo, GLOSSAlgorithm):
            indices, strategies = algo.recommend_discrete(
                X_obs, y_obs, candidates, y_true, n_points, rf)
        else:
            indices = algo.recommend_discrete(
                X_obs, y_obs, candidates, y_true, n_points, rf)
            strategies = ["recommended"] * len(indices)

        elapsed = time.time() - t0

        if len(indices) == 0:
            rng = np.random.default_rng(seed * 1000 + r)
            obs_set = set(map(tuple, np.round(X_obs, 10)))
            avail = [i for i in range(len(candidates))
                     if tuple(np.round(candidates[i], 10)) not in obs_set]
            indices = rng.choice(avail, size=min(n_points, len(avail)), replace=False).tolist()
            strategies = ["recommended"] * len(indices)

        new_X = candidates[indices]
        new_y = y_true[indices]
        X_obs = np.vstack([X_obs, new_X])
        y_obs = np.concatenate([y_obs, new_y])
        all_X_sampled = np.vstack([all_X_sampled, new_X])

        best_so_far = float(y_obs.max())
        diversity = _mean_pairwise_dist(all_X_sampled[N_INIT:])

        summary_rows.append({
            "dataset": dataset_name, "algorithm": algo_name, "seed": seed,
            "round": r, "best_so_far": best_so_far,
            "pct_opt": (best_so_far - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0,
            "time_s": elapsed, "diversity": diversity,
        })
        for i, (idx, strat) in enumerate(zip(indices, strategies)):
            point_rows.append({
                "round": r, "point_idx": i, "y": float(y_true[idx]),
                "strategy": strat, "candidate_idx": int(idx), "x0": "", "x1": "",
            })

        if r % 5 == 0:
            print(f"    [{algo_name}] seed={seed} round={r:2d}: "
                  f"best={best_so_far:.4f} ({(best_so_far-y_min)/(y_opt-y_min)*100:.1f}%)")

    _save_points(study_name, algo_name, dataset_name, seed, point_rows)
    return summary_rows


def run_continuous_seed(func, bounds, y_opt, dataset_name,
                        algo_name, algo, seed, study_name,
                        n_rounds=N_ROUNDS, n_points=N_POINTS,
                        init_bottom_pct=None, y_min=0.0):
    """Run one algorithm on one continuous 2D function for one seed.
    Returns list of round-level summary dicts, and saves point-level CSV.
    """
    init_X = _sample_init_continuous(bounds, N_INIT, seed,
                                     func=func, init_bottom_pct=init_bottom_pct)
    init_y = func(init_X)

    X_obs = init_X.copy()
    y_obs = init_y.copy()
    all_X_sampled = X_obs.copy()

    summary_rows = []
    point_rows = []

    best_so_far = float(y_obs.max())
    summary_rows.append({
        "dataset": dataset_name, "algorithm": algo_name, "seed": seed,
        "round": -1, "best_so_far": best_so_far,
        "pct_opt": (best_so_far - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0,
        "time_s": 0.0, "diversity": 0.0,
    })
    for i in range(len(init_X)):
        point_rows.append({
            "round": -1, "point_idx": i, "y": float(init_y[i]),
            "strategy": "init", "candidate_idx": "",
            "x0": float(init_X[i, 0]), "x1": float(init_X[i, 1]),
        })

    for r in range(n_rounds):
        t0 = time.time()
        rf = make_rf(seed=seed * 100 + r)
        rf.fit(X_obs, y_obs)

        if isinstance(algo, GLOSSAlgorithm):
            pts, strategies = algo.recommend_continuous(X_obs, y_obs, bounds, n_points, rf)
        else:
            pts = algo.recommend_continuous(X_obs, y_obs, bounds, n_points, rf)
            strategies = None

        elapsed = time.time() - t0

        pts = np.atleast_2d(pts)[:n_points]
        new_y = func(pts)

        X_obs = np.vstack([X_obs, pts])
        y_obs = np.concatenate([y_obs, new_y])
        all_X_sampled = np.vstack([all_X_sampled, pts])

        best_so_far = float(y_obs.max())
        diversity = _mean_pairwise_dist(all_X_sampled[N_INIT:])

        summary_rows.append({
            "dataset": dataset_name, "algorithm": algo_name, "seed": seed,
            "round": r, "best_so_far": best_so_far,
            "pct_opt": (best_so_far - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0,
            "time_s": elapsed, "diversity": diversity,
        })
        for i in range(len(pts)):
            strat = strategies[i] if strategies is not None else "recommended"
            point_rows.append({
                "round": r, "point_idx": i, "y": float(new_y[i]),
                "strategy": strat, "candidate_idx": "",
                "x0": float(pts[i, 0]), "x1": float(pts[i, 1]),
            })

        if r % 5 == 0:
            print(f"    [{algo_name}] seed={seed} round={r:2d}: "
                  f"best={best_so_far:.4f} ({(best_so_far-y_min)/(y_opt-y_min)*100:.1f}%)")

    _save_points(study_name, algo_name, dataset_name, seed, point_rows)
    return summary_rows


def run_study(study_name, datasets_discrete, datasets_continuous, seeds=SEEDS,
              init_bottom_pct=None, algo_factory=None):
    """Run all algorithms on all datasets for all seeds. Save CSV.

    init_bottom_pct: passed to run_discrete_seed — if set, initializes only from
    the bottom X% of y_true to ensure a challenging, consistent starting point.
    algo_factory: optional callable(seed) -> dict of {name: algo}. Defaults to
    _make_algorithms which returns GLOSS/BO/GA/Random.
    """
    if algo_factory is None:
        algo_factory = _make_algorithms
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []

    # Discrete datasets
    for ds in datasets_discrete:
        print(f"\n=== {ds['name']} ({len(ds['y_true'])} samples) ===")
        if init_bottom_pct is not None:
            print(f"    (init from bottom {int(init_bottom_pct*100)}% of y_true)")
        for seed in seeds:
            algos = algo_factory(seed)
            for algo_name, algo in algos.items():
                print(f"  {algo_name} seed={seed}")
                rows = run_discrete_seed(ds, algo_name, algo, seed,
                                         study_name=study_name,
                                         init_bottom_pct=init_bottom_pct)
                all_rows.extend(rows)

    # Continuous datasets
    for ds in datasets_continuous:
        print(f"\n=== {ds['name']} ===")
        y_opt, _ = ds["optimum"]()
        # Estimate y_min by sampling the function on a grid
        _bounds = ds["bounds"]
        _grid = np.linspace(0, 1, 500)
        _XX, _YY = np.meshgrid(_grid, _grid)
        _pts = np.column_stack([_XX.ravel(), _YY.ravel()])
        _y_grid = ds["func"](_pts)
        y_min_cont = float(_y_grid.min())
        for seed in seeds:
            algos = algo_factory(seed)
            for algo_name, algo in algos.items():
                print(f"  {algo_name} seed={seed}")
                rows = run_continuous_seed(
                    ds["func"], ds["bounds"], y_opt, ds["name"],
                    algo_name, algo, seed, study_name=study_name,
                    init_bottom_pct=init_bottom_pct,
                    y_min=y_min_cont,
                )
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, f"pub_{study_name}_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",
                        choices=["main", "scaling", "complexity", "zinc", "chembl",
                                 "zinc_sas", "egfr", "pcqm4m", "alchemy",
                                 "alchemy_scaling", "alpha_scaling",
                                 "gap_scaling", "ratio_scaling",
                                 "ratio_complexity", "all"],
                        default="all")
    parser.add_argument("--zinc-n", type=int, default=None,
                        help="ZINC space size (default: runs 10000, 50000, 100000)")
    args = parser.parse_args()

    if args.study in ("main", "all"):
        print("\n" + "="*60)
        print("STUDY 1: Main benchmark (Alchemy-gap, QM9-gap, QM9-alpha)")
        print("="*60)
        from benchmarks.datasets import load_qm9, load_alchemy

        print("Loading Alchemy-gap (20k)...")
        ds_alc = load_alchemy(n_samples=20000, seed=42)
        print("Loading QM9-gap (20k)...")
        ds_gap = load_qm9(n_samples=20000, target="gap")
        print("Loading QM9-alpha (20k)...")
        ds_alpha = load_qm9(n_samples=20000, target="alpha")

        run_study("main",
                  datasets_discrete=[ds_alc, ds_gap, ds_alpha],
                  datasets_continuous=[],
                  init_bottom_pct=0.2)

    if args.study in ("scaling", "all"):
        print("\n" + "="*60)
        print("STUDY 2: Scaling (QM9-gap at n=2000/5000/10000/15000/20000)")
        print("="*60)
        from benchmarks.datasets import load_qm9
        for n in [2000, 5000, 10000, 15000, 20000]:
            print(f"Loading QM9-gap ({n})...")
            ds = load_qm9(n_samples=n, target="gap")
            ds["name"] = f"QM9-gap-{n}"
            run_study(f"scaling_{n}",
                      datasets_discrete=[ds],
                      datasets_continuous=[],
                      init_bottom_pct=0.2)

    if args.study in ("complexity", "all"):
        print("\n" + "="*60)
        print("STUDY 3: Complexity (Simple/Medium/Complex 2D)")
        print("="*60)
        from benchmarks.virtual_2d import VIRTUAL_2D_SPACES, make_dataset
        ds_2d = [make_dataset(sp, seed=42) for sp in VIRTUAL_2D_SPACES]
        run_study("complexity",
                  datasets_discrete=ds_2d,
                  datasets_continuous=[],
                  init_bottom_pct=0.2)

    if args.study in ("zinc", "all"):
        print("\n" + "="*60)
        print("STUDY 4: ZINC-QED (n=10k / 50k / 100k)")
        print("="*60)
        from benchmarks.datasets import load_zinc

        zinc_sizes = [args.zinc_n] if args.zinc_n else [10000, 50000, 100000]
        for n in zinc_sizes:
            print(f"Loading ZINC-QED ({n:,})...")
            ds = load_zinc(n_candidates=n, seed=42)
            ds["name"] = f"ZINC-QED-{n}"
            run_study(f"zinc_{n}",
                      datasets_discrete=[ds],
                      datasets_continuous=[])


    if args.study in ("egfr",):
        print("\n" + "="*60)
        print("STUDY 7: ChEMBL-EGFR (10801 compounds, pIC50 target)")
        print("="*60)
        from benchmarks.datasets import load_chembl_egfr
        print("Loading ChEMBL-EGFR...")
        ds = load_chembl_egfr()
        print(f"  {len(ds['y_true'])} compounds, "
              f"y range [{ds['y_true'].min():.2f}, {ds['y_true'].max():.2f}]")
        run_study("chembl_egfr",
                  datasets_discrete=[ds],
                  datasets_continuous=[],
                  init_bottom_pct=0.5)

    if args.study in ("zinc_sas",):
        print("\n" + "="*60)
        print("STUDY 6: ZINC-SAS (100k compounds, synthetic accessibility)")
        print("="*60)
        from benchmarks.datasets import load_zinc_sas
        print("Loading ZINC-SAS (20k)...")
        ds = load_zinc_sas(n_candidates=20000, seed=42)
        print(f"  {len(ds['y_true'])} compounds, "
              f"y range [{ds['y_true'].min():.3f}, {ds['y_true'].max():.3f}]")
        run_study("zinc_sas",
                  datasets_discrete=[ds],
                  datasets_continuous=[],
                  init_bottom_pct=0.5)

    if args.study in ("chembl",):
        print("\n" + "="*60)
        print("STUDY 5: ChEMBL-DRD2 (9k compounds, pIC50 target)")
        print("="*60)
        from benchmarks.datasets import load_chembl_drd2
        print("Loading ChEMBL-DRD2...")
        ds = load_chembl_drd2()
        print(f"  {len(ds['y_true'])} compounds, "
              f"y range [{ds['y_true'].min():.2f}, {ds['y_true'].max():.2f}]")
        run_study("chembl_drd2",
                  datasets_discrete=[ds],
                  datasets_continuous=[],
                  init_bottom_pct=0.5)

    if args.study in ("pcqm4m",):
        print("\n" + "="*60)
        print("STUDY 8: PCQM4Mv2 HOMO-LUMO gap (20k molecules, DFT-computed)")
        print("="*60)
        from benchmarks.datasets import load_pcqm4m
        print("Loading PCQM4M-gap (20k)...")
        ds = load_pcqm4m(n_samples=20000, seed=42)
        print(f"  {len(ds['y_true'])} molecules, "
              f"y range [{ds['y_true'].min():.3f}, {ds['y_true'].max():.3f}]")
        run_study("pcqm4m",
                  datasets_discrete=[ds],
                  datasets_continuous=[])

    if args.study in ("alchemy",):
        print("\n" + "="*60)
        print("STUDY 9: Alchemy HOMO-LUMO gap (20k molecules, DFT B3LYP/cc-pVDZ)")
        print("="*60)
        from benchmarks.datasets import load_alchemy
        print("Loading Alchemy-gap (20k)...")
        ds = load_alchemy(n_samples=20000, seed=42)
        print(f"  {len(ds['y_true'])} molecules, "
              f"y range [{ds['y_true'].min():.3f}, {ds['y_true'].max():.3f}]")
        run_study("alchemy",
                  datasets_discrete=[ds],
                  datasets_continuous=[],
                  init_bottom_pct=0.2)

    if args.study in ("alchemy_scaling",):
        print("\n" + "="*60)
        print("STUDY 10: Alchemy scaling (n=2000/5000/10000/20000)")
        print("="*60)
        from benchmarks.datasets import load_alchemy
        for n in [2000, 5000, 10000, 20000]:
            print(f"Loading Alchemy-gap ({n:,})...")
            ds = load_alchemy(n_samples=n, seed=42)
            ds["name"] = f"Alchemy-gap-{n}"
            print(f"  {len(ds['y_true'])} molecules, "
                  f"y range [{ds['y_true'].min():.3f}, {ds['y_true'].max():.3f}]")
            run_study(f"alchemy_scaling_{n}",
                      datasets_discrete=[ds],
                      datasets_continuous=[],
                      init_bottom_pct=0.2)


    if args.study in ("alpha_scaling",):
        print("\n" + "="*60)
        print("STUDY: QM9-alpha scaling (n=1000/10000/100000)")
        print("="*60)
        from benchmarks.datasets import load_qm9
        for n in [1000, 10000, 100000]:
            print(f"Loading QM9-alpha ({n:,})...")
            ds = load_qm9(n_samples=n, target="alpha")
            ds["name"] = f"QM9-alpha-{n}"
            print(f"  {len(ds['y_true'])} molecules, "
                  f"y range [{ds['y_true'].min():.3f}, {ds['y_true'].max():.3f}]")
            run_study(f"alpha_scaling_{n}",
                      datasets_discrete=[ds],
                      datasets_continuous=[],
                      init_bottom_pct=0.2)

    if args.study in ("gap_scaling",):
        print("\n" + "="*60)
        print("STUDY: QM9-gap scaling (n=1000/10000/100000)")
        print("="*60)
        from benchmarks.datasets import load_qm9
        for n in [1000, 10000, 100000]:
            print(f"Loading QM9-gap ({n:,})...")
            ds = load_qm9(n_samples=n, target="gap")
            ds["name"] = f"QM9-gap-{n}"
            print(f"  {len(ds['y_true'])} molecules, "
                  f"y range [{ds['y_true'].min():.3f}, {ds['y_true'].max():.3f}]")
            run_study(f"gap_scaling_{n}",
                      datasets_discrete=[ds],
                      datasets_continuous=[],
                      init_bottom_pct=0.2)

    if args.study in ("ratio_scaling",):
        print("\n" + "="*60)
        print("STUDY: Ratio-Scaling (GLOSS 2:3:3 / 4:2:2 / 6:1:1, MultiCluster-5C-8D at n=5k/20k/50k)")
        print("="*60)
        from benchmarks.datasets import load_multicluster_chemical
        for n in [5000, 20000, 50000]:
            print(f"Loading MultiCluster-5C-8D ({n:,})...")
            ds = load_multicluster_chemical(n_candidates=n, ndim=8, n_clusters=5, seed=42)
            ds["name"] = f"MultiCluster-5C-8D-{n}"
            run_study(f"ratio_scaling_{n}",
                      datasets_discrete=[ds],
                      datasets_continuous=[],
                      init_bottom_pct=0.2,
                      algo_factory=_make_ratio_algorithms)

    if args.study in ("ratio_complexity",):
        print("\n" + "="*60)
        print("STUDY: Ratio-Complexity (GLOSS 2:3:3 / 4:2:2 / 6:1:1, MultiCluster 3C/5C/7C-8D)")
        print("="*60)
        from benchmarks.datasets import load_multicluster_chemical
        ds_list = []
        for nc in [3, 5, 7]:
            print(f"Loading MultiCluster-{nc}C-8D (n=5000)...")
            ds_list.append(load_multicluster_chemical(
                n_candidates=5000, ndim=8, n_clusters=nc, seed=42))
        run_study("ratio_complexity",
                  datasets_discrete=ds_list,
                  datasets_continuous=[],
                  init_bottom_pct=0.2,
                  algo_factory=_make_ratio_algorithms)


if __name__ == "__main__":
    main()
