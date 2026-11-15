"""S5 ablation: κ-schedule + UCB-1 bandit comparison.

Runs GLOSS with 5 configurations on QM9-gap (100k) and Arrhenius-2D
(10k), each 5 seeds × 20 rounds × q=8:

  1. κ=fixed,  bandit=off   (paper default)
  2. κ=decay,  bandit=off
  3. κ=cosine, bandit=off
  4. κ=fixed,  bandit=on    (UCB-1 reallocates {n_gb,n_lb,n_ue,n_uc})
  5. κ=cosine, bandit=on    (combined)

Crucially, this script holds ONE GLOSS instance per (config, seed) so
that the round counter and bandit win/try statistics persist across
rounds (the production GLOSSAlgorithm wrapper rebuilds the instance
each round and would lose this state).

Output: bench_kappa_bandit_summary.csv with one row per (config, dataset, seed, round).
"""

import os
import time
import numpy as np
import pandas as pd

from gloss import GLOSS
from benchmarks.datasets import load_qm9
from benchmarks.virtual_functions import get_virtual_dataset
from benchmarks.rf_surrogate import make_rf

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [47, 53, 59, 61, 67]
N_ROUNDS = 20
N_INIT = 8
Q = 8
DIVERSITY_RADIUS = 0.02
KAPPA = 2.0
KAPPA_MIN = 0.5

CONFIGS = [
    dict(name="fixed_static",      kappa_schedule="fixed",  adaptive=False),
    dict(name="decay_static",      kappa_schedule="decay",  adaptive=False),
    dict(name="cosine_static",     kappa_schedule="cosine", adaptive=False),
    dict(name="fixed_bandit",      kappa_schedule="fixed",  adaptive=True),
    dict(name="cosine_bandit",     kappa_schedule="cosine", adaptive=True),
]


def sample_init(y_true, n_init, seed):
    rng = np.random.default_rng(seed)
    threshold = np.quantile(y_true, 0.2)
    pool = np.where(y_true <= threshold)[0]
    return rng.choice(pool, size=n_init, replace=False).tolist()


def run_one(candidates, y_true, ds_name, config, seed):
    """Persistent GLOSS instance across all rounds (so _round counter and
    bandit state survive)."""
    y_opt = float(y_true.max())
    y_min = float(y_true.min())

    gloss = GLOSS(
        space={"candidates": candidates},
        mode="discrete",
        direction="maximize",
        ucb_kappa=KAPPA,
        kappa_schedule=config["kappa_schedule"],
        kappa_min=KAPPA_MIN,
        n_rounds=N_ROUNDS,
        adaptive_strategy=config["adaptive"],
        default_batch_size=Q,
        diversity_radius=DIVERSITY_RADIUS,
        seed=seed,
    )
    fixed_strategy = {"global_best": 4, "local_best": 2, "unexplored": 2,
                      "unconverged": 0}

    init_idx = sample_init(y_true, N_INIT, seed)
    X_obs = candidates[init_idx].copy()
    y_obs = y_true[init_idx].copy()

    rows = []
    best = float(y_obs.max())
    pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
    rows.append(dict(config=config["name"], dataset=ds_name, seed=seed,
                     round=-1, best_so_far=best, pct_opt=pct,
                     time_s=0.0))

    for r in range(N_ROUNDS):
        t0 = time.time()
        rf = make_rf(seed=seed * 100 + r)
        rf.fit(X_obs, y_obs)
        # When bandit on, pass strategy_points=None to let _bandit_allocation
        # decide. When off, pass the fixed 4:2:2.
        sp = None if config["adaptive"] else fixed_strategy
        results = gloss.recommend(surrogate=rf, X_existing=X_obs,
                                  strategy_points=sp)
        elapsed = time.time() - t0

        # Resolve recommended points to candidate indices
        indices, strategies = [], []
        for res in results:
            if "point" not in res:
                continue
            pt = np.array(res["point"])
            d = np.linalg.norm(candidates - pt, axis=1)
            idx = int(np.argmin(d))
            if idx not in indices:
                indices.append(idx)
                strategies.append(res.get("strategy", "unknown"))
        if len(indices) < Q:
            rng = np.random.default_rng(seed * 1000 + r)
            obs_set = set(map(tuple, np.round(X_obs, 10)))
            avail = [i for i in range(len(candidates))
                     if tuple(np.round(candidates[i], 10)) not in obs_set
                     and i not in indices]
            extra = rng.choice(avail,
                               size=min(Q - len(indices), len(avail)),
                               replace=False).tolist()
            indices.extend(extra)
            strategies.extend(["fallback"] * len(extra))

        new_X = candidates[indices]
        new_y = y_true[indices]

        # bandit feedback BEFORE updating X_obs (needs current_best_before)
        if config["adaptive"]:
            best_before = best
            results_with_y = []
            for res, y in zip(results, new_y):
                if "point" in res:
                    rd = dict(res)
                    rd["y"] = float(y)
                    results_with_y.append(rd)
            gloss.feedback(results_with_y, best_before)

        X_obs = np.vstack([X_obs, new_X])
        y_obs = np.concatenate([y_obs, new_y])

        best = float(y_obs.max())
        pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
        rows.append(dict(config=config["name"], dataset=ds_name, seed=seed,
                         round=r, best_so_far=best, pct_opt=pct,
                         time_s=elapsed))
    return rows


def main():
    print("Loading QM9-gap (100k)...")
    qm = load_qm9(n_samples=100000, target="gap")
    print("Loading Arrhenius-2D (10k)...")
    ar = get_virtual_dataset("arrhenius", n_candidates=10000, n_peaks=3,
                              noise_sigma=0.0, peak_width=0.08, seed=42)

    datasets = [("QM9-gap", qm["candidates"], qm["y_true"]),
                ("Arrhenius-2D", ar["candidates"], ar["y_true"])]

    all_rows = []
    for cfg in CONFIGS:
        for ds_name, cand, y in datasets:
            for seed in SEEDS:
                print(f"  [{cfg['name']}] {ds_name} seed={seed} ...", flush=True)
                rows = run_one(cand, y, ds_name, cfg, seed)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "bench_kappa_bandit_summary.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)")

    # Console summary
    print("\n" + "=" * 80)
    print("Summary: mean t95 over 5 seeds")
    print("=" * 80)
    print(f"{'config':<18} {'dataset':<14} {'mean t95':<10} {'reach 95%':<12}")
    for cfg in CONFIGS:
        for ds_name, _, _ in datasets:
            sub = df[(df["config"] == cfg["name"]) & (df["dataset"] == ds_name)]
            seeds_t95 = []
            for seed in SEEDS:
                ssub = sub[sub["seed"] == seed]
                r = ssub[ssub["pct_opt"] >= 95]
                seeds_t95.append(int(r["round"].min()) if len(r) > 0 else 20)
            reach = sum(1 for t in seeds_t95 if t < 20)
            print(f"{cfg['name']:<18} {ds_name:<14} {np.mean(seeds_t95):<10.2f} {reach}/5")


if __name__ == "__main__":
    main()
