"""S6 ablation: surrogate-model sensitivity (RF vs MLP).

Runs GLOSS (4:2:2) on QM9-gap (100k) with two surrogate types:
  1. RF: scikit-learn RandomForestRegressor (100 trees) — main-paper baseline
  2. MLP: scikit-learn MLPRegressor (3 hidden layers, ReLU, early stopping)

GP is omitted because Gaussian-process fitting is O(n³); on n=100k it
would take hours per round. A small-n GP comparison can be added in
follow-up if needed.

Output: bench_surrogate_sens_summary.csv (config, seed, round, pct_opt, time_s).
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from benchmarks.datasets import load_qm9
from benchmarks.algorithms import GLOSSAlgorithm
from benchmarks.rf_surrogate import make_rf

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [47, 53, 59, 61, 67]
N_ROUNDS = 20
N_INIT = 8
Q = 8
GLOSS_STRATEGY = {"global_best": 4, "local_best": 2, "unexplored": 2,
                  "unconverged": 0}
GLOSS_DIVERSITY_RADIUS = 0.02


class MLPSurrogate:
    """sklearn MLP wrapper that exposes predict(X) -> mu only.

    GLOSS gracefully falls back to mean-only prediction when the surrogate
    doesn't expose return_std. We do not attempt MLP uncertainty here;
    Global_best with no σ degenerates to greedy mean exploitation.
    """
    def __init__(self, seed):
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()
        self._mlp = MLPRegressor(
            hidden_layer_sizes=(64, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=seed,
        )

    def fit(self, X, y):
        Xs = self._scaler_x.fit_transform(X)
        ys = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self._mlp.fit(Xs, ys)
        return self

    def predict(self, X):
        Xs = self._scaler_x.transform(X)
        ys = self._mlp.predict(Xs)
        return self._scaler_y.inverse_transform(ys.reshape(-1, 1)).ravel()


def make_surrogate(kind, seed):
    if kind == "RF":
        return make_rf(seed=seed)
    elif kind == "MLP":
        return MLPSurrogate(seed=seed)
    raise ValueError(kind)


def sample_init(y_true, n_init, seed):
    rng = np.random.default_rng(seed)
    threshold = np.quantile(y_true, 0.2)
    pool = np.where(y_true <= threshold)[0]
    return rng.choice(pool, size=n_init, replace=False).tolist()


def run_one(candidates, y_true, surrogate_kind, seed):
    y_opt = float(y_true.max())
    y_min = float(y_true.min())

    algo = GLOSSAlgorithm(
        strategy_points=GLOSS_STRATEGY,
        ucb_kappa=2.0,
        diversity_radius=GLOSS_DIVERSITY_RADIUS,
        seed=seed,
    )

    init_idx = sample_init(y_true, N_INIT, seed)
    X_obs = candidates[init_idx].copy()
    y_obs = y_true[init_idx].copy()

    rows = []
    best = float(y_obs.max())
    pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
    rows.append(dict(surrogate=surrogate_kind, seed=seed, round=-1,
                     best_so_far=best, pct_opt=pct, time_s=0.0))

    for r in range(N_ROUNDS):
        t0 = time.time()
        sur = make_surrogate(surrogate_kind, seed=seed * 100 + r)
        sur.fit(X_obs, y_obs)
        indices, _ = algo.recommend_discrete(
            X_obs, y_obs, candidates, y_true, Q, sur)
        elapsed = time.time() - t0

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

        new_X = candidates[indices]
        new_y = y_true[indices]
        X_obs = np.vstack([X_obs, new_X])
        y_obs = np.concatenate([y_obs, new_y])

        best = float(y_obs.max())
        pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
        rows.append(dict(surrogate=surrogate_kind, seed=seed, round=r,
                         best_so_far=best, pct_opt=pct, time_s=elapsed))
    return rows


def main():
    print("Loading QM9-gap (100k)...")
    ds = load_qm9(n_samples=100000, target="gap")
    candidates, y_true = ds["candidates"], ds["y_true"]

    all_rows = []
    for kind in ["RF", "MLP"]:
        for seed in SEEDS:
            print(f"  [{kind}] seed={seed} ...", flush=True)
            rows = run_one(candidates, y_true, kind, seed)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "bench_surrogate_sens_summary.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: GLOSS on QM9-gap (n=100k) by surrogate type")
    print("=" * 70)
    print(f"{'surrogate':<12} {'mean t95':<10} {'reach 95%':<12} {'mean t/round (s)':<18}")
    for kind in ["RF", "MLP"]:
        sub = df[df["surrogate"] == kind]
        per_round = sub[sub["round"] >= 0]
        seeds_t95 = []
        for seed in SEEDS:
            ssub = per_round[per_round["seed"] == seed]
            r = ssub[ssub["pct_opt"] >= 95]
            seeds_t95.append(int(r["round"].min()) if len(r) > 0 else 20)
        reach = sum(1 for t in seeds_t95 if t < 20)
        print(f"{kind:<12} {np.mean(seeds_t95):<10.2f} {reach}/5         "
              f"{per_round['time_s'].mean():.2f}")


if __name__ == "__main__":
    main()
