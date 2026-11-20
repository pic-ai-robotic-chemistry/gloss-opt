"""Measure local refinement displacement under RF and GP surrogates."""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks import protocol  # noqa: E402
from benchmarks.algorithms import _expected_improvement  # noqa: E402
from benchmarks.bench_main import sample_init  # noqa: E402
from benchmarks.datasets import load_buchwald_hartwig, load_qm9  # noqa: E402
from benchmarks.gp_surrogate import GPSurrogate  # noqa: E402
from benchmarks.rf_surrogate import make_rf  # noqa: E402

SEEDS = (47, 53, 59)
#: Observations per probe: the published initial batch plus two rounds' worth.
N_OBS = 24
#: Starting points per probe, taken from the top of the acquisition ranking.
N_STARTS = 10
#: The optimizers the BO(EI) and UCB-BO arms use.
METHODS = ("L-BFGS-B", "SLSQP")
RESULTS = os.path.join(ROOT, "build", "paper", "reruns", "gp_refinement")
#: Bounds of the noise floor, only used to build the probe's candidate pool.
POOL_SIZE = 5000


def bounds_of(X):
    return list(zip(X.min(axis=0), X.max(axis=0)))


def datasets():
    """Three probe datasets with their measured values."""
    d = load_qm9(n_samples=100000, target="gap")
    bh = load_buchwald_hartwig()
    from benchmarks.virtual_functions import arrhenius_yield
    rng = np.random.default_rng(0)
    X_arr = rng.uniform(0.0, 1.0, (POOL_SIZE, 2))
    return [
        ("qm9", d["candidates"], d["y_true"]),
        ("bh", bh["candidates"], bh["y_true"]),
        ("arr", X_arr, arrhenius_yield(X_arr)),
    ]


def probe(surrogate, X_obs, y_obs, bounds, dataset, seed, family):
    """Refine N_STARTS points of one round's pool and record what happened."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed * 1000 + 7)
    pool = protocol.make_pool(bounds, rng, n=POOL_SIZE)
    # The pool the arms actually see: the distance filter has already run.
    pool = protocol.filter_pool(pool, X_obs, surrogate,
                                refinement_evidence=PROBE_EVIDENCE)
    if len(pool) == 0:
        return []
    mu, sd = surrogate.predict(pool, return_std=True)
    y_best = float(np.max(y_obs))
    acq = _expected_improvement(mu, sd, y_best=y_best)
    starts = pool[np.argsort(acq)[::-1][:N_STARTS]]

    def neg_acq(x, _s=surrogate, _yb=y_best):
        m, s = _s.predict(np.asarray(x, dtype=float).reshape(1, -1),
                          return_std=True)
        return -_expected_improvement(m, s, y_best=_yb)[0]

    rows = []
    for x0 in starts:
        for method in METHODS:
            res = minimize(neg_acq, x0, method=method, bounds=bounds,
                           options={"maxiter": 200})
            moved = float(np.linalg.norm(np.asarray(res.x) - np.asarray(x0)))
            gap = float(np.min(np.linalg.norm(
                np.asarray(res.x)[None, :] - X_obs, axis=1)))
            rows.append({
                "dataset": dataset, "seed": seed, "family": family,
                "method": method,
                "success": bool(res.success),
                "nfev": int(getattr(res, "nfev", 0)),
                "moved": moved,
                "acq_gain": float(-res.fun - (-neg_acq(x0))),
                "min_gap_to_measured": gap,
                "inside_zone": bool(gap < protocol.MIN_DISTANCE_TO_MEASURED),
                "n_obs": len(X_obs),
            })
    return rows


def main():
    import pandas as pd

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    rows = []
    for name, X_all, y_all in datasets():
        bounds = bounds_of(X_all)
        for seed in SEEDS:
            idx = sample_init(X_all, y_all, 8, seed)
            extra = np.random.default_rng(seed).choice(
                len(X_all), N_OBS - len(idx), replace=False)
            keep = np.concatenate([idx, extra])
            X_obs = np.asarray(X_all)[keep]
            y_obs = np.asarray(y_all)[keep]
            for family in ("rf", "gp"):
                surrogate = (make_rf(seed=seed) if family == "rf"
                             else GPSurrogate())
                surrogate.fit(X_obs, y_obs)
                rows.extend(probe(surrogate, X_obs, y_obs, bounds, name, seed,
                                  surrogate.SURROGATE_FAMILY))
                print(f"  {name:4s} seed={seed} {family:3s}: "
                      f"{len(rows)} rows so far", flush=True)

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "refinement_measurement.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(df)} rows, {time.time() - t0:.0f}s)")
    for (family, method), grp in df.groupby(["family", "method"]):
        print(f"  {family:4s} {method:10s} moved median {grp.moved.median():.4f} "
              f"max {grp.moved.max():.4f}  inside-zone {int(grp.inside_zone.sum())}"
              f"/{len(grp)}  acq gain median {grp.acq_gain.median():+.4f}")
    # The single number the protocol wants as evidence.
    for family, grp in df.groupby("family"):
        moved = int((grp.moved > 0).sum())
        print(f"  evidence for {family}: largest displacement "
              f"{grp.moved.max():.6f}, moved in {moved} of {len(grp)} probes")
    print("inside_zone counts refined endpoints within the exclusion distance "
          "of measured points. Starting points were drawn from the filtered pool.")


PROBE_EVIDENCE = "reproduction.studies.gp_refinement (probe measurement)"


if __name__ == "__main__":
    main()
