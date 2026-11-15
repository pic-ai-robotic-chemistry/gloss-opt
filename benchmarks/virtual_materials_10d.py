"""10D virtual materials optimization function for continuous-space benchmarking.

Models a perovskite-inspired materials design problem with 10 parameters:
  x0  Pb/Sn ratio in B-site cation          [0, 1]
  x1  I/Br ratio in X-site halide            [0, 1]
  x2  MA (methylammonium) fraction in A-site [0, 1]
  x3  FA (formamidinium) fraction in A-site  [0, 1]
  x4  Annealing temperature (norm 100–250°C) [0, 1]
  x5  Annealing time (norm 5–120 min)        [0, 1]
  x6  Film thickness (norm 200–600 nm)       [0, 1]
  x7  ETL work function offset               [0, 1]
  x8  DMF/DMSO solvent ratio                 [0, 1]
  x9  Ambient humidity (norm 0–60 %)         [0, 1]

Landscape:
  - 12 Gaussian peaks (heights 16 → 3, widths 0.07 → 0.35)
  - Non-separable interaction terms (cation-halide, temp-time, solvent-humidity)
  - Global optimum near x* ≈ [0.32, 0.73, 0.18, 0.61, 0.44, 0.78, 0.25, 0.62, 0.17, 0.09]
  - Global optimum value ≈ 19-21 (found by differential evolution)
"""

import numpy as np
from scipy.optimize import differential_evolution

BOUNDS = [(0.0, 1.0)] * 10
NDIM = 10

# ── Reproducible random peaks (seeded once at module load) ────────────────────
_rng = np.random.default_rng(20250325)
_N_PEAKS = 12

_CENTERS = _rng.uniform(0.05, 0.95, (_N_PEAKS, NDIM))
_HEIGHTS = np.array([16.0, 13.0, 11.0, 9.5, 8.0, 7.0, 6.0, 5.5, 5.0, 4.5, 4.0, 3.0])
_WIDTHS  = np.array([ 0.07,  0.14,  0.11, 0.19, 0.09, 0.17, 0.24, 0.13, 0.21, 0.27, 0.16, 0.35])

# Fix the global-optimum center to a known location
_CENTERS[0] = np.array([0.32, 0.73, 0.18, 0.61, 0.44, 0.78, 0.25, 0.62, 0.17, 0.09])


def virtual_materials_10d(X):
    """Evaluate the 10D virtual materials optimization function.

    Args:
        X: array-like, shape (n, 10) or (10,), all values in [0, 1].

    Returns:
        np.ndarray of shape (n,) — material performance scores (higher = better).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))

    y = np.zeros(len(X))

    # ── Gaussian peaks ────────────────────────────────────────────────────────
    for center, height, width in zip(_CENTERS, _HEIGHTS, _WIDTHS):
        dist2 = np.sum((X - center) ** 2, axis=1)
        y += height * np.exp(-dist2 / (2.0 * width ** 2))

    # ── Interaction terms ─────────────────────────────────────────────────────
    # Cation-halide synergy (MA + I combination improves absorption)
    y += 2.5 * X[:, 2] * X[:, 1] * (1.0 - np.abs(X[:, 2] + X[:, 3] - 1.0))

    # Annealing: optimal time depends on temperature (ridge in temp-time plane)
    optimal_time = 0.3 + 0.5 * X[:, 4]
    y += 3.0 * np.exp(-((X[:, 5] - optimal_time) ** 2) / 0.04)

    # Film thickness × ETL offset interaction
    y += 2.0 * np.sin(np.pi * X[:, 6]) * np.cos(np.pi * X[:, 7])

    # Solvent-humidity degradation (high humidity + high DMF is bad)
    y -= 3.5 * np.exp(-((X[:, 8] - 0.85) ** 2 + (X[:, 9] - 0.75) ** 2) / 0.04)

    # Halide oscillatory response (Br-rich compositions show quantum oscillations)
    y += 1.8 * np.sin(3.0 * np.pi * X[:, 1]) * X[:, 0]

    # Pb/Sn toxicity-efficiency trade-off: pure Sn (x0≈0) degrades quickly
    y -= 2.0 * np.exp(-X[:, 0] ** 2 / 0.02)

    return y


# ── True optimum (found once, cached) ─────────────────────────────────────────
_TRUE_OPTIMUM = None


def get_true_optimum():
    """Return (x_optimal, y_optimal) for the 10D function.

    Uses differential evolution with tight tolerance.  Result is cached
    in the module so it's only computed once per session.
    """
    global _TRUE_OPTIMUM
    if _TRUE_OPTIMUM is not None:
        return _TRUE_OPTIMUM

    print("  Computing true optimum of 10D virtual function (differential evolution)...")
    result = differential_evolution(
        lambda x: -virtual_materials_10d(x.reshape(1, -1))[0],
        bounds=BOUNDS,
        seed=42,
        maxiter=5000,
        tol=1e-10,
        popsize=25,
        mutation=(0.5, 1.5),
        recombination=0.9,
    )
    x_opt = result.x
    y_opt = float(-result.fun)
    _TRUE_OPTIMUM = (x_opt, y_opt)
    print(f"  True optimum: y* = {y_opt:.4f}")
    return _TRUE_OPTIMUM
