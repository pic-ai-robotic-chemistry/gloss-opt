"""Virtual optimization landscapes with distinct characteristics.

Six functions designed to test different algorithm strengths/weaknesses:

1. single_peak_6d    — Unimodal smooth Gaussian (easy baseline)
2. double_peak_6d    — Bimodal deceptive: wide local max + narrow global max
                       in a different corner. Tests whether methods escape local.
3. multi_peak_8d     — 7 Gaussian peaks of decreasing heights. Tests coverage.
4. rastrigin_8d      — Oscillatory: dense local optima traps. Tests robustness.
5. high_dim_15d      — Single Gaussian in 15D (σ=0.55). Tests dimensionality.
6. plateau_6d        — Wide flat plateau (y≈9) + narrow true peak (y=10).
                       EI sees tiny improvement signal; UCB still explores.

All functions defined on [0,1]^d with positive values; true max = 10.0 (Plateau-6D ≈ 9.88).
"""

import numpy as np


# ── 1. Single-Peak 6D ────────────────────────────────────────────────────────
_SP6_CENTER = np.array([0.40, 0.60, 0.35, 0.65, 0.50, 0.45])
_SP6_SIGMA  = 0.30
SP6_BOUNDS  = [(0.0, 1.0)] * 6


def single_peak_6d(x):
    """Unimodal smooth Gaussian. Easy baseline — all methods should converge."""
    x = np.atleast_2d(x)
    dist_sq = np.sum((x - _SP6_CENTER) ** 2, axis=1)
    return np.exp(-dist_sq / (2 * _SP6_SIGMA ** 2)) * 10.0


def single_peak_6d_optimum():
    return 10.0, _SP6_CENTER.copy()


# ── 2. Double-Peak 6D (deceptive) ────────────────────────────────────────────
_DP6_GLOBAL = np.array([0.80, 0.78, 0.82, 0.79, 0.77, 0.81])  # far corner, h=10
_DP6_LOCAL  = np.array([0.22, 0.25, 0.20, 0.28, 0.24, 0.21])  # near origin, h=9
_DP6_SG     = 0.18   # narrower global peak
_DP6_SL     = 0.28   # wider local peak (easier to discover via random sampling)
DP6_BOUNDS  = [(0.0, 1.0)] * 6


def double_peak_6d(x):
    """Bimodal deceptive: wide local max (h=9.0) near origin vs narrow global
    max (h=10.0) in far corner. Exploitation-heavy methods get trapped locally."""
    x       = np.atleast_2d(x)
    d_g     = np.sum((x - _DP6_GLOBAL) ** 2, axis=1)
    d_l     = np.sum((x - _DP6_LOCAL)  ** 2, axis=1)
    peak_global = np.exp(-d_g / (2 * _DP6_SG ** 2)) * 10.0
    peak_local  = np.exp(-d_l / (2 * _DP6_SL ** 2)) *  9.0
    return np.maximum(peak_global, peak_local)


def double_peak_6d_optimum():
    return 10.0, _DP6_GLOBAL.copy()


# ── 3. Multi-Peak 8D ─────────────────────────────────────────────────────────
_MP8_CENTERS = np.array([
    [0.12, 0.82, 0.40, 0.62, 0.22, 0.78, 0.50, 0.28],
    [0.78, 0.18, 0.62, 0.32, 0.82, 0.12, 0.48, 0.72],
    [0.50, 0.50, 0.18, 0.82, 0.48, 0.52, 0.28, 0.62],
    [0.28, 0.68, 0.78, 0.18, 0.68, 0.32, 0.72, 0.42],
    [0.88, 0.38, 0.28, 0.72, 0.18, 0.82, 0.62, 0.12],
    [0.62, 0.12, 0.88, 0.42, 0.42, 0.58, 0.18, 0.82],
    [0.38, 0.78, 0.52, 0.22, 0.78, 0.22, 0.82, 0.52],
])
_MP8_HEIGHTS = np.array([10.0, 8.5, 7.5, 6.8, 6.0, 5.2, 4.5])
_MP8_SIGMAS  = np.array([0.28, 0.28, 0.26, 0.28, 0.28, 0.26, 0.28])
MP8_BOUNDS   = [(0.0, 1.0)] * 8


def multi_peak_8d(x):
    """7 Gaussian peaks with decreasing heights (10, 8.5, ..., 4.5) spread
    across 8D. Tests multi-modal coverage — global (tallest) peak wins."""
    x = np.atleast_2d(x)
    y = np.zeros(len(x))
    for c, h, s in zip(_MP8_CENTERS, _MP8_HEIGHTS, _MP8_SIGMAS):
        d = np.sum((x - c) ** 2, axis=1)
        y = np.maximum(y, h * np.exp(-d / (2 * s ** 2)))
    return y


def multi_peak_8d_optimum():
    return 10.0, _MP8_CENTERS[0].copy()


# ── 4. Rastrigin 8D ──────────────────────────────────────────────────────────
RASTRIGIN_8D_BOUNDS = [(0.0, 1.0)] * 8
_R8_SCALE = 5.12   # maps [0,1] → [-5.12, 5.12]


def rastrigin_8d(x):
    """Rastrigin mapped to [0,1]^8. Dense grid of local optima; global max at
    center (0.5,...). Tests robustness to local trap escapes."""
    x   = np.atleast_2d(x)
    xr  = x * 2 * _R8_SCALE - _R8_SCALE   # → [-5.12, 5.12]
    n   = xr.shape[1]
    A   = 10.0
    val = A * n + np.sum(xr ** 2 - A * np.cos(2 * np.pi * xr), axis=1)
    # Min rastrigin = 0 at center; negate + normalise to [0,10]
    worst = A * n + n * _R8_SCALE ** 2
    return (worst - val) / worst * 10.0


def rastrigin_8d_optimum():
    return 10.0, np.full(8, 0.5)


# ── 5. High-Dimensional Smooth (15D) ─────────────────────────────────────────
_HD15_CENTER = np.array([0.45, 0.55, 0.40, 0.60, 0.50, 0.48, 0.52,
                          0.44, 0.56, 0.42, 0.58, 0.46, 0.54, 0.50, 0.47])
_HD15_SIGMA  = 0.55   # wide enough to be findable in 15D
HD15_BOUNDS  = [(0.0, 1.0)] * 15


def high_dim_15d(x):
    """Single smooth Gaussian (σ=0.55) in 15D. Tests algorithmic scaling with
    dimensionality — GP quality degrades, surrogate guidance matters more."""
    x = np.atleast_2d(x)
    dist_sq = np.sum((x - _HD15_CENTER) ** 2, axis=1)
    return np.exp(-dist_sq / (2 * _HD15_SIGMA ** 2)) * 10.0


def high_dim_15d_optimum():
    return 10.0, _HD15_CENTER.copy()


# ── 6. Plateau 6D ─────────────────────────────────────────────────────────────
_PL6_CENTER = np.array([0.72, 0.28, 0.68, 0.34, 0.70, 0.30])
_PL6_SIGMA  = 0.12   # narrow peak
_PL6_PLATEAU = 9.0
PL6_BOUNDS  = [(0.0, 1.0)] * 6


def plateau_6d(x):
    """Wide flat landscape (y≈9.0 everywhere) with a single narrow true peak
    (y=10.0, σ=0.12). Expected Improvement sees only +1.0 room to improve →
    BO(EI) barely distinguishes peak from plateau. UCB explores uncertainty."""
    x = np.atleast_2d(x)
    # Flat plateau with mild quadratic bowl (keeps plateau from being fully flat)
    d_center = np.sum((x - 0.5) ** 2, axis=1)
    background = _PL6_PLATEAU - 0.5 * d_center   # slight gradient toward center
    # Sharp true peak
    d_peak = np.sum((x - _PL6_CENTER) ** 2, axis=1)
    peak   = np.exp(-d_peak / (2 * _PL6_SIGMA ** 2)) * 1.0  # +1.0 above plateau
    return np.clip(background + peak, 0, 10.0)


def plateau_6d_optimum():
    y = float(plateau_6d(_PL6_CENTER.reshape(1, -1))[0])
    return y, _PL6_CENTER.copy()


# ── Registry ─────────────────────────────────────────────────────────────────
VIRTUAL_SPACES = [
    {
        "name":     "Single-Peak-6D",
        "tag":      "Unimodal smooth",
        "dim":      6,
        "func":     single_peak_6d,
        "optimum":  single_peak_6d_optimum,
        "bounds":   SP6_BOUNDS,
        "expected": "All methods converge — easy baseline",
    },
    {
        "name":     "Double-Peak-6D",
        "tag":      "Bimodal deceptive",
        "dim":      6,
        "func":     double_peak_6d,
        "optimum":  double_peak_6d_optimum,
        "bounds":   DP6_BOUNDS,
        "expected": "BO(EI) trapped in wide local max; UCB finds global",
    },
    {
        "name":     "Multi-Peak-8D",
        "tag":      "Multi-modal (7 peaks)",
        "dim":      8,
        "func":     multi_peak_8d,
        "optimum":  multi_peak_8d_optimum,
        "bounds":   MP8_BOUNDS,
        "expected": "Multi-strategy GLOSS advantage; local+unexplored aids coverage",
    },
    {
        "name":     "Rastrigin-8D",
        "tag":      "Oscillatory (local traps)",
        "dim":      8,
        "func":     rastrigin_8d,
        "optimum":  rastrigin_8d_optimum,
        "bounds":   RASTRIGIN_8D_BOUNDS,
        "expected": "Dense local optima; UCB exploration advantage over EI",
    },
    {
        "name":     "High-Dim-15D",
        "tag":      "High-dimensional smooth",
        "dim":      15,
        "func":     high_dim_15d,
        "optimum":  high_dim_15d_optimum,
        "bounds":   HD15_BOUNDS,
        "expected": "GP scales poorly in 15D; per-point optimisation helps",
    },
    {
        "name":     "Plateau-6D",
        "tag":      "Flat plateau + narrow peak",
        "dim":      6,
        "func":     plateau_6d,
        "optimum":  plateau_6d_optimum,
        "bounds":   PL6_BOUNDS,
        "expected": "BO(EI) hampered by tiny improvement signal; UCB explores freely",
    },
]
