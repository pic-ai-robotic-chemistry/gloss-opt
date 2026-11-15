"""Virtual optimization landscapes of increasing complexity.

Increasing complexity is achieved by increasing BOTH dimensionality AND
number of peaks. This is realistic: more complex chemical spaces tend to
be higher-dimensional with more local optima.

- simple   : 3D, 1 Gaussian peak (easy baseline)
- medium   : 5D, 3 Gaussian peaks (multi-modal, moderate dimensionality)
- complex  : 8D, 5 Gaussian peaks (high-D, strongly deceptive)

Higher dimensionality makes Random search increasingly inefficient (curse
of dimensionality), while surrogate-guided methods (GLOSS, BO, GA) maintain
effectiveness. GLOSS's multi-strategy exploration is most valuable in the
complex, high-dimensional setting.
"""

import numpy as np

# ─── Simple: 3D, 1 Gaussian peak ────────────────────────────────────────────
_S_CENTER = np.array([0.50, 0.50, 0.50])
_S_SIGMA  = 0.30


def simple_func(x):
    """3D unimodal: single Gaussian peak at center. Height = 10."""
    x = np.atleast_2d(x)
    d2 = np.sum((x - _S_CENTER) ** 2, axis=1)
    return 10.0 * np.exp(-d2 / (2 * _S_SIGMA ** 2))


# ─── Medium: 5D, 3 Gaussian peaks ───────────────────────────────────────────
_M_CENTERS = np.array([
    [0.20, 0.20, 0.80, 0.50, 0.50],  # global max (h=10)
    [0.80, 0.80, 0.20, 0.50, 0.50],  # secondary (h=7)
    [0.50, 0.50, 0.50, 0.20, 0.80],  # tertiary (h=5)
])
_M_HEIGHTS = np.array([10.0, 7.0, 5.0])
_M_SIGMAS  = np.array([0.35, 0.40, 0.35])


def medium_func(x):
    """5D trimodal: 3 Gaussian peaks at different regions."""
    x = np.atleast_2d(x)
    y = np.zeros(len(x))
    for c, h, s in zip(_M_CENTERS, _M_HEIGHTS, _M_SIGMAS):
        d2 = np.sum((x - c) ** 2, axis=1)
        y = np.maximum(y, h * np.exp(-d2 / (2 * s ** 2)))
    return y


# ─── Complex: 8D, 5 Gaussian peaks ──────────────────────────────────────────
_C_CENTERS = np.array([
    [0.15, 0.85, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],  # global (h=10)
    [0.85, 0.15, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],  # deceptive (h=9)
    [0.50, 0.50, 0.85, 0.15, 0.50, 0.50, 0.50, 0.50],  # moderate (h=6)
    [0.50, 0.50, 0.50, 0.50, 0.85, 0.15, 0.50, 0.50],  # moderate (h=5)
    [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.85, 0.15],  # moderate (h=4)
])
_C_HEIGHTS = np.array([10.0, 9.0, 6.0, 5.0, 4.0])
_C_SIGMAS  = np.array([0.50, 0.55, 0.45, 0.45, 0.45])


def complex_func(x):
    """8D multimodal: 5 Gaussian peaks, deceptive landscape."""
    x = np.atleast_2d(x)
    y = np.zeros(len(x))
    for c, h, s in zip(_C_CENTERS, _C_HEIGHTS, _C_SIGMAS):
        d2 = np.sum((x - c) ** 2, axis=1)
        y = np.maximum(y, h * np.exp(-d2 / (2 * s ** 2)))
    return y


# ─── Discretization helper ───────────────────────────────────────────────────
def discretize_nd(func, ndim, name, n_candidates=20000, seed=42):
    """Create a discrete dataset from an n-D function."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n_candidates, ndim))
    y = func(X)
    return {
        "candidates": X,
        "y_true": y,
        "name": name,
    }


def make_dataset(space_entry, n_candidates=None, seed=42):
    """Create a discrete dataset from a VIRTUAL_2D_SPACES entry."""
    n = n_candidates or space_entry["n_candidates"]
    return discretize_nd(space_entry["func"], space_entry["ndim"],
                         space_entry["name"], n_candidates=n, seed=seed)


# ─── Original 2D functions (for heatmap/trajectory visualization) ────────────
BOUNDS_2D = [(0.0, 1.0), (0.0, 1.0)]

_S2_CENTER = np.array([0.50, 0.50])
_S2_SIGMA  = 0.18


def simple_2d(x):
    """Unimodal: single Gaussian peak at center. Height = 10."""
    x = np.atleast_2d(x)
    d2 = np.sum((x - _S2_CENTER) ** 2, axis=1)
    return 10.0 * np.exp(-d2 / (2 * _S2_SIGMA ** 2))


def simple_2d_optimum():
    return float(simple_2d(_S2_CENTER.reshape(1, -1))[0]), _S2_CENTER.copy()


_M2_CENTERS = np.array([
    [0.20, 0.20],
    [0.80, 0.75],
    [0.55, 0.20],
])
_M2_HEIGHTS = np.array([10.0, 7.0, 5.0])
_M2_SIGMAS  = np.array([0.12, 0.14, 0.14])


def medium_2d(x):
    """3 Gaussian peaks at different corners."""
    x = np.atleast_2d(x)
    y = np.zeros(len(x))
    for c, h, s in zip(_M2_CENTERS, _M2_HEIGHTS, _M2_SIGMAS):
        d2 = np.sum((x - c) ** 2, axis=1)
        y = np.maximum(y, h * np.exp(-d2 / (2 * s ** 2)))
    return y


def medium_2d_optimum():
    return float(medium_2d(_M2_CENTERS[0].reshape(1, -1))[0]), _M2_CENTERS[0].copy()


_C2_CENTERS = np.array([
    [0.15, 0.85],
    [0.85, 0.15],
    [0.50, 0.50],
    [0.15, 0.15],
    [0.85, 0.85],
])
_C2_HEIGHTS = np.array([10.0, 9.0, 6.0, 5.0, 4.0])
_C2_SIGMAS  = np.array([0.08, 0.18, 0.12, 0.10, 0.10])


def complex_2d(x):
    """5 peaks + sinusoidal deceptive background."""
    x = np.atleast_2d(x)
    y = np.zeros(len(x))
    for c, h, s in zip(_C2_CENTERS, _C2_HEIGHTS, _C2_SIGMAS):
        d2 = np.sum((x - c) ** 2, axis=1)
        y = np.maximum(y, h * np.exp(-d2 / (2 * s ** 2)))
    y += 0.5 * np.sin(8 * np.pi * x[:, 0]) * np.cos(6 * np.pi * x[:, 1])
    return y


def complex_2d_optimum():
    g = np.linspace(0, 1, 1000)
    xx, yy = np.meshgrid(g, g)
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])
    y_grid = complex_2d(X_grid)
    best_idx = int(np.argmax(y_grid))
    return float(y_grid[best_idx]), X_grid[best_idx].copy()


# ─── Registry ─────────────────────────────────────────────────────────────────
VIRTUAL_2D_SPACES = [
    {
        "name": "Simple-3D",
        "tag": "Unimodal (1 peak, 3D)",
        "func": simple_func,
        "ndim": 3,
        "n_candidates": 10000,
        "complexity": 1,
    },
    {
        "name": "Medium-5D",
        "tag": "Trimodal (3 peaks, 5D)",
        "func": medium_func,
        "ndim": 5,
        "n_candidates": 20000,
        "complexity": 2,
    },
    {
        "name": "Complex-8D",
        "tag": "Multimodal (5 peaks, 8D)",
        "func": complex_func,
        "ndim": 8,
        "n_candidates": 50000,
        "complexity": 3,
    },
]
