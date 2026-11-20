"""Starting designs on the original feature coordinates for the initialization benchmark."""
import numpy as np
from scipy.stats import qmc

from benchmarks.bench_main import _initial_points, sample_init


def _pairwise(points):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if len(points) < 2:
        return np.zeros((0, 0))
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def min_pair_distance(points):
    """Smallest distance between any two of the points."""
    dist = _pairwise(points)
    if len(dist) == 0:
        return 0.0
    upper = dist[np.triu_indices(len(dist), 1)]
    return float(upper.min())


def mean_nn_distance(points):
    """Mean distance from each point to its nearest other point."""
    dist = _pairwise(points)
    if len(dist) == 0:
        return 0.0
    np.fill_diagonal(dist, np.inf)
    return float(dist.min(axis=1).mean())


# -- discrete designs ---------------------------------------------------

def _uniform_indices(candidates, n, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(len(candidates), size=n, replace=False).tolist()


def _lhs_indices(candidates, n, seed):
    """Latin hypercube in the feature box, snapped to the nearest free row.

    Snapping is greedy in target order and never reuses a row: a design that
    returned the same candidate twice would spend two of the eight starting
    evaluations on one point.
    """
    candidates = np.asarray(candidates, dtype=float)
    lo = candidates.min(axis=0)
    hi = candidates.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)

    sampler = qmc.LatinHypercube(d=candidates.shape[1], seed=seed)
    targets = lo + sampler.random(n) * span

    chosen, taken, hits = [], np.zeros(len(candidates), dtype=bool), 0
    snap = []
    for target in targets:
        dists = np.linalg.norm(candidates - target, axis=1)
        nearest = int(np.argmin(dists))
        free = dists.copy()
        free[taken] = np.inf
        idx = int(np.argmin(free))
        if idx != nearest:
            hits += 1
        snap.append(float(free[idx]))
        taken[idx] = True
        chosen.append(idx)
    return chosen, {"snap_dist_mean": float(np.mean(snap)),
                    "snap_dist_max": float(np.max(snap)),
                    "snap_taken_hits": hits}


def _maxmin_indices(candidates, n, seed):
    """Greedy furthest-point design over the candidate library.

    The first point is drawn, not fixed, so the design is a function of the
    seed like the others. After that each step takes the candidate whose
    distance to the chosen set is largest, which is the standard max-min
    construction and the reason to call this a space-filling design at all.

    Ties are broken at random, not by index. On one-hot libraries the
    distances are integers, so ties are the norm rather than the exception:
    np.argmax alone walks the first few rows of the table and produces a
    design that is not space-filling at all. That was measured, not feared:
    the unbroken version returned indices 0, 2, 3, 6, 7, 9, 10 on
    Buchwald-Hartwig.
    """
    candidates = np.asarray(candidates, dtype=float)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(candidates)))]
    nearest = np.linalg.norm(candidates - candidates[chosen[0]], axis=1)
    for _ in range(n - 1):
        best = nearest.max()
        tied = np.flatnonzero(nearest >= best - 1e-12)
        idx = int(rng.choice(tied))
        chosen.append(idx)
        nearest = np.minimum(nearest,
                             np.linalg.norm(candidates - candidates[idx],
                                            axis=1))
    return chosen


def discrete_design(kind, candidates, y_true, seed, n=8):
    """Returns (indices, diagnostics) for one starting set.

    Indices are Python ints into `candidates`, all distinct and in range.
    """
    candidates = np.asarray(candidates, dtype=float)
    diag = {"snap_dist_mean": 0.0, "snap_dist_max": 0.0, "snap_taken_hits": 0}

    if kind == "bottom20":
        # The published rule, delegated so it cannot drift.
        indices = list(sample_init(candidates, y_true, n, seed))
    elif kind == "uniform":
        indices = _uniform_indices(candidates, n, seed)
    elif kind == "lhs":
        indices, extra = _lhs_indices(candidates, n, seed)
        diag.update(extra)
    elif kind == "maxmin":
        indices = _maxmin_indices(candidates, n, seed)
    else:
        raise ValueError(f"no discrete starting design named {kind!r}")

    indices = [int(i) for i in indices]
    _assert_valid_indices(indices, len(candidates), kind)
    points = candidates[indices]
    diag.update({"init_min_pair_dist": min_pair_distance(points),
                 "init_mean_nn_dist": mean_nn_distance(points)})
    return indices, diag


# -- continuous designs -------------------------------------------------

def _maxmin_points(landscape, seed, n, pool_n):
    lows = np.array([b[0] for b in landscape["bounds"]], dtype=float)
    highs = np.array([b[1] for b in landscape["bounds"]], dtype=float)
    rng = np.random.default_rng(seed)
    pool = rng.uniform(lows, highs, size=(pool_n, len(lows)))
    first = int(rng.integers(pool_n))
    chosen = [first]
    nearest = np.linalg.norm(pool - pool[first], axis=1)
    for _ in range(n - 1):
        idx = int(np.argmax(nearest))
        chosen.append(idx)
        nearest = np.minimum(nearest,
                             np.linalg.norm(pool - pool[idx], axis=1))
    return pool[chosen]


def continuous_design(kind, landscape, seed, n=8, pool_n=10_000):
    """Returns (points, diagnostics) for one continuous starting set."""
    if kind in ("bottom20", "uniform", "lhs"):
        # Delegated: these three already exist in the benchmark package, and
        # the published runs used the first of them.
        points = _initial_points(landscape, seed, n, kind)
    elif kind == "maxmin":
        points = _maxmin_points(landscape, seed, n, pool_n)
    else:
        raise ValueError(f"no continuous starting design named {kind!r}")

    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape != (n, landscape["ndim"]):
        raise AssertionError(
            f"design {kind!r} returned shape {points.shape}, "
            f"expected {(n, landscape['ndim'])}"
        )
    lows = np.array([b[0] for b in landscape["bounds"]], dtype=float)
    highs = np.array([b[1] for b in landscape["bounds"]], dtype=float)
    if np.any(points < lows) or np.any(points > highs):
        raise AssertionError(f"design {kind!r} left the box")
    diag = {"init_min_pair_dist": min_pair_distance(points),
            "init_mean_nn_dist": mean_nn_distance(points),
            "snap_dist_mean": 0.0, "snap_dist_max": 0.0,
            "snap_taken_hits": 0}
    return points, diag


def design(kind, spec, seed, n=8, candidates=None, y_true=None,
           landscape=None, pool_n=10_000):
    """Dispatch on the dataset path. Returns (start, diagnostics)."""
    if spec["path"] == "discrete":
        return discrete_design(kind, candidates, y_true, seed, n=n)
    return continuous_design(kind, landscape, seed, n=n, pool_n=pool_n)


def _assert_valid_indices(indices, n_rows, kind):
    if len(indices) != len(set(indices)):
        raise AssertionError(f"design {kind!r} returned a repeated index")
    for idx in indices:
        if not (0 <= idx < n_rows):
            raise AssertionError(
                f"design {kind!r} returned index {idx}, outside [0, {n_rows})"
            )
