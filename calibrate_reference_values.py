"""Recalculate the reference extrema and write build/paper/reference_values.json."""
import json
import os
import platform
import sys
from datetime import date

import numpy as np
import scipy
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
sys.path.insert(0, _ROOT)

from benchmarks.reference_values import (  # noqa: E402
    PCT_OPT_RTOL,
    key_for,
)
from benchmarks.virtual_functions import (  # noqa: E402
    RATIO_COMPLEXITY_LEVELS,
    arrhenius_yield,
    lennard_jones,
    muller_brown,
    nd_clusters,
)

OUT_PATH = os.path.join(_ROOT, "build", "paper", "reference_values.json")
GRID_RES = 500      # the resolution arrhenius_optimum() already uses
CHECK_RES = 2000    # independent cross-check, finer than the working grid
MIN_STARTS = 8
MAX_STARTS = 12
CHUNK = 500_000     # rows per function call on the cross-check grid


def _grid(res):
    xs = np.linspace(0.0, 1.0, res)
    xx, yy = np.meshgrid(xs, xs)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _evaluate_in_chunks(func, points):
    """Keep peak memory bounded on the four-million-row cross-check grid.

    Measured: chunked evaluation leaves peak memory flat, evaluating the whole
    grid at once adds about 214 MiB. Both give the same value bit for bit.
    """
    best = -np.inf
    for start in range(0, len(points), CHUNK):
        best = max(best, float(func(points[start:start + CHUNK]).max()))
    return best


def _evaluate_min_in_chunks(func, points):
    """The minimum over the cross-check grid, in the same chunked style.

    arrhenius gets y_min = 0.0 from a strict lower-bound argument, but
    muller_brown and lennard_jones have no such argument: their background
    keeps the attained minimum positive. Their y_min is measured here, on the
    finer cross-check grid, because the minimum moves between the 500-point
    and 2000-point grids.
    """
    best = np.inf
    for start in range(0, len(points), CHUNK):
        best = min(best, float(func(points[start:start + CHUNK]).min()))
    return best


def _spread_starts(points, values, min_distance, min_starts, max_starts):
    """Best grid points that are at least min_distance apart.

    Taking the top n points outright does not work: on the default Arrhenius
    surface the top 20 grid points lie within 0.009 of each other while the
    peak width is 0.08, so every one of them sits in the same basin and the
    extra refinements repeat the first. Walking down the value-ordered list
    and keeping only points far enough from the ones already accepted buys
    real basin coverage for the same effort.

    A wide peak leaves no room for min_starts points at that spacing, so the
    spacing halves and the walk repeats. The distance actually used is
    returned, because it belongs in the record.
    """
    order = np.argsort(values)[::-1]
    for _ in range(4):
        starts = []
        for idx in order:
            point = points[idx]
            if all(np.linalg.norm(point - taken) > min_distance
                   for taken in starts):
                starts.append(point)
                if len(starts) >= max_starts:
                    break
        if len(starts) >= min_starts:
            return starts, min_distance
        min_distance /= 2.0
    raise ValueError(
        f"only {len(starts)} starts at a spacing of {min_distance!r}, "
        f"below the {min_starts} this landscape is meant to get"
    )


def _distinct_maxima(refined, tolerance=1e-9):
    """Group the refined results by value, best first.

    How many distinct local maxima the starts reached is the only direct
    evidence that they covered more than one basin, so it is recorded. On the
    default Arrhenius surface there are exactly two: 18.3522, reached from
    seven starts, and 13.5908, reached from five. Peaks 1 and 2 of that
    surface sit closer together than one peak width and merge into the first.
    """
    groups = []
    for value, point in sorted(refined, key=lambda pair: -pair[0]):
        for group in groups:
            if abs(group["y"] - value) <= tolerance * max(1.0, abs(value)):
                group["n_starts"] += 1
                break
        else:
            groups.append({"y": value, "x": [float(v) for v in point],
                           "n_starts": 1})
    return groups


def arrhenius_reference(n_peaks=3, peak_width=0.08, noise_sigma=0.0, seed=42):
    if noise_sigma > 0:
        raise ValueError(
            "noise_sigma > 0 has no reference value. The noise is drawn per "
            "call by row index, so y is not a function of x; a realised value "
            "can also exceed the noise-free optimum."
        )

    # The working grid only has to put one start inside the global basin, but
    # the cross-check grid has to resolve every peak, or a missed basin goes
    # unnoticed. Measured: at peak_width 0.08 the grid spacing is 2.004e-3,
    # forty times finer than the peak, and refinement adds only 0.0033%. At
    # peak_width 0.005 the same grid reads 10.059311 against a refined
    # 10.300006, an error of 2.4%.
    needed = 10.0 / peak_width + 1
    if CHECK_RES < needed:
        raise ValueError(
            f"peak_width={peak_width} needs a cross-check grid of at least "
            f"{needed:.0f} points per axis, and CHECK_RES is {CHECK_RES}"
        )

    def surface(points):
        return arrhenius_yield(points, n_peaks=n_peaks, noise_sigma=0.0,
                               peak_width=peak_width, seed=seed)

    points = _grid(GRID_RES)
    values = surface(points)
    grid_best = float(values.max())

    # The peak width is given in x units on both axes: the temperature axis
    # maps 0-1 onto a 400 K range and sigma_T is peak_width * 400 K, and the
    # catalyst axis uses peak_width directly.
    starts, spacing = _spread_starts(points, values, 2.0 * peak_width,
                                     MIN_STARTS, MAX_STARTS)

    refined = []
    for x0 in starts:
        res = minimize(lambda x: -float(surface(np.atleast_2d(x))[0]), x0,
                       method="L-BFGS-B", bounds=[(0.0, 1.0), (0.0, 1.0)])
        refined.append((-float(res.fun), res.x))
    best_value, best_x = max(refined, key=lambda pair: pair[0])

    if best_value < grid_best:
        # The grid point itself is always available, so this cannot lose.
        best_value, best_x = grid_best, points[int(np.argmax(values))]

    check_best = _evaluate_in_chunks(surface, _grid(CHECK_RES))
    if check_best > best_value:
        raise AssertionError(
            f"the {CHECK_RES}-point cross-check grid found {check_best!r}, "
            f"above the refined optimum {best_value!r}: the starts missed a basin"
        )

    # Zero is a strict lower bound because every term is non-negative: the
    # background is 0.5 * sin(pi x0) * sin(pi x1) with both sines non-negative
    # on the unit square, and each peak term is a product of exponentials. The
    # trailing np.maximum(y, 0) in the surface never fires without noise. The
    # infimum actually attained on the domain is about 5.3e-18, at a corner.
    y_min = 0.0

    return {
        "landscape": "arrhenius",
        "params": {"n_peaks": n_peaks, "peak_width": peak_width,
                   "noise_sigma": 0.0, "seed": seed},
        "y_opt": best_value,
        "y_min": y_min,
        "x_opt": [float(v) for v in best_x],
        "threshold_95": y_min + 0.95 * (best_value - y_min),
        "y_opt_basis": (
            "global maximum on [0,1]^2. Verified by a cross-check grid of "
            f"{CHECK_RES} points per axis and by L-BFGS-B refinement from "
            f"starts at least {spacing!r} apart in x units."),
        "y_min_basis": (
            "strict lower bound: every term of the surface is non-negative. "
            "The minimum actually attained on the domain is about 5.3e-18."),
        "guard_grid_res": GRID_RES,
        "guard_grid_max": grid_best,
        "cross_check_grid_res": CHECK_RES,
        "cross_check_grid_max": check_best,
        "n_starts": len(starts),
        "start_min_distance": spacing,
        "local_maxima": _distinct_maxima(refined),
    }


def nd_clusters_reference(ndim, n_peaks, peak_width, seed=42):
    """Both ends are exact here, so nothing is searched.

    Each cluster contributes 10.0 * exp(-d^2 / 2 sigma^2) and the clusters are
    combined with a running maximum rather than a sum, so no cluster can lift
    another. A flat 0.5 is added afterwards. Cluster centres are drawn inside
    [0.15, 0.85], so a centre is always inside the search box and the peak
    value 10.0 + 0.5 is reached exactly. Away from every centre the peak terms
    vanish, so 0.5 is approached but never reached.

    The distance uses only a subset of the axes, so the optimum is attained on
    a whole subspace rather than at an isolated point. The centres and that
    subset are recorded here because they are drawn from one seeded stream in
    a fixed order and cannot be recovered afterwards for a different ndim.
    """
    rng_params = np.random.RandomState(seed + 600)
    centers = rng_params.uniform(0.15, 0.85, (n_peaks, ndim))
    n_key = max(2, ndim // 2)
    key_dims = rng_params.choice(ndim, size=n_key, replace=False)

    y_opt, y_min = 10.5, 0.5

    at_centres = nd_clusters(centers, n_peaks=n_peaks, ndim=ndim,
                             peak_width=peak_width, noise_sigma=0.0, seed=seed)
    gap = float(np.abs(at_centres - y_opt).max())
    if gap > 0.0:
        raise AssertionError(
            f"ndim={ndim}: the surface reads {at_centres} at its own cluster "
            f"centres, off the analytic {y_opt} by {gap!r}"
        )

    return {
        "landscape": "nd_clusters",
        "params": {"ndim": ndim, "n_peaks": n_peaks,
                   "peak_width": peak_width, "noise_sigma": 0.0, "seed": seed},
        "y_opt": float(y_opt),
        "y_min": float(y_min),
        "x_opt": [float(v) for v in centers[0]],
        "threshold_95": y_min + 0.95 * (y_opt - y_min),
        "y_opt_basis": (
            "exact: the clusters combine by running maximum so the peak value "
            "is 10.0, the flat background adds 0.5, and a cluster centre lies "
            "inside the search box. Attained on a whole subspace, because the "
            "distance uses only the key axes."),
        "y_min_basis": "approached, not attained: every peak term is positive",
        "centres": [[float(v) for v in row] for row in centers],
        "key_dims": [int(v) for v in key_dims],
    }


def _search_reference(landscape_name, n_peaks, peak_width, seed, surface):
    """The shared grid + L-BFGS-B + cross-check search for the 2D terrains.

    muller_brown and lennard_jones need the same search as arrhenius: they
    are smooth, cheap, two-dimensional, with no closed-form optimum. The
    search is shared; arrhenius keeps its own function because its y_min is a
    strict lower bound (0.0) rather than a measurement, but the search body is
    identical. y_min here is measured on the cross-check grid, and the second
    peak is the second-highest distinct refined maximum.
    """
    needed = 10.0 / peak_width + 1
    if CHECK_RES < needed:
        raise ValueError(
            f"peak_width={peak_width} needs a cross-check grid of at least "
            f"{needed:.0f} points per axis, and CHECK_RES is {CHECK_RES}"
        )

    points = _grid(GRID_RES)
    values = surface(points)
    grid_best = float(values.max())

    starts, spacing = _spread_starts(points, values, 2.0 * peak_width,
                                     MIN_STARTS, MAX_STARTS)

    refined = []
    for x0 in starts:
        res = minimize(lambda x: -float(surface(np.atleast_2d(x))[0]), x0,
                       method="L-BFGS-B", bounds=[(0.0, 1.0), (0.0, 1.0)])
        refined.append((-float(res.fun), res.x))
    best_value, best_x = max(refined, key=lambda pair: pair[0])

    if best_value < grid_best:
        best_value, best_x = grid_best, points[int(np.argmax(values))]

    check_best = _evaluate_in_chunks(surface, _grid(CHECK_RES))
    if check_best > best_value:
        raise AssertionError(
            f"the {CHECK_RES}-point cross-check grid found {check_best!r}, "
            f"above the refined optimum {best_value!r}: the starts missed a basin"
        )

    y_min = _evaluate_min_in_chunks(surface, _grid(CHECK_RES))
    maxima = _distinct_maxima(refined)

    return {
        "landscape": landscape_name,
        "params": {"n_peaks": n_peaks, "peak_width": peak_width,
                   "noise_sigma": 0.0, "seed": seed},
        "y_opt": best_value,
        "y_min": y_min,
        "x_opt": [float(v) for v in best_x],
        "threshold_95": y_min + 0.95 * (best_value - y_min),
        "y_opt_basis": (
            "global maximum on [0,1]^2. Verified by a cross-check grid of "
            f"{CHECK_RES} points per axis and by L-BFGS-B refinement from "
            f"starts at least {spacing!r} apart in x units."),
        "y_min_basis": (
            f"minimum measured on the {CHECK_RES}-point cross-check grid. "
            "No closed-form lower bound: the background keeps the attained "
            "minimum positive."),
        "guard_grid_res": GRID_RES,
        "guard_grid_max": grid_best,
        "cross_check_grid_res": CHECK_RES,
        "cross_check_grid_max": check_best,
        "n_starts": len(starts),
        "start_min_distance": spacing,
        "local_maxima": maxima,
    }


def _second_peak(maxima, peak_width):
    """The highest distinct maximum at least one peak width from the global.

    The Lennard-Jones Morse well is a ring: the maximum sits on a circle of
    radius re = peak_width / 2, and the centre is lower. L-BFGS-B can stop at
    two points on that ring a fraction of a peak width apart, and those are
    the same peak, not a second one. Requiring a whole peak width of separation
    skips such stops and returns the next real well instead. For a Gaussian
    well (muller_brown) the second peak is already farther than one width, so
    this is a no-op there.
    """
    global_x = np.array(maxima[0]["x"])
    for group in maxima[1:]:
        if np.linalg.norm(np.array(group["x"]) - global_x) >= peak_width:
            return group["y"], group["x"]
    raise AssertionError(
        f"no second basin at least {peak_width} from the global optimum: "
        f"{len(maxima)} maxima, none far enough")


def muller_brown_reference(n_peaks=3, peak_width=0.08, noise_sigma=0.0, seed=42):
    if noise_sigma > 0:
        raise ValueError(
            "noise_sigma > 0 has no reference value. The noise is drawn per "
            "call by row index, so y is not a function of x; a realised value "
            "can also exceed the noise-free optimum."
        )

    def surface(points):
        return muller_brown(points, n_peaks=n_peaks, noise_sigma=0.0,
                            peak_width=peak_width, seed=seed)

    record = _search_reference("muller_brown", n_peaks, peak_width, seed,
                               surface)
    maxima = record["local_maxima"]
    if len(maxima) < 2:
        raise AssertionError(
            f"muller_brown n_peaks={n_peaks} refined to only {len(maxima)} "
            f"distinct maxima; a second peak is required for the multipeak steps"
        )
    record["second_peak"], record["second_peak_x"] = _second_peak(maxima,
                                                                  peak_width)
    return record


def lennard_jones_reference(n_peaks=3, peak_width=0.08, noise_sigma=0.0, seed=42):
    if noise_sigma > 0:
        raise ValueError(
            "noise_sigma > 0 has no reference value. The noise is drawn per "
            "call by row index, so y is not a function of x; a realised value "
            "can also exceed the noise-free optimum."
        )

    def surface(points):
        return lennard_jones(points, n_peaks=n_peaks, noise_sigma=0.0,
                             peak_width=peak_width, seed=seed)

    record = _search_reference("lennard_jones", n_peaks, peak_width, seed,
                               surface)
    maxima = record["local_maxima"]
    if len(maxima) < 2:
        raise AssertionError(
            f"lennard_jones n_peaks={n_peaks} refined to only {len(maxima)} "
            f"distinct maxima; a second peak is required for the multipeak steps"
        )
    record["second_peak"], record["second_peak_x"] = _second_peak(maxima,
                                                                  peak_width)
    return record


def main():
    entries = {}

    arrhenius = arrhenius_reference()
    entries[key_for("arrhenius", **arrhenius["params"])] = arrhenius

    for level in RATIO_COMPLEXITY_LEVELS:
        record = nd_clusters_reference(ndim=level["ndim"],
                                       n_peaks=level["n_peaks"],
                                       peak_width=level["peak_width"])
        entries[key_for("nd_clusters", **record["params"])] = record

    # Two density controls. Same ndim and n_peaks as RC3 (6D, 5 peaks);
    # only peak_width moves, to the two ends of the ratio-complexity axis. They
    # exist to separate the peak-width effect from the dimension effect, so the
    # axis can no longer be read as a pure complexity trend. Fixed before the
    # run; the guard below keeps every committed entry bit-for-bit.
    for peak_width in (0.10, 0.20):
        record = nd_clusters_reference(ndim=6, n_peaks=5,
                                       peak_width=peak_width)
        entries[key_for("nd_clusters", **record["params"])] = record

    # Calibrate both multipeak families at each tested peak count and width.
    for name, ref_fn in (("muller_brown", muller_brown_reference),
                         ("lennard_jones", lennard_jones_reference)):
        for peak_width in (0.08, 0.05, 0.03):
            for n_peaks in (3, 5):
                record = ref_fn(n_peaks=n_peaks, peak_width=peak_width)
                entries[key_for(name, **record["params"])] = record

    # Reproducibility guard: entries already on file must come back bit-for-bit.
    # New configurations only ADD keys; they never rewrite an existing one.
    # The comparison is exact now. An earlier version stripped second_peak_x
    # from the recomputed record only, while the committed record already had
    # it, so the guard failed on every 2D terrain for a reason that had nothing
    # to do with reproducibility. Comparing every field is stricter and, since
    # second_peak_x is itself committed, the right thing to hold.
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as handle:
            committed = json.load(handle)["entries"]
        for key, rec in committed.items():
            if key not in entries:
                raise AssertionError(
                    f"calibration dropped a committed key {key!r}")
            fresh = entries[key]
            if fresh != rec:
                differing = sorted(
                    k for k in set(fresh) | set(rec)
                    if fresh.get(k) != rec.get(k))
                raise AssertionError(
                    f"calibration changed a committed key {key!r}; "
                    f"differing fields: {differing}")

    document = {
        "generated": date.today().isoformat(),
        "generator": "calibrate_reference_values.py",
        "note": ("Only the continuous benchmark path reads this file. The "
                 "discrete path keeps taking both ends from its candidate "
                 "pool, and arrhenius_optimum() is unchanged."),
        "pct_opt_rtol": PCT_OPT_RTOL,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "entries": entries,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {OUT_PATH}")
    for key, record in sorted(entries.items()):
        print(f"  {key}")
        print(f"      y_opt {record['y_opt']!r}  y_min {record['y_min']!r}"
              f"  95% at {record['threshold_95']!r}")
        if "second_peak" in record:
            print(f"      second_peak {record['second_peak']!r} at "
                  f"{record['second_peak_x']!r}")
        for group in record.get("local_maxima", []):
            print(f"      local max {group['y']!r} from "
                  f"{group['n_starts']} starts")


if __name__ == "__main__":
    main()
