"""RF benchmark runners for the three paper datasets and historical studies.

Use --study paper for all eleven configurations, or --study main9 for the
nine-arm comparison used for paired inference. Legacy five-arm experiments
and the single-seed pilot have separate study names.
"""

import argparse
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.spatial.distance import pdist
import pandas as pd

from benchmarks import protocol
from benchmarks import reference_values
from benchmarks.virtual_functions import nd_clusters
from benchmarks.rf_surrogate import make_rf
from benchmarks.local_penalty import LocalPenaltyAlgorithm
from benchmarks.algorithms import (GLOSSAlgorithm, BOAlgorithm, UCBOAlgorithm,
                                   GeneticAlgorithm, RandomAlgorithm,
                                   QPOAlgorithm, ThompsonSamplingAlgorithm)
from benchmarks.datasets import load_qm9, load_buchwald_hartwig
from gloss.gloss import GLOSS
from benchmarks.virtual_functions import (
    arrhenius_yield, arrhenius_optimum, COMPLEXITY_LEVELS,
    RATIO_COMPLEXITY_LEVELS, get_virtual_dataset, get_nd_cluster_dataset,
    muller_brown, lennard_jones,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RERUN_DIR = Path(__file__).resolve().parents[1] / "build" / "paper" / "reruns"

N_INIT = 8
N_ROUNDS = 20
N_POINTS = 8
SEEDS = [47, 53, 59, 61, 67]
# The scaling study uses the first ten seeds; the main study uses all twenty.
SEEDS_10 = SEEDS + [71, 73, 79, 83, 89]
SEEDS_20 = SEEDS_10 + [97, 101, 103, 107, 109, 113, 127, 131, 137, 139]

GLOSS_STRATEGY = {"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0}
GLOSS_DIVERSITY_RADIUS = 0.02

GLOSS_RATIOS = {
    "GLOSS(2:3:3)": {"global_best": 2, "local_best": 3, "unexplored": 3, "unconverged": 0},
    "GLOSS(4:2:2)": {"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0},
    "GLOSS(6:1:1)": {"global_best": 6, "local_best": 1, "unexplored": 1, "unconverged": 0},
}


def make_algorithms(seed):
    return {
        "GLOSS": GLOSSAlgorithm(
            strategy_points=GLOSS_STRATEGY, ucb_kappa=2.0,
            diversity_radius=GLOSS_DIVERSITY_RADIUS, seed=seed),
        "UCB-BO": UCBOAlgorithm(kappa=2.0, seed=seed),
        "BO(EI)": BOAlgorithm(seed=seed),
        "GA": GeneticAlgorithm(kappa=2.0, seed=seed),
        "Random": RandomAlgorithm(seed=seed),
    }


def make_ratio_algorithms(seed):
    return {
        name: GLOSSAlgorithm(
            strategy_points=sp, ucb_kappa=2.0,
            diversity_radius=GLOSS_DIVERSITY_RADIUS, seed=seed)
        for name, sp in GLOSS_RATIOS.items()
    }


GLOBAL_ONLY_STRATEGY = {"global_best": 8, "local_best": 0,
                        "unexplored": 0, "unconverged": 0}


def _gloss_arm(ratio, radius, novelty_signal="distance", novelty_quantile=None):
    def build(seed):
        return GLOSSAlgorithm(
            strategy_points={"global_best": ratio[0], "local_best": ratio[1],
                             "unexplored": ratio[2], "unconverged": 0},
            ucb_kappa=2.0, diversity_radius=radius, seed=seed,
            novelty_signal=novelty_signal, novelty_quantile=novelty_quantile)
    return build


# One place where an arm gets constructed. Before this, nine files each
# carried their own copy of the list, three of them as byte-identical
# necessarily the arms the next run used.
#
# make_algorithms() and make_ratio_algorithms() stay as they are. They are
# the entry points of Study 2 to Study 5, whose numbers must not move.
ARM_BUILDERS = {
    "GLOSS(4:2:2)": _gloss_arm((4, 2, 2), GLOSS_DIVERSITY_RADIUS),
    "GLOSS(2:3:3)": _gloss_arm((2, 3, 3), GLOSS_DIVERSITY_RADIUS),
    "GLOSS(6:1:1)": _gloss_arm((6, 1, 1), GLOSS_DIVERSITY_RADIUS),
    # The two arms of the diversity ablation. They differ from each other in
    # the radius alone, which is what that experiment isolates.
    "GLOSS(8:0:0)": _gloss_arm((8, 0, 0), 0.0),
    "GLOSS(8:0:0)+div": _gloss_arm((8, 0, 0), GLOSS_DIVERSITY_RADIUS),
    "UCB-BO": lambda seed: UCBOAlgorithm(kappa=2.0, seed=seed),
    "BO(EI)": lambda seed: BOAlgorithm(seed=seed),
    "GA": lambda seed: GeneticAlgorithm(kappa=2.0, seed=seed),
    "qPO": lambda seed: QPOAlgorithm(seed=seed),
    "Thompson": lambda seed: ThompsonSamplingAlgorithm(seed=seed),
    "Random": lambda seed: RandomAlgorithm(seed=seed),
    # three differ from it in the novelty gate alone.
    #   novdist: distance, rank gate at q=0.5 (the control, same gate shape)
    #   novsigma: sigma, rank gate at q=0.5 (the arm under test)
    #   GLOSS(4:2:2) with unconverged=2 (the library's own pure-sigma arm)
    "novdist": _gloss_arm((4, 2, 2), GLOSS_DIVERSITY_RADIUS,
                          novelty_quantile=0.5),
    "novsigma": _gloss_arm((4, 2, 2), GLOSS_DIVERSITY_RADIUS,
                           novelty_signal="sigma", novelty_quantile=0.5),
    "GLOSS(4:2:2)uc": lambda seed: GLOSSAlgorithm(
        strategy_points={"global_best": 4, "local_best": 2,
                         "unexplored": 0, "unconverged": 2},
        ucb_kappa=2.0, diversity_radius=GLOSS_DIVERSITY_RADIUS, seed=seed),
    # acquisition they penalize, so each pairs with the greedy arm of its own
    # acquisition: LocalPenalty(EI) against BO(EI), LocalPenalty(UCB) against
    # UCB-BO. That isolates batch construction from acquisition choice.
    "LocalPenalty(EI)": lambda seed: LocalPenaltyAlgorithm(
        base="ei", seed=seed),
    "LocalPenalty(UCB)": lambda seed: LocalPenaltyAlgorithm(
        base="ucb", kappa=2.0, seed=seed),
}

# test ran them. The order is pinned by a test as well as written here: no
# algorithm here touches the global numpy state, so the order does not change
# a result today, and the test is what keeps that true.
CONTINUOUS_ARMS = ["GLOSS(4:2:2)", "GLOSS(2:3:3)", "GLOSS(6:1:1)",
                   "UCB-BO", "BO(EI)", "GA", "qPO", "Thompson", "Random"]

# response experiments need. It swaps the two ratio arms, which belong to the
# ratio sweep, for the two arms of the diversity ablation.
RERUN_ARMS = ["GLOSS(4:2:2)", "GLOSS(8:0:0)", "GLOSS(8:0:0)+div",
              "UCB-BO", "BO(EI)", "GA", "qPO", "Thompson", "Random"]


PAPER_ARMS = RERUN_ARMS + ["GLOSS(2:3:3)", "GLOSS(6:1:1)"]

def make_arms(seed, names):
    """One algorithm object per (arm, seed). Never reuse one across seeds.

    Only GeneticAlgorithm.reset() really resets. The other classes inherit a
    reset() that does nothing, and two kinds of state outlive it: the rng that
    the two BO arms, Random, Thompson and qPO advance once per round, and
    GLOSSAlgorithm._call_count, which feeds seed * 10000 + _call_count
    straight into the random stream GLOSS draws from. A caller that builds the
    arms once and loops the seeds inside therefore gets different numbers with
    no error anywhere.
    """
    unknown = [n for n in names if n not in ARM_BUILDERS]
    if unknown:
        raise KeyError(f"no arm named {unknown!r}; the registry holds "
                       f"{sorted(ARM_BUILDERS)}")
    return {name: ARM_BUILDERS[name](seed) for name in names}


#: How many rows the per-round sigma statistics are computed on. A full
#: 100k-candidate pool would cost more than the run itself, and the plateau
#: share is a proportion, so a fixed subsample estimates it without bias in
#: any way that matters. The seed is fixed so the number is reproducible.
SIGMA_SAMPLE = 5000
SIGMA_SAMPLE_SEED = 20261025


def row_keys(a, decimals=10):
    """Rows as comparable scalars, for a vectorised "is this row in that set".

    Comparing rounded tuples in Python costs about 0.27 s per round on a 100k
    library, which is more than the sigma predictions it guards. Viewing each
    row as one fixed-width value makes membership a numpy operation.
    """
    a = np.round(np.ascontiguousarray(a, dtype=float), decimals)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def sigma_sample_pool(pool, observed=None):
    """The rows the sigma statistics are computed on.

    observed, when given, is removed first: those rows are already measured
    and no arm can propose them, so counting them would bias the share
    towards whichever part of the space has been visited.
    """
    pool = np.asarray(pool)
    if observed is not None and len(observed):
        pool = pool[~np.isin(row_keys(pool), row_keys(np.asarray(observed)))]
    if len(pool) > SIGMA_SAMPLE:
        rng = np.random.default_rng(SIGMA_SAMPLE_SEED)
        pool = pool[rng.choice(len(pool), SIGMA_SAMPLE, replace=False)]
    return pool


def sigma_share_and_cv(surrogate, X):
    """Share of X on the sigma plateau, and the coefficient of variation."""
    if len(X) == 0:
        return 0.0, 0.0
    if hasattr(surrogate, "sigma_stats"):
        stats = surrogate.sigma_stats(X)
        return stats["sigma_platform_share"], stats["sigma_cv"]
    sd = np.asarray(surrogate.predict(X, return_std=True)[1], dtype=float)
    top = float(np.max(sd))
    return (float(np.mean(sd >= top - 1e-6)),
            float(np.std(sd) / np.mean(sd)) if sd.size else 0.0)


# not pass n_top_for_optimization, so the package default applies; the
# continuous path passes protocol.N_TOP_FOR_OPTIMIZATION, which is 50. Read
# from the package instead of written down, so the two cannot drift apart.
_DISCRETE_N_TOP = inspect.signature(
    GLOSS.__init__).parameters["n_top_for_optimization"].default


def sample_init(candidates, y_true, n_init, seed):
    """Sample init points from bottom 20%."""
    rng = np.random.default_rng(seed)
    threshold = np.quantile(y_true, 0.2)
    pool = np.where(y_true <= threshold)[0]
    return rng.choice(pool, size=n_init, replace=False).tolist()


def run_discrete(dataset, algo_name, algo, seed, n_rounds=N_ROUNDS,
                 init_idx=None, surrogate_factory=None, surrogate_tag="rf"):
    candidates = dataset["candidates"]
    y_true = dataset["y_true"]
    y_opt = float(y_true.max())
    y_min = float(y_true.min())
    ds_name = dataset["name"]

    if init_idx is None:
        init_idx = sample_init(candidates, y_true, N_INIT, seed)
    X_obs = candidates[init_idx].copy()
    y_obs = y_true[init_idx].copy()

    # The genetic algorithm carries a population between rounds, so it has
    # to be told that a new run is starting. The others are stateless and
    # ignore this.
    if hasattr(algo, "reset"):
        algo.reset()

    summary = []
    points = []

    # The two paths divide pct_opt by different things: here the largest value
    # in the finite pool, in run_continuous the true optimum of the surface.
    # On Arrhenius the two differ by 1.40%, so a merged table that carries
    # only the name pct_opt states two quantities in one column. The three
    # columns below put the convention and both ends of it in every row.
    provenance = {
        "denominator": "pool_optimum", "y_opt": y_opt, "y_min": y_min,
        "protocol_version": protocol.VERSION, "code_revision": CODE_REVISION,
        "source_hash": SOURCE_HASH,
        # The protocol version alone cannot separate the two paths: both
        # write "1.0" while GLOSS scores a different number of candidates in
        # each. recommend_continuous passes the protocol's n_top, and
        # recommend_discrete passes nothing and gets the package default. So
        # the number that was actually used goes in the row, read from the
        # package rather than copied, and a continuous row is recognisable by
        "n_top_for_optimization": _DISCRETE_N_TOP,
    }

    best = float(y_obs.max())
    pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
    summary.append({"dataset": ds_name, "algorithm": algo_name, "seed": seed,
                     "round": -1, "best_so_far": best, "pct_opt": pct,
                     "time_s": 0.0, "diversity": 0.0, **provenance})
    for i, idx in enumerate(init_idx):
        row = {"dataset": ds_name, "algorithm": algo_name, "seed": seed,
               "round": -1, "point_idx": i, "candidate_idx": int(idx),
               "y": float(y_true[idx]), "strategy": "init"}
        # store first 2 features for 2D compatibility
        row["x0"] = float(candidates[idx, 0]) if candidates.shape[1] > 0 else ""
        row["x1"] = float(candidates[idx, 1]) if candidates.shape[1] > 1 else ""
        points.append(row)

    make_surrogate = surrogate_factory or (lambda s: make_rf(seed=s))
    for r in range(n_rounds):
        t0 = time.time()
        rf = make_surrogate(seed * 100 + r)
        rf.fit(X_obs, y_obs)

        # The measured set as the arm sees it this round, before this round's
        # batch is appended. The sigma statistics below are defined on that
        # set, not on the set that already contains the points being chosen.
        X_obs_round = X_obs
        if isinstance(algo, GLOSSAlgorithm):
            indices, strategies = algo.recommend_discrete(
                X_obs, y_obs, candidates, y_true, N_POINTS, rf)
        else:
            indices = algo.recommend_discrete(
                X_obs, y_obs, candidates, y_true, N_POINTS, rf)
            strategies = ["recommended"] * len(indices)

        elapsed = time.time() - t0

        if len(indices) == 0:
            rng = np.random.default_rng(seed * 1000 + r)
            obs_set = set(map(tuple, np.round(X_obs, 10)))
            avail = [i for i in range(len(candidates))
                     if tuple(np.round(candidates[i], 10)) not in obs_set]
            indices = rng.choice(avail, size=min(N_POINTS, len(avail)), replace=False).tolist()
            strategies = ["fallback"] * len(indices)

        new_X = candidates[indices]
        new_y = y_true[indices]
        X_obs = np.vstack([X_obs, new_X])
        y_obs = np.concatenate([y_obs, new_y])

        best = float(y_obs.max())
        pct = (best - y_min) / (y_opt - y_min) * 100 if y_opt > y_min else 0.0
        from scipy.spatial.distance import pdist
        div = float(pdist(X_obs[N_INIT:]).mean()) if len(X_obs) > N_INIT + 1 else 0.0

        row_sum = {"dataset": ds_name, "algorithm": algo_name, "seed": seed,
                   "round": r, "best_so_far": best, "pct_opt": pct,
                   "time_s": elapsed, "diversity": div,
                   "surrogate": surrogate_tag,
                   "n_points": len(indices), **provenance}
        # Sigma statistics for this round, on the candidates the arm could
        # choose from. The share sitting on the plateau is what decides
        # whether a sigma-based gate can rank anything at all, so it is
        # measured here rather than in a side experiment.
        scored = sigma_sample_pool(candidates, X_obs_round)
        if len(scored):
            _share, _cv = sigma_share_and_cv(rf, scored)
            row_sum["sigma_platform_share"] = _share
            row_sum["sigma_cv"] = _cv
        # Novelty-gate statistics come from the strategy layer, which is the
        # only place that knows how many candidates passed the gate.
        for field in ("novelty_pass_frac", "all_tied"):
            stats = getattr(algo, "last_novelty_stats", None)
            if stats and field in stats:
                row_sum[field] = stats[field]
            elif hasattr(algo, field):
                row_sum[field] = getattr(algo, field)
        # Discrete selection does not refine coordinates.
        row_sum["postfilter_dropped"] = int(getattr(algo, "postfilter_dropped", 0))
        # Compare plateau shares from the scored pool and the diagnostic subsample.
        _stats = getattr(algo, "last_novelty_stats", None) or {}
        if "sigma_platform_share" in _stats:
            row_sum["sigma_platform_share_gate"] = _stats["sigma_platform_share"]
        # Whatever an algorithm knows about its own round goes into the
        # result file, so the numbers quoted in the paper come out of the run
        # that made the figures instead of a separate measurement.
        for field in ("genotype_kind", "sigma_share", "n_ucb_requested",
                      "n_predictions", "n_library_scored", "n_fill"):
            if hasattr(algo, field):
                row_sum[field] = getattr(algo, field)
        # The same quantity goes by n_fill here and n_fill_internal in
        # run_continuous, so a merged table turns both into half a column of
        # NaN. n_fill_internal is the name to use; n_fill stays for the
        # scripts that already read it.
        if "n_fill" in row_sum:
            row_sum["n_fill_internal"] = row_sum["n_fill"]
        summary.append(row_sum)

        filled = set(getattr(algo, "fill_indices", []))
        for i, (idx, strat) in enumerate(zip(indices, strategies)):
            if int(idx) in filled:
                strat = "ga_fill"
            row = {"dataset": ds_name, "algorithm": algo_name, "seed": seed,
                   "round": r, "point_idx": i, "candidate_idx": int(idx),
                   "y": float(y_true[idx]), "strategy": strat}
            row["x0"] = float(candidates[idx, 0]) if candidates.shape[1] > 0 else ""
            row["x1"] = float(candidates[idx, 1]) if candidates.shape[1] > 1 else ""
            points.append(row)

        if r % 5 == 0:
            print(f"    [{algo_name:12s}] seed={seed} r={r:2d}: "
                  f"best={best:.4f} pct={pct:.1f}%  ({elapsed:.1f}s)")

    return summary, points


def make_continuous_algorithms(seed, names=None):
    return make_arms(seed, names or CONTINUOUS_ARMS)


def _runner_version():
    """Fingerprint of the runner's source, recorded in each result file."""
    # The functions whose source affects the result numbers.
    source = "".join(inspect.getsource(fn) for fn in (
        run_continuous, _top_up, _initial_points, continuous_landscape))
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def _code_revision():
    """Identify either the Git checkout or the sources in a downloaded archive."""
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        try:
            head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           text=True, timeout=10, cwd=root).strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                            text=True, timeout=10, cwd=root)
            return head + ("+dirty" if dirty.strip() else "")
        except (OSError, subprocess.SubprocessError):
            pass
    digest = hashlib.sha256()
    for package in ("benchmarks", "gloss"):
        for path in sorted((root / package).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return "source-" + digest.hexdigest()[:16]


CODE_REVISION = _code_revision()


def _source_hash():
    """Fingerprint of the modules whose behaviour the results depend on."""
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(here, "algorithms.py"),
             os.path.join(here, "gp_surrogate.py"),
             os.path.join(here, "protocol.py"),
             os.path.join(here, "rf_surrogate.py"),
             os.path.join(os.path.dirname(here), "gloss", "gloss.py"),
             os.path.join(os.path.dirname(here), "gloss", "strategies",
                          "unexplored.py")]
    digest = hashlib.sha256()
    for path in paths:
        try:
            with open(path, "rb") as handle:
                digest.update(handle.read())
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()[:16]


SOURCE_HASH = _source_hash()


def continuous_landscape(name, **params):
    """One spec that yields the surface, its bounds and its two metric ends.

    Both come from the same parameters on purpose. The surface takes a
    landscape seed (42 for every shipped configuration) while a run takes its
    own seed, and building the closure from one while looking the reference
    values up under the other gives a wrong denominator with no error at all.
    """
    reference = reference_values.get(name, **params)
    ndim = len(reference["x_opt"])

    if name == "arrhenius":
        def surface(points):
            return arrhenius_yield(points, n_peaks=params["n_peaks"],
                                   noise_sigma=params["noise_sigma"],
                                   peak_width=params["peak_width"],
                                   seed=params["seed"])
    elif name == "nd_clusters":
        def surface(points):
            return nd_clusters(points, n_peaks=params["n_peaks"],
                               ndim=params["ndim"],
                               peak_width=params["peak_width"],
                               noise_sigma=params["noise_sigma"],
                               seed=params["seed"])
    elif name == "muller_brown":
        def surface(points):
            return muller_brown(points, n_peaks=params["n_peaks"],
                                noise_sigma=params["noise_sigma"],
                                peak_width=params["peak_width"],
                                seed=params["seed"])
    elif name == "lennard_jones":
        def surface(points):
            return lennard_jones(points, n_peaks=params["n_peaks"],
                                 noise_sigma=params["noise_sigma"],
                                 peak_width=params["peak_width"],
                                 seed=params["seed"])
    else:
        raise ValueError(f"no continuous surface named {name!r}")

    # The closure and the stored optimum have to describe the same surface.
    #
    # Two-sided, because the error this guards against is one-sided the other
    # way: a closure built with the wrong parameters reads LOWER at the stored
    # optimum, not higher. Measured with the peak width set to 0.05 it reads
    # 7.80 against a stored 18.35, and with the landscape seed replaced by a
    # run seed it reads 0.4953. A one-sided check waved both through.
    at_opt = float(surface(np.atleast_2d(reference["x_opt"]))[0])
    slack = reference["y_opt"] * reference_values.PCT_OPT_RTOL
    assert abs(at_opt - reference["y_opt"]) <= slack, (
        f"{name}: the surface reads {at_opt!r} at the stored optimum "
        f"{reference['y_opt']!r}. The closure and the reference values "
        f"describe different surfaces."
    )

    # One point of agreement is not much, so each landscape gets a second,
    # independent check of its own shape.
    if name in ("arrhenius", "muller_brown", "lennard_jones"):
        # The stored guard is the maximum over the same grid the calibration
        # walked. Recomputing it costs 0.01 s and every wrong-parameter
        # closure above moves it. muller_brown's peak height is width-
        # independent, so this guard is weak to a wrong width there; the
        # at-opt check above is the backstop.
        res = reference["guard_grid_res"]
        axis = np.linspace(0.0, 1.0, res)
        mesh = np.meshgrid(axis, axis)
        grid_max = float(surface(
            np.column_stack([m.ravel() for m in mesh])).max())
        assert abs(grid_max - reference["guard_grid_max"]) <= (
            reference["guard_grid_max"] * reference_values.PCT_OPT_RTOL), (
            f"{name}: the {res}-point grid reads {grid_max!r} against the "
            f"stored {reference['guard_grid_max']!r}"
        )
    else:
        # No grid is affordable in ten dimensions, so the shape is probed at a
        # fixed offset from each stored cluster centre, along a key axis. The
        # expected value follows from the peak width in the lookup key, so a
        # closure built with a different width disagrees at once. This branch
        # is nd_clusters only; the new 2D terrains take the grid branch above.
        offset = 0.05
        width = params["peak_width"]
        expected = 10.0 * np.exp(-offset ** 2 / (2 * width ** 2)) + 0.5
        probes = np.array(reference["centres"], dtype=float)
        at_centres = float(surface(probes).min())
        assert abs(at_centres - reference["y_opt"]) <= slack, (
            f"{name}: a stored cluster centre reads {at_centres!r}, not the "
            f"stored {reference['y_opt']!r}"
        )
        probes[:, reference["key_dims"][0]] += offset
        for value in surface(probes):
            assert abs(value - expected) <= expected * 1e-9, (
                f"{name}: {offset} from a centre reads {value!r}, and a peak "
                f"width of {width!r} says {expected!r}"
            )

    # Which axes the objective actually depends on. The cluster surface takes
    # its distance over a subset, so the rest carry no selection pressure at
    # all; a spread measured over every axis then reports the axes the
    # objective ignores. Measured at ten dimensions, that reverses the sign:
    # the full-axis figure says GLOSS spreads its batch wider than a random
    # draw (ratio 1.013) while on the five axes that matter it packs 17%
    # tighter (0.826).
    key_dims = reference.get("key_dims") or list(range(ndim))

    return {
        "name": name,
        "params": dict(params),
        "surface": surface,
        "bounds": [(0.0, 1.0)] * ndim,
        "ndim": ndim,
        "key_dims": list(key_dims),
        "y_opt": reference["y_opt"],
        "y_min": reference["y_min"],
    }


def _initial_points(landscape, seed, n_init, kind):
    rng = np.random.default_rng(seed)
    lows = np.array([b[0] for b in landscape["bounds"]])
    highs = np.array([b[1] for b in landscape["bounds"]])
    ndim = landscape["ndim"]

    if kind == "bottom20":
        scouts = rng.uniform(lows, highs, size=(50_000, ndim))
        values = landscape["surface"](scouts)
        worst = scouts[values <= np.quantile(values, 0.2)]
        return worst[rng.choice(len(worst), size=n_init, replace=False)]
    if kind == "uniform":
        return rng.uniform(lows, highs, size=(n_init, ndim))
    if kind == "lhs":
        cut = (np.arange(n_init) + rng.random((ndim, n_init))) / n_init
        return np.column_stack([
            lows[d] + (highs[d] - lows[d]) * rng.permutation(cut[d])
            for d in range(ndim)
        ])
    raise ValueError(f"no initialisation rule named {kind!r}")


def _top_up(chosen, pool, measured, rng):
    free = np.arange(len(pool))
    if len(chosen):
        chosen = np.atleast_2d(np.asarray(chosen, dtype=float))
        gaps = np.linalg.norm(pool[:, None, :] - chosen[None, :, :],
                              axis=-1).min(axis=1)
        free = free[gaps >= protocol.MIN_DISTANCE_WITHIN_BATCH]
    if len(free) == 0:
        raise RuntimeError(
            f"the filtered pool holds {len(pool)} rows and none of them is "
            f"{protocol.MIN_DISTANCE_WITHIN_BATCH} clear of this batch, so "
            f"there is nothing left to top up with"
        )
    point = pool[int(rng.choice(free, size=1, replace=False)[0])]
    # The pool was filtered against the measured set, so this has to hold.
    # Checking it here keeps the guarantee local to the function that makes
    # it, and catches an unfiltered pool being handed in.
    assert protocol.far_enough(point, measured), (
        "a top-up point sat inside the exclusion zone, so the pool handed in "
        "was not the filtered one"
    )
    return point


def run_continuous(landscape, ds_name, algo_name, algo, seed,
                   n_rounds=N_ROUNDS, n_points=N_POINTS, n_init=N_INIT,
                   init_kind="bottom20", init_X=None, pool_size=None,
                   pool_coverage=False, surrogate_factory=None,
                   surrogate_tag="rf", refinement_evidence=None):
    surface = landscape["surface"]
    bounds = landscape["bounds"]
    ndim = landscape["ndim"]
    y_opt, y_min = landscape["y_opt"], landscape["y_min"]

    if pool_size is None:
        pool_size = protocol.POOL_SIZE
    if pool_coverage:
        noise = float(landscape["params"].get("noise_sigma") or 0.0)
        if noise != 0.0:
            raise ValueError(
                f"pool_coverage is defined only on a noiseless surface, and "
                f"{landscape['name']} has noise_sigma={noise!r}. A noisy "
                f"surface draws its noise per call, so a candidate does not "
                f"have one y and the count is not a set of qualifying points."
            )
    threshold_95 = y_min + 0.95 * (y_opt - y_min)

    if init_X is None:
        init_X = _initial_points(landscape, seed, n_init, init_kind)
    init_X = np.atleast_2d(np.asarray(init_X, dtype=float))
    n_init = len(init_X)
    init_y = surface(init_X)

    # The genetic algorithm carries a population between rounds, so it has
    # to be told that a new run is starting. The others are stateless and
    # ignore this.
    if hasattr(algo, "reset"):
        algo.reset()

    requested = getattr(algo, "_strategy_points", None) or {}

    X_obs, y_obs = init_X.copy(), init_y.copy()
    summary, points = [], []

    # Initialised here so the closure below can read them on the round -1 row,
    # which is written before the first round assigns either.
    pool = None
    rf = None

    def record_summary(r, best, elapsed, bought, n_fill, n_internal,
                       pool_rows, raw_pool_hash=None,
                       delivered_by_strategy=None, coverage=None):
        gap = y_opt - y_min
        pct = (best - y_min) / gap * 100 if gap > 0 else 0.0
        fresh = X_obs[n_init:]
        spread = float(pdist(fresh).mean()) if len(fresh) > 1 else 0.0
        key_dims = landscape["key_dims"]
        key_spread = (float(pdist(fresh[:, key_dims]).mean())
                      if len(fresh) > 1 else 0.0)
        key_reference = protocol.uniform_mean_distance(len(key_dims))
        row = {
            "dataset": ds_name, "algorithm": algo_name, "seed": seed,
            "round": r, "ndim": ndim, "best_so_far": best, "pct_opt": pct,
            "time_s": elapsed, "diversity": spread,
            # The plain mean distance grows with dimension on its own: 0.54 in
            # two dimensions against 1.28 in ten, for uniform points. Dividing
            # by the square root of the dimension takes most of that out but
            # leaves about 9%, and it still measures every axis.
            "diversity_norm": spread / np.sqrt(ndim),
            # These two are the ones to compare across dimension. The first
            # looks only at the axes the objective depends on; the second
            # states it as a ratio to drawing at random in those axes, so
            # below one means the batch is packed tighter than chance.
            "diversity_key": key_spread,
            "diversity_vs_uniform": (key_spread / key_reference
                                     if key_reference else 0.0),
            "n_key_dims": len(key_dims),
            "n_points": bought, "n_fill_runner": n_fill,
            "n_fill_internal": n_internal, "pool_rows": pool_rows,
            "raw_pool_hash": raw_pool_hash,
            "y_opt": y_opt, "y_min": y_min,
            "denominator": "true_optimum",
            "landscape": landscape["name"],
            # compares; the reference key carries every parameter.
            "landscape_key": reference_values.key_for(
                landscape["name"], **landscape["params"]),
            "init_kind": init_kind,
            "n_init": n_init, "init_y_mean": float(init_y.mean()),
            "init_y_max": float(init_y.max()),
            # choice, and one that changes between configurations has to be
            # visible in the file rather than inferred from the code.
            "pool_size": pool_size,
            "surrogate": surrogate_tag,
            "protocol_version": protocol.VERSION,
            "runner_version": RUNNER_VERSION,
            "code_revision": CODE_REVISION,
            "source_hash": SOURCE_HASH,
        }
        # The round's surrogate statistics, on the pool the arm could choose
        # decides whether a sigma-based gate can rank at all.
        if rf is not None and pool is not None and len(pool):
            _share, _cv = sigma_share_and_cv(rf, sigma_sample_pool(pool, None))
            row["sigma_platform_share"] = _share
            row["sigma_cv"] = _cv
        _stats = getattr(algo, "last_novelty_stats", None)
        if _stats:
            for _f in ("novelty_pass_frac", "all_tied"):
                if _f in _stats:
                    row[_f] = _stats[_f]
            if "sigma_platform_share" in _stats:
                row["sigma_platform_share_gate"] = _stats["sigma_platform_share"]
        # Coverage is written only when it was asked for, so an existing
        # analysis that reads an old file does not see a column of NaNs and
        # conclude the pool was empty.
        if coverage is not None:
            row.update(coverage)
        # Store algorithm diagnostics. Internal fill is already recorded in n_fill_internal.
        for field in ("genotype_kind", "sigma_share", "n_ucb_requested",
                      "n_predictions", "n_library_scored",
                      "postfilter_dropped"):
            if hasattr(algo, field):
                row[field] = getattr(algo, field)
        # The initial row has no strategy quota to fall short of, so it gets
        # no shortfall columns at all. Writing the whole quota there, as an
        # earlier version did, inflates any average taken over every row.
        if delivered_by_strategy is not None:
            for name, want in requested.items():
                row[f"short_{name}"] = want - delivered_by_strategy.get(name, 0)
        summary.append(row)
        return pct

    # The point rows carried no provenance at all, so an assertion on the
    # summary file said nothing about them and an older point file could be
    # analysed with nothing to object. The threshold is here too because the
    # analysis that asks which point first crossed the 95% line needs it, and
    # recomputing it somewhere else is how a denominator gets lost.
    point_provenance = {
        "denominator": "true_optimum", "y_opt": y_opt, "y_min": y_min,
        "threshold_95": y_min + 0.95 * (y_opt - y_min),
        "protocol_version": protocol.VERSION,
        "runner_version": RUNNER_VERSION, "code_revision": CODE_REVISION,
    }

    record_summary(-1, float(y_obs.max()), 0.0, n_init, 0, 0, 0)
    for i in range(n_init):
        points.append({"dataset": ds_name, "algorithm": algo_name,
                       "seed": seed, "round": -1, "point_idx": i,
                       "strategy": "init", "is_fill": False,
                       "improved_best": False, "y": float(init_y[i]),
                       **{f"x{d}": float(init_X[i, d]) for d in range(ndim)},
                       **point_provenance})

    for r in range(n_rounds):
        started = time.time()
        rf = (surrogate_factory or (lambda s: make_rf(seed=s)))(seed * 100 + r)
        rf.fit(X_obs, y_obs)

        # The pool depends only on the dataset, the seed and the round, so
        # every arm of a study scores the same candidates even though a run
        # covers one algorithm at a time. It is built and filtered for every
        # algorithm, including the ones that never score it, because the
        # filter is where the surrogate family gets checked. The raw draw is
        # hashed so an assertion can prove no arm got its own pool.
        raw_pool = protocol.make_pool(bounds, np.random.default_rng([seed, r, 0]),
                                      n=pool_size)
        # The declared candidate count, offered to algorithms that derive their
        # own budget from it. The filtered pool has a slightly different length
        # every round, so reading that instead would make the genetic
        # algorithm's generation count drift and would stop the published
        # 10,000-pool run from reproducing.
        algo.pool_size_hint = int(len(raw_pool))
        raw_pool_hash = hashlib.sha256(raw_pool.tobytes()).hexdigest()
        pool = protocol.filter_pool(raw_pool, X_obs, rf,
                                    refinement_evidence=refinement_evidence)

        # Two counts, not one. The raw pool says whether the round could have
        # reached the line at all; the filtered pool says whether it still
        # could after the distance rule removed the neighbourhood of every
        # measured point. Neither is fed to the algorithm.
        coverage = None
        coverage_seconds = 0.0
        if pool_coverage:
            diagnostic_started = time.time()
            raw_y = surface(raw_pool)
            pool_y = surface(pool) if len(pool) else np.empty(0)
            coverage = {
                "raw_pool_size": int(len(raw_pool)),
                "raw_pool_n_above95": int((raw_y >= threshold_95).sum()),
                "raw_pool_max_y": float(raw_y.max()) if len(raw_y) else float("nan"),
                "pool_n_above95": int((pool_y >= threshold_95).sum()),
                "pool_max_y": float(pool_y.max()) if len(pool_y) else float("nan"),
                "threshold_95": float(threshold_95),
            }
            coverage_seconds = time.time() - diagnostic_started

        before = float(y_obs.max())
        if isinstance(algo, GLOSSAlgorithm):
            new_X, strategies = algo.recommend_continuous(
                X_obs, y_obs, bounds, n_points, rf, pool=pool)
        else:
            new_X = algo.recommend_continuous(
                X_obs, y_obs, bounds, n_points, rf, pool=pool)
            strategies = ["recommended"] * n_points
        new_X = np.atleast_2d(np.asarray(new_X, dtype=float))
        strategies = list(strategies)[:len(new_X)]
        delivered = len(new_X)
        delivered_by_strategy = {
            name: sum(1 for s in strategies if s == name) for name in requested}

        # A short batch is topped up by length alone. No algorithm is given an
        # exemption: the genetic algorithm never comes back short -- it fills
        # its own population gaps and raises if it cannot -- so a rule keyed
        # on its identity would guard nothing and would hide a real failure.
        filled = []
        while len(new_X) + len(filled) < n_points:
            k = len(filled)
            extra = _top_up(np.vstack([new_X, filled]) if filled else new_X,
                            pool, X_obs, np.random.default_rng([seed, r, k + 1]))
            filled.append(extra)
        if filled:
            new_X = np.vstack([new_X, np.array(filled)])
            strategies = strategies + ["fill"] * len(filled)

        new_y = surface(new_X)
        X_obs = np.vstack([X_obs, new_X])
        y_obs = np.concatenate([y_obs, new_y])

        # The timer keeps its published span -- surrogate fit, pool, filter,
        # recommendation -- minus the diagnostic, which evaluates the surface
        # on the whole pool and would otherwise report a cost no algorithm
        # incurred.
        elapsed = time.time() - started - coverage_seconds
        best = float(y_obs.max())
        internal = int(getattr(algo, "n_fill", 0) or 0)
        pct = record_summary(r, best, elapsed, len(new_X), len(filled),
                             internal, len(pool), raw_pool_hash,
                             delivered_by_strategy, coverage)

        for i in range(len(new_X)):
            points.append({
                "dataset": ds_name, "algorithm": algo_name, "seed": seed,
                "round": r, "point_idx": i,
                "strategy": strategies[i] if i < len(strategies) else "unknown",
                "is_fill": i >= delivered,
                "improved_best": bool(new_y[i] > before),
                "y": float(new_y[i]),
                **{f"x{d}": float(new_X[i, d]) for d in range(ndim)},
                **point_provenance})

        if r % 5 == 0:
            note = f" +{len(filled)} filled" if filled else ""
            print(f"    [{algo_name:12s}] seed={seed} r={r:2d}: "
                  f"best={best:.4f} pct={pct:.1f}%  ({elapsed:.1f}s){note}")

    return summary, points


RUNNER_VERSION = _runner_version()


# ═══════════════════════════════════════════════════════════════════════════════
# Study runners
# ═══════════════════════════════════════════════════════════════════════════════

def study_main(seeds=SEEDS, n_rounds=N_ROUNDS, out_suffix="", arms=None,
               base_name="bench_main", title="STUDY 1: Main Comparison", output_dir=RERUN_DIR):
    arms_of = (lambda s: make_arms(s, arms)) if arms else make_algorithms
    print("=" * 60)
    print(title)
    print("=" * 60)

    all_summary = []
    all_points = []

    # --- QM9-gap 100k ---
    print("\n[1/3] Loading QM9-gap (100k)...")
    ds_qm9 = load_qm9(n_samples=100000, target="gap")
    for seed in seeds:
        algos = arms_of(seed)
        for aname, algo in algos.items():
            print(f"  QM9-gap-100k / {aname} / seed={seed}")
            s, p = run_discrete(ds_qm9, aname, algo, seed, n_rounds)
            all_summary.extend(s)
            all_points.extend(p)

    # --- Buchwald-Hartwig ---
    print("\n[2/3] Loading Buchwald-Hartwig...")
    ds_bh = load_buchwald_hartwig()
    for seed in seeds:
        algos = arms_of(seed)
        for aname, algo in algos.items():
            print(f"  Buchwald-Hartwig / {aname} / seed={seed}")
            s, p = run_discrete(ds_bh, aname, algo, seed, n_rounds)
            all_summary.extend(s)
            all_points.extend(p)

    # Continuous progress uses the calibrated surface optimum. Finite-pool
    # studies use the best candidate in the library.
    print("\n[3/3] Arrhenius (continuous surface)...")
    land = continuous_landscape("arrhenius", n_peaks=3, peak_width=0.08,
                                noise_sigma=0.0, seed=42)
    for seed in seeds:
        algos = arms_of(seed)
        for aname, algo in algos.items():
            print(f"  Arrhenius-2D-cont / {aname} / seed={seed}")
            s, p = run_continuous(land, "Arrhenius-2D-cont", aname, algo,
                                  seed, n_rounds)
            all_summary.extend(s)
            all_points.extend(p)

    df_sum = pd.DataFrame(all_summary)
    df_pts = pd.DataFrame(all_points)
    base = f"{base_name}{out_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    df_sum.to_csv(os.path.join(output_dir, f"{base}_summary.csv"), index=False)
    df_pts.to_csv(os.path.join(output_dir, f"{base}_points.csv"), index=False)
    print(f"\nSaved: {base}_summary.csv, {base}_points.csv")
    return df_sum


def study_main9(seeds=SEEDS_20, n_rounds=N_ROUNDS, out_suffix=""):
    return study_main(seeds=seeds, n_rounds=n_rounds, out_suffix=out_suffix,
                      arms=RERUN_ARMS, base_name="bench_main9",
                      title="STUDY 1: Main Comparison")


def _save_sweep(rows, filename):
    frame = pd.DataFrame(rows)
    RERUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RERUN_DIR / filename
    frame.to_csv(path, index=False)
    print(f"Saved: {path}")
    return frame


def study_scaling(seeds=SEEDS, n_rounds=N_ROUNDS, out_suffix=""):
    """Study 2: QM9-gap at n=5k,10k,...,100k × 5 algorithms."""
    print("=" * 60)
    print("STUDY 2: Scaling Study")
    print("=" * 60)

    all_summary = []
    scales = list(range(5000, 105000, 5000))  # 5k to 100k step 5k

    for n in scales:
        print(f"\n--- Scale n={n} ---")
        ds = load_qm9(n_samples=n, target="gap")
        ds["name"] = f"QM9-gap-{n//1000}k"
        for seed in seeds:
            algos = make_algorithms(seed)
            for aname, algo in algos.items():
                print(f"  {ds['name']} / {aname} / seed={seed}")
                s, _ = run_discrete(ds, aname, algo, seed, n_rounds)
                all_summary.extend(s)

    return _save_sweep(all_summary, f"bench_scaling{out_suffix}_summary.csv")


def study_complexity(seeds=SEEDS, n_rounds=N_ROUNDS, out_suffix=""):
    """Study 3: Virtual Arrhenius at 5 complexity levels × 5 algorithms (discrete)."""
    print("=" * 60)
    print("STUDY 3: Complexity Study (discrete, 5k to 50k candidates)")
    print("=" * 60)

    all_summary = []

    for level in COMPLEXITY_LEVELS:
        label = level["label"]
        n_peaks = level["n_peaks"]
        noise_sigma = level["noise_sigma"]
        peak_width = level["peak_width"]

        n_cand = level.get("n_candidates", 10000)
        ds = get_virtual_dataset("arrhenius", n_candidates=n_cand,
                                  n_peaks=n_peaks, noise_sigma=noise_sigma,
                                  peak_width=peak_width, seed=42)
        ds["name"] = label
        y = ds["y_true"]
        thresh = y.min() + 0.95 * (ds["y_opt"] - y.min())
        n_top = (y >= thresh).sum()
        p8 = (1 - (1 - n_top / len(y)) ** 8) * 100

        print(f"\n--- {label} (peaks={n_peaks}, noise={noise_sigma}, width={peak_width}, n={n_cand}) ---")
        print(f"    y_opt={ds['y_opt']:.2f}, n_top={n_top}, P(8)={p8:.2f}%")

        for seed in seeds:
            algos = make_algorithms(seed)
            for aname, algo in algos.items():
                print(f"  {label} / {aname} / seed={seed}")
                s, _ = run_discrete(ds, aname, algo, seed, n_rounds)
                all_summary.extend(s)

    return _save_sweep(all_summary, f"bench_complexity{out_suffix}_summary.csv")


def study_ratio_scaling(seeds=SEEDS, n_rounds=N_ROUNDS, out_suffix=""):
    """Study 4: QM9-gap at n=5k,...,100k × 3 GLOSS ratios."""
    print("=" * 60)
    print("STUDY 4: Ratio Scaling Study")
    print("=" * 60)

    all_summary = []
    scales = list(range(5000, 105000, 5000))

    for n in scales:
        print(f"\n--- Scale n={n} ---")
        ds = load_qm9(n_samples=n, target="gap")
        ds["name"] = f"QM9-gap-{n//1000}k"
        for seed in seeds:
            algos = make_ratio_algorithms(seed)
            for aname, algo in algos.items():
                print(f"  {ds['name']} / {aname} / seed={seed}")
                s, _ = run_discrete(ds, aname, algo, seed, n_rounds)
                all_summary.extend(s)

    return _save_sweep(all_summary, f"bench_ratio_scaling{out_suffix}_summary.csv")


def study_ratio_complexity(seeds=SEEDS, n_rounds=N_ROUNDS, out_suffix=""):
    """Study 5: nD clusters with increasing dimensionality × 3 GLOSS ratios."""
    print("=" * 60)
    print("STUDY 5: Ratio Complexity (nD clusters, dim increases)")
    print("=" * 60)

    all_summary = []

    for level in RATIO_COMPLEXITY_LEVELS:
        label = level["label"]
        ndim = level["ndim"]
        n_peaks = level["n_peaks"]
        peak_width = level["peak_width"]
        n_cand = level["n_candidates"]

        ds = get_nd_cluster_dataset(ndim=ndim, n_peaks=n_peaks,
                                     peak_width=peak_width,
                                     n_candidates=n_cand, seed=42)
        ds["name"] = label
        y = ds["y_true"]
        thresh = y.min() + 0.95 * (ds["y_opt"] - y.min())
        n_top = (y >= thresh).sum()
        p8 = (1 - (1 - n_top / len(y)) ** 8) * 100

        print(f"\n--- {label} (ndim={ndim}, peaks={n_peaks}, n={n_cand}) ---")
        print(f"    n_top={n_top}, P(8)={p8:.2f}%")

        for seed in seeds:
            algos = make_ratio_algorithms(seed)
            for aname, algo in algos.items():
                print(f"  {label} / {aname} / seed={seed}")
                s, _ = run_discrete(ds, aname, algo, seed, n_rounds)
                all_summary.extend(s)

    return _save_sweep(all_summary, f"bench_ratio_complexity{out_suffix}_summary.csv")


def study_pilot(seeds=None, n_rounds=10):
    """Quick pilot: 1 seed, 10 rounds, all 3 datasets."""
    print("=" * 60)
    print("PILOT: Quick validation (1 seed, 10 rounds)")
    print("=" * 60)

    pilot_seeds = seeds or [47]
    pilot_rounds = n_rounds
    all_summary = []

    # QM9-gap 20k (quick)
    print("\n[1/3] QM9-gap 20k...")
    ds = load_qm9(n_samples=20000, target="gap")
    ds["name"] = "QM9-gap-20k"
    for seed in pilot_seeds:
        algos = make_algorithms(seed)
        for aname, algo in algos.items():
            s, _ = run_discrete(ds, aname, algo, seed, pilot_rounds)
            all_summary.extend(s)

    # Buchwald-Hartwig
    print("\n[2/3] Buchwald-Hartwig...")
    ds_bh = load_buchwald_hartwig()
    for seed in pilot_seeds:
        algos = make_algorithms(seed)
        for aname, algo in algos.items():
            s, _ = run_discrete(ds_bh, aname, algo, seed, pilot_rounds)
            all_summary.extend(s)

    # Arrhenius, continuous. The pilot follows study_main: a pilot that runs
    # a path the real study no longer uses validates nothing.
    print("\n[3/3] Arrhenius-2D-cont (continuous surface)...")
    land = continuous_landscape("arrhenius", n_peaks=3, peak_width=0.08,
                                noise_sigma=0.0, seed=42)
    for seed in pilot_seeds:
        algos = make_algorithms(seed)
        for aname, algo in algos.items():
            s, _ = run_continuous(land, "Arrhenius-2D-cont", aname, algo,
                                  seed, pilot_rounds)
            all_summary.extend(s)

    # Print results
    df = pd.DataFrame(all_summary)
    print("\n" + "=" * 60)
    print("PILOT RESULTS")
    print("=" * 60)
    for ds_name in df["dataset"].unique():
        print(f"\n--- {ds_name} ---")
        sub = df[df["dataset"] == ds_name]
        final = sub[sub["round"] == sub["round"].max()]
        for _, row in final.sort_values("pct_opt", ascending=False).iterrows():
            print(f"  {row['algorithm']:12s}: pct_opt={row['pct_opt']:.1f}%  best={row['best_so_far']:.4f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="pilot",
                        choices=["pilot", "main", "main9", "paper", "scaling",
                                 "complexity", "ratio_scaling",
                                 "ratio_complexity", "all"])
    # Verifying that a study still runs used to mean editing the source, and
    # the number of repeats rises to 15-20 in the final locked re-run, so
    # both are arguments now.
    parser.add_argument("--seeds", type=int, default=None,
                        help=f"how many seeds to use: at most {len(SEEDS_20)} "
                             f"for --study main9 or paper (default all "
                             f"{len(SEEDS_20)}), at most {len(SEEDS_10)} for "
                             f"the other studies (default the five published "
                             f"ones).")
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--out-suffix", default="",
                        help="appended to the output file names; use it for "
                             "a verification run to keep other "
                             "result files separate")
    args = parser.parse_args()

    is_paper_study = args.study in ("main9", "paper")
    available = SEEDS_20 if is_paper_study else SEEDS_10
    if args.seeds is not None and not 1 <= args.seeds <= len(available):
        parser.error(f"--seeds must be between 1 and {len(available)}")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    default_seeds = SEEDS_20 if is_paper_study else SEEDS
    seeds = default_seeds if args.seeds is None else available[:args.seeds]
    opts = dict(seeds=seeds, n_rounds=args.rounds,
                out_suffix=args.out_suffix)

    if args.study == "pilot":
        study_pilot(seeds=seeds if args.seeds is not None else None,
                    n_rounds=args.rounds if args.rounds != N_ROUNDS else 10)
    elif args.study == "main":
        study_main(**opts)
    elif args.study == "paper":
        study_main(**opts, arms=PAPER_ARMS, base_name="bench_paper",
                   title="Paper benchmark: eleven algorithms on three datasets")
    elif args.study == "main9":
        study_main9(**opts)
    elif args.study == "scaling":
        study_scaling(**opts)
    elif args.study == "complexity":
        study_complexity(**opts)
    elif args.study == "ratio_scaling":
        study_ratio_scaling(**opts)
    elif args.study == "ratio_complexity":
        study_ratio_complexity(**opts)
    elif args.study == "all":
        study_main(**opts)
        study_scaling(**opts)
        study_complexity(**opts)
        study_ratio_scaling(**opts)
        study_ratio_complexity(**opts)
