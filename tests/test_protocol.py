import numpy as np
import pytest

from benchmarks import protocol
from benchmarks.rf_surrogate import RFSurrogate


class _NotARandomForest:
    SURROGATE_FAMILY = "gaussian_process"


# -- the constants are what they were measured to be ---------------------

def test_the_constants_are_pinned():
    """A changed constant has to break a test, not just change a result."""
    assert protocol.POOL_SIZE == 10_000
    assert protocol.MIN_DISTANCE_TO_MEASURED == 0.02
    assert protocol.MIN_DISTANCE_WITHIN_BATCH == 1e-6
    assert protocol.N_TOP_FOR_OPTIMIZATION == 50
    assert protocol.SURROGATE_FAMILY == "random_forest"


def test_the_two_distances_stay_separate():
    """Reusing one for both would hand one arm a batch-diversity rule.

    UCB-BO is the arm with no batch diversity, so it would stop being a
    control. The gap between the two values is what keeps them distinct.
    """
    assert protocol.MIN_DISTANCE_WITHIN_BATCH < protocol.MIN_DISTANCE_TO_MEASURED


# -- the exclusion rule does work ----------------------------------------

def test_the_filter_removes_rows_and_had_rows_to_remove():
    """Both halves matter.

    If the filter kept everything the rule would be vacuous, and if the
    unfiltered pool held no close rows the test would pass for the wrong
    reason. Measured in two dimensions with 168 points down, about a fifth of
    the pool sits inside the zone.
    """
    rng = np.random.default_rng(0)
    bounds = [(0.0, 1.0)] * 2
    pool = protocol.make_pool(bounds, rng)
    measured = rng.uniform(0.0, 1.0, size=(168, 2))

    gaps = np.linalg.norm(pool[:, None, :] - measured[None, :, :],
                          axis=-1).min(axis=1)
    inside = int((gaps < protocol.MIN_DISTANCE_TO_MEASURED).sum())
    assert inside > 0, "the unfiltered pool held nothing to remove"

    kept = protocol.filter_pool(pool, measured, RFSurrogate())
    assert len(kept) == len(pool) - inside
    survivors = np.linalg.norm(kept[:, None, :] - measured[None, :, :],
                               axis=-1).min()
    assert survivors >= protocol.MIN_DISTANCE_TO_MEASURED


def test_the_filter_is_a_no_op_in_ten_dimensions():
    """Worth knowing, and worth stating: the rule only bites in low dimension.

    A ball of radius 0.02 has negligible volume in ten dimensions, so nothing
    is removed there and the rule constrains only the two-dimensional runs.
    """
    rng = np.random.default_rng(0)
    bounds = [(0.0, 1.0)] * 10
    pool = protocol.make_pool(bounds, rng)
    measured = rng.uniform(0.0, 1.0, size=(168, 10))
    assert len(protocol.filter_pool(pool, measured, RFSurrogate())) == len(pool)


def test_a_point_exactly_at_the_distance_is_allowed_by_both_rules():
    """One convention, or a boundary point is kept by one rule and cut by the other."""
    measured = np.array([[0.5, 0.5]])
    point = np.array([0.5 + protocol.MIN_DISTANCE_TO_MEASURED, 0.5])
    assert protocol.far_enough(point, measured)
    kept = protocol.filter_pool(point[None, :][0][None, :], measured, RFSurrogate())
    assert len(kept) == 1


# -- the guard is reachable and refuses --------------------------------

def test_the_filter_refuses_a_surrogate_the_protocol_never_measured():
    """The guard sits in the chokepoint, so no caller can forget it.

    With a Gaussian process in place refinement does move points, and the two
    Bayesian-optimisation baselines select straight after refining without
    re-checking the distance. Measured there, a third of UCB-BO's proposals
    land back inside the zone.
    """
    rng = np.random.default_rng(0)
    pool = protocol.make_pool([(0.0, 1.0)] * 2, rng, n=100)
    with pytest.raises(ValueError) as caught:
        protocol.filter_pool(pool, rng.uniform(0, 1, (4, 2)), _NotARandomForest())
    assert "refinement_evidence" in str(caught.value)


def test_the_guard_admits_another_family_only_against_evidence():
    protocol.require("gaussian_process",
                     refinement_evidence="measure_protocol.py, largest step 0.31")
    with pytest.raises(ValueError):
        protocol.require("gaussian_process")
    with pytest.raises(ValueError):
        protocol.require("gaussian_process", refinement_evidence="")


def test_the_family_comes_off_the_surrogate():
    assert protocol.family_of(RFSurrogate()) == "random_forest"
    assert protocol.family_of(_NotARandomForest()) == "gaussian_process"


# -- the shared pool is shared -----------------------------------------

def test_the_sampler_refuses_to_hand_back_a_prefix():
    """A prefix would give every strategy the same rows and shrink the set."""
    rng = np.random.default_rng(0)
    pool = protocol.make_pool([(0.0, 1.0)] * 2, rng, n=100)
    sampler = protocol.pool_sampler(pool)
    assert sampler(100, None, None, rng) is pool
    assert sampler(None, None, None, rng) is pool
    with pytest.raises(ValueError):
        sampler(50, None, None, rng)


def test_an_unknown_constant_raises():
    with pytest.raises(KeyError):
        protocol.get("no_such_constant")
