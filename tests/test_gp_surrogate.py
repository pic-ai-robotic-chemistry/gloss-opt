
import warnings

import numpy as np
import pytest

from benchmarks import protocol
from benchmarks.algorithms import GLOSSAlgorithm, _refined_selection
from benchmarks.bench_main import ARM_BUILDERS, make_arms
from benchmarks.gp_surrogate import GPSurrogate
from benchmarks.rf_surrogate import make_rf
from gloss.gloss import GLOSS


def _toy(n=60, ndim=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n, ndim))
    y = -(X ** 2).sum(axis=1)
    return X, y


class _FakeSigma:
    """Surrogate whose sigma is a chosen function of the row index."""

    SURROGATE_FAMILY = "fake"

    def __init__(self, mu, sigma):
        self._mu = np.asarray(mu, dtype=float)
        self._sigma = np.asarray(sigma, dtype=float)

    def predict(self, X, return_std=False):
        n = len(np.asarray(X))
        mu = self._mu[:n]
        if return_std:
            return mu, self._sigma[:n]
        return mu


# -- the surrogate -----------------------------------------------------------

def test_gp_surrogate_is_its_own_family():
    gp = GPSurrogate()
    assert gp.SURROGATE_FAMILY == "gaussian_process"
    assert protocol.family_of(gp) == "gaussian_process"
    assert protocol.family_of(make_rf(seed=0)) == "random_forest"


def test_gp_predict_is_finite_and_uncertain():
    X, y = _toy()
    gp = GPSurrogate().fit(X[:20], y[:20])
    mu, sd = gp.predict(X, return_std=True)
    assert mu.shape == (len(X),)
    assert sd.shape == (len(X),)
    assert np.isfinite(mu).all() and np.isfinite(sd).all()
    assert (sd > 0).all()
    stats = gp.sigma_stats(X)
    assert 0.0 <= stats["sigma_platform_share"] <= 1.0
    assert stats["sigma_cv"] >= 0.0


def test_gp_fit_is_deterministic():
    X, y = _toy()
    a = GPSurrogate().fit(X[:20], y[:20]).predict(X)
    b = GPSurrogate().fit(X[:20], y[:20]).predict(X)
    assert np.array_equal(a, b)


def test_gp_refuses_joint_sampling():
    """Thompson and qPO are out of scope for this family, on purpose."""
    gp = GPSurrogate()
    assert not hasattr(gp, "sample_predictions")
    X, y = _toy()
    gp.fit(X[:20], y[:20])
    with pytest.raises(TypeError):
        from benchmarks.algorithms import ThompsonSamplingAlgorithm
        algo = ThompsonSamplingAlgorithm(seed=0)
        algo.recommend_continuous(X[:20], y[:20], [(0.0, 1.0)] * 3, 4, gp)


def test_protocol_refuses_gp_without_evidence():
    with pytest.raises(ValueError, match="refinement_evidence"):
        protocol.require("gaussian_process")
    protocol.require("gaussian_process",
                     refinement_evidence="measure.py, largest displacement 1.18")


def test_filter_pool_forwards_the_evidence():
    X, y = _toy()
    gp = GPSurrogate().fit(X[:20], y[:20])
    pool = np.random.default_rng(0).uniform(0.0, 1.0, (50, 3))
    with pytest.raises(ValueError, match="refinement_evidence"):
        protocol.filter_pool(pool, X[:20], gp)
    kept = protocol.filter_pool(pool, X[:20], gp,
                                refinement_evidence="measure.py, 1.18")
    assert len(kept) <= len(pool)


# -- the rank gate -----------------------------------------------------------

def _gloss(**kw):
    X, _ = _toy()
    base = dict(mode="discrete", space={"candidates": X}, direction="maximize",
                seed=0)
    base.update(kw)
    return GLOSS(**base)


_MU = np.array([0.1, 0.9, 0.5, 0.2])


def test_rank_gate_breaks_signal_ties_towards_the_best_prediction():
    """All-equal signal: the kept set must be the best predictions."""
    X = np.array([[0.0] * 3, [1.0] * 3, [2.0] * 3, [3.0] * 3])
    gloss = GLOSS(mode="discrete", space={"candidates": X},
                  direction="maximize", seed=0, n_random_samples=10,
                  novelty_quantile=0.5, novelty_signal="sigma")
    fake = _FakeSigma(_MU, np.array([1.0, 1.0, 1.0, 1.0]))
    from gloss.strategies.unexplored import find_unexplored
    stats = {}
    out = find_unexplored(
        surrogate=fake, space=gloss.space, n_points=2, explored=X[:1],
        excluded=X[:1], direction="maximize", novelty_signal="sigma",
        novelty_quantile=0.5, n_measured=1, stats_out=stats)
    picked = sorted(round(float(p["point"][0]), 6) for p in out)
    # Row 0 is excluded, so the fake surrogate scores the three remaining
    # candidates 0.1, 0.9, 0.5; the two best are the rows at 2.0 and 3.0.
    assert picked == [2.0, 3.0]
    assert stats["all_tied"] is True
    # ceil(0.5 * 3) = 2 of the three remaining candidates.
    assert stats["novelty_pass_frac"] == pytest.approx(2 / 3)


def test_rank_gate_keeps_exactly_the_quantile():
    n = 40
    X = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    gloss = GLOSS(mode="discrete", space={"candidates": X},
                  direction="maximize", seed=0, novelty_quantile=0.25,
                  novelty_signal="sigma")
    rng = np.random.default_rng(0)
    fake = _FakeSigma(rng.uniform(size=n), rng.uniform(size=n))
    from gloss.strategies.unexplored import find_unexplored
    stats = {}
    out = find_unexplored(
        surrogate=fake, space=gloss.space, n_points=n, explored=X[:2],
        excluded=X[:2], direction="maximize", novelty_signal="sigma",
        novelty_quantile=0.25, n_measured=2, stats_out=stats)
    n_cand = len(X) - 2  # the two explored rows are not candidates
    n_keep = int(np.ceil(0.25 * n_cand))
    assert stats["novelty_pass_frac"] == pytest.approx(n_keep / n_cand)
    assert len(out) <= n_keep


def test_default_gate_is_unchanged_by_the_new_parameters():
    X, y = _toy(n=200, ndim=3)
    sp = {"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0}
    a = _gloss(seed=1)
    b = _gloss(seed=1, novelty_signal="sigma", novelty_quantile=None)
    rf = make_rf(seed=0).fit(X[:20], y[:20])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_a = a.recommend(surrogate=rf, X_existing=X[:20], strategy_points=sp)
        out_b = b.recommend(surrogate=rf, X_existing=X[:20], strategy_points=sp)
    assert [p["point"] for p in out_a] == [p["point"] for p in out_b]
    assert a.last_novelty_stats is None
    assert b.last_novelty_stats is None


def test_distance_and_sigma_gates_pick_different_points():
    X, y = _toy(n=200, ndim=3)
    sp = {"global_best": 0, "local_best": 0, "unexplored": 4, "unconverged": 0}
    rf = make_rf(seed=0).fit(X[:20], y[:20])
    picks = {}
    for signal in ("distance", "sigma"):
        g = _gloss(seed=1, novelty_signal=signal, novelty_quantile=0.5)
        out = g.recommend(surrogate=rf, X_existing=X[:20], strategy_points=sp)
        picks[signal] = [tuple(p["point"]) for p in out]
        assert g.last_novelty_stats["novelty_pass_frac"] == pytest.approx(0.5)
    assert picks["distance"] != picks["sigma"]


# -- the post-refinement re-filter -------------------------------------------

class _Owner:
    pass


def test_refined_point_inside_the_zone_is_replaced_and_counted():
    X_obs = np.array([[0.5, 0.5]])
    pool = np.array([[0.1, 0.1], [0.9, 0.9]])
    scores = np.array([0.9, 0.8])
    owner = _Owner()
    # First refined point lands 0.01 away, inside the 0.02 rule.
    refined = [np.array([0.51, 0.5]), np.array([0.2, 0.2])]
    selected = _refined_selection(refined, pool, scores, X_obs, 2, 0.02, owner)
    assert owner.postfilter_dropped == 1
    assert len(selected) == 2
    assert all(np.linalg.norm(np.asarray(p) - X_obs[0]) >= 0.02
               for p in selected)


def test_refinement_is_a_no_op_for_the_random_forest():
    """The published arms must not move: refined points are kept as they are.

    The two refined points sit 0.005 apart, which is far below the 0.02
    measured rule and far above the 1e-6 within-batch rule. Applying the
    measured rule to batch mates would replace one of them and change the
    published BO(EI) and UCB-BO columns; this case is the one that catches it.
    """
    owner = _Owner()
    X_obs = np.array([[0.0, 0.0]])
    pool = np.array([[0.9, 0.9], [0.8, 0.8]])
    refined = [np.array([0.5, 0.5]), np.array([0.505, 0.5])]
    selected = _refined_selection(refined, pool, np.array([1.0, 0.9]), X_obs,
                                  2, 0.02, owner)
    assert owner.postfilter_dropped == 0
    assert len(selected) == 2
    assert np.allclose(selected[0], [0.5, 0.5])
    assert np.allclose(selected[1], [0.505, 0.5])


def test_quantile_out_of_range_is_rejected():
    X, _ = _toy()
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="novelty_quantile"):
            GLOSS(mode="discrete", space={"candidates": X},
                  direction="maximize", novelty_quantile=bad)


# -- the arms ----------------------------------------------------------------

def test_step_9_arms_exist_and_differ_in_one_factor():
    for name in ("novdist", "novsigma", "GLOSS(4:2:2)uc"):
        assert name in ARM_BUILDERS
    arms = make_arms(0, ["GLOSS(4:2:2)", "novdist", "novsigma"])
    a, d, s = (arms["GLOSS(4:2:2)"], arms["novdist"], arms["novsigma"])
    # The control shares the published gate; the arm under test is sigma.
    assert a._novelty_quantile is None and d._novelty_quantile == 0.5
    assert d._novelty_signal == "distance" and s._novelty_signal == "sigma"
    assert a._strategy_points == d._strategy_points == s._strategy_points
    assert isinstance(a, GLOSSAlgorithm)
