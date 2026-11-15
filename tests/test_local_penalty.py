
import numpy as np
import pytest
from scipy.special import erfc

from benchmarks.algorithms import _expected_improvement
from benchmarks.local_penalty import (
    LocalPenaltyAlgorithm, estimate_lipschitz, neighbour_pairs,
    penalty_factors, positive,
)


class _StubSurrogate:
    """Predictions from callables, in place of a fitted forest."""

    def __init__(self, mu_fn, sd_fn):
        self.mu_fn = mu_fn
        self.sd_fn = sd_fn

    def predict(self, X, return_std=False):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mu = self.mu_fn(X)
        if not return_std:
            return mu
        return mu, self.sd_fn(X)


def _quadratic(x):
    """-(x - 0.5)^2 on [0, 1]: slope magnitude peaks at 1.0 on the ends."""
    return -(x - 0.5) ** 2


def test_lipschitz_recovers_known_slope():
    x = np.linspace(0.0, 1.0, 501).reshape(-1, 1)
    mu = _quadratic(x[:, 0])
    assert estimate_lipschitz(mu, neighbour_pairs(x)) == pytest.approx(1.0, rel=0.01)


def test_phi_half_crossing_is_the_radius():
    """phi = 0.5 exactly where the distance to x_j equals r_j."""
    mu_j, sd_j, m_hat, lipschitz = 0.2, 0.05, 1.4, 3.0
    radius = (m_hat - mu_j) / lipschitz
    x_j = np.array([0.0, 0.0])
    points = np.array([[radius, 0.0], [0.1 * radius, 0.0],
                       [radius + 10 * radius, 0.0]])
    phi, _, r_last = penalty_factors(points, [x_j], [mu_j], [sd_j], m_hat,
                                     lipschitz, 0.0)
    assert r_last == pytest.approx(radius, rel=1e-12)
    assert phi[0] == pytest.approx(0.5, abs=1e-9)
    assert phi[1] < 0.05
    assert phi[2] > 0.99


def test_phi_is_monotone_in_distance():
    """Inside the radius the factor is below 0.5, outside it above."""
    x_j = np.array([0.0])
    d = np.linspace(0.0, 2.0, 201)
    points = d.reshape(-1, 1)
    mu_j, sd_j, m_hat, lipschitz = 0.2, 0.05, 1.4, 3.0
    phi, _, _ = penalty_factors(points, [x_j], [mu_j], [sd_j], m_hat,
                                lipschitz, 1e-6)
    assert np.all(np.diff(phi) >= -1e-12)
    radius = (m_hat - mu_j) / lipschitz
    assert np.all(phi[d < radius] < 0.5)
    assert np.all(phi[d > radius] > 0.5)


def test_phi_depends_on_distance_not_on_the_candidate_mean():
    """Four candidates at the same radius get the same factor."""
    mu_j, sd_j, m_hat, lipschitz = 0.0, 0.2, 1.0, 2.0
    x_j = np.array([0.5, 0.5])
    points = np.array([[0.9, 0.5], [0.5, 0.9], [0.1, 0.5], [0.5, 0.1]])
    phi, _, _ = penalty_factors(points, [x_j], [mu_j], [sd_j], m_hat,
                                lipschitz, 0.0)
    assert np.allclose(phi, phi[0])


def test_zero_lipschitz_makes_phi_constant():
    """L -> 0 leaves the factor at 1 everywhere: tests the mechanism, not luck."""
    points = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    phi, _, r_last = penalty_factors(points, [np.array([0.0])], [0.0], [0.1],
                                     1.0, 0.0, 1e-9)
    assert r_last == 0.0
    assert np.all(phi == 1.0)
    assert float(np.std(phi)) < 1e-12


def test_sigma_floor_prevents_division_by_zero():
    """A zero sigma(x_j) must give finite numbers, not inf or nan."""
    points = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    phi, z, _ = penalty_factors(points, [np.array([0.0])], [0.0], [0.0], 1.0,
                                1.0, 1e-3)
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(z))


def test_positive_shift_preserves_order():
    acq = np.array([-5.0, -0.2, 0.0, 0.3, 2.0])
    out = positive(acq)
    assert np.all(out > 0.0)
    assert np.all(np.diff(out) > 0.0)


def test_positive_shift_leaves_positive_input_alone():
    acq = np.array([0.1, 0.5, 2.0])
    assert np.array_equal(positive(acq), acq)


def test_greedy_equals_base_acquisition_without_penalty():
    """L -> 0 recovers plain greedy top-k, and phi is constant."""
    x = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    mu = _quadratic(x[:, 0]) + 1.0
    sd = np.full(len(x), 0.1)
    algo = LocalPenaltyAlgorithm(base="ei", seed=3)
    picks, phi_std = _run_select(algo, x, mu, sd, n_points=8, lipschitz=0.0)
    # The comparison is against greedy selection under the same tie rule:
    # sequential argmax, first maximum wins. A plain argsort would differ on
    # exact ties, which expected improvement produces by symmetry, and that
    # difference would say nothing about the penalty.
    alpha = _expected_improvement(mu, sd, y_best=0.0)
    greedy = []
    for _ in range(8):
        remaining = alpha.copy()
        remaining[greedy] = -np.inf
        greedy.append(int(np.argmax(remaining)))
    assert picks == greedy
    assert phi_std < 1e-12


def test_penalty_changes_the_second_pick():
    x = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    mu = np.exp(-((x[:, 0] - 0.5) ** 2) / 0.02)
    sd = np.full(len(x), 0.02)
    algo = LocalPenaltyAlgorithm(base="ei", seed=3)
    picks, _ = _run_select(algo, x, mu, sd, n_points=4, lipschitz=8.0)
    greedy = list(np.argsort(-mu)[:4])
    assert picks[0] == greedy[0]
    assert picks[1] != greedy[1]


def test_batch_rows_are_distinct_and_exclude_observed():
    x = np.linspace(0.0, 1.0, 300).reshape(-1, 1)
    mu = _quadratic(x[:, 0])
    algo = LocalPenaltyAlgorithm(base="ei", seed=11)
    gp = _StubSurrogate(lambda X: _quadratic(X[:, 0]),
                        lambda X: np.full(len(X), 0.05))
    idx = algo.recommend_discrete(x[[0, 1, 2]], mu[[0, 1, 2]], x, mu, 5, gp)
    assert len(idx) == 5
    assert len(set(idx)) == 5
    assert not set(idx) & {0, 1, 2}


def test_continuous_returns_requested_count():
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    pool = np.random.default_rng(0).uniform(0, 1, size=(500, 2))
    algo = LocalPenaltyAlgorithm(base="ucb", seed=5)
    gp = _StubSurrogate(lambda X: np.sum(X, axis=1),
                        lambda X: np.full(len(X), 0.1))
    out = algo.recommend_continuous(np.zeros((2, 2)), np.zeros(2), bounds, 8,
                                    gp, pool=pool)
    assert out.shape == (8, 2)


def test_diagnostics_are_recorded_and_finite():
    x = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    mu = _quadratic(x[:, 0]) + 1.0
    algo = LocalPenaltyAlgorithm(base="ei", seed=2)
    gp = _StubSurrogate(lambda X: _quadratic(X[:, 0]) + 1.0,
                        lambda X: np.full(len(X), 0.05))
    algo.recommend_discrete(x[:3], mu[:3], x, mu, 8, gp)
    assert len(algo.diagnostics) == 1
    row = algo.diagnostics[0]
    for key in ("lp_L", "lp_M_hat", "lp_r_min", "lp_r_median", "lp_occ_k2",
                "lp_occ_k8", "lp_z_p10", "lp_z_p90", "lp_sigma_floor"):
        assert key in row, key
        assert np.isfinite(row[key]), key
    assert row["lp_L"] > 0.0
    assert 0.0 <= row["lp_occ_k2"] <= 1.0
    assert 0.0 <= row["lp_occ_k8"] <= 1.0


def test_bad_base_is_rejected():
    with pytest.raises(ValueError):
        LocalPenaltyAlgorithm(base="ucb2")


def test_registry_arms_differ_only_in_base():
    from benchmarks.bench_main import ARM_BUILDERS

    ei = ARM_BUILDERS["LocalPenalty(EI)"](7)
    ucb = ARM_BUILDERS["LocalPenalty(UCB)"](7)
    assert (ei.base, ucb.base) == ("ei", "ucb")
    assert ei.kappa == ucb.kappa == 2.0


def _run_select(algo, x, mu, sd, n_points, lipschitz):
    """Drive _select with a fixed Lipschitz constant instead of an estimate."""
    import benchmarks.local_penalty as module

    original = module.estimate_lipschitz_restricted
    module.estimate_lipschitz_restricted = lambda m, a, sp, nb: lipschitz
    try:
        picks = algo._select(x, mu, sd, np.arange(len(x)), n_points, 0.0)
        phi, _, _ = module.penalty_factors(
            x, [x[picks[0]]], [float(mu[picks[0]])], [float(sd[picks[0]])],
            float(np.max(mu)), lipschitz, 0.0)
        phi_std = float(np.std(phi))
    finally:
        module.estimate_lipschitz_restricted = original
    algo.diagnostics = []
    return picks, phi_std


def _select_reference(mu, sd, n_points, y_best, lipschitz, use_candidate_mean):
    """An independent copy of the loop, with either mu(x_j) or mu(x).

    The two variants are the mutation this test exists to catch: with the
    candidate's own mean inside z, the factor stops being a function of
    distance and starts penalizing high-mean candidates.
    """
    x = np.linspace(0.0, 1.0, len(mu)).reshape(-1, 1)
    alpha = np.maximum(_expected_improvement(mu, sd, y_best=y_best), 0.0)
    m_hat = float(np.max(mu))
    chosen, picked = [], []
    for _ in range(n_points):
        phi = np.ones(len(mu))
        for index in chosen:
            distance = np.abs(x[:, 0] - x[index, 0])
            mean = mu if use_candidate_mean else np.full(len(mu), mu[index])
            z = (lipschitz * distance - m_hat + mean) / (
                np.sqrt(2.0) * max(float(sd[index]), 1e-3 * float(np.max(sd))))
            phi = phi * 0.5 * erfc(-z)
        score = alpha * phi
        score[picked] = -np.inf
        pick = int(np.argmax(score))
        picked.append(pick)
        chosen.append(pick)
    return picked


def test_candidate_mean_is_not_used():
    """The factor uses mu(x_j); using mu(x) changes the batch, and the test says so."""
    x = np.linspace(0.0, 1.0, 400)[:, None]
    mu = np.exp(-((x[:, 0] - 0.5) ** 2) / 0.02)
    sd = np.full(len(x), 0.02)
    algo = LocalPenaltyAlgorithm(base="ei", seed=3)
    picks, _ = _run_select(algo, x, mu, sd, n_points=4, lipschitz=8.0)
    right = _select_reference(mu, sd, 4, 0.0, 8.0, use_candidate_mean=False)
    wrong = _select_reference(mu, sd, 4, 0.0, 8.0, use_candidate_mean=True)
    assert wrong != right, "the two variants must differ for this to be a test"
    assert picks == right
    assert picks != wrong


def test_measured_points_are_never_reselected():
    """A discrete batch must not spend budget on candidates already measured."""
    x = np.linspace(0.0, 1.0, 300).reshape(-1, 1)
    mu = _quadratic(x[:, 0])
    algo = LocalPenaltyAlgorithm(base="ei", seed=4)
    gp = _StubSurrogate(lambda X: _quadratic(X[:, 0]),
                        lambda X: np.full(len(X), 0.05))
    observed = [10, 40, 150, 151, 152]
    idx = algo.recommend_discrete(x[observed], mu[observed], x, mu, 6, gp)
    assert not set(idx) & set(observed)


def test_discrete_available_mask_is_used_by_the_ranking():
    """With every good candidate measured, the arm must pick the rest, not repeats."""
    x = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
    mu = -np.abs(x[:, 0] - 0.5)
    top = list(np.argsort(-mu)[:20])
    algo = LocalPenaltyAlgorithm(base="ucb", seed=8)
    gp = _StubSurrogate(lambda X: -np.abs(X[:, 0] - 0.5),
                        lambda X: np.full(len(X), 0.05))
    idx = algo.recommend_discrete(x[top], mu[top], x, mu, 8, gp)
    assert len(idx) == 8
    assert not set(idx) & set(top)
