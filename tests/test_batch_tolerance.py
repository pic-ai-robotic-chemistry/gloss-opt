import numpy as np
import pytest

from gloss import GLOSS
from gloss.utils import is_batch_rejected, is_duplicate, is_rejected


class StepRamp:
    """A surrogate that mimics a random forest: flat between steps.

    The value falls as x grows, so the best candidate is the one with the
    smallest x. The steps matter: a smooth ramp would let the continuous
    refinement slide every point to the boundary and the batch spacing would
    not be what the test measured. For the benchmark RF, the
    refinement moves a point by exactly zero.
    """

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        mu = -np.round(X[:, 0], 2)
        if return_std:
            # Uncertainty grows with x, so the unconverged strategy has a
            # strict order to follow instead of a tie across the whole pool.
            return mu, np.round(X[:, 0], 2)
        return mu


def line_pool(n=20, step=0.01, y=0.5):
    """n points on one line, step apart. x = 0.00, 0.01, ... by default."""
    return np.array([[round(i * step, 10), y] for i in range(n)])


def sampler_for(pool):
    def sampler(n, bounds, constraints, rng):
        return pool
    return sampler


def xs_of(results):
    return [round(r["point"][0], 10) for r in results]


# --------------------------------------------------------------------------
# 1. the helper itself
# --------------------------------------------------------------------------

def test_is_rejected_inactive_is_one_distance_for_both_groups():
    excluded = np.array([[0.0, 0.0], [1.0, 0.0]])
    point = np.array([0.3, 0.0])
    # measured row at 0.3, batch row at 0.7; one distance of 0.4 rejects it
    assert is_rejected(point, excluded, 1, 0.4, None) is True
    assert is_rejected(point, excluded, 1, 0.2, None) is False


def test_is_rejected_active_uses_each_group_distance():
    excluded = np.array([[0.0, 0.0],    # measured
                         [1.0, 0.0]])   # batch mate
    near_measured = np.array([0.3, 0.0])
    near_batch = np.array([0.7, 0.0])
    # measured 0.5, batch 0.1: only the point near the measured one goes
    assert is_rejected(near_measured, excluded, 1, 0.5, 0.1) is True
    assert is_rejected(near_batch, excluded, 1, 0.5, 0.1) is False
    # two parameters wired to one variable, cannot produce both lines.
    assert is_rejected(near_measured, excluded, 1, 0.1, 0.5) is False
    assert is_rejected(near_batch, excluded, 1, 0.1, 0.5) is True


def test_is_rejected_equal_values_take_the_single_distance_branch():
    excluded = np.array([[0.0, 0.0], [1.0, 0.0]])
    point = np.array([0.3, 0.0])
    assert is_rejected(point, excluded, 1, 0.4, 0.4) is True
    assert is_rejected(point, excluded, 1, 0.2, 0.2) is False


def test_is_rejected_handles_an_out_of_range_split():
    # This one records a convention rather than guarding a mechanism: a slice
    # past the end is the whole array, so a caller that counts every row as
    # measured gets an empty batch group and one distance.
    excluded = np.array([[0.0, 0.0]])
    point = np.array([0.3, 0.0])
    # n_measured past the end: every row is measured, the batch group is empty
    assert is_rejected(point, excluded, 9, 0.4, 0.1) is True
    assert is_rejected(point, excluded, 9, 0.2, 0.1) is False


def test_is_batch_rejected_is_silent_while_inactive():
    batch = [np.array([0.0, 0.0])]
    point = np.array([0.0, 0.0])
    # the same point, and still no rejection, because the batch distance is
    assert is_batch_rejected(point, batch, 0.0, None) is False
    assert is_batch_rejected(point, batch, 0.02, 0.02) is False
    # active: the batch distance now holds
    assert is_batch_rejected(point, batch, 0.02, 0.0) is True
    assert is_batch_rejected(np.array([0.03, 0.0]), batch, 0.0, 0.05) is True
    assert is_batch_rejected(np.array([0.06, 0.0]), batch, 0.0, 0.05) is False


def test_is_batch_rejected_with_no_batch_yet():
    assert is_batch_rejected(np.array([0.0, 0.0]), [], 0.0, 0.05) is False


# --------------------------------------------------------------------------
# 2. the boundary conventions, as they stand today
# --------------------------------------------------------------------------

def test_the_three_boundary_conventions_are_recorded():
    from benchmarks import protocol
    # is_duplicate rejects a point exactly at the tolerance
    assert is_duplicate(np.array([0.5, 0.0]), np.array([[0.0, 0.0]]), 0.5) is True
    # far_enough keeps a point exactly at the gap
    assert protocol.far_enough(np.array([0.5, 0.0]),
                               np.array([[0.0, 0.0]]), 0.5) is True
    # the diversity radius keeps a point exactly at the radius: the check is
    # the three; this test remembers where they stand.
    pool = np.array([[0.0, 0.5], [0.5, 0.5]])
    algo = GLOSS(space={"bounds": [(0.0, 1.0), (0.0, 1.0)]},
                 mode="continuous", direction="maximize",
                 n_random_samples=len(pool), n_top_for_optimization=len(pool),
                 ucb_kappa=0.0, duplicate_tolerance=0.0,
                 diversity_radius=0.5, custom_sampler=sampler_for(pool),
                 seed=0)
    got = algo.recommend(surrogate=StepRamp(),
                         X_existing=np.empty((0, 2)),
                         strategy_points={"global_best": 2, "local_best": 0,
                                          "unexplored": 0, "unconverged": 0})
    assert xs_of(got) == [0.0, 0.5]


# --------------------------------------------------------------------------
# 3. each strategy respects the batch distance
# --------------------------------------------------------------------------

def run_one_strategy(strategy, batch_tolerance, n_points=3, pool=None,
                     **extra):
    pool = line_pool() if pool is None else pool
    algo = GLOSS(space={"bounds": [(0.0, 1.0), (0.0, 1.0)]},
                 mode="continuous", direction="maximize",
                 n_random_samples=len(pool),
                 n_top_for_optimization=len(pool),
                 ucb_kappa=0.0,
                 duplicate_tolerance=0.0,
                 batch_tolerance=batch_tolerance,
                 custom_sampler=sampler_for(pool),
                 seed=0, **extra)
    points = {"global_best": 0, "local_best": 0,
              "unexplored": 0, "unconverged": 0}
    points[strategy] = n_points
    return algo.recommend(surrogate=StepRamp(),
                          X_existing=np.empty((0, 2)),
                          strategy_points=points)


def test_global_best_spaces_the_batch():
    # inactive: the three best candidates, one step apart
    assert xs_of(run_one_strategy("global_best", None)) == [0.0, 0.01, 0.02]
    # active at 0.05: the best, then the first candidate FARTHER than 0.05,
    # because is_duplicate rejects a point exactly at the distance (the "<="
    # convention recorded above). So 0.05 is out and 0.06 is in, then 0.11 is
    # 0.05 from 0.06 and out, and 0.12 is in.
    assert xs_of(run_one_strategy("global_best", 0.05)) == [0.0, 0.06, 0.12]


def test_local_best_spaces_the_batch():
    # The window has to be narrower than the pool step, or this strategy finds
    # one local optimum on a ramp and there is no batch to space out. At 0.005
    # every pool point stands alone in its window, so all of them qualify and
    # the merge distance (half the window, 0.0025) stays below the batch
    # distance under test.
    got = run_one_strategy("local_best", None, window_radius=0.005)
    assert xs_of(got) == [0.0, 0.01, 0.02]
    got = run_one_strategy("local_best", 0.05, window_radius=0.005)
    assert xs_of(got) == [0.0, 0.06, 0.12]


def test_unexplored_spaces_the_batch():
    # threshold 0 makes every candidate far enough, so the order is the
    # predicted value and what is left to see is the spacing
    got = run_one_strategy("unexplored", None, unexplored_threshold=0.0)
    assert xs_of(got) == [0.0, 0.01, 0.02]
    got = run_one_strategy("unexplored", 0.05, unexplored_threshold=0.0)
    assert xs_of(got) == [0.0, 0.06, 0.12]


def test_unconverged_spaces_the_batch():
    # uncertainty grows with x, so this strategy walks the pool the other way.
    # No benchmark configuration gives it a point, so this test is the only
    # thing that holds it to the same semantics as the other three.
    got = run_one_strategy("unconverged", None)
    assert xs_of(got) == [0.19, 0.18, 0.17]
    got = run_one_strategy("unconverged", 0.05)
    assert xs_of(got) == [0.19, 0.13, 0.07]


def test_global_best_fallback_loop_also_spaces_the_batch():
    pool = line_pool()
    algo = GLOSS(space={"bounds": [(0.0, 1.0), (0.0, 1.0)]},
                 mode="continuous", direction="maximize",
                 n_random_samples=len(pool),
                 n_top_for_optimization=2,      # shorter than the batch
                 ucb_kappa=0.0, duplicate_tolerance=0.0, batch_tolerance=0.05,
                 custom_sampler=sampler_for(pool), seed=0)
    got = algo.recommend(surrogate=StepRamp(),
                         X_existing=np.empty((0, 2)),
                         strategy_points={"global_best": 3, "local_best": 0,
                                          "unexplored": 0, "unconverged": 0})
    # The shortlist of two gives 0.00 and rejects 0.01; the fallback loop then
    # has to keep the same distance, so 0.06 and 0.12 and nothing nearer.
    assert xs_of(got) == [0.0, 0.06, 0.12]


def test_local_best_keeps_its_distance_from_an_earlier_strategy():
    """The points an earlier strategy chose are batch mates, not measurements.

    If local_best were handed the whole exclusion as measured points, it would
    judge its distance to the global_best point by the measured distance (zero
    here) and could sit right next to it.
    """
    pool = line_pool()
    algo = GLOSS(space={"bounds": [(0.0, 1.0), (0.0, 1.0)]},
                 mode="continuous", direction="maximize",
                 n_random_samples=len(pool), n_top_for_optimization=len(pool),
                 ucb_kappa=0.0, duplicate_tolerance=0.0, batch_tolerance=0.05,
                 window_radius=0.005, custom_sampler=sampler_for(pool), seed=0)
    got = algo.recommend(surrogate=StepRamp(),
                         X_existing=np.empty((0, 2)),
                         strategy_points={"global_best": 1, "local_best": 2,
                                          "unexplored": 0, "unconverged": 0})
    assert xs_of(got) == [0.0, 0.06, 0.12]


def test_unexplored_keeps_its_distance_from_an_earlier_strategy():
    pool = line_pool()
    algo = GLOSS(space={"bounds": [(0.0, 1.0), (0.0, 1.0)]},
                 mode="continuous", direction="maximize",
                 n_random_samples=len(pool), n_top_for_optimization=len(pool),
                 ucb_kappa=0.0, duplicate_tolerance=0.0, batch_tolerance=0.05,
                 unexplored_threshold=0.0, custom_sampler=sampler_for(pool),
                 seed=0)
    got = algo.recommend(surrogate=StepRamp(),
                         X_existing=np.empty((0, 2)),
                         strategy_points={"global_best": 1, "local_best": 0,
                                          "unexplored": 2, "unconverged": 0})
    assert xs_of(got) == [0.0, 0.06, 0.12]


def test_the_batch_distance_also_holds_across_two_strategies():
    # global_best walks up from x=0, unconverged walks down from x=0.19.
    # One batch, two strategies: the distance has to hold between them too.
    pool = line_pool(n=8)          # x = 0.00 .. 0.07
    algo = GLOSS(space={"bounds": [(0.0, 1.0), (0.0, 1.0)]},
                 mode="continuous", direction="maximize",
                 n_random_samples=len(pool), n_top_for_optimization=len(pool),
                 ucb_kappa=0.0, duplicate_tolerance=0.0, batch_tolerance=0.05,
                 custom_sampler=sampler_for(pool), seed=0)
    got = algo.recommend(surrogate=StepRamp(),
                         X_existing=np.empty((0, 2)),
                         strategy_points={"global_best": 1, "local_best": 0,
                                          "unexplored": 0, "unconverged": 2})
    # global_best takes 0.00; unconverged wants 0.07 then 0.06, 0.05, 0.04,
    # but everything within 0.05 of 0.00 is out, so it gets 0.07 and nothing
    # else: 0.06 and 0.05 are too near 0.07, and 0.04 and below are too near
    # 0.00. A short batch is the honest outcome here.
    assert xs_of(got) == [0.0, 0.07]


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

def run_discrete_strategy(strategy, batch_tolerance, n_points=3, pool=None,
                          **extra):
    pool = line_pool() if pool is None else pool
    algo = GLOSS(space={"candidates": pool}, mode="discrete",
                 direction="maximize", ucb_kappa=0.0,
                 duplicate_tolerance=0.0, batch_tolerance=batch_tolerance,
                 seed=0, **extra)
    points = {"global_best": 0, "local_best": 0,
              "unexplored": 0, "unconverged": 0}
    points[strategy] = n_points
    return algo.recommend(surrogate=StepRamp(),
                          X_existing=np.empty((0, 2)),
                          strategy_points=points)


def test_discrete_global_best_spaces_the_batch_only_when_asked():
    assert xs_of(run_discrete_strategy("global_best", None)) == [0.0, 0.01, 0.02]
    assert xs_of(run_discrete_strategy("global_best", 0.05)) == [0.0, 0.06, 0.12]


def test_discrete_local_best_spaces_the_batch_only_when_asked():
    got = run_discrete_strategy("local_best", None, window_radius=0.005)
    assert xs_of(got) == [0.0, 0.01, 0.02]
    got = run_discrete_strategy("local_best", 0.05, window_radius=0.005)
    assert xs_of(got) == [0.0, 0.06, 0.12]


def test_the_discrete_path_still_hands_back_two_rows_that_coincide():
    pool = np.array([[0.0, 0.5], [0.0, 0.5], [0.3, 0.5]])
    got = run_discrete_strategy("global_best", None, n_points=2, pool=pool)
    assert xs_of(got) == [0.0, 0.0]
    # and with the batch distance in use, the second copy goes
    got = run_discrete_strategy("global_best", 0.1, n_points=2, pool=pool)
    assert xs_of(got) == [0.0, 0.3]


# --------------------------------------------------------------------------
# 4. a batch distance of zero still refuses two points in one place
# --------------------------------------------------------------------------

def test_zero_batch_distance_still_refuses_a_coincident_point():
    pool = np.array([[0.0, 0.5], [0.0, 0.5], [0.3, 0.5], [0.6, 0.5]])
    got = run_one_strategy("global_best", 0.0, n_points=3, pool=pool,
                           )
    assert xs_of(got) == [0.0, 0.3, 0.6]


# --------------------------------------------------------------------------
# 4b. the shortlist is re-ranked by predicted mean, which is why the
#     global-only arm is not the same algorithm as UCB-BO
# --------------------------------------------------------------------------

class Disagree:

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        mu = -np.round(X[:, 0], 2)
        if return_std:
            return mu, np.round(X[:, 0], 2)
        return mu


def test_the_shortlist_is_reranked_by_predicted_mean():
    pool = line_pool()
    algo = GLOSS(space={"bounds": [(0.0, 1.0), (0.0, 1.0)]},
                 mode="continuous", direction="maximize",
                 n_random_samples=len(pool),
                 n_top_for_optimization=5,     # the shortlist, by UCB
                 ucb_kappa=2.0, duplicate_tolerance=0.0,
                 custom_sampler=sampler_for(pool), seed=0)
    got = algo.recommend(surrogate=Disagree(),
                         X_existing=np.empty((0, 2)),
                         strategy_points={"global_best": 1, "local_best": 0,
                                          "unexplored": 0, "unconverged": 0})
    # UCB (mu + 2 sigma = -x + 2x = x) puts the shortlist at the large-x end:
    # 0.19, 0.18, 0.17, 0.16, 0.15. The mean (-x) then picks the smallest of
    # those five. A UCB-ordered pick would have returned 0.19.
    assert xs_of(got) == [0.15]


# --------------------------------------------------------------------------
# 5. the constructor defaults
# --------------------------------------------------------------------------

def test_discrete_default_tolerance_is_zero():
    algo = GLOSS(space={"candidates": line_pool()}, mode="discrete")
    assert algo.duplicate_tolerance == 0.0
    assert algo.batch_tolerance is None


def test_continuous_default_tolerance_follows_the_radius():
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    algo = GLOSS(space={"bounds": bounds}, mode="continuous",
                 diversity_radius=0.02)
    assert algo.duplicate_tolerance == 0.02
    assert algo.batch_tolerance is None
    # With no radius the fallback is a thousandth of the diagonal. The 2-D
    # unit box has a diagonal of sqrt(2), so this is 0.0014142135623730952.
    algo = GLOSS(space={"bounds": bounds}, mode="continuous",
                 diversity_radius=0.0)
    assert algo.duplicate_tolerance == pytest.approx(np.sqrt(2) * 0.001)
    assert algo.batch_tolerance is None


def test_an_explicit_tolerance_wins_over_both_defaults():
    algo = GLOSS(space={"bounds": [(0.0, 1.0), (0.0, 1.0)]},
                 mode="continuous", diversity_radius=0.02,
                 duplicate_tolerance=0.5, batch_tolerance=0.1)
    assert algo.duplicate_tolerance == 0.5
    assert algo.batch_tolerance == 0.1
