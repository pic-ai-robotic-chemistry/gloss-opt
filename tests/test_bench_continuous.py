"""The continuous runner, and the one claim it exists to make true.

Every algorithm is supposed to spend the same budget. That is what the batch
top-up is for, and before this runner there was nothing holding it: the
continuous path had no top-up at all and GLOSS came back with seven points
out of eight. So the first test here is the budget identity, and the rest
guard the ways it could be met dishonestly -- by buying the same point twice,
by topping up from outside the pool, or by quietly reaching into the
exclusion zone.

Distances are compared against literals rather than the protocol constants.
A check written around the constant moves with it: setting the exclusion
distance to zero made an earlier version of this suite pass while every
algorithm walked straight into the zone.
"""
import numpy as np
import pytest

from benchmarks import protocol, reference_values
from benchmarks.algorithms import (
    BOAlgorithm,
    GeneticAlgorithm,
    GLOSSAlgorithm,
    QPOAlgorithm,
    RandomAlgorithm,
)
from benchmarks.bench_main import continuous_landscape, run_continuous

EXCLUSION = 0.02   # literal on purpose; see the module docstring
WITHIN_BATCH = 1e-6

ARRHENIUS = dict(n_peaks=3, peak_width=0.08, noise_sigma=0.0, seed=42)
CLUSTERS_10D = dict(ndim=10, n_peaks=7, peak_width=0.2, noise_sigma=0.0, seed=42)


class _Stingy:
    """An algorithm that hands back fewer points than it was asked for.

    Nothing shipped comes back short on demand, so without this the top-up
    never runs during a test and the assertions around it cannot fail.
    """

    def __init__(self, delivers):
        self.delivers = delivers
        self.n_fill = 0

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp, pool=None):
        return pool[:self.delivers]

    def reset(self):
        pass


def _landscape(ndim=2):
    if ndim == 2:
        return continuous_landscape("arrhenius", **ARRHENIUS)
    return continuous_landscape("nd_clusters", **CLUSTERS_10D)


def _gloss(seed, ratio=(4, 2, 2)):
    return GLOSSAlgorithm(
        strategy_points={"global_best": ratio[0], "local_best": ratio[1],
                         "unexplored": ratio[2], "unconverged": 0},
        ucb_kappa=2.0, diversity_radius=0.02, seed=seed)


# -- the budget identity -------------------------------------------------

@pytest.mark.parametrize("delivers", [8, 6, 3, 0])
def test_every_round_buys_the_full_batch(delivers):
    """However few points the algorithm offers, the round buys eight."""
    land = _landscape()
    summary, points = run_continuous(land, "Arrhenius-2D", "stingy",
                                     _Stingy(delivers), 47, n_rounds=3)
    rounds = [row for row in summary if row["round"] >= 0]
    assert [row["n_points"] for row in rounds] == [8, 8, 8]
    assert [row["n_fill_runner"] for row in rounds] == [8 - delivers] * 3
    bought = [p for p in points if p["round"] >= 0]
    assert len(bought) == 3 * 8
    assert sum(1 for p in bought if p["is_fill"]) == 3 * (8 - delivers)


def test_a_full_batch_is_not_topped_up():
    land = _landscape()
    summary, _ = run_continuous(land, "Arrhenius-2D", "qPO",
                                QPOAlgorithm(seed=47), 47, n_rounds=3)
    assert all(row["n_fill_runner"] == 0 for row in summary if row["round"] >= 0)


def test_the_observed_set_grows_by_the_batch_size_every_round():
    """The identity the whole top-up exists to preserve."""
    land = _landscape()
    summary, points = run_continuous(land, "Arrhenius-2D", "GLOSS(2:3:3)",
                                     _gloss(47, (2, 3, 3)), 47, n_rounds=4)
    n_init = summary[0]["n_init"]
    for row in summary:
        if row["round"] < 0:
            continue
        seen = n_init + (row["round"] + 1) * 8
        got = n_init + len([p for p in points
                            if 0 <= p["round"] <= row["round"]])
        assert got == seen, f"round {row['round']}: {got} points, expected {seen}"


# -- the batch obeys both distance rules ---------------------------------

@pytest.mark.parametrize("name,make", [
    ("GLOSS", lambda s: _gloss(s)),
    ("BO(EI)", lambda s: BOAlgorithm(seed=s)),
    ("GA", lambda s: GeneticAlgorithm(seed=s)),
    ("Random", lambda s: RandomAlgorithm(seed=s)),
    ("stingy", lambda s: _Stingy(5)),
])
def test_no_point_lands_near_a_measured_one_or_near_a_batch_mate(name, make):
    land = _landscape()
    _, points = run_continuous(land, "Arrhenius-2D", name, make(47), 47,
                               n_rounds=4)
    history = np.array([[p["x0"], p["x1"]] for p in points
                        if p["round"] == -1], dtype=float)
    for r in range(4):
        batch = np.array([[p["x0"], p["x1"]] for p in points
                          if p["round"] == r], dtype=float)
        gaps = np.linalg.norm(batch[:, None, :] - history[None, :, :],
                              axis=-1).min(axis=1)
        assert gaps.min() >= EXCLUSION, (
            f"{name} round {r}: a point sat {gaps.min():.2e} from a measured "
            f"one, inside the {EXCLUSION} floor")
        pairs = np.linalg.norm(batch[:, None, :] - batch[None, :, :], axis=-1)
        np.fill_diagonal(pairs, np.inf)
        assert pairs.min() >= WITHIN_BATCH, (
            f"{name} round {r}: two points in one batch sat "
            f"{pairs.min():.2e} apart")
        history = np.vstack([history, batch])


def test_the_top_up_never_returns_a_row_the_batch_already_holds():
    """Tested directly, because the integration test cannot see this.

    A ten-thousand-row pool against an eight-point batch means a collision
    about once in a thousand draws, so a few rounds of a full run will not
    exhibit it even when the exclusion is removed. The contract is checked
    here instead, on a pool small enough that a wrong implementation has
    nowhere to hide.
    """
    from benchmarks.bench_main import _top_up

    pool = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    chosen = pool[:2]
    for k in range(20):
        extra = _top_up(chosen, pool, np.empty((0, 2)),
                        np.random.default_rng(k))
        assert np.allclose(extra, pool[2]), (
            "the top-up returned a row this batch already holds")

    # And when nothing is left it says so rather than repeating a point.
    with pytest.raises(RuntimeError):
        _top_up(pool, pool, np.empty((0, 2)), np.random.default_rng(0))


def test_fill_points_come_from_the_round_pool():
    """Topping up from the box instead would dodge the exclusion filter."""
    land = _landscape()
    summary, points = run_continuous(land, "Arrhenius-2D", "stingy",
                                     _Stingy(2), 47, n_rounds=2)
    for row in summary:
        if row["round"] < 0:
            continue
        rng = np.random.default_rng([47, row["round"], 0])
        pool = protocol.filter_pool(
            protocol.make_pool(land["bounds"], rng),
            np.array([[p["x0"], p["x1"]] for p in points
                      if p["round"] < row["round"]], dtype=float),
            _Surrogate())
        known = {r.tobytes() for r in pool}
        fills = [p for p in points
                 if p["round"] == row["round"] and p["is_fill"]]
        assert len(fills) == 6
        for p in fills:
            row_bytes = np.array([p["x0"], p["x1"]], dtype=float).tobytes()
            assert row_bytes in known, "a fill point was not a pool row"


class _Surrogate:
    SURROGATE_FAMILY = "random_forest"


# -- the metric and the columns ------------------------------------------

def test_the_denominator_is_the_calibrated_one():
    """Nothing else looks at it, and everything is divided by it.

    Doubling the stored optimum halves every percentage and leaves an upper
    bound of 100 comfortably satisfied, so the bound cannot stand in for this.
    """
    flat = continuous_landscape("arrhenius", **ARRHENIUS)
    assert flat["y_opt"] == 18.352184511262436
    assert flat["y_min"] == 0.0
    assert flat["y_opt"] == reference_values.get("arrhenius", **ARRHENIUS)["y_opt"]

    clustered = continuous_landscape("nd_clusters", **CLUSTERS_10D)
    assert clustered["y_opt"] == 10.5
    assert clustered["y_min"] == 0.5


def test_a_landscape_built_on_the_wrong_parameters_is_refused():
    """The error is one-sided: a wrong closure reads LOWER at the optimum.

    Measured with the peak width at 0.05 the surface reads 7.80 against a
    stored 18.35, and with the landscape seed replaced by a run seed it reads
    0.4953. Both passed while the check was one-sided.
    """
    import benchmarks.bench_main as runner
    from benchmarks.virtual_functions import arrhenius_yield as real

    for wrong in ({"peak_width": 0.05}, {"seed": 47}, {"n_peaks": 1}):
        runner.arrhenius_yield = (
            lambda X, _w=wrong, **kw: real(X, **{**kw, **_w}))
        try:
            with pytest.raises(AssertionError):
                continuous_landscape("arrhenius", **ARRHENIUS)
        finally:
            runner.arrhenius_yield = real


def test_the_percentage_never_passes_a_hundred():
    """The denominator is the true optimum, so this is the real check.

    The slack is the protocol's: the stored optimum sits a couple of units in
    the last place below the largest value float64 reaches near the argmax.
    """
    for ndim in (2, 10):
        land = _landscape(ndim)
        summary, _ = run_continuous(land, f"L{ndim}", "qPO",
                                    QPOAlgorithm(seed=47), 47, n_rounds=3)
        worst = max(row["pct_opt"] for row in summary)
        assert worst <= 100.0 * (1 + reference_values.PCT_OPT_RTOL), worst


def test_ten_dimensions_run_and_record_ten_coordinates():
    land = _landscape(10)
    summary, points = run_continuous(land, "Clusters-10D", "GLOSS(4:2:2)",
                                     _gloss(47), 47, n_rounds=2)
    assert summary[0]["ndim"] == 10
    assert all(f"x{d}" in points[0] for d in range(10))
    assert "x10" not in points[0]


def test_the_key_axis_spread_ignores_the_axes_the_objective_ignores():
    land = _landscape(10)
    record = reference_values.get("nd_clusters", **CLUSTERS_10D)
    assert land["key_dims"] == record["key_dims"]
    assert len(land["key_dims"]) == 5
    assert len(land["key_dims"]) < land["ndim"]

    summary, _ = run_continuous(land, "C10", "GLOSS(4:2:2)", _gloss(47), 47,
                                n_rounds=4)
    last = [row for row in summary if row["round"] >= 0][-1]
    assert last["n_key_dims"] == 5
    # Over a strict subset of the axes the mean distance has to be smaller
    # than over all of them, so the two columns cannot be the same number.
    assert last["diversity_key"] < last["diversity"]
    # And the ratio column states it against a random draw in those axes.
    expected = last["diversity_key"] / protocol.uniform_mean_distance(5)
    assert last["diversity_vs_uniform"] == pytest.approx(expected, rel=1e-12)


def test_a_landscape_with_no_subset_uses_every_axis():
    """Arrhenius depends on both its axes, so the key set is all of them."""
    land = _landscape(2)
    assert land["key_dims"] == [0, 1]
    summary, _ = run_continuous(land, "A2", "qPO", QPOAlgorithm(seed=47), 47,
                                n_rounds=2)
    last = [row for row in summary if row["round"] >= 0][-1]
    assert last["n_key_dims"] == 2
    assert last["diversity_key"] == pytest.approx(last["diversity"], rel=1e-12)


def test_the_spread_column_is_comparable_across_dimension():
    rows = {}
    for ndim in (2, 10):
        land = _landscape(ndim)
        summary, _ = run_continuous(land, f"L{ndim}", "Random",
                                    RandomAlgorithm(seed=47), 47, n_rounds=3)
        last = [r for r in summary if r["round"] >= 0][-1]
        rows[ndim] = last
    raw = rows[10]["diversity"] / rows[2]["diversity"]
    normalised = rows[10]["diversity_norm"] / rows[2]["diversity_norm"]
    assert normalised < raw
    # A band, not just "smaller than the raw ratio": dividing by ndim squared
    # would also be smaller, and would be wrong. Measured on uniform points
    # the normalised ratio is about 1.08, so a residual of up to 15% is
    # expected and anything beyond it is a different normalisation.
    assert 0.85 <= normalised <= 1.15, normalised


def test_the_result_names_the_protocol_and_the_code_that_made_it():
    land = _landscape()
    summary, _ = run_continuous(land, "Arrhenius-2D", "qPO",
                                QPOAlgorithm(seed=47), 47, n_rounds=1)
    row = summary[-1]
    assert row["protocol_version"] == "1.0"
    assert row["init_kind"] == "bottom20"
    assert row["denominator"] == "true_optimum"
    assert row["code_revision"] and row["code_revision"] != "unknown"
    for column in ("dataset", "algorithm", "seed", "y_opt", "y_min",
                   "n_init", "pool_rows", "n_fill_internal"):
        assert column in row


# -- the initialisation hook --------------------------------------------

@pytest.mark.parametrize("kind", ["bottom20", "uniform", "lhs"])
def test_each_initialisation_rule_runs_and_is_recorded(kind):
    land = _landscape()
    summary, points = run_continuous(land, "Arrhenius-2D", "qPO",
                                     QPOAlgorithm(seed=47), 47, n_rounds=1,
                                     init_kind=kind)
    assert summary[0]["init_kind"] == kind
    assert summary[0]["n_init"] == 8
    assert len([p for p in points if p["round"] == -1]) == 8


@pytest.mark.parametrize("ndim", [2, 10])
def test_the_space_filling_rule_really_fills_the_space(ndim):
    from benchmarks.bench_main import _initial_points

    land = _landscape(ndim)
    points = _initial_points(land, seed=47, n_init=8, kind="lhs")
    assert points.shape == (8, ndim)
    for axis in range(ndim):
        bins = np.floor(points[:, axis] * 8).astype(int)
        assert sorted(bins) == list(range(8)), (
            f"axis {axis} put {len(set(bins))} points in 8 intervals")


def test_the_worst_fifth_really_starts_lower_than_a_uniform_draw():
    """Otherwise the published initialisation is not the handicap it claims."""
    land = _landscape()
    low, _ = run_continuous(land, "L", "qPO", QPOAlgorithm(seed=47), 47,
                            n_rounds=1, init_kind="bottom20")
    even, _ = run_continuous(land, "L", "qPO", QPOAlgorithm(seed=47), 47,
                             n_rounds=1, init_kind="uniform")
    assert low[0]["init_y_mean"] < even[0]["init_y_mean"]


def test_given_points_are_used_as_they_are():
    land = _landscape()
    mine = np.array([[0.11, 0.22], [0.33, 0.44], [0.55, 0.66]])
    summary, points = run_continuous(land, "L", "qPO", QPOAlgorithm(seed=47),
                                     47, n_rounds=1, init_X=mine)
    assert summary[0]["n_init"] == 3
    start = sorted((p["x0"], p["x1"]) for p in points if p["round"] == -1)
    assert np.allclose(start, sorted(map(tuple, mine)))


def test_a_fill_point_does_not_move_when_more_are_added():
    """The property the top-up's own stream exists for.

    A single run-level stream would make the second fill point of a round
    depend on how many came before it, so the same round would sample
    differently once a strategy delivered one point fewer.
    """
    land = _landscape()
    short, _ = run_continuous(land, "L", "stingy", _Stingy(6), 47, n_rounds=2)
    shorter, _ = run_continuous(land, "L", "stingy", _Stingy(2), 47, n_rounds=2)
    assert short[1]["n_fill_runner"] == 2
    assert shorter[1]["n_fill_runner"] == 6

    # The contract, asserted directly. Comparing the points instead does not
    # work: with every fill drawing from one freshly seeded stream, each call
    # starts from the same state, so the points agree across runs and only
    # their independence within a round is lost. What has to hold is that the
    # runner hands each fill its own stream.
    import benchmarks.bench_main as runner

    states = []
    real_top_up = runner._top_up

    def spy(chosen, pool, measured, rng):
        states.append(str(rng.bit_generator.state["state"]))
        return real_top_up(chosen, pool, measured, rng)

    runner._top_up = spy
    try:
        run_continuous(land, "L", "stingy", _Stingy(3), 47, n_rounds=1)
    finally:
        runner._top_up = real_top_up

    assert len(states) == 5, states
    assert len(set(states)) == 5, (
        "two fill points of one round were drawn from the same stream state, "
        "so they are not independent draws")


def test_topping_up_does_not_disturb_the_initial_points():
    """The top-up draws from its own stream, so the start cannot shift."""
    land = _landscape()
    full, _ = run_continuous(land, "L", "qPO", QPOAlgorithm(seed=47), 47,
                             n_rounds=2)
    short, _ = run_continuous(land, "L", "stingy", _Stingy(1), 47, n_rounds=2)
    assert full[0]["init_y_mean"] == short[0]["init_y_mean"]


# -- reproducibility and the shared pool --------------------------------

def test_the_same_seed_gives_the_same_run():
    land = _landscape()
    first, _ = run_continuous(land, "L", "GLOSS", _gloss(47), 47, n_rounds=3)
    second, _ = run_continuous(land, "L", "GLOSS", _gloss(47), 47, n_rounds=3)
    assert [r["best_so_far"] for r in first] == [r["best_so_far"] for r in second]


def test_two_arms_see_the_same_pool_in_the_same_round():
    """Bit for bit, not just the same number of rows.

    A row count is a lossy summary: two different pools of the same size
    would pass it. The pool must depend on the dataset, the seed and the
    round, and on nothing about which algorithm is running.
    """
    land = _landscape()
    seen = {}

    real_make = protocol.make_pool

    def spy(bounds, rng, n=None):
        pool = real_make(bounds, rng, n)
        seen.setdefault(len(seen), pool.copy())
        return pool

    protocol.make_pool = spy
    try:
        run_continuous(land, "L", "qPO", QPOAlgorithm(seed=47), 47, n_rounds=1)
        run_continuous(land, "L", "Random", RandomAlgorithm(seed=47), 47,
                       n_rounds=1)
    finally:
        protocol.make_pool = real_make

    assert len(seen) == 2
    assert np.array_equal(seen[0], seen[1]), "the two arms drew different pools"


def test_a_population_search_that_cannot_fill_its_batch_raises():
    """It must not come back short and let the runner paper over it.

    Squeezed into a box narrower than the exclusion distance, the genetic
    algorithm has nowhere legal to put an offspring. Raising there is what
    keeps the top-up from hiding a real failure.
    """
    land = _landscape()
    tight = dict(land)
    tight["bounds"] = [(0.5, 0.52), (0.5, 0.52)]
    with pytest.raises(RuntimeError):
        run_continuous(tight, "L", "GA", GeneticAlgorithm(seed=47), 47,
                       n_rounds=6, init_X=np.array([[0.505, 0.505],
                                                    [0.515, 0.515]]))



def test_the_discrete_runner_does_not_use_the_continuous_references():
    """Its denominator stays the candidate pool, by decision."""
    import inspect

    from benchmarks.bench_main import run_discrete
    body = inspect.getsource(run_discrete)
    assert "reference_values" not in body
    assert "y_true.max()" in body
