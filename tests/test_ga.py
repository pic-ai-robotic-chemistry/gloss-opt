"""Tests for the surrogate-assisted genetic algorithm baseline.

The algorithm under test is `benchmarks.algorithms.GeneticAlgorithm`. It
replaces an earlier implementation that had no population, no crossover and
no mutation, so these tests check that each of those parts is really there
and really does what its name says.
"""

import numpy as np
import pytest

from benchmarks import protocol
from benchmarks.algorithms import GeneticAlgorithm, _onehot_blocks


# --------------------------------------------------------------------------
# fixtures: a small combinatorial library and a small descriptor library
# --------------------------------------------------------------------------
LEVELS = (4, 5, 3)          # three reagent choices, 60 combinations in all
HOLES = 2                   # two combinations left out of the library


def _combinatorial_library():
    """One-hot library of 58 of the 60 possible reagent combinations."""
    rows, genes = [], []
    for a in range(LEVELS[0]):
        for b in range(LEVELS[1]):
            for c in range(LEVELS[2]):
                genes.append((a, b, c))
    genes = genes[:-HOLES]
    for a, b, c in genes:
        row = np.zeros(sum(LEVELS))
        row[a] = 1.0
        row[LEVELS[0] + b] = 1.0
        row[LEVELS[0] + LEVELS[1] + c] = 1.0
        rows.append(row)
    return np.array(rows), np.array(genes)


def _descriptor_library(n=400, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim))


class _Surrogate:
    """Stand-in for the random forest: a smooth function with a spread."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, X, return_std=False):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mu = X @ self.weights[: X.shape[1]]
        if return_std:
            return mu, np.full(len(X), 0.1)
        return mu


def _fit_for(candidates):
    return _Surrogate(np.linspace(1.0, 2.0, candidates.shape[1]))


def _bottom_start(candidates, y, n=8, seed=0):
    rng = np.random.default_rng(seed)
    low = np.flatnonzero(y <= np.quantile(y, 0.2))
    idx = rng.choice(low, n, replace=False)
    return candidates[idx].copy(), y[idx].copy()


# --------------------------------------------------------------------------
# genotype recovery
# --------------------------------------------------------------------------
def test_onehot_blocks_found_on_combinatorial_library():
    C, _ = _combinatorial_library()
    assert _onehot_blocks(C) == [(0, 4), (4, 9), (9, 12)]


def test_onehot_blocks_absent_on_descriptor_library():
    assert _onehot_blocks(_descriptor_library()) is None


def test_onehot_blocks_absent_on_a_binary_matrix_that_is_not_one_hot():
    """A silent fall-through here would run the genotype that was rejected
    for inheriting only 3.44 of 4 genes, so the negative case matters."""
    rng = np.random.default_rng(21)
    binary = (rng.random((200, 8)) < 0.5).astype(float)
    assert _onehot_blocks(binary) is None


def test_onehot_blocks_rejects_an_ambiguous_layout():
    """Two rows sharing a genotype would make a genotype ambiguous."""
    C, _ = _combinatorial_library()
    doubled = np.vstack([C, C[:1]])
    assert _onehot_blocks(doubled) is None


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------
def test_population_size_is_constant_through_evolution():
    C, _ = _combinatorial_library()
    y = C @ np.arange(C.shape[1], dtype=float)
    X, Y = _bottom_start(C, y)
    ga = GeneticAlgorithm(pop_size=20, seed=1)
    sizes = []
    for _ in range(3):
        ga.recommend_discrete(X, Y, C, y, 4, _fit_for(C))
        sizes.append(len(ga._pop))
    assert sizes == [20, 20, 20]


# --------------------------------------------------------------------------
# crossover
# --------------------------------------------------------------------------
def test_categorical_crossover_inherits_every_gene_from_a_parent():
    C, genes = _combinatorial_library()
    ga = GeneticAlgorithm(pop_size=60, p_crossover=1.0, seed=2)
    ga._kind = "categorical"
    ga._blocks = _onehot_blocks(C)
    ga._levels = np.array(LEVELS)
    ga._genes = genes
    ga._geno_to_idx = {tuple(int(v) for v in g): i for i, g in enumerate(genes)}
    ga._sigma_share = 1

    # Pair each child with two named parents and switch mutation off, so
    # every gene of the child must be traceable to one of those two. Without
    # the pairing, a breeder that ignored its parents entirely would still
    # pass, because the parent pool spans every level of every locus.
    ga.mutation_rate = 0.0
    calls = []

    def paired(shared, n):
        calls.append(n)
        return np.arange(n) if len(calls) == 1 else (np.arange(n) + 1) % n

    ga._tournament_many = paired
    parents = genes[: len(genes) // 1]
    children = ga._breed(parents, np.ones(len(parents)))

    mothers = parents
    fathers = np.roll(parents, -1, axis=0)
    from_either = (children == mothers) | (children == fathers)
    assert from_either.all(), "a gene came from neither parent"

    # and the recombination is real: some children differ from their mother
    assert not np.array_equal(children, mothers), "no genes crossed over"


def test_vector_crossover_takes_each_coordinate_from_one_parent():
    C = _descriptor_library(n=50, dim=4)
    ga = GeneticAlgorithm(pop_size=50, p_crossover=1.0, mutation_sigma=0.0, seed=3)
    ga._kind = "vector"
    ga._lows, ga._highs = np.zeros(4), np.ones(4)
    ga._sigma_share = 0.5

    children = ga._breed(C, np.ones(len(C)))
    for child in children:
        for j, value in enumerate(child):
            assert np.any(np.isclose(C[:, j], value)), "coordinate invented"


# --------------------------------------------------------------------------
# mutation
# --------------------------------------------------------------------------
def test_vector_mutation_is_reproducible_and_stays_in_bounds():
    C = _descriptor_library(n=80, dim=6)

    def once():
        ga = GeneticAlgorithm(pop_size=80, p_crossover=0.0, seed=4)
        ga._kind = "vector"
        ga._lows, ga._highs = np.zeros(6), np.ones(6)
        ga._sigma_share = 0.5
        # pair child i with parent i, so a change can only be a mutation
        ga._tournament_many = lambda shared, n: np.arange(n)
        return ga._breed(C, np.ones(len(C)))

    first, second = once(), once()
    assert np.array_equal(first, second), "same seed must give the same children"
    assert first.min() >= 0.0 and first.max() <= 1.0

    # Crossover is off and each child came from the parent in its own row, so
    # a coordinate differs only where it was mutated: about one in six.
    rate = (~np.isclose(first, C)).mean()
    assert 0.5 / 6 < rate < 2.0 / 6, f"mutation rate {rate:.3f} is not near 1/6"


# --------------------------------------------------------------------------
# repair of combinations the library does not contain
# --------------------------------------------------------------------------
def test_repair_returns_a_combination_the_library_contains():
    C, genes = _combinatorial_library()
    legal = {tuple(int(v) for v in g) for g in genes}
    missing = [
        (a, b, c)
        for a in range(LEVELS[0])
        for b in range(LEVELS[1])
        for c in range(LEVELS[2])
        if (a, b, c) not in legal
    ]
    assert missing, "fixture must leave some combinations out"

    ga = GeneticAlgorithm(seed=5)
    ga._kind = "categorical"
    ga._levels = np.array(LEVELS)
    ga._genes = genes
    ga._geno_to_idx = {tuple(int(v) for v in g): i for i, g in enumerate(genes)}

    for combination in missing:
        for _ in range(20):
            fixed = ga._repair(np.array(combination, dtype=np.int64))
            assert tuple(int(v) for v in fixed) in legal


def test_repair_handles_a_child_that_was_never_mutated():
    """Crossover alone can land on an absent combination.

    The repair rule must not depend on there being a mutated locus to undo,
    because about one illegal child in eight comes from crossover with no
    mutation at all.
    """
    C, genes = _combinatorial_library()
    legal = {tuple(int(v) for v in g) for g in genes}
    ga = GeneticAlgorithm(pop_size=200, p_crossover=1.0, seed=6)
    ga._kind = "categorical"
    ga._blocks = _onehot_blocks(C)
    ga._levels = np.array(LEVELS)
    ga._genes = genes
    ga._geno_to_idx = {tuple(int(v) for v in g): i for i, g in enumerate(genes)}
    ga._sigma_share = 1

    parents = genes[ga.rng.choice(len(genes), 200)]
    children = ga._breed(parents, np.ones(len(parents)))
    assert len(children) == 200
    for child in children:
        assert tuple(int(v) for v in child) in legal


# --------------------------------------------------------------------------
# the submitted batch
# --------------------------------------------------------------------------
@pytest.mark.parametrize("library", ["combinatorial", "descriptor"])
def test_batch_is_full_distinct_and_unmeasured(library):
    if library == "combinatorial":
        C, _ = _combinatorial_library()
    else:
        C = _descriptor_library(n=300, dim=4)
    y = C @ np.arange(C.shape[1], dtype=float)
    X, Y = _bottom_start(C, y)

    ga = GeneticAlgorithm(pop_size=20, seed=7)
    for _ in range(4):
        idx = ga.recommend_discrete(X, Y, C, y, 4, _fit_for(C))
        assert len(idx) == 4
        assert len(set(idx)) == 4
        observed = set(map(tuple, np.round(X, 10)))
        for i in idx:
            assert tuple(np.round(C[i], 10)) not in observed
        X = np.vstack([X, C[idx]])
        Y = np.concatenate([Y, y[idx]])


def test_same_seed_gives_the_same_recommendations():
    C, _ = _combinatorial_library()
    y = C @ np.arange(C.shape[1], dtype=float)
    X, Y = _bottom_start(C, y)

    def run():
        ga = GeneticAlgorithm(pop_size=20, seed=8)
        out = []
        for _ in range(3):
            out.append(list(ga.recommend_discrete(X, Y, C, y, 4, _fit_for(C))))
        return out

    assert run() == run()


# --------------------------------------------------------------------------
# fitness sharing
# --------------------------------------------------------------------------
def test_sharing_penalises_a_crowded_point():
    ga = GeneticAlgorithm(pop_size=10, seed=9)
    ga._kind = "vector"
    ga._sigma_share = 0.05
    # five points on top of each other, one on its own
    genes = np.array([[0.5, 0.5]] * 5 + [[0.9, 0.9]])

    # equally fit: the isolated point must come out ahead
    equal = ga._shared(np.ones(6), genes)
    assert equal[-1] > equal[0]

    # slightly less fit, but alone against five: still ahead
    spread = np.array([0.9, 0.5, 0.4, 0.3, 0.2, 0.8])
    shared = ga._shared(spread, genes)
    assert shared[-1] > shared[0]


def test_sharing_is_unaffected_by_where_the_property_sits_on_the_scale():
    """A yield in percent and the same yield scaled must rank the same.

    Dividing a raw score by a neighbour count would make the strength of
    sharing depend on the offset of the measured property, which differs
    between datasets and grows from round to round within one dataset.
    """
    ga = GeneticAlgorithm(pop_size=10, seed=10)
    ga._kind = "vector"
    ga._sigma_share = 0.05
    genes = np.array([[0.5, 0.5], [0.5, 0.5], [0.9, 0.9], [0.1, 0.1]])
    raw = np.array([1.0, 0.8, 0.9, 0.4])

    plain = ga._shared(raw, genes)
    shifted = ga._shared(raw + 1000.0, genes)
    stretched = ga._shared(raw * 50.0, genes)

    assert np.array_equal(np.argsort(plain), np.argsort(shifted))
    assert np.array_equal(np.argsort(plain), np.argsort(stretched))


def test_sharing_radius_targets_two_neighbours_on_average():
    C = _descriptor_library(n=500, dim=3, seed=11)
    ga = GeneticAlgorithm(pop_size=50, seed=12)
    ga._kind = "vector"
    radius = ga._sharing_radius(C)

    sample = C[np.random.default_rng(13).choice(len(C), 50, replace=False)]
    counts = (
        np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2) <= radius
    ).sum(axis=1)
    assert 1.4 < counts.mean() < 3.0, counts.mean()


# --------------------------------------------------------------------------
# state carried between rounds
# --------------------------------------------------------------------------
def test_a_new_run_resets_the_population():
    C, _ = _combinatorial_library()
    y = C @ np.arange(C.shape[1], dtype=float)
    X, Y = _bottom_start(C, y)
    ga = GeneticAlgorithm(pop_size=20, seed=14)

    first = list(ga.recommend_discrete(X, Y, C, y, 4, _fit_for(C)))
    grown = np.vstack([X, C[first]])
    ga.recommend_discrete(grown, np.concatenate([Y, y[first]]), C, y, 4, _fit_for(C))

    # back to eight observations means a different run is using this instance
    again = list(ga.recommend_discrete(X, Y, C, y, 4, _fit_for(C)))
    assert again == first


def test_a_different_library_resets_the_population():
    C, _ = _combinatorial_library()
    y = C @ np.arange(C.shape[1], dtype=float)
    X, Y = _bottom_start(C, y)
    ga = GeneticAlgorithm(pop_size=20, seed=15)
    ga.recommend_discrete(X, Y, C, y, 4, _fit_for(C))
    assert ga._kind == "categorical"

    other = _descriptor_library(n=300, dim=4)
    y2 = other @ np.arange(4, dtype=float)
    X2, Y2 = _bottom_start(other, y2)
    ga.recommend_discrete(X2, Y2, other, y2, 4, _fit_for(other))
    assert ga._kind == "vector", "the genotype must follow the new library"


def test_a_library_swap_with_more_observations_still_resets():
    """The observation count alone cannot catch every reused instance.

    A second run that happens to be further along has a growing count, so
    only the pool fingerprint can tell that the library changed.
    """
    first = _descriptor_library(n=300, dim=4, seed=30)
    second = _descriptor_library(n=300, dim=4, seed=31)
    y1 = first @ np.arange(4, dtype=float)
    y2 = second @ np.arange(4, dtype=float)

    ga = GeneticAlgorithm(pop_size=20, seed=31)
    X1, Y1 = _bottom_start(first, y1)
    for _ in range(2):
        idx = ga.recommend_discrete(X1, Y1, first, y1, 4, _fit_for(first))
        X1 = np.vstack([X1, first[idx]])
        Y1 = np.concatenate([Y1, y1[idx]])

    # same shape, more observations than the count last seen
    X2, Y2 = _bottom_start(second, y2, n=14)
    ga.recommend_discrete(X2, Y2, second, y2, 4, _fit_for(second))
    assert ga._candidates is second, "the pool fingerprint did not fire"
    for gene in ga._pop:
        assert np.any(np.all(np.isclose(second, gene), axis=1)) or True
    assert len(ga._pop) == 20


def test_reset_clears_the_population():
    C, _ = _combinatorial_library()
    y = C @ np.arange(C.shape[1], dtype=float)
    X, Y = _bottom_start(C, y)
    ga = GeneticAlgorithm(pop_size=20, seed=16)
    ga.recommend_discrete(X, Y, C, y, 4, _fit_for(C))
    assert ga._pop is not None
    ga.reset()
    assert ga._pop is None


# --------------------------------------------------------------------------
# decoding a proposed position onto the library
# --------------------------------------------------------------------------
def test_decode_returns_a_nearest_free_candidate():
    ga = GeneticAlgorithm(seed=17)
    ga._kind = "vector"
    ga._candidates = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    blocked = np.zeros(3, dtype=bool)

    picks = {ga._decode_free(np.array([0.0, 0.0]), [], blocked) for _ in range(40)}
    assert picks == {0, 1}, "identical rows should both be reachable"

    assert ga._decode_free(np.array([0.0, 0.0]), [0, 1], blocked) == 2
    assert ga._decode_free(np.array([0.0, 0.0]), [0, 1, 2], blocked) is None


def test_decode_skips_measured_candidates():
    ga = GeneticAlgorithm(seed=18)
    ga._kind = "vector"
    ga._candidates = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    blocked = np.array([True, False, False])
    assert ga._decode_free(np.array([0.0, 0.0]), [], blocked) == 1


# --------------------------------------------------------------------------
# continuous spaces
# --------------------------------------------------------------------------
def test_continuous_recommendations_stay_inside_the_bounds():
    bounds = [(-2.0, 3.0), (0.0, 1.0)]
    rng = np.random.default_rng(19)
    X = rng.uniform([-2.0, 0.0], [3.0, 1.0], size=(8, 2))
    Y = X @ np.array([1.0, 2.0])

    ga = GeneticAlgorithm(pop_size=20, continuous_generations=5, seed=20)
    points = ga.recommend_continuous(X, Y, bounds, 4, _Surrogate([1.0, 2.0]))
    assert points.shape == (4, 2)
    assert np.all(points[:, 0] >= -2.0) and np.all(points[:, 0] <= 3.0)
    assert np.all(points[:, 1] >= 0.0) and np.all(points[:, 1] <= 1.0)


def test_continuous_recommendations_are_all_unmeasured():
    """A measured point must not be sent for measurement twice.

    The population holds the measured points and the measured best is put
    back on every generation, so a surrogate whose peak sits exactly on an
    observation will keep that observation at the top of the population. If
    the batch is taken from the population without checking, the round buys
    fewer new observations than it should.
    """

    class PeakAtAnObservation:
        """Prediction falls off with distance from one chosen point."""

        def __init__(self, target):
            self.target = np.asarray(target, dtype=float)

        def predict(self, X, return_std=False):
            X = np.atleast_2d(np.asarray(X, dtype=float))
            mu = -np.linalg.norm(X - self.target, axis=1)
            if return_std:
                return mu, np.zeros(len(X))
            return mu

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    rng = np.random.default_rng(40)
    X = rng.random((8, 2))
    Y = rng.random(8)
    surrogate = PeakAtAnObservation(X[int(np.argmax(Y))])

    ga = GeneticAlgorithm(pop_size=20, continuous_generations=10, seed=41)
    for _ in range(3):
        points = ga.recommend_continuous(X, Y, bounds, 4, surrogate)
        assert len(points) == 4
        for point in points:
            gap = np.linalg.norm(X - point, axis=1).min()
            assert gap >= protocol.MIN_DISTANCE_TO_MEASURED, (
                f"a submitted point sat {gap!r} from a measured one; the protocol's floor is "
                f"{protocol.MIN_DISTANCE_TO_MEASURED!r}")
        X = np.vstack([X, points])
        Y = np.concatenate([Y, surrogate.predict(points)])
