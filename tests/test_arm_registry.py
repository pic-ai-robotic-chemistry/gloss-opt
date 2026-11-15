import pytest

from benchmarks import bench_main
from benchmarks.algorithms import (GLOSSAlgorithm, BOAlgorithm, UCBOAlgorithm,
                                   GeneticAlgorithm, RandomAlgorithm,
                                   QPOAlgorithm, ThompsonSamplingAlgorithm)

SEED = 47

# test ran them. No algorithm here touches the global numpy state, so the
# order does not change a result today; this test is what keeps that true.
EXPECTED_CONTINUOUS = ["GLOSS(4:2:2)", "GLOSS(2:3:3)", "GLOSS(6:1:1)",
                       "UCB-BO", "BO(EI)", "GA", "qPO", "Thompson", "Random"]

# response experiments need.
EXPECTED_RERUN = ["GLOSS(4:2:2)", "GLOSS(8:0:0)", "GLOSS(8:0:0)+div",
                  "UCB-BO", "BO(EI)", "GA", "qPO", "Thompson", "Random"]

# (quota, diversity radius, kappa) for each GLOSS arm. The two 8:0:0 arms come
# from benchmarks/bench_ablation_diversity.py, where they differ from each
# other in the radius alone: that difference is the whole experiment.
EXPECTED_GLOSS = {
    "GLOSS(4:2:2)": ({"global_best": 4, "local_best": 2, "unexplored": 2,
                      "unconverged": 0}, 0.02, 2.0),
    "GLOSS(2:3:3)": ({"global_best": 2, "local_best": 3, "unexplored": 3,
                      "unconverged": 0}, 0.02, 2.0),
    "GLOSS(6:1:1)": ({"global_best": 6, "local_best": 1, "unexplored": 1,
                      "unconverged": 0}, 0.02, 2.0),
    "GLOSS(8:0:0)": ({"global_best": 8, "local_best": 0, "unexplored": 0,
                      "unconverged": 0}, 0.0, 2.0),
    "GLOSS(8:0:0)+div": ({"global_best": 8, "local_best": 0, "unexplored": 0,
                          "unconverged": 0}, 0.02, 2.0),
}

EXPECTED_CLASS = {
    "UCB-BO": UCBOAlgorithm, "BO(EI)": BOAlgorithm, "GA": GeneticAlgorithm,
    "qPO": QPOAlgorithm, "Thompson": ThompsonSamplingAlgorithm,
    "Random": RandomAlgorithm,
}


def test_continuous_arm_names_and_order():
    assert bench_main.CONTINUOUS_ARMS == EXPECTED_CONTINUOUS
    assert list(bench_main.make_continuous_algorithms(SEED)) == \
        EXPECTED_CONTINUOUS


def test_rerun_arm_names_and_order():
    assert bench_main.RERUN_ARMS == EXPECTED_RERUN


def test_registry_covers_both_lists():
    for name in EXPECTED_CONTINUOUS + EXPECTED_RERUN:
        assert name in bench_main.ARM_BUILDERS


@pytest.mark.parametrize("name", sorted(EXPECTED_GLOSS))
def test_gloss_arm_settings(name):
    quota, radius, kappa = EXPECTED_GLOSS[name]
    algo = bench_main.make_arms(SEED, [name])[name]
    assert isinstance(algo, GLOSSAlgorithm)
    assert algo._strategy_points == quota
    assert algo._diversity_radius == radius
    assert algo._ucb_kappa == kappa
    assert sum(quota.values()) == 8, "a GLOSS arm has to spend the whole batch"


@pytest.mark.parametrize("name", sorted(EXPECTED_CLASS))
def test_other_arm_classes(name):
    algo = bench_main.make_arms(SEED, [name])[name]
    assert isinstance(algo, EXPECTED_CLASS[name])


def test_two_ablation_arms_differ_in_the_radius_alone():
    """The diversity ablation rests on this and on nothing else."""
    plain = bench_main.make_arms(SEED, ["GLOSS(8:0:0)"])["GLOSS(8:0:0)"]
    with_div = bench_main.make_arms(
        SEED, ["GLOSS(8:0:0)+div"])["GLOSS(8:0:0)+div"]
    assert plain._strategy_points == with_div._strategy_points
    assert plain._ucb_kappa == with_div._ucb_kappa
    assert plain._diversity_radius == 0.0
    assert with_div._diversity_radius == 0.02


def test_kappa_of_the_pointwise_baselines():
    """UCB-BO is the ablation control, so its kappa has to match GLOSS's."""
    ucb = bench_main.make_arms(SEED, ["UCB-BO"])["UCB-BO"]
    ga = bench_main.make_arms(SEED, ["GA"])["GA"]
    assert ucb.kappa == 2.0
    assert ga.kappa == 2.0


def test_a_new_object_per_call():
    """make_arms must not hand out a shared object.

    Only GeneticAlgorithm.reset() really resets. GLOSSAlgorithm._call_count
    feeds seed * 10000 + _call_count into the random stream GLOSS draws from,
    so a caller that reuses an object across seeds gets different numbers with
    no error anywhere.
    """
    first = bench_main.make_arms(SEED, EXPECTED_RERUN)
    second = bench_main.make_arms(SEED, EXPECTED_RERUN)
    for name in EXPECTED_RERUN:
        assert first[name] is not second[name], name


def test_seed_reaches_the_object():
    gloss_a = bench_main.make_arms(47, ["GLOSS(4:2:2)"])["GLOSS(4:2:2)"]
    gloss_b = bench_main.make_arms(53, ["GLOSS(4:2:2)"])["GLOSS(4:2:2)"]
    assert gloss_a._seed == 47
    assert gloss_b._seed == 53
    assert gloss_a._call_count == 0


def test_unknown_arm_is_an_error():
    with pytest.raises(KeyError):
        bench_main.make_arms(SEED, ["GLOSS(9:9:9)"])


def test_ten_seeds_are_defined_once():
    """The scaling sweep used to carry its own copy of this list."""
    assert bench_main.SEEDS_10[:5] == bench_main.SEEDS
    assert bench_main.SEEDS_10 == [47, 53, 59, 61, 67, 71, 73, 79, 83, 89]
    from benchmarks import bench_scaling_batch_aware
    assert bench_scaling_batch_aware.ALL_SEEDS is bench_main.SEEDS_10
