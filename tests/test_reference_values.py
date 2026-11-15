import importlib.util
import os

import numpy as np
import pytest

from benchmarks import reference_values
from benchmarks.virtual_functions import (
    RATIO_COMPLEXITY_LEVELS,
    arrhenius_yield,
    get_virtual_dataset,
    nd_clusters,
)

ARRHENIUS = dict(n_peaks=3, peak_width=0.08, noise_sigma=0.0, seed=42)

# What the calibration produced. A sixty-digit Newton solve puts the true
# optimum at 18.35218451126244377, so this sits two units in the last place
# below it, and the largest value float64 reaches near the argmax is about
# 18.352184511262468. That is why the percentage needs PCT_OPT_RTOL rather
# than a bare "at most 100".
CALIBRATED_Y_OPT = 18.352184511262436
CALIBRATED_GRID_MAX = 18.35158018116758
SECOND_LOCAL_MAX = 13.59082620005476


def _surface(points):
    return arrhenius_yield(points, n_peaks=3, noise_sigma=0.0,
                           peak_width=0.08, seed=42)


def _grid_max(res):
    xs = np.linspace(0.0, 1.0, res)
    xx, yy = np.meshgrid(xs, xs)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    best = -np.inf
    for start in range(0, len(points), 500_000):
        best = max(best, float(_surface(points[start:start + 500_000]).max()))
    return best


def _calibration_module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "calibrate_reference_values.py")
    if not os.path.exists(path):
        pytest.skip("calibration script not present in this tree")
    spec = importlib.util.spec_from_file_location("calibrate_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the stored numbers are the calibrated ones ---------------------------

def test_the_stored_optimum_is_the_calibrated_value():
    """Without this, a hand-edited or truncated optimum passes everything else.

    Six digits of the optimum is 18.352184, which is BELOW the true value, so
    a run that reaches the top would report over 100%.
    """
    record = reference_values.get("arrhenius", **ARRHENIUS)
    assert record["y_opt"] == pytest.approx(CALIBRATED_Y_OPT, rel=1e-12)
    assert record["y_min"] == 0.0


def test_the_threshold_agrees_with_the_two_ends():
    for key in sorted(reference_values._document()["entries"]):
        record = reference_values._document()["entries"][key]
        expected = record["y_min"] + 0.95 * (record["y_opt"] - record["y_min"])
        assert record["threshold_95"] == pytest.approx(expected, rel=1e-15), key


def test_every_optimum_lies_inside_the_search_box():
    for key in sorted(reference_values._document()["entries"]):
        x_opt = reference_values._document()["entries"][key]["x_opt"]
        assert all(0.0 <= v <= 1.0 for v in x_opt), key


def test_the_tolerance_travels_with_the_values():
    assert reference_values.PCT_OPT_RTOL == 1e-6
    assert reference_values._document()["pct_opt_rtol"] == 1e-6


# -- the optimum really is the highest value ------------------------------

def test_the_guard_value_still_matches_the_surface():
    """A change to the surface must not leave the stored values in place.

    The guard is the maximum over the same 500-point grid the calibration
    walked, so any edit to arrhenius_yield moves it. The tolerance absorbs a
    one-ulp difference between maths libraries and still catches a relative
    change of 1e-10, which is a hundred times smaller than it allows.
    """
    record = reference_values.get("arrhenius", **ARRHENIUS)
    assert record["guard_grid_max"] == pytest.approx(CALIBRATED_GRID_MAX, rel=1e-12)
    assert _grid_max(record["guard_grid_res"]) == pytest.approx(
        record["guard_grid_max"], rel=1e-12)


def test_a_finer_grid_finds_nothing_above_the_stored_optimum():
    record = reference_values.get("arrhenius", **ARRHENIUS)
    finer = _grid_max(record["cross_check_grid_res"])
    assert finer <= record["y_opt"]
    assert finer == pytest.approx(record["cross_check_grid_max"], rel=1e-12)


def test_the_starts_reached_more_than_one_basin():
    """The only direct evidence that the start selection covered the surface.

    Peaks 1 and 2 merge, so the default surface has exactly two basins: the
    merged pair, and the nominally tallest peak on its own at 13.5908.
    """
    record = reference_values.get("arrhenius", **ARRHENIUS)
    maxima = record["local_maxima"]
    assert len(maxima) == 2
    assert maxima[0]["y"] == pytest.approx(CALIBRATED_Y_OPT, rel=1e-12)
    assert maxima[1]["y"] == pytest.approx(SECOND_LOCAL_MAX, rel=1e-12)
    assert record["n_starts"] >= 8


def test_a_million_random_points_stay_under_the_stored_optimum():
    """Only a smoke check: a run that exceeds y_opt by a lot shows up here.

    One million points is the floor for this to mean anything. The optimum
    sits 0.003% above the working grid, so ten million uniform points hold
    only about ten that beat the grid at all.
    """
    record = reference_values.get("arrhenius", **ARRHENIUS)
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1.0, size=(1_000_000, 2))
    assert float(_surface(points).max()) <= record["y_opt"]


# -- the equal-height clusters -------------------------------------------

@pytest.mark.parametrize(
    "level", RATIO_COMPLEXITY_LEVELS,
    ids=[level["label"] for level in RATIO_COMPLEXITY_LEVELS])
def test_cluster_centres_read_exactly_the_stored_optimum(level):
    """Both ends are exact for this landscape, so equality is the right test.

    The parameters come from the shipped level list, so a new level fails here
    until it is calibrated instead of being silently left out.
    """
    record = reference_values.get(
        "nd_clusters", ndim=level["ndim"], n_peaks=level["n_peaks"],
        peak_width=level["peak_width"], noise_sigma=0.0, seed=42)
    centres = np.asarray(record["centres"], dtype=float)
    at_centres = nd_clusters(centres, n_peaks=level["n_peaks"],
                             ndim=level["ndim"], peak_width=level["peak_width"],
                             noise_sigma=0.0, seed=42)
    assert np.all(at_centres == record["y_opt"])


@pytest.mark.parametrize(
    "level", RATIO_COMPLEXITY_LEVELS,
    ids=[level["label"] for level in RATIO_COMPLEXITY_LEVELS])
def test_random_cluster_points_stay_between_the_two_ends(level):
    record = reference_values.get(
        "nd_clusters", ndim=level["ndim"], n_peaks=level["n_peaks"],
        peak_width=level["peak_width"], noise_sigma=0.0, seed=42)
    rng = np.random.default_rng(1)
    points = rng.uniform(0.0, 1.0, size=(200_000, level["ndim"]))
    y = nd_clusters(points, n_peaks=level["n_peaks"], ndim=level["ndim"],
                    peak_width=level["peak_width"], noise_sigma=0.0, seed=42)
    assert float(y.max()) <= record["y_opt"]
    assert float(y.min()) >= record["y_min"]



def test_the_discrete_pool_still_has_the_numbers_the_kept_results_used():
    """The kept discrete results divide by these two, so they must not move."""
    y = get_virtual_dataset("arrhenius", n_candidates=10000)["y_true"]
    assert float(y.max()) == pytest.approx(18.09932209205458, rel=1e-12)
    assert float(y.min()) == pytest.approx(7.171182667492678e-06, rel=1e-12)


# -- the guards against misuse -------------------------------------------

def test_an_unknown_landscape_raises_and_says_what_is_on_file():
    with pytest.raises(KeyError) as caught:
        reference_values.get("arrhenius", n_peaks=12, peak_width=0.03,
                             noise_sigma=0.0, seed=42)
    assert "Calibrate them first" in str(caught.value)


def test_a_noisy_landscape_is_refused_rather_than_sent_to_calibration():
    with pytest.raises(KeyError) as caught:
        reference_values.get("arrhenius", n_peaks=5, peak_width=0.07,
                             noise_sigma=0.05, seed=42)
    assert "cannot get any" in str(caught.value)


def test_calibrating_a_noisy_surface_raises():
    with pytest.raises(ValueError) as caught:
        _calibration_module().arrhenius_reference(noise_sigma=0.05)
    assert "no reference value" in str(caught.value)


def test_a_numpy_scalar_builds_the_same_key_as_a_plain_float():
    plain = reference_values.key_for("arrhenius", **ARRHENIUS)
    as_numpy = reference_values.key_for(
        "arrhenius", n_peaks=np.int64(3), peak_width=np.float64(0.08),
        noise_sigma=np.float64(0.0), seed=np.int64(42))
    assert plain == as_numpy
