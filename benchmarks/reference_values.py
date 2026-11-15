import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "reference_values.json")
_CACHE = None

#: Relative slack a continuous run's percentage may exceed 100 by.
#:
#: The stored optimum is a refined numerical value, not a closed form, and it
#: can sit a couple of units in the last place below the largest value float64
#: reaches near the argmax. Measured on the Arrhenius surface, a run that
#: lands there reports 100.00000000000018. A check for "at most 100" would
#: fail on that; a check with this slack still catches any real overshoot,
#: which would be larger by many orders of magnitude.
PCT_OPT_RTOL = 1e-6


def _document():
    global _CACHE
    if _CACHE is None:
        with open(_PATH) as handle:
            _CACHE = json.load(handle)
    return _CACHE


def _plain(value):
    """A numpy scalar prints as np.float64(0.08) from numpy 2 on, so unwrap it.

    Without this a caller that pulls a parameter out of an array builds a
    different key from a caller that wrote the literal, and the lookup fails
    for no reason the caller can see.
    """
    return value.item() if hasattr(value, "item") else value


def key_for(landscape, **params):
    """One stable key per landscape and parameter set."""
    parts = [landscape] + [f"{k}={_plain(params[k])!r}" for k in sorted(params)]
    return "|".join(parts)


def get(landscape, **params):
    """Reference values for one landscape, or an error naming what is on file.

    Returns a dict with at least y_opt, y_min, x_opt and threshold_95.
    """
    document = _document()
    key = key_for(landscape, **params)
    entries = document["entries"]
    if key not in entries:
        if _plain(params.get("noise_sigma") or 0) != 0:
            raise KeyError(
                f"{key} has no reference values and cannot get any. A noisy "
                f"surface draws its noise per call by row index, so y is not "
                f"a function of x and neither end of the metric is defined. "
                f"Use the discrete path for noisy landscapes."
            )
        known = "\n  ".join(sorted(entries))
        raise KeyError(
            f"no reference values for {key}.\n"
            f"Calibrate them first with "
            f"the calibration script.\n"
            f"On file:\n  {known}"
        )
    return entries[key]


def ends(landscape, **params):
    """Just (y_opt, y_min), which is what the benchmark runners need."""
    record = get(landscape, **params)
    return record["y_opt"], record["y_min"]


def environment():
    """The numpy and scipy versions the values were calibrated under."""
    return _document()["environment"]
