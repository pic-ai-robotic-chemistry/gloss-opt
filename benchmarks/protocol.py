"""The protocol the continuous benchmark holds every algorithm to.

The discrete benchmark is even-handed by construction: every algorithm scores
the same candidate library. The continuous benchmark had no such construction,
and the algorithms drifted apart. Measured before this module existed, on the
two-dimensional Arrhenius surface, per round:

    GLOSS      10,704 points sent to the surrogate, 6.95 of 8 points delivered
    BO(EI)      5,030 points, 8.00 delivered
    UCB-BO      5,030 points, 8.00 delivered
    GA         10,134 points, 8.00 delivered
    qPO        10,000 points, 8.00 delivered
    Thompson   10,000 points, 8.00 delivered

GLOSS was sampling a separate pool per strategy, so it saw two to three times
as many candidates as the baselines. Only GLOSS refused to propose a point
close to one already measured: qPO spent 17% of its experiments within 0.002
of a measured point, Thompson 18%, GLOSS none. Both asymmetries work against
the batch-aware baselines, which is the opposite of what the continuous
rewrite was meant to fix.

Everything the protocol fixes lives here, is read through get() alone, and
raises rather than falling back, for the same reason the reference values do:
a benchmark that silently runs two algorithms under different rules produces
a number that looks fine and means nothing.
"""
import numpy as np

from benchmarks.reference_values import PCT_OPT_RTOL  # noqa: F401

VERSION = "1.0"

# PCT_OPT_RTOL is defined in reference_values, where the optimum it applies to
# comes from, and re-exported here because callers reach for it next to the
# other protocol constants. One definition, two names: it has already been
# looked up under the wrong module twice.

#: Candidate pool size, shared by every algorithm, every round.
#:
#: One number for every landscape and every dimension. A pool that grew with
#: pool sized from the 95% hit rate would put the answer into the setup: by
#: that rule the ten-dimensional level would get 122,000 candidates against
#: 512 for the two-dimensional one, while dimension is the very thing being
#: compared. Coverage is reported as a diagnostic instead (see
#: fraction_above_threshold).
#:
#: 10,000 is what qPO and Thompson already used, and it keeps GLOSS near two
#: seconds per round at ten dimensions. The cost of a larger pool falls almost
#: entirely on GLOSS: measured at ten dimensions, 5,000 candidates cost it
#: 1.46 s per round and 250,000 cost 50.3 s, of which 87-98% is the per-point
#: neighbourhood loop in local_best, while the random forest needs 0.85 s and
#: qPO needs 0.04 s on the same 250,000.
POOL_SIZE = 10_000

#: How far a proposal has to stay from every point already measured.
#:
#: This is the value GLOSS already enforced through duplicate_tolerance, so
#: adopting it changes nothing for GLOSS and only holds the baselines to the
#: same rule. The discrete benchmark has the same rule in a stronger form: a
#: candidate that has been measured is removed from the library outright.
MIN_DISTANCE_TO_MEASURED = 0.02

#: How far two points inside one batch have to stay from each other.
#:
#: Separate from the distance to measured points on purpose. Reusing that one
#: here would give whichever algorithm applied it a batch-diversity rule the
#: others do not have, and UCB-BO exists precisely as the arm with no batch
#: diversity: it would stop being a control. The value matches what the two
#: Bayesian-optimisation baselines already use to drop a duplicate.
MIN_DISTANCE_WITHIN_BATCH = 1e-6

#: How many of the pool's top-scoring candidates go to local refinement.
#:
#: It has to be at least the largest batch any strategy asks for, or the
#: surplus points fall through to a fallback that ranks by predicted mean with
#: no uncertainty term. Measured at 3, the four global_best points had UCB
#: ranks 0, 2217, 499 and 2132 while their predicted-mean ranks were 1483, 0,
#: 3 and 8: three of the four were chosen by mean, not by the acquisition
#: function the paper describes.
#:
#: Ten is not enough. Measured over 20 rounds x 5 seeds, the share of points
#: that still came from the mean-ranked fallback was 8.5% at a quota of four
#: and 28.7% at a quota of six. Fifty takes both to zero. Refinement is a
#: no-op on this surrogate family, so the extra starts cost almost nothing.
#:
#: Even at fifty, global_best is not pure UCB: it takes the UCB top n_top and
#: then re-sorts those by predicted mean (gloss/strategies/global_best.py).
#: The constant bounds how far down the UCB ranking a point can come from; it
#: does not change the ranking rule. The manuscript has to say so.
N_TOP_FOR_OPTIMIZATION = 50

#: The surrogate family the protocol was measured under.
#:
#: Local refinement is a no-op on a random forest, whose prediction is
#: piecewise constant: measured over both optimizers and both dimensions, the
#: displacement is exactly zero, nfev is ndim+1, and a finite difference
#: returns a zero gradient at every step size from 1e-8 to 1e-3. On a Gaussian
#: largest 1.18. So this protocol does not transfer to another surrogate
#: family without measuring refinement again. require() enforces that.
SURROGATE_FAMILY = "random_forest"

_ENTRIES = {
    "pool_size": POOL_SIZE,
    "min_distance_to_measured": MIN_DISTANCE_TO_MEASURED,
    "min_distance_within_batch": MIN_DISTANCE_WITHIN_BATCH,
    "n_top_for_optimization": N_TOP_FOR_OPTIMIZATION,
    "surrogate_family": SURROGATE_FAMILY,
    "version": VERSION,
}


def get(name):
    """One protocol constant, or an error naming the ones that exist."""
    if name not in _ENTRIES:
        known = ", ".join(sorted(_ENTRIES))
        raise KeyError(f"no protocol constant named {name!r}. On file: {known}")
    return _ENTRIES[name]


def family_of(surrogate):
    """The surrogate's family name, for require()."""
    return getattr(surrogate, "SURROGATE_FAMILY", type(surrogate).__name__)


def require(surrogate_family, refinement_evidence=None):
    if surrogate_family == SURROGATE_FAMILY:
        return
    if refinement_evidence:
        print(f"protocol: {surrogate_family!r} admitted on this evidence: "
              f"{refinement_evidence}")
        return
    raise ValueError(
        f"this protocol was measured under {SURROGATE_FAMILY!r}, not "
        f"{surrogate_family!r}. Local refinement is a no-op on a random "
        f"forest and is not on a Gaussian process, so the protocol does not "
        f"carry over. Measure refinement on the new surrogate, then pass "
        f"refinement_evidence='<script>, largest displacement <value>'."
    )


def make_pool(bounds, rng, n=None):
    """The candidate pool for one round, shared by every algorithm."""
    # The pool is drawn once per round and handed to all seven algorithms, so
    # the candidate set is identical and only the number of surrogate calls
    # differs between them. That difference is a property of the algorithms,
    # not a thumb on the scale.
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    return rng.uniform(lows, highs, size=(n or POOL_SIZE, len(bounds)))


def pool_sampler(pool):
    """A custom_sampler for GLOSS that hands back the shared pool.

    GLOSS samples once per strategy through Space.sample, so without this each
    of its three strategies draws its own pool and it sees three times the
    candidates the baselines do. The signature is the one Space.sample calls.
    """
    def sampler(n, bounds, constraints, rng):
        if n is None or n >= len(pool):
            return pool
        # Handing back pool[:n] would give every strategy the same prefix and
        # quietly shrink the shared set. The caller sets n_random_samples from
        # the pool length, so this is a wiring mistake, not a case to absorb.
        raise ValueError(
            f"the shared pool holds {len(pool)} candidates and {n} were asked "
            f"for. Set n_random_samples from the pool length."
        )
    return sampler


def filter_pool(pool, measured, surrogate, minimum=None, refinement_evidence=None):
    """Drop the pool rows that sit too close to a point already measured.

    One filter, applied to the shared pool before any algorithm sees it, is
    what makes the rule uniform. Patching each algorithm's own selection loop
    would give four more places for the rule to drift.

    This mirrors the discrete benchmark exactly: there a measured candidate is
    removed from the library, so no algorithm can propose it again. The
    continuous path had no equivalent, because the only exclusion on offer
    matched candidate rows exactly and a freshly drawn random point never
    equals a measured one. That is why qPO and Thompson were spending a sixth
    of their experiments within 0.002 of a point already measured.

    Filtering the pool is enough ONLY because local refinement is a no-op
    under this protocol's surrogate family: a refined point cannot walk back
    into the exclusion zone if it does not move at all. On a surrogate where
    refinement does move points, the filter would have to be re-applied after
    refinement. require() is what keeps that case from slipping through.

    refinement_evidence is forwarded to require(). A caller running a surrogate
    whose refinement does move points passes the measured displacement here;
    the algorithms that refine then re-apply this filter to their refined
    points, so nothing is admitted by this call alone.
    """
    # Every algorithm that draws from the pool comes through here, which makes
    # this the one place the surrogate family can be checked without relying
    # on each caller to remember.
    require(family_of(surrogate), refinement_evidence=refinement_evidence)
    if len(measured) == 0:
        return pool
    gap = minimum if minimum is not None else MIN_DISTANCE_TO_MEASURED
    measured = np.atleast_2d(np.asarray(measured, dtype=float))
    keep = np.ones(len(pool), dtype=bool)
    chunk = 20_000  # a full pairwise matrix is pool x measured x ndim floats
    for start in range(0, len(pool), chunk):
        block = pool[start:start + chunk]
        gaps = np.linalg.norm(block[:, None, :] - measured[None, :, :], axis=-1)
        keep[start:start + chunk] = gaps.min(axis=1) >= gap
    return pool[keep]


def far_enough(point, measured, minimum=None):
    """Is this proposal far enough from everything already measured?"""
    if len(measured) == 0:
        return True
    gap = minimum if minimum is not None else MIN_DISTANCE_TO_MEASURED
    # Strictly inside the gap is a violation, exactly at it is not: one
    # convention, matching filter_pool, so a point at the boundary is not
    # kept by one rule and rejected by the other.
    return bool(np.min(np.linalg.norm(np.asarray(measured) - point, axis=1)) >= gap)


_UNIFORM_DISTANCE = {}


def uniform_mean_distance(ndim, samples=200_000, seed=0):
    """Mean distance between two uniform points in the unit cube.

    The plain mean pairwise distance of a batch grows with dimension on its
    own -- it tends to sqrt(ndim/6) -- so comparing the figure across
    dimensions says more about the dimension than about the algorithm.
    Dividing by this reference removes that, and what is left is how tightly
    an algorithm packs its batch relative to drawing at random.

    It depends only on the dimension, so it is computed once and kept.
    """
    if ndim not in _UNIFORM_DISTANCE:
        rng = np.random.default_rng(seed)
        left = rng.uniform(0.0, 1.0, size=(samples, ndim))
        right = rng.uniform(0.0, 1.0, size=(samples, ndim))
        _UNIFORM_DISTANCE[ndim] = float(
            np.linalg.norm(left - right, axis=1).mean())
    return _UNIFORM_DISTANCE[ndim]


def fraction_above_threshold(func, bounds, threshold, n=4_000_000, seed=0):
    """Share of the box that scores at or above the threshold.

    This is the coverage diagnostic the pool size is deliberately NOT derived
    from. It is reported for every landscape and dimension, including the
    levels where it is poor: measured against the step-1 reference thresholds,
    the eight-dimensional cluster level holds 1.34e-4 of the box and the
    ten-dimensional one 4.10e-5, so a 10,000-point pool is expected to contain
    1.3 and 0.4 qualifying candidates. Those numbers belong in the report,
    not in the protocol.
    """
    rng = np.random.default_rng(seed)
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    hits, seen = 0, 0
    chunk = 500_000
    while seen < n:
        take = min(chunk, n - seen)
        points = rng.uniform(lows, highs, size=(take, len(bounds)))
        hits += int((func(points) >= threshold).sum())
        seen += take
    return hits / seen
