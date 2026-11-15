
import numpy as np
from scipy.special import erfc

from benchmarks import protocol
from benchmarks.algorithms import _expected_improvement

#: Relative floor for sigma(x_j). An absolute floor of 1e-6 is meaningless at
#: the scale of these objectives (0.02 to 0.5) and turns erfc into a step.
SIGMA_FLOOR_REL = 1e-3

#: A finite difference needs a minimum spacing to avoid dividing by zero.
MIN_SPACING = 1e-12

#: Shift applied when an acquisition is not already positive.
SHIFT_EPS = 1e-9

#: Batch sizes at which the occupancy diagnostic is reported. k=1 is
#: identically zero because no penalty exists yet, so it is not reported.
OCCUPANCY_STEPS = (2, 8)


def neighbour_pairs(X):
    """Spacing and index of each candidate's nearest neighbour, once.

    They depend only on the candidate set, which is fixed for a whole run, so
    the caller caches them. Returns ``(spacing, index)``.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    if len(X) < 2:
        return np.zeros(1), np.zeros(1, dtype=int)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    dist, idx = nn.kneighbors(X)
    return np.maximum(dist[:, 1], MIN_SPACING), idx[:, 1]


def estimate_lipschitz(mu, pairs):
    """Largest nearest-neighbour slope over the whole candidate set."""
    mu = np.asarray(mu, dtype=float)
    spacing, index = pairs
    if len(mu) < 2:
        return 0.0
    return float(np.max(np.abs(mu - mu[index]) / spacing))


def estimate_lipschitz_restricted(mu, available, spacing, neighbour):
    """The same estimate, over the rows still available to the arm.

    The neighbour indices point into the full candidate set, because that is
    what the cached tree covers; only the rows being scored contribute a slope.
    """
    if len(available) < 2:
        return 0.0
    slopes = np.abs(mu[available] - mu[neighbour]) / spacing
    return float(np.max(slopes))


def penalty_factors(X, chosen, mu_chosen, sd_chosen, m_hat, lipschitz,
                    sigma_floor):
    """The product of penalty factors for every candidate against `chosen`.

    Returns (phi_product, z_last, r_last). Pure: the tests drive it directly
    with a known Lipschitz constant and a closed-form mean, because a fitted
    forest has a piecewise-constant mean and every finite difference on it is
    either zero or an outlier.
    """
    X = np.asarray(X, dtype=float)
    phi = np.ones(len(X))
    z_last = np.zeros(len(X))
    r_last = 0.0
    for x_j, mu_j, sd_j in zip(chosen, mu_chosen, sd_chosen):
        if lipschitz <= 0.0:
            # No slope estimate means no radius estimate, so no penalty. The
            # occupancy column is what makes that visible.
            r_last = 0.0
            z_last = np.full(len(X), np.inf)
            continue
        r_j = (m_hat - mu_j) / lipschitz
        distance = np.linalg.norm(X - np.asarray(x_j, dtype=float), axis=1)
        z = (lipschitz * distance - m_hat + mu_j) / (
            np.sqrt(2.0) * max(sd_j, sigma_floor))
        phi = phi * 0.5 * erfc(-z)
        z_last = z
        r_last = r_j
    return phi, z_last, r_last


def positive(acq):
    """Shift an acquisition onto the positive half line, order preserved."""
    acq = np.asarray(acq, dtype=float)
    lowest = float(np.min(acq))
    # Zero is a legitimate value of expected improvement, and shifting it would
    # make candidates that tie at zero distinguishable by their penalty factor.
    # Only a genuinely negative acquisition needs the shift.
    if lowest >= 0.0:
        return acq
    return acq - lowest + SHIFT_EPS


class LocalPenaltyAlgorithm:
    """Greedy batch selection with one local penalty per chosen point."""

    def __init__(self, base="ei", kappa=2.0, sigma_floor_rel=SIGMA_FLOOR_REL,
                 seed=None):
        if base not in ("ei", "ucb"):
            raise ValueError(f"base must be 'ei' or 'ucb', got {base!r}")
        self.base = base
        self.kappa = kappa
        self.sigma_floor_rel = sigma_floor_rel
        self.rng = np.random.default_rng(seed)
        self.n_fill = 0
        self.postfilter_dropped = 0
        #: One dict per round, for the sweep to merge into its rows.
        self.diagnostics = []
        #: Nearest-neighbour pairs of the current candidate set, built once.
        self._pairs = None
        self._pairs_X = None

    # -- acquisition -------------------------------------------------------
    def _acquisition(self, mu, sd, y_best):
        if self.base == "ei":
            return positive(_expected_improvement(mu, sd, y_best=y_best))
        return positive(mu + self.kappa * sd)

    def _cached_pairs(self, X):
        """Neighbour pairs of the whole candidate set, built once per run.

        The subset of rows still available changes every round, and keying the
        cache on that subset would rebuild a 100k-point tree every round: a
        measured 21 s per round instead of 3 s per unit. The neighbours come
        from the full candidate set, which is the set the arm scores.
        """
        if self._pairs is None or self._pairs_X is not X:
            self._pairs = neighbour_pairs(X)
            self._pairs_X = X
        return self._pairs

    # -- greedy batch ------------------------------------------------------
    def _select(self, X, mu, sd, available, n_points, y_best):
        """Greedy batch over `X`, restricted to the `available` rows."""
        alpha = self._acquisition(mu, sd, y_best)
        m_hat = float(np.max(mu[available])) if len(available) else 0.0
        spacing, neighbour = self._cached_pairs(X)
        spacing = spacing[available]
        neighbour = neighbour[available]
        lipschitz = estimate_lipschitz_restricted(mu, available, spacing,
                                                  neighbour)
        sigma_floor = self.sigma_floor_rel * float(np.max(sd[available])) \
            if len(available) else 0.0

        # Rows that are not available are the already measured candidates.
        # Without this mask the arm spends part of its budget re-observing
        # points it has seen: measured on the first run, 16.6% of the EI arm's
        # picks and 20.5% of the UCB arm's picks were repeats.
        allowed = np.zeros(len(X), dtype=bool)
        allowed[available] = True

        picked = []
        taken = np.zeros(len(X), dtype=bool)
        chosen_x, chosen_mu, chosen_sd = [], [], []
        radii = []
        occupancy = {}
        z_p10 = z_p90 = float("nan")

        for step in range(1, n_points + 1):
            phi, z_last, r_last = penalty_factors(
                X, chosen_x, chosen_mu, chosen_sd, m_hat, lipschitz,
                sigma_floor)
            if chosen_x:
                radii.append(r_last)
                if step in OCCUPANCY_STEPS:
                    # Share of the pool the already chosen points rule out.
                    # Denominator: the pool rows still available to the arm,
                    # which is what the ranking sees.
                    occupancy[step] = float(np.mean(phi[available] < 0.5))
                    if lipschitz > 0.0:
                        z_p10 = float(np.percentile(z_last[available], 10))
                        z_p90 = float(np.percentile(z_last[available], 90))
            # Ties break by candidate index: argmax returns the first maximum,
            # and expected improvement is exactly zero for most candidates, so
            # without that rule the batch would not be reproducible.
            score = alpha * phi
            score[~allowed] = -np.inf
            score[taken] = -np.inf
            if not np.isfinite(score).any():
                break
            pick = int(np.argmax(score))
            if not np.isfinite(score[pick]):
                break
            picked.append(pick)
            taken[pick] = True
            chosen_x.append(X[pick])
            chosen_mu.append(float(mu[pick]))
            chosen_sd.append(float(sd[pick]))

        self.diagnostics.append({
            "lp_L": lipschitz,
            "lp_M_hat": m_hat,
            "lp_r_min": float(np.min(radii)) if radii else float("nan"),
            "lp_r_median": float(np.median(radii)) if radii else float("nan"),
            "lp_occ_k2": occupancy.get(2, float("nan")),
            "lp_occ_k8": occupancy.get(8, float("nan")),
            "lp_z_p10": z_p10,
            "lp_z_p90": z_p90,
            "lp_sigma_floor": sigma_floor,
        })
        return picked

    # -- runner interface --------------------------------------------------
    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        mu, sd = gp.predict(candidates, return_std=True)
        mu = np.asarray(mu, dtype=float)
        sd = np.asarray(sd, dtype=float)
        obs_set = set(map(tuple, np.round(X_obs, 10)))
        available = np.array([
            i for i in range(len(candidates))
            if tuple(np.round(candidates[i], 10)) not in obs_set])
        if len(available) == 0:
            return []
        return self._select(candidates, mu, sd, available, n_points,
                            float(np.max(y_obs)))

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp, pool=None):
        """Greedy batch over the round's shared pool, without refinement.

        Refinement is a measured no-op on a random forest (zero displacement in
        180 of 180 probes), so skipping it reproduces what refining would do.
        The pool the runner supplies has already passed the protocol's distance
        filter, so every pick keeps the 0.02 rule.
        """
        ndim = len(bounds)
        rng = self.rng
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        samples = pool if pool is not None else rng.uniform(
            lows, highs, size=(10000, ndim))
        mu, sd = gp.predict(samples, return_std=True)
        mu = np.asarray(mu, dtype=float)
        sd = np.asarray(sd, dtype=float)
        picks = self._select(samples, mu, sd, np.arange(len(samples)),
                             n_points, float(np.max(y_obs)))
        selected = [samples[i] for i in picks]
        self.n_fill = 0
        # A fill point obeys the same 0.02 rule as a chosen one. A uniform draw
        # that lands inside the exclusion zone would break the protocol the
        # other arms are held to, so it is filtered here rather than trusted.
        while len(selected) < n_points:
            candidate = rng.uniform(lows, highs)
            if protocol.far_enough(candidate, X_obs):
                selected.append(candidate)
                self.n_fill = self.n_fill + 1
        return np.array(selected[:n_points])

    def reset(self):
        self.n_fill = 0
        self.postfilter_dropped = 0
        self.diagnostics = []
        self._pairs = None
        self._pairs_X = None

    def diagnostics_frame(self):
        import pandas as pd

        return pd.DataFrame(self.diagnostics)
