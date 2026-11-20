"""Batch recommendation algorithms for the discrete and continuous benchmarks.

The main experiment supplies a random-forest surrogate to model-based arms;
the GP sensitivity study supplies a Gaussian process. Random sampling uses
the candidate pool directly. The runner calls recommend_discrete() or
recommend_continuous() according to the search space.
"""

import time
import numpy as np

from benchmarks import protocol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm


def make_fixed_gp():
    """Create a GP with fixed hyperparameters (no kernel optimization)."""
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        nu=2.5, length_scale=1.0, length_scale_bounds="fixed"
    )
    return GaussianProcessRegressor(
        kernel=kernel, alpha=1e-5, optimizer=None, normalize_y=True
    )


def fit_gp(X, y):
    """Fit a fixed-hyperparameter GP on observed data."""
    gp = make_fixed_gp()
    gp.fit(X, y)
    return gp


def _nearest_free_index(candidates, point, taken):
    """Index of the candidate nearest to point, skipping ones already taken.

    An algorithm that proposes a position, rather than an index, has to be
    mapped back onto the candidate list. Several candidates can carry exactly
    the same feature row, and a plain argmin always returns the first of them,
    so a second proposed point mapping to that row used to be dropped and the
    batch came out short, by as much as three points of eight. Here the rows
    already taken by this batch are put out of reach and the next nearest free
    candidate is used instead.

    Points already measured are not excluded here, deliberately. Each strategy
    already draws from a pool with those removed, so the exact position it
    proposes is always still available and this function returns it unchanged.
    Adding a second exclusion rule on top only risks disagreeing with the
    first, and a disagreement would silently hand a strategy a different
    molecule under its own name.

    Which of several identical rows gets picked is arbitrary, and has to be:
    the surrogate sees the same features for all of them and cannot prefer
    one. Their measured values do differ, so the choice does affect the run;
    the baselines already take both copies when they rank by score, so this
    removes a handicap rather than granting an advantage.
    """
    dists = np.linalg.norm(candidates - point, axis=1)
    if len(taken):
        dists[list(taken)] = np.inf
    idx = int(np.argmin(dists))
    return idx if np.isfinite(dists[idx]) else None


# --------------------------------------------------------------------------
# GLOSS
# --------------------------------------------------------------------------
class GLOSSAlgorithm:
    """GLOSS wrapper for benchmarking."""

    def __init__(self, strategy_points=None, ucb_kappa=2.0, diversity_radius=0.0,
                 diversity_metric="euclidean", local_top_k=None, seed=None,
                 novelty_signal="distance", novelty_quantile=None):
        from gloss import GLOSS
        self._gloss_cls = GLOSS
        self._strategy_points = strategy_points
        self._ucb_kappa = ucb_kappa
        self._diversity_radius = diversity_radius
        self._diversity_metric = diversity_metric
        self._local_top_k = local_top_k
        # Rank gate for the unexplored strategy; None keeps the published
        # absolute threshold. See gloss/strategies/unexplored.py.
        self._novelty_signal = novelty_signal
        self._novelty_quantile = novelty_quantile
        self._seed = seed
        self._call_count = 0

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Recommend from discrete candidates.

        Returns:
            (indices, strategies): candidate indices and their per-point strategy labels.
        """
        from gloss import GLOSS

        call_seed = self._seed * 10000 + self._call_count if self._seed is not None else None
        self._call_count += 1
        gloss = GLOSS(
            space={"candidates": candidates},
            mode="discrete",
            direction="maximize",
            ucb_kappa=self._ucb_kappa,
            diversity_radius=self._diversity_radius,
            diversity_metric=self._diversity_metric,
            local_top_k=self._local_top_k,
            novelty_signal=self._novelty_signal,
            novelty_quantile=self._novelty_quantile,
            seed=call_seed,
        )
        strategy_points = self._strategy_points or _distribute_points(n_points)

        results = gloss.recommend(
            surrogate=gp,
            X_existing=X_obs,
            strategy_points=strategy_points,
        )

        all_indices = []
        strategies = []
        for r in results:
            if "point" not in r:
                continue
            idx = _nearest_free_index(candidates, np.array(r["point"]),
                                      all_indices)
            if idx is None:
                continue
            all_indices.append(idx)
            strategies.append(r.get("strategy", "unknown"))

        # The gate's own record of the round, read by the runner. GLOSS fills
        # it during recommend(); it is None under the published threshold gate.
        self.last_novelty_stats = getattr(gloss, "last_novelty_stats", None)
        return all_indices[:n_points], strategies[:n_points]

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp, pool=None):
        """Recommend from continuous space.

        A pool, when given, is the round's shared candidate set and is handed
        to every strategy through the custom sampler. Without it each strategy
        draws its own, which is how GLOSS came to see two to three times the
        candidates the baselines did.

        Returns:
            (points, strategies): coordinates and their per-point strategy labels.
        """
        from gloss import GLOSS

        call_seed = self._seed * 10000 + self._call_count if self._seed is not None else None
        self._call_count += 1
        n_samples = len(pool) if pool is not None else protocol.POOL_SIZE
        gloss = GLOSS(
            space={"bounds": bounds},
            mode="continuous",
            direction="maximize",
            n_random_samples=n_samples,
            # Refine the high-UCB starts before ranking by predicted mean.
            n_top_for_optimization=protocol.N_TOP_FOR_OPTIMIZATION,
            ucb_kappa=self._ucb_kappa,
            diversity_radius=self._diversity_radius,
            diversity_metric=self._diversity_metric,
            # Two distances, not one. The protocol value holds a proposal
            # away from every point already measured, and every arm is held to
            # it. The arm's own radius holds the points of one batch apart,
            # and that is the mechanism the ablation switches off: with one
            # knob for both, GLOSS(8:0:0) and GLOSS(8:0:0)+div returned the
            # same 1,680 points and the ablation had no "off" arm.
            duplicate_tolerance=protocol.MIN_DISTANCE_TO_MEASURED,
            batch_tolerance=self._diversity_radius,
            custom_sampler=protocol.pool_sampler(pool) if pool is not None else None,
            novelty_signal=self._novelty_signal,
            novelty_quantile=self._novelty_quantile,
            seed=call_seed,
        )
        strategy_points = self._strategy_points or _distribute_points(n_points)

        results = gloss.recommend(
            surrogate=gp,
            X_existing=X_obs,
            strategy_points=strategy_points,
        )

        self.n_fill = 0
        all_points = []
        strategies = []
        for r in results:
            if "point" in r:
                all_points.append(r["point"])
                strategies.append(r.get("strategy", "unknown"))

        if len(all_points) == 0:
            # Same rule as everywhere else: a fallback point still has to sit
            # outside the exclusion zone, and it comes from the shared pool
            # when there is one. Counted, because a round that falls back
            # entirely spends its whole budget outside the strategies and the
            # result file has to say so.
            fallback_rng = np.random.default_rng(call_seed)
            if pool is not None:
                idx = fallback_rng.choice(
                    len(pool), size=min(n_points, len(pool)), replace=False)
                # Count what was actually returned, not what was asked for: a
                # pool smaller than the batch gives fewer.
                self.n_fill = len(idx)
                return pool[idx], ["unknown"] * len(idx)
            self.n_fill = n_points
            lows = np.array([b[0] for b in bounds])
            highs = np.array([b[1] for b in bounds])
            fallback = []
            while len(fallback) < n_points:
                candidate = fallback_rng.uniform(lows, highs)
                if protocol.far_enough(candidate, X_obs):
                    fallback.append(candidate)
            return np.array(fallback), ["unknown"] * n_points

        self.last_novelty_stats = getattr(gloss, "last_novelty_stats", None)
        return np.array(all_points[:n_points]), strategies[:n_points]

    def reset(self):
        pass


def _distribute_points(n_points):
    """Distribute n_points across 4 strategies."""
    base = n_points // 4
    remainder = n_points % 4
    counts = [base] * 4
    for i in range(remainder):
        counts[i] += 1
    return {
        "global_best": counts[0],
        "local_best": counts[1],
        "unexplored": counts[2],
        "unconverged": counts[3],
    }


# --------------------------------------------------------------------------
# Bayesian Optimization (Expected Improvement)
# --------------------------------------------------------------------------
class BOAlgorithm:
    """Bayesian Optimization with Expected Improvement."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Recommend from discrete candidates using EI."""
        # Compute EI for all candidates
        mu, sigma = gp.predict(candidates, return_std=True)
        ei = _expected_improvement(mu, sigma, y_best=np.max(y_obs))

        # Exclude already observed points
        obs_set = set(map(tuple, np.round(X_obs, 10)))
        for i in range(len(candidates)):
            if tuple(np.round(candidates[i], 10)) in obs_set:
                ei[i] = -np.inf

        # Select top-n by EI, greedily avoiding duplicates
        indices = []
        order = np.argsort(ei)[::-1]
        for idx in order:
            if len(indices) >= n_points:
                break
            if idx not in indices:
                indices.append(idx)

        return indices

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp, pool=None):
        """Recommend from continuous space using EI on random samples."""
        ndim = len(bounds)
        rng = self.rng
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])

        # The round's shared pool when the runner supplies one, so that
        # every algorithm scores the same candidates.
        samples = pool if pool is not None else rng.uniform(
            lows, highs, size=(protocol.POOL_SIZE, ndim))
        mu, sigma = gp.predict(samples, return_std=True)
        ei = _expected_improvement(mu, sigma, y_best=np.max(y_obs))

        # Refine top candidates with scipy
        from scipy.optimize import minimize

        top_indices = np.argsort(ei)[::-1][:10]
        best_points = []

        for idx in top_indices:
            x0 = samples[idx]

            def neg_ei(x):
                m, s = gp.predict(x.reshape(1, -1), return_std=True)
                return -_expected_improvement(m, s, y_best=np.max(y_obs))[0]

            try:
                res = minimize(
                    neg_ei, x0, method="L-BFGS-B",
                    bounds=[(b[0], b[1]) for b in bounds],
                )
                if res.success:
                    best_points.append(res.x)
                else:
                    best_points.append(x0)
            except Exception:
                best_points.append(x0)

        # Deduplicate and select, re-applying the protocol's distance rule to
        # the refined points. On a random forest they never move, so this is a
        # no-op there and the published numbers stand; on a Gaussian process it
        # is what stops a refined point from being spent inside the exclusion
        # zone the pool filter cleared.
        selected = _refined_selection(
            best_points, samples, ei, X_obs, n_points,
            protocol.MIN_DISTANCE_TO_MEASURED, self)

        # A fill point obeys the same rule as a chosen one. Measured over 100
        # rounds this branch never fires, because refinement does not move a
        # point on this surrogate family and so the top ten stay distinct.
        # It is counted anyway: without a counter there is no way to check the
        # claim that every algorithm spends the same budget.
        self.n_fill = 0
        while len(selected) < n_points:
            candidate = rng.uniform(lows, highs)
            if protocol.far_enough(candidate, X_obs):
                selected.append(candidate)
                self.n_fill += 1

        return np.array(selected[:n_points])

    def reset(self):
        pass


def _keeps_protocol_distance(point, X_obs, selected, minimum=None):
    """Does this point keep the protocol's distance rules?

    Two rules, and they are not the same number. A point must be at least
    ``MIN_DISTANCE_TO_MEASURED`` from every measured point, which is what the
    pool filter enforces; and it must not duplicate another point of its own
    batch, which is ``MIN_DISTANCE_WITHIN_BATCH``. Using the 0.02 measured rule
    for batch mates as well would add batch diversity to arms that the
    protocol deliberately leaves without it, and would change the published
    BO(EI) and UCB-BO columns: with a random forest those two arms emit
    near-duplicate pairs, and a 0.02 batch rule replaces 6 to 8 of the 8 points
    every round.
    """
    minimum = protocol.MIN_DISTANCE_TO_MEASURED if minimum is None else minimum
    point = np.asarray(point, dtype=float)
    if not protocol.far_enough(point, X_obs, minimum):
        return False
    for other in selected:
        if np.linalg.norm(point - np.asarray(other, dtype=float)) < \
                protocol.MIN_DISTANCE_WITHIN_BATCH:
            return False
    return True


def _refined_selection(best_points, samples, scores, X_obs, n_points, minimum,
                       owner):
    """Select refined points, replacing any that walks back into the zone.

    Local refinement is a no-op on a random forest, so the published protocol
    could select refined points without re-checking them. On a Gaussian process
    refinement does move points, and a moved point can land inside the
    exclusion zone the pool filter just cleared. Dropping such a point and
    taking the next best unrefined pool row keeps the budget at n_points and
    keeps every arm measured under the same rule.

    postfilter_dropped records the number of refined points replaced.
    """
    order = np.argsort(scores)[::-1]
    selected = []
    dropped = 0
    cursor = 0
    for point in best_points:
        if len(selected) >= n_points:
            break
        if _keeps_protocol_distance(point, X_obs, selected, minimum):
            selected.append(np.asarray(point, dtype=float))
            continue
        dropped += 1
        while cursor < len(order):
            candidate = np.asarray(samples[order[cursor]], dtype=float)
            cursor += 1
            if _keeps_protocol_distance(candidate, X_obs, selected, minimum):
                selected.append(candidate)
                break
    owner.postfilter_dropped = dropped
    return selected


# --------------------------------------------------------------------------
# GA: surrogate-assisted genetic algorithm
# --------------------------------------------------------------------------
def _onehot_blocks(candidates):
    """Column blocks of a one-hot encoded library, or None if it is not one.

    A combinatorial library is the Cartesian product of a few reagent
    choices, and one-hot encoding lays those choices out as consecutive
    blocks of columns with exactly one column set in each block. Recovering
    the blocks recovers the choices themselves, which is what lets a genetic
    algorithm recombine whole reagents (this ligand with that base) instead
    of doing arithmetic on indicator columns. A descriptor library has no
    such layout and falls through to the real-vector genotype.
    """
    if candidates.ndim != 2 or candidates.shape[1] < 2:
        return None
    if not np.all((candidates == 0.0) | (candidates == 1.0)):
        return None

    blocks = []
    start = 0
    running = np.zeros(len(candidates))
    for j in range(candidates.shape[1]):
        running = running + candidates[:, j]
        if np.all(running == len(blocks) + 1):
            blocks.append((start, j + 1))
            start = j + 1
    if start != candidates.shape[1] or len(blocks) < 2:
        return None

    # The choices must also reconstruct the library one to one. Without that
    # a genotype would stand for several different reactions and the
    # algorithm could not know which one it had proposed.
    genes = np.column_stack([candidates[:, a:b].argmax(axis=1) for a, b in blocks])
    if len(np.unique(genes, axis=0)) != len(candidates):
        return None
    return blocks


class GeneticAlgorithm:
    """Surrogate-assisted genetic algorithm.

    A population of candidate experiments evolves under the surrogate's UCB
    score, which costs nothing to evaluate, and only the fittest handful goes
    for real measurement each round. That is how an evolutionary search is
    run when every true evaluation is an experiment, and it keeps this
    baseline on the same surrogate as every other algorithm here.

    The genotype follows the library. A combinatorial library is recombined
    on the reagent choices themselves, so a child always inherits whole
    choices from its parents and never needs to be mapped anywhere. A
    descriptor library has no such structure, so the genotype is the
    descriptor vector and only the points actually submitted are mapped onto
    their nearest unmeasured library members. Mapping inside the generation
    loop would both cost a nearest-neighbour scan per individual per
    generation and drag individuals away from the genotype they were bred
    from, so it is not done there.

    Diversity comes from fitness sharing (Goldberg & Richardson 1987): a
    point crowded by others in the population has its score divided by how
    many neighbours it has. The score is rescaled to a common range first,
    because dividing a raw UCB value would make the sharing strength depend
    on where the measured property happens to sit on the number line, and
    that changes from one dataset and one round to the next.

    The population carries across rounds. This algorithm is therefore the
    only one here whose recommendation is not a function of (X_obs, y_obs)
    alone, which is why it watches for a reused instance and resets itself.
    """

    def __init__(self, kappa=2.0, tournament_size=3, pop_size=100,
                 p_crossover=0.9, mutation_rate=None, mutation_sigma=0.1,
                 continuous_generations=None, seed=None):
        self.kappa = kappa
        self.tournament_size = tournament_size
        self.pop_size = pop_size
        self.continuous_generations = continuous_generations
        self.p_crossover = p_crossover
        # One gene per individual on average, the standard setting, unless a
        # caller asks for something else.
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Forget the population so the instance can start a fresh run."""
        self.rng = np.random.default_rng(self.seed)
        self._pop = None
        self._raw = None
        self._kind = None
        self._blocks = None
        self._levels = None
        self._genes = None
        self._geno_to_idx = None
        self._candidates = None
        self._sigma_share = None
        self._lows = None
        self._highs = None
        self._fingerprint = None
        self._n_obs_seen = -1
        self._cache = {}
        self.genotype_kind = None
        self.sigma_share = None
        self.n_fill = 0
        self.fill_indices = []
        self.n_ucb_requested = 0
        self.n_predictions = 0
        self.n_library_scored = 0

    def _zero_counters(self):
        """Start the per-round tally.

        Three separate numbers, because they answer three questions. The
        first is the search effort this baseline was granted, which is what
        makes the comparison with a pointwise acquisition fair. The second is
        what it actually cost, which is lower on a combinatorial library
        because the same reagent combination comes up again and again. The
        third is how much of the library it really looked at, which only has
        a meaning when a genotype is a library member.
        """
        self.n_ucb_requested = 0
        self.n_predictions = 0
        self.n_library_scored = 0
        self.n_fill = 0
        self.fill_indices = []

    # -- fitness ----------------------------------------------------------
    def _ucb(self, gp, X):
        mu, sigma = gp.predict(X, return_std=True)
        return mu + self.kappa * sigma

    def _score(self, gp, genes):
        """UCB of each genotype, reusing values already computed this round.

        On a combinatorial library the same reagent combination comes up
        again and again across generations, so its score is cached. Without
        the cache the real number of surrogate calls would be several times
        the budget this baseline is supposed to get.
        """
        self.n_ucb_requested += len(genes)
        if self._kind != "categorical":
            # A descriptor genotype is not a library member, so there is no
            # library coverage to count here: only the eight points that are
            # submitted ever touch the library.
            self.n_predictions += len(genes)
            return self._ucb(gp, genes)

        keys = [tuple(int(v) for v in g) for g in genes]
        missing = [k for k in dict.fromkeys(keys) if k not in self._cache]
        if missing:
            rows = self._candidates[[self._geno_to_idx[k] for k in missing]]
            values = self._ucb(gp, rows)
            self._cache.update(zip(missing, values))
            self.n_predictions += len(missing)
            self.n_library_scored += len(missing)
        return np.array([self._cache[k] for k in keys])

    def _distances(self, genes):
        if self._kind == "categorical":
            return (genes[:, None, :] != genes[None, :, :]).sum(axis=2)
        # Squared-distance identity rather than an explicit difference array:
        # the population is scored on every generation, and building a
        # (P, P, d) array each time dominates the run on a wide library.
        square = (genes * genes).sum(axis=1)
        gram = genes @ genes.T
        d2 = square[:, None] + square[None, :] - 2.0 * gram
        return np.sqrt(np.maximum(d2, 0.0))

    def _shared(self, raw, genes):
        """Fitness after sharing, on a scale that does not depend on y."""
        lo, hi = float(np.min(raw)), float(np.max(raw))
        if hi > lo:
            scaled = 1e-3 + (1.0 - 1e-3) * (raw - lo) / (hi - lo)
        else:
            scaled = np.ones(len(raw))
        counts = (self._distances(genes) <= self._sigma_share).sum(axis=1)
        return scaled / counts

    def _sharing_radius(self, genes):
        """Radius at which a random population of this size averages two
        neighbours per individual.

        A fixed number cannot serve here: the same radius that holds half the
        population in a two-dimensional pool holds nobody in a
        twenty-dimensional one, so sharing would be either the only thing
        that matters or no operation at all depending on the dataset. Fixing
        the average neighbour count instead keeps the selection pressure the
        same everywhere, and the radius it implies is reported for each
        library.
        """
        rng = np.random.default_rng(0)
        take = min(1000, len(genes))
        sample = genes[rng.choice(len(genes), take, replace=False)]
        iu = np.triu_indices(take, 1)
        flat = self._distances(sample)[iu]
        target = 1.0 / max(1, self.pop_size - 1)

        if self._kind == "categorical":
            widths = np.arange(1, genes.shape[1] + 1)
            fractions = np.array([(flat <= w).mean() for w in widths])
            return int(widths[int(np.argmin(np.abs(fractions - target)))])
        return float(np.quantile(flat, target))

    # -- genotypes --------------------------------------------------------
    def _genotype_of(self, rows):
        rows = np.atleast_2d(rows)
        if self._kind == "categorical":
            return np.column_stack(
                [rows[:, a:b].argmax(axis=1) for a, b in self._blocks]
            ).astype(np.int64)
        return rows.astype(float)

    def _key(self, gene):
        if self._kind == "categorical":
            return tuple(int(v) for v in gene)
        return tuple(np.round(gene, 10))

    def _random_genotypes(self, n):
        if self._kind == "categorical":
            return self._genes[self.rng.choice(len(self._genes), n, replace=True)]
        return self.rng.uniform(self._lows, self._highs, size=(n, len(self._lows)))

    def _repair(self, child):
        """Pull an illegal reagent combination back into the library.

        A few combinations of the reagent choices were never run, so uniform
        crossover can land on one. Resampling one choice at random reaches a
        combination that does exist in nearly every case; the nearest legal
        combination is the fallback.
        """
        if self._key(child) in self._geno_to_idx:
            return child
        trial = child.copy()
        j = int(self.rng.integers(0, trial.size))
        trial[j] = int(self.rng.integers(0, self._levels[j]))
        if self._key(trial) in self._geno_to_idx:
            return trial
        hamming = (self._genes != child[None, :]).sum(axis=1)
        ties = np.flatnonzero(hamming == hamming.min())
        return self._genes[int(self.rng.choice(ties))].copy()

    # -- evolution --------------------------------------------------------
    def _tournament(self, shared):
        pick = self.rng.integers(0, len(shared), size=self.tournament_size)
        return int(pick[int(np.argmax(shared[pick]))])

    def _tournament_many(self, shared, n):
        """Winners of n independent tournaments."""
        picks = self.rng.integers(0, len(shared), size=(n, self.tournament_size))
        return picks[np.arange(n), np.argmax(shared[picks], axis=1)]

    def _breed(self, parents, shared):
        """One generation of children from the whole population at once."""
        n_children, n_genes = parents.shape
        mothers = parents[self._tournament_many(shared, n_children)]
        fathers = parents[self._tournament_many(shared, n_children)]

        crossing = self.rng.random((n_children, 1)) < self.p_crossover
        from_mother = self.rng.random((n_children, n_genes)) < 0.5
        children = np.where(crossing & ~from_mother, fathers, mothers)

        rate = 1.0 / n_genes if self.mutation_rate is None else self.mutation_rate
        mutating = self.rng.random((n_children, n_genes)) < rate
        if self._kind == "categorical":
            draws = self.rng.integers(0, self._levels, size=(n_children, n_genes))
            children = np.where(mutating, draws, children)
            children = self._repair_all(children)
        else:
            noise = self.rng.normal(0.0, self.mutation_sigma, children.shape)
            children = np.clip(children + mutating * noise, self._lows, self._highs)
        return children

    def _repair_all(self, children):
        """Replace any reagent combination the library does not contain."""
        for i, child in enumerate(children):
            if self._key(child) not in self._geno_to_idx:
                children[i] = self._repair(child)
        return children

    def _select_survivors(self, gp, pool, raw, elite_gene):
        """Keep the best of parents and children, with the measured best safe.

        Duplicates go first and truncation second. The other order would
        discard distinct individuals to the size limit and then refill the
        gaps with random ones, which loses inheritance for nothing.
        """
        keys = pool if self._kind == "categorical" else np.round(pool, 10)
        _, first = np.unique(keys, axis=0, return_index=True)
        keep = np.sort(first)  # keep the earliest copy, so a parent outlives
        pool, raw = pool[keep], raw[keep]  # an identical child

        shared = self._shared(raw, pool)
        order = np.argsort(shared)[::-1][: self.pop_size]
        pool, raw = pool[order], raw[order]

        if elite_gene is not None and self._key(elite_gene) not in {
            self._key(g) for g in pool
        }:
            elite_raw = self._score(gp, elite_gene[None, :])
            if len(pool) < self.pop_size:
                pool = np.vstack([pool, elite_gene[None, :]])
                raw = np.concatenate([raw, elite_raw])
            else:
                pool[-1] = elite_gene
                raw[-1] = elite_raw[0]

        if len(pool) < self.pop_size:
            extra = self._random_genotypes(self.pop_size - len(pool))
            pool = np.vstack([pool, extra])
            raw = np.concatenate([raw, self._score(gp, extra)])
        return pool, raw

    # -- batch ------------------------------------------------------------
    def _decode_free(self, point, taken, blocked):
        """Nearest library member that is free, ties broken at random.

        Equidistant candidates are common: an untried reagent choice leaves
        a whole block of the library exactly as far away, and taking the
        first of them every time would pull the batch towards whichever
        corner of the spreadsheet happens to come first.
        """
        dist = np.linalg.norm(self._candidates - point, axis=1)
        if len(taken):
            dist[list(taken)] = np.inf
        dist[blocked] = np.inf
        closest = dist.min()
        if not np.isfinite(closest):
            return None
        ties = np.flatnonzero(dist == closest)
        return int(ties[0]) if len(ties) == 1 else int(self.rng.choice(ties))

    def _submit(self, observed, n_points):
        shared = self._shared(self._raw, self._pop)
        indices = []
        for i in np.argsort(shared)[::-1]:
            if len(indices) >= n_points:
                break
            if self._kind == "categorical":
                idx = self._geno_to_idx[self._key(self._pop[i])]
                if observed[idx] or idx in indices:
                    continue
            else:
                idx = self._decode_free(self._pop[i], indices, observed)
                if idx is None:
                    continue
            indices.append(idx)

        if len(indices) < n_points:
            free = np.setdiff1d(
                np.flatnonzero(~observed), np.array(indices, dtype=int)
            )
            draw = min(n_points - len(indices), len(free))
            if draw:
                extra = self.rng.choice(free, size=draw, replace=False)
                indices.extend(int(i) for i in extra)
                self.fill_indices = [int(i) for i in extra]
                self.n_fill += draw
        return indices

    # -- public interface -------------------------------------------------
    def _fingerprint_of(self, candidates):
        stride = max(1, len(candidates) // 64)
        sample = np.ascontiguousarray(candidates[::stride])
        return (candidates.shape, hash(sample.tobytes()))

    def _start_run_if_needed(self, candidates, X_obs, y_obs):
        """Reset on a new run.

        The number of observations only ever grows inside a run, so a count
        that has not grown means a different run is using this instance. The
        candidate pool is fingerprinted as well, since a different dataset of
        the same width and length would otherwise go unnoticed.
        """
        fingerprint = self._fingerprint_of(candidates)
        if len(X_obs) <= self._n_obs_seen or fingerprint != self._fingerprint:
            self._init_population(candidates, X_obs, y_obs)
            self._fingerprint = fingerprint
        self._n_obs_seen = len(X_obs)

    def _init_population(self, candidates, X_obs, y_obs):
        self.rng = np.random.default_rng(self.seed)
        self._candidates = candidates
        self._cache = {}
        self._zero_counters()

        self._blocks = _onehot_blocks(candidates)
        if self._blocks is not None:
            self._kind = "categorical"
            self._levels = np.array([b - a for a, b in self._blocks])
            self._genes = self._genotype_of(candidates)
            self._geno_to_idx = {
                self._key(g): i for i, g in enumerate(self._genes)
            }
            seed_genes = self._genes
        else:
            self._kind = "vector"
            self._levels = None
            self._genes = None
            self._geno_to_idx = None
            seed_genes = candidates
        self._lows = np.zeros(candidates.shape[1])
        self._highs = np.ones(candidates.shape[1])

        self._sigma_share = self._sharing_radius(seed_genes)
        self.genotype_kind = self._kind
        self.sigma_share = self._sigma_share
        self._pop = None
        self._raw = None

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Evolve the population on the surrogate, submit its fittest points."""
        self._start_run_if_needed(candidates, X_obs, y_obs)
        observed = _observed_mask(candidates, X_obs)
        self._zero_counters()

        if self._pop is None:
            start = self._genotype_of(X_obs)
            filler = self._random_genotypes(max(0, self.pop_size - len(start)))
            self._pop = np.vstack([start, filler])[: self.pop_size]
            self._raw = self._score(gp, self._pop)
        else:
            self._cache = {}
            self._raw = self._score(gp, self._pop)

        elite_gene = None
        if len(y_obs):
            elite_gene = self._genotype_of(X_obs[int(np.argmax(y_obs))])[0]

        n_generations = max(1, round(len(candidates) / self.pop_size))
        for _ in range(n_generations):
            shared = self._shared(self._raw, self._pop)
            children = self._breed(self._pop, shared)
            child_raw = self._score(gp, children)
            self._pop, self._raw = self._select_survivors(
                gp,
                np.vstack([self._pop, children]),
                np.concatenate([self._raw, child_raw]),
                elite_gene,
            )

        return self._submit(observed, n_points)

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp, pool=None):
        """Same algorithm on a continuous space, with blend crossover.

        A population search has no candidate pool, so the shared pool is
        accepted and ignored: the runner calls every algorithm the same way.
        What the pool size does set here is the generation count, through the
        same round(candidates / population) rule the discrete branch uses.
        """
        lows = np.array([b[0] for b in bounds], dtype=float)
        highs = np.array([b[1] for b in bounds], dtype=float)
        span = highs - lows
        ndim = len(bounds)
        alpha = 0.5

        fingerprint = (ndim, tuple(lows), tuple(highs))
        if len(X_obs) <= self._n_obs_seen or fingerprint != self._fingerprint:
            self.rng = np.random.default_rng(self.seed)
            self._kind = "vector"
            self._blocks = None
            self._candidates = None
            self._lows, self._highs = lows, highs
            box = np.random.default_rng(0).uniform(lows, highs, size=(1000, ndim))
            self._sigma_share = self._sharing_radius(box)
            self._pop = None
            self._fingerprint = fingerprint
            self.genotype_kind = "vector"
            self.sigma_share = self._sigma_share
        self._n_obs_seen = len(X_obs)
        self._zero_counters()

        if self._pop is None:
            filler = self.rng.uniform(
                lows, highs, size=(max(0, self.pop_size - len(X_obs)), ndim)
            )
            self._pop = np.vstack([np.atleast_2d(X_obs), filler])[: self.pop_size]
        self._raw = self._score(gp, self._pop)

        elite = None
        if len(y_obs):
            elite = np.asarray(X_obs[int(np.argmax(y_obs))], dtype=float)

        # Each generation scores a population. Match the nominal search budget
        # to the size of the shared candidate pool.
        generations = self.continuous_generations
        if generations is None:
            # Use the pool size before filtering so the generation budget stays
            # fixed as observations accumulate.
            n_candidates = getattr(self, "pool_size_hint", None)
            if n_candidates is None:
                n_candidates = (len(pool) if pool is not None
                                else protocol.POOL_SIZE)
            generations = max(1, round(n_candidates / self.pop_size))
        # Recorded so a test can check the depth against the pool size that was
        # actually used, rather than against a constant that happens to match.
        self._last_generations = generations
        for _ in range(generations):
            shared = self._shared(self._raw, self._pop)
            children = np.empty_like(self._pop)
            for i in range(len(children)):
                mother = self._pop[self._tournament(shared)]
                if self.rng.random() < self.p_crossover:
                    father = self._pop[self._tournament(shared)]
                    low = np.minimum(mother, father)
                    high = np.maximum(mother, father)
                    reach = alpha * (high - low)
                    child = self.rng.uniform(low - reach, high + reach)
                else:
                    child = mother.copy()
                rate = 1.0 / ndim if self.mutation_rate is None else self.mutation_rate
                mutate = self.rng.random(ndim) < rate
                if mutate.any():
                    child[mutate] += self.rng.normal(
                        0.0, self.mutation_sigma * span[mutate], int(mutate.sum())
                    )
                children[i] = np.clip(child, lows, highs)

            child_raw = self._score(gp, children)
            self._pop, self._raw = self._select_survivors(
                gp,
                np.vstack([self._pop, children]),
                np.concatenate([self._raw, child_raw]),
                elite,
            )

        # The measured points sit in the population and the measured best is
        # put back on every generation, so without this they would be sent
        # for measurement a second time and the round would buy fewer than
        # n_points new observations.
        measured = np.atleast_2d(np.asarray(X_obs, dtype=float))

        # A population search proposes its own offspring rather than picking
        # from the shared pool, so filtering that pool cannot hold it to the
        # minimum distance from measured points. The rule has to be applied
        # where the points are proposed. The discrete branch has had the
        # equivalent all along: it skips a candidate that has been measured.
        # Measured before this check existed, a quarter of the offspring this
        # branch submitted sat inside the exclusion zone.
        gap = protocol.MIN_DISTANCE_TO_MEASURED
        # Two different rules. Reusing the distance-to-measured value for the
        # spacing inside one batch would make this the only baseline with
        # batch diversity, and UCB-BO is the arm that is supposed to have none.
        within = protocol.MIN_DISTANCE_WITHIN_BATCH

        def is_new(point):
            if not protocol.far_enough(point, measured, gap):
                return False
            return all(np.linalg.norm(point - c) >= within for c in chosen)

        shared = self._shared(self._raw, self._pop)
        chosen = []
        for i in np.argsort(shared)[::-1]:
            if len(chosen) >= n_points:
                break
            if is_new(self._pop[i]):
                chosen.append(self._pop[i])
        # A fill point is held to the same rule, or the round would buy back
        # what the check just rejected.
        attempts = 0
        while len(chosen) < n_points:
            candidate = self.rng.uniform(lows, highs)
            attempts += 1
            if attempts > 1000:
                # Accepting the point here would break the rule silently and,
                # worse, leave every later gap in this batch unchecked. The
                # exclusion zone would have to cover nearly the whole box for
                # this to happen: measured rejection rates are 18% in two
                # dimensions and 0% in ten.
                raise RuntimeError(
                    f"no point outside the exclusion zone after {attempts} "
                    f"draws, with {len(measured)} points measured. The zone "
                    f"now covers most of the box."
                )
            if is_new(candidate):
                chosen.append(candidate)
                self.n_fill += 1
        return np.array(chosen[:n_points])


def _expected_improvement(mu, sigma, y_best):
    """Compute Expected Improvement."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - y_best) / (sigma + 1e-10)
        ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-10] = 0.0
    return ei


# --------------------------------------------------------------------------
# UCB-BO: Bayesian Optimization with Upper Confidence Bound acquisition
# --------------------------------------------------------------------------
class UCBOAlgorithm:
    """Bayesian Optimization with UCB acquisition (mu + kappa*sigma).

    Uses the supplied surrogate to rank candidates by UCB. BOAlgorithm uses EI.
    """

    def __init__(self, kappa=2.0, seed=None):
        self.kappa = kappa
        self.rng = np.random.default_rng(seed)

    def _ucb(self, gp, X):
        mu, sigma = gp.predict(X, return_std=True)
        return mu + self.kappa * sigma

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Recommend from discrete candidates using UCB."""
        acq = self._ucb(gp, candidates)

        obs_set = set(map(tuple, np.round(X_obs, 10)))
        for i in range(len(candidates)):
            if tuple(np.round(candidates[i], 10)) in obs_set:
                acq[i] = -np.inf

        indices = []
        for idx in np.argsort(acq)[::-1]:
            if len(indices) >= n_points:
                break
            if idx not in indices:
                indices.append(idx)

        return indices

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp, pool=None):
        """Recommend from continuous space using UCB on random samples + refinement."""
        ndim = len(bounds)
        rng = self.rng
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])

        # The round's shared pool when the runner supplies one, so that
        # every algorithm scores the same candidates.
        samples = pool if pool is not None else rng.uniform(
            lows, highs, size=(protocol.POOL_SIZE, ndim))
        acq = self._ucb(gp, samples)

        from scipy.optimize import minimize

        top_indices = np.argsort(acq)[::-1][:10]
        best_points = []

        for idx in top_indices:
            x0 = samples[idx]

            def neg_ucb(x):
                m, s = gp.predict(x.reshape(1, -1), return_std=True)
                return -(m[0] + self.kappa * s[0])

            try:
                res = minimize(
                    neg_ucb, x0, method="L-BFGS-B",
                    bounds=[(b[0], b[1]) for b in bounds],
                )
                best_points.append(res.x if res.success else x0)
            except Exception:
                best_points.append(x0)

        selected = _refined_selection(
            best_points, samples, acq, X_obs, n_points,
            protocol.MIN_DISTANCE_TO_MEASURED, self)

        # Same rule for a fill point as for a chosen one, and counted for the
        # same reason; see BO(EI).
        self.n_fill = 0
        while len(selected) < n_points:
            candidate = rng.uniform(lows, highs)
            if protocol.far_enough(candidate, X_obs):
                selected.append(candidate)
                self.n_fill += 1

        return np.array(selected[:n_points])

    def reset(self):
        pass


# --------------------------------------------------------------------------
# Random Search
# --------------------------------------------------------------------------
class RandomAlgorithm:
    """Uniform random sampling."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Randomly select from unobserved candidates."""
        obs_set = set(map(tuple, np.round(X_obs, 10)))
        available = [
            i for i in range(len(candidates))
            if tuple(np.round(candidates[i], 10)) not in obs_set
        ]
        if len(available) <= n_points:
            return available
        chosen = self.rng.choice(available, size=n_points, replace=False)
        return chosen.tolist()

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp, pool=None):
        """Random sampling, from the shared pool when the runner supplies one.

        The lower bound has to obey the protocol too. Drawing freshly from the
        box instead put 13% of its proposals within the exclusion distance of
        a point already measured, which the discrete benchmark makes
        impossible: there a measured candidate leaves the library.
        """
        if pool is not None:
            idx = self.rng.choice(len(pool), size=min(n_points, len(pool)),
                                  replace=False)
            return pool[idx]
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        return self.rng.uniform(lows, highs, size=(n_points, len(bounds)))

    def reset(self):
        pass


# --------------------------------------------------------------------------
# Batch-aware acquisition functions
#
# Both build a batch from joint samples of the surrogate, i.e. whole predicted
# functions over the candidate set, rather than from a pointwise score. With a
# random forest each tree is one such sample, so the shared RF surrogate gives
# 100 joint samples per round and no separate model is needed. That keeps the
# "same surrogate for every algorithm" condition of the benchmark.
#
# The two differ only in how the batch is drawn from those samples:
#   Thompson sampling picks the best point of a randomly drawn sample,
#   qPO picks the points most often best across all samples.
# --------------------------------------------------------------------------


def _joint_samples(gp, X):
    """Joint surrogate samples over X, shape (n_samples, len(X))."""
    if hasattr(gp, "sample_predictions"):
        return gp.sample_predictions(X)
    raise TypeError(
        "Thompson sampling and qPO need joint posterior samples; the surrogate "
        "must provide sample_predictions(X)."
    )


def _observed_mask(candidates, X_obs):
    obs_set = set(map(tuple, np.round(X_obs, 10)))
    return np.array([tuple(np.round(c, 10)) in obs_set for c in candidates])


class ThompsonSamplingAlgorithm:
    """Parallel Thompson sampling (Hernandez-Lobato et al., ICML 2017).

    For each slot of the batch, draw one function from the surrogate posterior
    and take its best candidate. Points already observed, and points already
    chosen for this batch, are skipped. Diversity is not imposed; it arises
    only because different draws disagree about where the optimum is.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        samples = _joint_samples(gp, candidates)
        samples = samples.copy()
        samples[:, _observed_mask(candidates, X_obs)] = -np.inf

        n_draws = samples.shape[0]
        indices = []
        for _ in range(n_points):
            row = samples[int(self.rng.integers(n_draws))].copy()
            if indices:
                row[indices] = -np.inf
            idx = int(np.argmax(row))
            if not np.isfinite(row[idx]):
                break
            indices.append(idx)
        return indices

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp,
                             pool=None, *, n_random=None):
        if pool is None:
            lows = np.array([b[0] for b in bounds])
            highs = np.array([b[1] for b in bounds])
            pool = self.rng.uniform(
                lows, highs, size=(n_random or protocol.POOL_SIZE, len(bounds)))
        picks = self.recommend_discrete(X_obs, y_obs, pool, None, n_points, gp)
        return pool[picks]

    def reset(self):
        pass


class QPOAlgorithm:
    """qPO, multipoint probability of optimality (Fromer et al., JCIM 2025).

    Score each candidate by how often it is the best point across the joint
    samples, which estimates the probability that it is the true optimum, then
    take the top q. Because the probability that the batch contains the optimum
    is the sum of these scores, the best batch is simply the top-scoring
    candidates and no combinatorial search is needed. Candidates that are never
    best all score zero and are ranked among themselves by predicted mean, as
    in the original work.

    Reference implementation: https://github.com/jenna-fromer/qPO (Gaussian
    process only). Here the scores come from the shared random forest, so the
    estimate uses 100 samples where the original used 10,000 draws from a GP
    posterior. The score of a rarely-optimal candidate is therefore coarse.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _scores(self, candidates, X_obs, gp):
        samples = _joint_samples(gp, candidates)
        samples = samples.copy()
        observed = _observed_mask(candidates, X_obs)
        samples[:, observed] = -np.inf

        best_per_sample = samples.argmax(axis=1)
        alpha = np.bincount(best_per_sample,
                            minlength=len(candidates)).astype(float)
        alpha /= samples.shape[0]
        mean = samples.mean(axis=0)

        alpha[observed] = -1.0          # never re-acquire an observed point
        mean[observed] = -np.inf
        return alpha, mean

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        alpha, mean = self._scores(candidates, X_obs, gp)
        # Primary key is the probability of being optimal; ties fall back to
        # the predicted mean. np.lexsort takes the last key as primary.
        order = np.lexsort((-mean, -alpha))
        return [int(i) for i in order[:n_points] if alpha[i] >= 0.0]

    def recommend_continuous(self, X_obs, y_obs, bounds, n_points, gp,
                             pool=None, *, n_random=None):
        if pool is None:
            lows = np.array([b[0] for b in bounds])
            highs = np.array([b[1] for b in bounds])
            pool = self.rng.uniform(
                lows, highs, size=(n_random or protocol.POOL_SIZE, len(bounds)))
        picks = self.recommend_discrete(X_obs, y_obs, pool, None, n_points, gp)
        return pool[picks]

    def reset(self):
        pass
