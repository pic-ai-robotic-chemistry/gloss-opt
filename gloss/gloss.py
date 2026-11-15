import numpy as np
from gloss.space import SearchSpace
from gloss.surrogate.auto_select import auto_select_surrogate
from gloss.strategies.global_best import find_global_best
from gloss.strategies.local_best import find_local_best
from gloss.strategies.unexplored import find_unexplored
from gloss.strategies.unconverged import find_unconverged


class GLOSS:
    """Global-Local Optimization Surrogate Strategy."""

    def __init__(self, space, mode="continuous", direction="minimize",
                 window_radius=None, unexplored_threshold=None,
                 n_random_samples=10000, n_top_for_optimization=10,
                 cv_folds=5, scoring="neg_root_mean_squared_error",
                 duplicate_tolerance=None, batch_tolerance=None,
                 custom_sampler=None,
                 ucb_kappa=2.0,
                 kappa_schedule="fixed",   # 'fixed', 'decay', 'cosine'
                 kappa_min=0.5,
                 kappa_decay=0.9,
                 n_rounds=20,
                 adaptive_strategy=False,
                 default_batch_size=8,
                 distance_metric="euclidean",    # passed to local_best and unexplored
                 diversity_radius=0.0,           # passed to global_best
                 diversity_metric="euclidean",   # passed to global_best
                 novelty_signal="distance",      # 'distance' or 'sigma', see strategies/unexplored
                 novelty_quantile=None,          # None uses the absolute threshold
                 local_top_k=None,               # O(K) truncation for local_best (None=default 500; 0=full O(n); int=override)
                 seed=None):                     # RNG seed for reproducibility
        if direction not in ("minimize", "maximize"):
            raise ValueError(f"direction must be 'minimize' or 'maximize', got '{direction}'")

        self.space = SearchSpace(
            mode=mode,
            bounds=space.get("bounds"),
            candidates=space.get("candidates"),
            param_grid=space.get("param_grid"),
            constraints=space.get("constraints"),
            custom_sampler=custom_sampler,
            seed=seed,
        )
        self.mode = mode
        self.direction = direction
        self.window_radius = window_radius
        self.unexplored_threshold = unexplored_threshold
        self.n_random_samples = n_random_samples
        self.n_top_for_optimization = n_top_for_optimization
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.ucb_kappa = ucb_kappa
        self.kappa_schedule = kappa_schedule
        self.kappa_min = kappa_min
        self.kappa_decay = kappa_decay
        self.n_rounds = n_rounds
        self._round = 0
        self.adaptive_strategy = adaptive_strategy
        self.default_batch_size = default_batch_size
        self.distance_metric = distance_metric
        self.local_top_k = local_top_k
        self.diversity_radius = diversity_radius
        self.diversity_metric = diversity_metric
        # Rank gate for the unexplored strategy. Defaults use an absolute distance threshold.
        if novelty_signal not in ("distance", "sigma"):
            raise ValueError(
                f"novelty_signal must be 'distance' or 'sigma', got {novelty_signal!r}"
            )
        if novelty_quantile is not None and not (0.0 < float(novelty_quantile) <= 1.0):
            raise ValueError(
                "novelty_quantile must be in (0, 1] or None, got "
                f"{novelty_quantile!r}"
            )
        self.novelty_signal = novelty_signal
        self.novelty_quantile = novelty_quantile
        #: Set by the unexplored strategy on each recommend() call.
        self.last_novelty_stats = None
        if adaptive_strategy:
            self._strategy_wins = {"global_best": 0, "local_best": 0,
                                   "unexplored": 0, "unconverged": 0}
            # Laplace smoothing: initial tries = 1 avoids division-by-zero in UCB-1.
            self._strategy_tries = {"global_best": 1, "local_best": 1,
                                    "unexplored": 1, "unconverged": 1}
        else:
            self._strategy_wins = None
            self._strategy_tries = None

        # duplicate_tolerance keeps a new point away from measured points;
        # batch_tolerance keeps points of one batch apart. None uses one
        # distance for both.
        self.batch_tolerance = batch_tolerance
        # In discrete mode the candidate filter already removes rows within
        # duplicate_tolerance, so a smaller batch_tolerance has no effect.
        if (mode == "discrete" and batch_tolerance is not None
                and duplicate_tolerance is not None
                and batch_tolerance < duplicate_tolerance):
            raise ValueError(
                f"batch_tolerance={batch_tolerance} is below "
                f"duplicate_tolerance={duplicate_tolerance} in discrete mode, "
                f"where the candidate filter has already removed those rows. "
                f"A batch distance below the measured distance has no effect "
                f"there, so this combination would be silently ignored.")
        if duplicate_tolerance is not None:
            self.duplicate_tolerance = duplicate_tolerance
        elif mode == "discrete":
            self.duplicate_tolerance = 0.0
        else:
            # In continuous mode, keep new points at least diversity_radius (or
            # 0.1% of the space diagonal) from observed points.
            self.duplicate_tolerance = max(
                diversity_radius, self.space.diagonal * 0.001
            )

    def reset(self):
        """Clear internal state."""
        self._round = 0
        if self.adaptive_strategy:
            self._strategy_wins = {"global_best": 0, "local_best": 0,
                                   "unexplored": 0, "unconverged": 0}
            self._strategy_tries = {"global_best": 1, "local_best": 1,
                                    "unexplored": 1, "unconverged": 1}

    def _get_kappa(self, round_num):
        """Return the acquisition weight kappa for the current round."""
        import math
        if self.kappa_schedule == "fixed":
            return self.ucb_kappa
        elif self.kappa_schedule == "decay":
            return max(self.kappa_min,
                       self.ucb_kappa * (self.kappa_decay ** round_num))
        elif self.kappa_schedule == "cosine":
            cos_val = math.cos(math.pi * round_num / max(self.n_rounds, 1))
            return self.kappa_min + 0.5 * (self.ucb_kappa - self.kappa_min) * (1 + cos_val)
        else:
            raise ValueError(
                f"Unknown kappa_schedule: {self.kappa_schedule!r}. "
                "Expected 'fixed', 'decay', or 'cosine'."
            )

    def feedback(self, results_with_y, current_best_before):
        """Update the bandit allocation after evaluating a batch."""
        if not self.adaptive_strategy:
            return

        # Best y in this batch
        if self.direction == "maximize":
            best_y = max((r.get("y_actual", -np.inf) for r in results_with_y),
                         default=-np.inf)
            improved = best_y > current_best_before
        else:
            best_y = min((r.get("y_actual", np.inf) for r in results_with_y),
                         default=np.inf)
            improved = best_y < current_best_before

        # Best y per strategy
        strategy_best = {}
        for r in results_with_y:
            s = r.get("strategy")
            y = r.get("y_actual", -np.inf)
            if s and (s not in strategy_best or y > strategy_best[s]):
                strategy_best[s] = y

        for s in self._strategy_tries:
            if strategy_best.get(s, -np.inf) > -np.inf:
                self._strategy_tries[s] += 1
            if self.direction == "maximize":
                if improved and strategy_best.get(s, -np.inf) >= best_y:
                    self._strategy_wins[s] += 1
            else:
                if improved and strategy_best.get(s, np.inf) <= best_y:
                    self._strategy_wins[s] += 1

    def _bandit_allocation(self, total_points, base_strategy_points=None):
        """Compute the bandit allocation summing to total_points."""
        import math
        if base_strategy_points is None:
            base_strategy_points = {
                "global_best": 6, "local_best": 1, "unexplored": 1, "unconverged": 0
            }

        total_rounds = max(sum(self._strategy_tries.values()), 1)
        scores = {}
        for s in base_strategy_points:
            if base_strategy_points[s] == 0:
                scores[s] = 0.0  # respect disabled strategies
                continue
            wins = self._strategy_wins[s]
            tries = max(self._strategy_tries[s], 1)
            scores[s] = (wins / tries) + math.sqrt(2 * math.log(total_rounds) / tries)

        # Softmax over active strategies → proportional allocation
        active = {s: v for s, v in scores.items() if v > 0}
        if not active:
            return {s: max(1, base_strategy_points.get(s, 0)) if base_strategy_points.get(s, 0) > 0 else 0
                    for s in base_strategy_points}

        max_s = max(active.values())
        exp_scores = {s: np.exp(v - max_s) for s, v in active.items()}
        total_exp = sum(exp_scores.values())
        proportions = {s: v / total_exp for s, v in exp_scores.items()}

        raw = {s: proportions[s] * total_points for s in active}
        allocation = {s: max(1, int(v)) for s, v in raw.items()}
        for s in base_strategy_points:
            if s not in allocation:
                allocation[s] = 0

        # Adjust total to match total_points (give/take from global_best)
        current_total = sum(allocation.values())
        diff = total_points - current_total
        if diff != 0:
            allocation["global_best"] = max(1, allocation.get("global_best", 1) + diff)

        return allocation

    def recommend(self, X_train=None, y_train=None, surrogate=None,
                  X_existing=None, strategy_points=None):
        """Recommend sample points using four strategies."""
        if surrogate is None and (X_train is None or y_train is None):
            raise ValueError(
                "Must provide either 'surrogate' or both 'X_train' and 'y_train'."
            )
        if X_train is not None and y_train is not None:
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(
                    f"X_train and y_train must have same length, "
                    f"got {X_train.shape[0]} and {y_train.shape[0]}."
                )

        if strategy_points is None:
            if self.adaptive_strategy and self._strategy_wins is not None:
                strategy_points = self._bandit_allocation(
                    total_points=self.default_batch_size,
                    base_strategy_points={
                        "global_best": 6, "local_best": 1, "unexplored": 1, "unconverged": 0
                    }
                )
            else:
                strategy_points = {
                    "global_best": 1, "local_best": 1, "unexplored": 1, "unconverged": 1
                }

        if surrogate is not None:
            current_surrogate = surrogate
        else:
            current_surrogate = auto_select_surrogate(
                X_train, y_train, cv_folds=self.cv_folds, scoring=self.scoring
            )

        if X_existing is not None:
            excluded = np.asarray(X_existing, dtype=float)
        else:
            excluded = np.empty((0, self.space.ndim))

        tol = self.duplicate_tolerance
        # Everything in excluded up to this row is a measured point. The rows
        # the strategies add below are batch mates and get batch_tolerance.
        n_measured = excluded.shape[0]
        batch_tol = self.batch_tolerance
        results = []

        current_kappa = self._get_kappa(self._round)

        # Strategy 1: Global Best
        n_gb = strategy_points.get("global_best", 0)
        if n_gb > 0:
            gb_results = find_global_best(
                surrogate=current_surrogate,
                space=self.space,
                n_points=n_gb,
                excluded=excluded,
                direction=self.direction,
                tolerance=tol,
                n_random_samples=self.n_random_samples,
                n_top=self.n_top_for_optimization,
                kappa=current_kappa,
                diversity_radius=self.diversity_radius,
                diversity_metric=self.diversity_metric,
                n_measured=n_measured,
                batch_tolerance=batch_tol,
            )
            results.extend(gb_results)
            for r in gb_results:
                if "point" in r:
                    excluded = np.vstack([excluded, np.array(r["point"]).reshape(1, -1)])

        # Strategy 2: Local Best
        n_lb = strategy_points.get("local_best", 0)
        if n_lb > 0:
            lb_results = find_local_best(
                surrogate=current_surrogate,
                space=self.space,
                n_points=n_lb,
                excluded=excluded,
                direction=self.direction,
                tolerance=tol,
                window_radius=self.window_radius,
                n_random_samples=self.n_random_samples,
                distance_metric=self.distance_metric,
                top_k=self.local_top_k,
                n_measured=n_measured,
                batch_tolerance=batch_tol,
            )
            results.extend(lb_results)
            for r in lb_results:
                if "point" in r:
                    excluded = np.vstack([excluded, np.array(r["point"]).reshape(1, -1)])

        # Strategy 3: Unexplored Sampling
        n_ue = strategy_points.get("unexplored", 0)
        if n_ue > 0:
            explored = excluded.copy()
            novelty_stats = {}
            ue_results = find_unexplored(
                surrogate=current_surrogate,
                space=self.space,
                n_points=n_ue,
                explored=explored,
                excluded=excluded,
                direction=self.direction,
                tolerance=tol,
                unexplored_threshold=self.unexplored_threshold,
                n_random_samples=self.n_random_samples,
                distance_metric=self.distance_metric,
                n_measured=n_measured,
                batch_tolerance=batch_tol,
                novelty_signal=self.novelty_signal,
                novelty_quantile=self.novelty_quantile,
                stats_out=novelty_stats,
            )
            self.last_novelty_stats = novelty_stats or None
            results.extend(ue_results)
            for r in ue_results:
                if "point" in r:
                    excluded = np.vstack([excluded, np.array(r["point"]).reshape(1, -1)])

        # Strategy 4: Unconverged Regions
        n_uc = strategy_points.get("unconverged", 0)
        if n_uc > 0:
            uc_results = find_unconverged(
                surrogate=current_surrogate,
                space=self.space,
                n_points=n_uc,
                excluded=excluded,
                X_obs=X_train if X_train is not None else np.empty((0, self.space.ndim)),
                tolerance=tol,
                n_random_samples=self.n_random_samples,
                n_measured=n_measured,
                batch_tolerance=batch_tol,
            )
            results.extend(uc_results)
            for r in uc_results:
                if "point" in r:
                    excluded = np.vstack([excluded, np.array(r["point"]).reshape(1, -1)])

        self._round += 1
        return results
