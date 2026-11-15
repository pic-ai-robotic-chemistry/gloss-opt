import warnings
import numpy as np
from gloss.utils import compute_min_distances, is_rejected


def find_unexplored(surrogate, space, n_points, explored, excluded, direction,
                    tolerance=0.0, unexplored_threshold=None, n_random_samples=10000,
                    distance_metric="euclidean",
                    n_measured=None, batch_tolerance=None,
                    novelty_signal="distance", novelty_quantile=None,
                    stats_out=None):
    """Select points in unexplored regions with the best predicted values."""
    sign = 1.0 if direction == "maximize" else -1.0

    if space.mode == "discrete":
        candidates = space.get_candidates_excluding(excluded, tolerance)
    else:
        candidates = space.sample(n_random_samples)

    if len(candidates) == 0:
        warnings.warn("unexplored: no candidates available.")
        return []

    if novelty_quantile is None:
        # Distances to the full explored set (used by the threshold path).
        if distance_metric == "jaccard":
            from sklearn.neighbors import BallTree
            if len(explored) == 0:
                # No observed points → treat everything as unexplored
                min_dists = np.full(len(candidates), np.inf)
            else:
                _tree = BallTree(explored.astype(float), metric="jaccard")
                min_dists, _ = _tree.query(candidates.astype(float), k=1)
                min_dists = min_dists.ravel()
        else:
            min_dists = compute_min_distances(candidates, explored)

    all_excluded = excluded.copy() if excluded.shape[0] > 0 else np.empty((0, space.ndim))
    n_meas = excluded.shape[0] if n_measured is None else n_measured

    if novelty_quantile is not None:
        # Rank gate: keep the most novel candidates, break ties by prediction.
        ref = explored[:n_meas] if 0 < n_meas <= explored.shape[0] else explored
        if novelty_signal == "sigma":
            try:
                mu, sigma = surrogate.predict(candidates, return_std=True)
            except TypeError as exc:
                if "return_std" not in str(exc):
                    raise
                raise TypeError(
                    "novelty_signal='sigma' needs a surrogate with "
                    "predict(..., return_std=True)"
                ) from exc
            sigma = np.asarray(sigma, dtype=float)
            signal = sigma
        elif novelty_signal == "distance":
            if distance_metric == "jaccard":
                from sklearn.neighbors import BallTree
                if len(ref) == 0:
                    signal = np.full(len(candidates), np.inf)
                else:
                    _tree = BallTree(ref.astype(float), metric="jaccard")
                    signal = _tree.query(candidates.astype(float), k=1)[0].ravel()
            else:
                signal = compute_min_distances(candidates, ref)
            mu = np.asarray(surrogate.predict(candidates), dtype=float)
        else:
            raise ValueError(
                f"novelty_signal must be 'distance' or 'sigma', got {novelty_signal!r}"
            )

        # Keep exactly ceil(q * n) candidates.
        n_keep = min(int(np.ceil(float(novelty_quantile) * len(candidates))),
                     len(candidates))
        # -signal sorts most novel first; -sign*mu breaks ties toward the best prediction.
        keep = np.lexsort((-sign * mu, -signal))[:n_keep]
        far_candidates = candidates[keep]
        preds = mu[keep]

        if stats_out is not None:
            span = float(np.max(signal) - np.min(signal))
            stats_out["novelty_pass_frac"] = n_keep / len(candidates)
            stats_out["all_tied"] = bool(span <= 0.0)
            if novelty_signal == "sigma":
                top = float(np.max(sigma))
                stats_out["sigma_platform_share"] = float(np.mean(sigma >= top - 1e-6))
                stats_out["sigma_cv"] = (
                    float(np.std(sigma) / np.mean(sigma)) if sigma.size else 0.0
                )
    else:
        if unexplored_threshold is None:
            if explored.shape[0] >= 2:
                from scipy.spatial.distance import pdist
                avg_dist = np.mean(pdist(explored))
                unexplored_threshold = avg_dist * 0.5
            else:
                unexplored_threshold = space.diagonal * 0.1

        far_mask = min_dists > unexplored_threshold
        far_candidates = candidates[far_mask]

        if len(far_candidates) == 0:
            warnings.warn(
                f"unexplored: no candidates above threshold {unexplored_threshold:.4f}. "
                "Returning best among all remaining candidates."
            )
            far_candidates = candidates

        preds = surrogate.predict(far_candidates)

    order = np.argsort(sign * preds)[::-1]
    results = []
    for idx in order:
        if len(results) >= n_points:
            break
        point = far_candidates[idx]
        if is_rejected(point, all_excluded, n_meas, tolerance, batch_tolerance):
            continue
        results.append({
            "point": point.tolist(),
            "strategy": "unexplored",
            "predicted_value": float(preds[idx]),
        })
        all_excluded = np.vstack([all_excluded, point.reshape(1, -1)])

    if len(results) < n_points:
        warnings.warn(f"unexplored: only found {len(results)}/{n_points} points.")
    return results
