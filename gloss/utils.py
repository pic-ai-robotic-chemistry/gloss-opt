import numpy as np
from scipy.spatial.distance import cdist


def space_diagonal(bounds):
    """Compute the diagonal length of the bounding box."""
    bounds = np.array(bounds)
    ranges = bounds[:, 1] - bounds[:, 0]
    return np.linalg.norm(ranges)


def compute_min_distances(points, reference):
    """Compute the minimum distance from each point to the reference set."""
    if reference.shape[0] == 0:
        return np.full(points.shape[0], np.inf)
    dists = cdist(points, reference, metric="euclidean")
    return dists.min(axis=1)


def is_rejected(point, excluded, n_measured=None, tolerance=0.0,
                batch_tolerance=None):
    """Reject a candidate too close to a measured point or a batch mate."""
    if batch_tolerance is None or batch_tolerance == tolerance:
        return is_duplicate(point, excluded, tolerance)
    # If n_measured is None, treat every row as measured.
    n = excluded.shape[0] if n_measured is None else n_measured
    return (is_duplicate(point, excluded[:n], tolerance)
            or is_duplicate(point, excluded[n:], batch_tolerance))


def is_batch_rejected(point, batch_points, tolerance=0.0, batch_tolerance=None):
    """Reject a candidate too close to another point in the same batch."""
    if batch_tolerance is None or batch_tolerance == tolerance:
        return False
    if len(batch_points) == 0:
        return False
    return is_duplicate(point, np.asarray(batch_points, dtype=float),
                        batch_tolerance)


def is_duplicate(point, excluded, tolerance=0.0):
    """Check if point is a duplicate of any point in excluded set."""
    if excluded.shape[0] == 0:
        return False
    point = point.reshape(1, -1)
    dists = cdist(point, excluded, metric="euclidean")[0]
    return bool(np.any(dists <= tolerance))
