"""Random Forest surrogate with ensemble uncertainty estimation.

predict(X, return_std=True) returns (mean, std) where std is computed
as the standard deviation of individual tree predictions (ensemble variance).
This is a well-established approach for uncertainty quantification with RF.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RFSurrogate:
    """RF surrogate compatible with GLOSS/BO/GA surrogate interface."""

    #: Which family this is, for benchmarks.protocol.require().
    #:
    #: The continuous protocol leaves local refinement switched on and treats
    #: it as a no-op, which holds for a piecewise-constant prediction and not
    #: for a smooth one. Naming the family here is what lets the protocol
    #: refuse a surrogate it was not measured under.
    SURROGATE_FAMILY = "random_forest"

    def __init__(self, n_estimators=100, random_state=None, n_jobs=1):
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            min_samples_leaf=1,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._rf.fit(X, y)
        return self

    def sample_predictions(self, X):
        """Per-tree predictions, shape (n_estimators, n_samples).

        Each row is one tree's prediction over the whole of X, that is one
        joint sample of the surrogate across all candidates. Thompson sampling
        and qPO both need joint samples rather than per-point marginals: which
        candidate looks best depends on how the predictions co-vary, and that
        information is lost if every candidate is sampled independently.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([tree.predict(X) for tree in self._rf.estimators_])

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        tree_preds = self.sample_predictions(X)
        mean = tree_preds.mean(axis=0)
        if return_std:
            std = tree_preds.std(axis=0)
            # Floor std to prevent zero uncertainty in UCB acquisition (mu + kappa*sigma)
            # when all trees agree on a prediction
            std = np.maximum(std, 1e-6)
            return mean, std
        return mean


def make_rf(seed=None):
    """Create a fresh unfitted RFSurrogate."""
    return RFSurrogate(n_estimators=100, random_state=seed, n_jobs=1)
