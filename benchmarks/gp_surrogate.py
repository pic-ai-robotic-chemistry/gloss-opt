
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

GP_ALPHA = 1e-2
LENGTH_SCALE_BOUNDS = (1e-2, 1e3)
CONSTANT_BOUNDS = (1e-3, 1e3)


class GPSurrogate:
    """GP surrogate compatible with the GLOSS/BO/GA surrogate interface."""

    #: Which family this is, for ``benchmarks.protocol.require()``.
    #:
    #: The continuous protocol treats local refinement as a no-op, which holds
    #: for the random forest and not for a smooth surrogate. Measured on this
    #: machine (``reproduction.studies.gp_refinement``, 180 probes per family): the
    #: random forest never moves a point, the Gaussian process moves in 92 of
    #: 180 probes with a median of 0.0007 and a maximum of 1.18. A run under
    #: this family must therefore pass measured refinement evidence.
    SURROGATE_FAMILY = "gaussian_process"

    def __init__(self, alpha=GP_ALPHA, random_state=None):
        # random_state is accepted for interface parity and deliberately not
        # used: the fit is deterministic given the data, and there is no
        # sampling in this class.
        self.alpha = alpha
        self._random_state = random_state
        self._gp = None
        self._n_features = None

    def _make_kernel(self, n_features):
        return ConstantKernel(1.0, constant_value_bounds=CONSTANT_BOUNDS) * Matern(
            nu=2.5,
            length_scale=np.ones(n_features),
            length_scale_bounds=LENGTH_SCALE_BOUNDS,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"fit expects a 2-D X, got shape {X.shape}")
        self._n_features = X.shape[1]
        self._gp = GaussianProcessRegressor(
            kernel=self._make_kernel(self._n_features),
            alpha=self.alpha,
            normalize_y=True,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            random_state=0,
        )
        self._gp.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        if self._gp is None:
            raise RuntimeError("predict called before fit")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not return_std:
            return self._gp.predict(X, return_std=False)
        mean, std = self._gp.predict(X, return_std=True)
        # Floor, for the same reason RFSurrogate floors: UCB multiplies sigma
        # by kappa and a numerically zero sigma would turn the acquisition into
        # plain greedy selection. The floor is on the scale of a fitted GP's
        # sigma, not a constant in y units.
        return mean, np.maximum(std, 1e-12)

    # -- diagnostics used by the runner and by the analysis ---------------

    def sigma_stats(self, X):
        """Posterior std statistics over X, for the identifiability gate.

        Returns a dict with the share of candidates sitting on the sigma
        plateau (within 1e-6 of the batch maximum), the coefficient of
        variation, and the threshold that a median-quantile gate would use.

        The plateau share is what decides whether sigma can rank candidates at
        all. It is measured on the candidate set scored by the arm.
        """
        std = self.predict(X, return_std=True)[1]
        top = float(np.max(std))
        return {
            "sigma_platform_share": float(np.mean(std >= top - 1e-6)),
            "sigma_cv": float(np.std(std) / np.mean(std)) if std.size else 0.0,
            "sigma_max": top,
        }
