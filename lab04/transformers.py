
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        self.lower_bounds_ = np.nanquantile(
            X, self.lower, axis=0
        )
        self.upper_bounds_ = np.nanquantile(
            X, self.upper, axis=0
        )

        return self

    def transform(self, X):
        check_is_fitted(
            self, ["lower_bounds_", "upper_bounds_"]
        )

        X = np.asarray(X, dtype=float)

        return np.clip(
            X,
            self.lower_bounds_,
            self.upper_bounds_
        )
