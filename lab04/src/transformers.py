
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PercentileClipper(BaseEstimator, TransformerMixin):
    """
    Clips numerical values using percentile limits
    calculated from the training dataset.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        """
        Calculate percentile limits using training data only.
        """

        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError(
                "Percentile limits must satisfy "
                "0 <= lower < upper <= 1."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Input data must be two-dimensional.")

        self.n_features_in_ = X.shape[1]

        self.lower_bounds_ = np.percentile(
            X,
            self.lower * 100,
            axis=0
        )

        self.upper_bounds_ = np.percentile(
            X,
            self.upper * 100,
            axis=0
        )

        return self

    def transform(self, X):
        """
        Clip numerical values using the limits
        learned during fitting.
        """

        check_is_fitted(
            self,
            ["lower_bounds_", "upper_bounds_"]
        )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Input must have the same number of features "
                "as the training data."
            )

        X_clipped = np.clip(
            X,
            self.lower_bounds_,
            self.upper_bounds_
        )

        return X_clipped

    def get_feature_names_out(self, input_features=None):
        """
        Preserve numerical feature names after clipping.
        Required for Task 7.
        """

        check_is_fitted(
            self,
            ["lower_bounds_", "upper_bounds_"]
        )

        if input_features is None:
            return np.asarray(
                [
                    f"x{i}"
                    for i in range(self.n_features_in_)
                ],
                dtype=object
            )

        if len(input_features) != self.n_features_in_:
            raise ValueError(
                "Number of input feature names does not match "
                "the fitted data."
            )

        return np.asarray(input_features, dtype=object)
