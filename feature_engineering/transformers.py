"""Generic numeric transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to numeric columns to reduce skew."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        # log1p of absolute value, preserve sign
        return np.sign(X_arr) * np.log1p(np.abs(X_arr))

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([f"{f}_log1p" for f in input_features])
        return input_features
