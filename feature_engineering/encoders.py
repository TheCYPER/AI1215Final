"""Custom encoders for categorical features."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Mean target encoding with Bayesian smoothing.

    For each category c of each column, the encoded value is
        (count(c) * mean_y(c) + smoothing * global_mean_y) / (count(c) + smoothing)
    which pulls rare categories toward the global mean to avoid overfitting
    on low-support cats.

    Works on integer-like targets (e.g., ordinal RiskTier) and continuous
    targets (e.g., InterestRate) alike.

    Leakage note: fitted on the training fold's y inside CrossValidator,
    which refits preprocessor per fold, so cross-fold leakage is zero.
    Within-fold leakage exists but is bounded by the smoothing term.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.cat_means_ = {}
        self.global_mean_ = 0.0

    def fit(self, X, y):
        X = self._to_df(X)
        y = pd.Series(np.asarray(y))
        self.global_mean_ = float(y.mean())
        self.cat_means_ = {}
        for col in X.columns:
            cat = X[col].astype(str)
            grp = y.groupby(cat)
            count = grp.count()
            mean = grp.mean()
            smoothed = (
                count * mean + self.smoothing * self.global_mean_
            ) / (count + self.smoothing)
            self.cat_means_[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X = self._to_df(X)
        result = np.zeros((len(X), len(X.columns)), dtype=np.float64)
        for i, col in enumerate(X.columns):
            mapping = self.cat_means_.get(col, {})
            result[:, i] = (
                X[col].astype(str).map(mapping).fillna(self.global_mean_).values
            )
        return result

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([f"{f}_tgt" for f in input_features])
        return np.array([f"{f}_tgt" for f in self.cat_means_])

    @staticmethod
    def _to_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features by their frequency (proportion) in the training set."""

    def __init__(self):
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        X = self._to_df(X)
        for col in X.columns:
            counts = X[col].value_counts(normalize=True)
            self.freq_maps_[col] = counts.to_dict()
        return self

    def transform(self, X):
        X = self._to_df(X)
        result = pd.DataFrame(index=X.index)
        for col in X.columns:
            freq_map = self.freq_maps_.get(col, {})
            result[col] = X[col].map(freq_map).fillna(0.0)
        return result.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([f"{f}_freq" for f in input_features])
        return np.array([f"{f}_freq" for f in self.freq_maps_])

    @staticmethod
    def _to_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)
