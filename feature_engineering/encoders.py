"""Custom encoders for categorical features."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
