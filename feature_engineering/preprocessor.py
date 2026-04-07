"""
Build the preprocessing pipeline as a ColumnTransformer.

Each transformer group is independent and can be swapped or extended
without touching the rest of the pipeline.
"""

from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering.credit_features import CreditFeatureBuilder
from feature_engineering.encoders import FrequencyEncoder
from feature_engineering.transformers import Log1pTransformer


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    freq_encoding_cols: List[str],
    log_transform_cols: List[str],
    enable_credit_features: bool = True,
) -> ColumnTransformer:
    """
    Compose all preprocessing into a single ColumnTransformer.

    Args:
        numeric_cols: Numeric feature columns.
        categorical_cols: Categorical feature columns (excl. freq-encoded).
        freq_encoding_cols: High-cardinality cols for frequency encoding.
        log_transform_cols: Columns to log-transform before scaling.
        enable_credit_features: Whether to add domain credit features.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    # Separate log-transform cols from regular numeric cols
    log_cols = [c for c in log_transform_cols if c in numeric_cols]
    regular_num_cols = [c for c in numeric_cols if c not in log_cols]

    # Remove freq-encoded cols from categorical pipeline
    onehot_cat_cols = [c for c in categorical_cols if c not in freq_encoding_cols]
    freq_cols = [c for c in freq_encoding_cols if c in categorical_cols]

    transformers = []

    # 1. Regular numeric: impute -> scale
    if regular_num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, regular_num_cols))

    # 2. Log-transformed numeric: impute -> log1p -> scale
    if log_cols:
        log_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", Log1pTransformer()),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("log_num", log_pipe, log_cols))

    # 3. Categorical: impute -> one-hot
    if onehot_cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, onehot_cat_cols))

    # 4. Frequency encoding for high-cardinality categoricals
    if freq_cols:
        freq_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("freq", FrequencyEncoder()),
        ])
        transformers.append(("freq", freq_pipe, freq_cols))

    # 5. Domain credit features (operates on all numeric input)
    if enable_credit_features:
        credit_pipe = Pipeline([
            ("builder", CreditFeatureBuilder()),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ])
        # Pass all columns — CreditFeatureBuilder picks what it needs via .get()
        all_cols = regular_num_cols + log_cols
        if all_cols:
            transformers.append(("credit", credit_pipe, all_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        n_jobs=-1,
    )
