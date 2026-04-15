"""
Build the preprocessing pipeline as a ColumnTransformer.

Each transformer group is independent and can be swapped or extended
without touching the rest of the pipeline.

F4 change (2026-04-15): categoricals are ordinal-encoded (one integer per
category) and surfaced via `categorical_feature_indices_` so LightGBM can
do multi-way native splits on them, instead of one-hot expanding into ~80
binary columns. A subset additionally gets smoothed target encoding as
extra floating-point features.
"""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from feature_engineering.credit_features import CreditFeatureBuilder
from feature_engineering.encoders import FrequencyEncoder, SmoothedTargetEncoder
from feature_engineering.transformers import Log1pTransformer


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    freq_encoding_cols: List[str],
    log_transform_cols: List[str],
    enable_credit_features: bool = True,
    target_encoding_cols: Optional[List[str]] = None,
    native_categorical: bool = True,
) -> ColumnTransformer:
    """Compose all preprocessing into a single ColumnTransformer.

    Returns a ColumnTransformer. After it's fit, use
    `get_categorical_feature_indices(preprocessor)` to retrieve the output
    column indices that LightGBM should treat as categorical.
    """
    target_encoding_cols = target_encoding_cols or []

    log_cols = [c for c in log_transform_cols if c in numeric_cols]
    regular_num_cols = [c for c in numeric_cols if c not in log_cols]

    freq_cols = [c for c in freq_encoding_cols if c in categorical_cols]
    # Cats left for the main cat path (excludes freq-encoded).
    main_cat_cols = [c for c in categorical_cols if c not in freq_cols]
    # Target-encoded cols are a subset of categorical_cols; they may ALSO
    # appear in main_cat_cols (native encoding in parallel).
    tgt_cols = [c for c in target_encoding_cols if c in categorical_cols]

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

    # 3. Main categoricals: either one-hot (legacy) or ordinal (LGBM-native).
    if main_cat_cols:
        if native_categorical:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("ordinal", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ])
            transformers.append(("cat_native", cat_pipe, main_cat_cols))
        else:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat_ohe", cat_pipe, main_cat_cols))

    # 4. High-cardinality frequency encoding (unchanged path).
    if freq_cols:
        freq_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("freq", FrequencyEncoder()),
        ])
        transformers.append(("cat_freq", freq_pipe, freq_cols))

    # 5. Target encoding for mid-cardinality cats (additive to native path).
    if tgt_cols:
        tgt_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("target", SmoothedTargetEncoder(smoothing=10.0)),
        ])
        transformers.append(("cat_target", tgt_pipe, tgt_cols))

    # 6. Domain credit features.
    if enable_credit_features:
        credit_pipe = Pipeline([
            ("builder", CreditFeatureBuilder()),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ])
        all_cols = regular_num_cols + log_cols
        if all_cols:
            transformers.append(("credit", credit_pipe, all_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        n_jobs=-1,
    )


def get_categorical_feature_indices(preprocessor: ColumnTransformer) -> List[int]:
    """Return the list of output-column indices that LightGBM should treat
    as categorical. Must be called AFTER `preprocessor.fit(...)`.

    Only the `cat_native` branch is flagged as categorical. Frequency and
    target encodings are numeric floats.
    """
    if not hasattr(preprocessor, "output_indices_"):
        # sklearn < 1.2 fallback: recompute from named_transformers_ output shapes.
        return _indices_from_shapes(preprocessor, "cat_native")
    indices_dict = preprocessor.output_indices_
    if "cat_native" not in indices_dict:
        return []
    slc = indices_dict["cat_native"]
    return list(range(slc.start, slc.stop))


def _indices_from_shapes(preprocessor: ColumnTransformer, target_name: str) -> List[int]:
    offset = 0
    for name, _, _ in preprocessor.transformers_:
        trans = preprocessor.named_transformers_.get(name)
        if trans is None or trans == "drop":
            continue
        # Probe output width via get_feature_names_out or a dummy transform.
        try:
            width = len(trans.get_feature_names_out())
        except Exception:
            width = 0
        if name == target_name:
            return list(range(offset, offset + width))
        offset += width
    return []
