"""
Smoke test for the ported TabularTransformer wrapper.

Runs a minimal fit + predict cycle on a 500-row sample of the training CSV
with a tiny backbone (d_model=32, n_layers=1, max_epochs=2) so a CPU or MPS
can complete it in well under a minute. Validates that:

- `model_factory` can build a `tabular_transformer` (classification & regression)
- `fit(X, y, eval_set=..., categorical_feature=...)` runs without errors
- `predict` returns the right shape and lies in the expected value space
- `predict_proba` rows sum to ~1.0

Not a performance test — numbers are nonsense at this size. This script exists
purely to catch wiring bugs before we ship to a GPU machine for the real run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import Config, TaskType  # noqa: E402
from data_cleaning.column_types import infer_column_types  # noqa: E402
from feature_engineering.preprocessor import (  # noqa: E402
    build_preprocessor,
    get_categorical_feature_indices,
)
from modeling import model_factory  # noqa: E402

SAMPLE_N = 500
TINY_OVERRIDES = {
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 1,
    "d_ff": 64,
    "dropout": 0.1,
    "max_epochs": 2,
    "patience": 2,
    "batch_size": 64,
    "lr": 1e-3,
    "seed": 0,
}


def _load_sample(config: Config, task: TaskType, n: int) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(config.paths.train_csv)
    # Use the smallest possible sample that still preserves all 5 RiskTier
    # classes — StratifiedKFold and the CE loss both want every class represented.
    if task == TaskType.CLASSIFICATION:
        target = config.columns.classification_target
        df = (
            df.groupby(target, group_keys=False)
            .apply(lambda g: g.head(max(1, n // 5)))
            .reset_index(drop=True)
        )
    else:
        target = config.columns.regression_target
        df = df.head(n).reset_index(drop=True)
    drop = [c for c in config.columns.targets if c in df.columns]
    X = df.drop(columns=drop)
    y = df[target]
    return X, y


def _preprocess(config: Config, X_tr, X_va):
    num_cols, cat_cols = infer_column_types(
        X_tr,
        targets=config.columns.targets,
        forced_categorical=config.columns.forced_categorical,
        drop_columns=config.columns.drop_columns,
    )
    pre = build_preprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        freq_encoding_cols=config.features.freq_encoding_cols,
        log_transform_cols=config.features.log_transform_cols,
        enable_credit_features=config.features.enable_credit_features,
        target_encoding_cols=config.features.target_encoding_cols,
        native_categorical=config.features.native_categorical_for_lgbm,
    )
    return pre, num_cols, cat_cols


def _run(task: TaskType) -> None:
    print(f"--- {task.value.upper()} smoke ---")
    config = Config()
    config.training.task_type = task
    config.training.n_splits = 2
    config.training.verbose = False

    if task == TaskType.CLASSIFICATION:
        config.models.clf_model_type = "tabular_transformer"
        params = config.models.tabular_transformer_clf_params
    else:
        config.models.reg_model_type = "tabular_transformer"
        params = config.models.tabular_transformer_reg_params

    params.update(TINY_OVERRIDES)

    X, y = _load_sample(config, task, SAMPLE_N)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)
    pre, num_cols, cat_cols = _preprocess(config, X_tr, X_va)

    X_tr_t = pre.fit_transform(X_tr, y_tr)
    X_va_t = pre.transform(X_va)
    cat_indices = get_categorical_feature_indices(pre)

    model = model_factory(config)

    t0 = time.time()
    model.fit(
        X_tr_t,
        y_tr.values,
        eval_set=[(X_va_t, y_va.values)],
        categorical_feature=cat_indices,
    )
    dt_fit = time.time() - t0

    pred = model.predict(X_va_t)
    assert pred.shape == (len(X_va_t),), f"pred shape {pred.shape} != ({len(X_va_t)},)"
    assert np.all(np.isfinite(pred)), "predictions contain NaN / inf"

    if task == TaskType.CLASSIFICATION:
        assert pred.min() >= 0 and pred.max() < config.training.n_classes, (
            f"cls pred out of range: min={pred.min()} max={pred.max()}"
        )
        proba = model.predict_proba(X_va_t)
        assert proba.shape == (len(X_va_t), config.training.n_classes), (
            f"proba shape {proba.shape}"
        )
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-4), (
            f"proba rows don't sum to 1; got min={row_sums.min()} max={row_sums.max()}"
        )
        print(f"  fit: {dt_fit:.2f}s  pred unique: {sorted(set(pred.tolist()))}")
        print(f"  proba row sum range: [{row_sums.min():.4f}, {row_sums.max():.4f}]")
    else:
        lo, hi = config.training.interest_rate_range
        # Predictions can stray a bit outside the training band with only 2 epochs.
        assert pred.min() > lo - 20 and pred.max() < hi + 20, (
            f"reg pred wildly out of range: {pred.min():.2f}..{pred.max():.2f}"
        )
        print(f"  fit: {dt_fit:.2f}s  pred range: [{pred.min():.2f}, {pred.max():.2f}]")

    print("  OK")


def main() -> None:
    _run(TaskType.CLASSIFICATION)
    _run(TaskType.REGRESSION)
    print("all smoke checks passed")


if __name__ == "__main__":
    main()
