"""Train the reg 12-TabNet stacking ensemble on FULL training data (no CV).

Saves an ensemble_dict artifact at outputs/models/regression_pipeline.joblib
in the format that scripts/generate_gated_submission.py expects:
    {
      "format": "ensemble_dict",
      "model": <fitted EnsembleModel>,
      "preprocessor": <fitted ColumnTransformer>,
    }

Why no-aug: row #51 (two-stage v2 with SOTA OOF) only beat no-aug by +0.0016
on 5-fold CV; the std 0.0156 dwarfs that gap. Not worth the extra complexity
of running cls inference on test to augment test features.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \\
        nohup python -u scripts/train_reg_on_full_data.py \\
        > /tmp/train_reg_full.log 2>&1 &
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from configs.config import (  # noqa: E402
    Config,
    TaskType,
    pure_tabnet_reg_stacking_components,
)
from data_cleaning.column_types import infer_column_types  # noqa: E402
from feature_engineering.preprocessor import (  # noqa: E402
    build_preprocessor,
    get_categorical_feature_indices,
)
from modeling import model_factory  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def main() -> None:
    log("train_reg_on_full_data start")

    cfg = Config()
    cfg.training.task_type = TaskType.REGRESSION
    cfg.models.reg_model_type = "ensemble"
    cfg.models.ensemble_reg_mode = "stacking"
    cfg.models.ensemble_meta_learner_type = "ridge"
    cfg.models.ensemble_reg_components = pure_tabnet_reg_stacking_components()

    log("loading train data...")
    train_df = pd.read_csv(cfg.paths.train_csv)
    target_col = cfg.get_target()
    drop = [c for c in cfg.columns.targets if c in train_df.columns]
    X = train_df.drop(columns=drop)
    y = train_df[target_col].values
    log(f"  X shape: {X.shape}, y shape: {y.shape}")

    log("building + fitting preprocessor on FULL train...")
    num_cols, cat_cols = infer_column_types(
        X,
        targets=cfg.columns.targets,
        forced_categorical=cfg.columns.forced_categorical,
        drop_columns=cfg.columns.drop_columns,
    )
    preprocessor = build_preprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        freq_encoding_cols=cfg.features.freq_encoding_cols,
        log_transform_cols=cfg.features.log_transform_cols,
        enable_credit_features=cfg.features.enable_credit_features,
        target_encoding_cols=cfg.features.target_encoding_cols,
        native_categorical=cfg.features.native_categorical_for_lgbm,
    )
    X_t = preprocessor.fit_transform(X, y)
    cat_indices = get_categorical_feature_indices(preprocessor)
    log(f"  transformed X shape: {X_t.shape}, cat_indices: {len(cat_indices)}")

    log("building + fitting reg ensemble on FULL train...")
    t0 = time.time()
    model = model_factory(cfg)
    fit_kwargs = {}
    if cat_indices:
        fit_kwargs["categorical_feature"] = cat_indices
    model.fit(X_t, y, **fit_kwargs)
    elapsed_min = round((time.time() - t0) / 60, 1)
    log(f"  reg ensemble trained in {elapsed_min} min")

    artifact_path = REPO_ROOT / "outputs" / "models" / "regression_pipeline.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    # Back up the old pipeline before overwriting
    if artifact_path.exists():
        backup = artifact_path.with_suffix(".xgb_backup.joblib")
        log(f"  backing up old pipeline → {backup}")
        artifact_path.rename(backup)

    log(f"saving ensemble_dict → {artifact_path}")
    joblib.dump(
        {
            "format": "ensemble_dict",
            "model": model,
            "preprocessor": preprocessor,
            "cat_indices": cat_indices,
        },
        artifact_path,
    )
    log("done")


if __name__ == "__main__":
    main()
