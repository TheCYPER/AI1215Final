"""Train on full data + generate gated submission for Kaggle.

Pipeline:
  1. Fit preprocessor on full train.
  2. Fit Stack(12 TabNet + 5 CORN-MLP) on full train.
  3. Fit TabNet single (tuned row #24 config) on full train.
  4. Transform test; get proba from both.
  5. Apply gated rule @ t=0.59: where stack top-1 proba < 0.59, use TabNet's
     argmax; else use stack's argmax.
  6. Regression: reuse the existing regression pipeline from outputs/models/
     (classification is the only thing we're improving on this round).
  7. Write submission CSV.

The threshold t=0.59 was chosen by half-split validation on the OOF from
stack_12tabnet_5coral and single-TabNet CV runs (experiments row #48),
honest OOF acc 0.8718 vs SOTA 0.8687.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from configs.config import Config, TaskType, tabnet_plus_coral_components
from data_cleaning.column_types import infer_column_types
from feature_engineering.preprocessor import (
    build_preprocessor,
    get_categorical_feature_indices,
)
from modeling import model_factory
from modeling.tabnet_model import TabNetModel
from modeling.ensemble_model import EnsembleModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gated_submission")


def build_full_data_preprocessor(cfg: Config, X: pd.DataFrame):
    num_cols, cat_cols = infer_column_types(
        X,
        targets=cfg.columns.targets,
        forced_categorical=cfg.columns.forced_categorical,
        drop_columns=cfg.columns.drop_columns,
    )
    return build_preprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        freq_encoding_cols=cfg.features.freq_encoding_cols,
        log_transform_cols=cfg.features.log_transform_cols,
        enable_credit_features=cfg.features.enable_credit_features,
        target_encoding_cols=cfg.features.target_encoding_cols,
        native_categorical=cfg.features.native_categorical_for_lgbm,
    )


def main(threshold: float, reg_pipeline_path: str, output_path: str):
    # Config setup
    cfg = Config()
    cfg.training.task_type = TaskType.CLASSIFICATION
    cfg.models.clf_model_type = "ensemble"
    cfg.models.ensemble_clf_components = tabnet_plus_coral_components(5)
    cfg.models.ensemble_clf_mode = "stacking"
    cfg.models.ensemble_meta_learner_type = "logreg"
    cfg.models.ensemble_stack_method = "holdout"

    # ---- Load train + test ----
    logger.info("Loading train/test data...")
    train_df = pd.read_csv(cfg.paths.train_csv)
    test_df = pd.read_csv(cfg.paths.test_csv)
    y_train = train_df[cfg.get_target()].values
    drop = [c for c in cfg.columns.targets if c in train_df.columns]
    X_train_raw = train_df.drop(columns=drop)
    X_test_raw = test_df.copy()
    # Test CSV may or may not include target columns — drop if present.
    for c in cfg.columns.targets:
        if c in X_test_raw.columns:
            X_test_raw = X_test_raw.drop(columns=[c])
    logger.info(f"Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # ---- Preprocessor on full train ----
    preprocessor = build_full_data_preprocessor(cfg, X_train_raw)
    X_train_t = preprocessor.fit_transform(X_train_raw, y_train)
    X_test_t = preprocessor.transform(X_test_raw)
    cat_indices = get_categorical_feature_indices(preprocessor)
    logger.info(f"Features after preprocessing: {X_train_t.shape[1]}")

    # ---- Train Stack(12 TabNet + 5 CORN-MLP) on full data ----
    logger.info("=== [1/2] Training Stack(12 TabNet + 5 CORN-MLP) on full data ===")
    t0 = time.time()
    stack_model = model_factory(cfg)
    assert isinstance(stack_model, EnsembleModel)
    fit_kwargs = {"categorical_feature": cat_indices} if cat_indices else {}
    stack_model.fit(X_train_t, y_train, **fit_kwargs)
    logger.info(f"Stack trained in {(time.time()-t0)/60:.1f} min")

    # ---- Train TabNet single on full data ----
    logger.info("=== [2/2] Training TabNet single on full data ===")
    cfg_tn = Config()
    cfg_tn.training.task_type = TaskType.CLASSIFICATION
    cfg_tn.models.clf_model_type = "tabnet"
    t0 = time.time()
    tabnet_model = model_factory(cfg_tn)
    assert isinstance(tabnet_model, TabNetModel)
    fit_kwargs_tn = {"categorical_feature": cat_indices} if cat_indices else {}
    tabnet_model.fit(X_train_t, y_train, **fit_kwargs_tn)
    logger.info(f"TabNet single trained in {(time.time()-t0)/60:.1f} min")

    # ---- Inference on test ----
    logger.info("=== Inference with gating ===")
    stack_proba = np.asarray(stack_model.predict_proba(X_test_t))
    tabnet_proba = np.asarray(tabnet_model.predict_proba(X_test_t))
    assert stack_proba.shape == tabnet_proba.shape, (stack_proba.shape, tabnet_proba.shape)

    top1 = stack_proba.max(axis=1)
    mask_low = top1 < threshold
    stack_pred = stack_proba.argmax(axis=1)
    tabnet_pred = tabnet_proba.argmax(axis=1)
    risk_tier = np.where(mask_low, tabnet_pred, stack_pred).astype(int)
    logger.info(
        f"Threshold={threshold}: {int(mask_low.sum())} / {len(mask_low)} samples "
        f"({100*mask_low.mean():.1f}%) went to TabNet specialist"
    )
    logger.info(f"RiskTier distribution: {np.bincount(risk_tier, minlength=5)}")

    # Save model artifacts (bundle) for reproducibility
    artifacts_path = PROJECT / "outputs" / "models" / "gated_cls_artifact.joblib"
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "stack_model": stack_model,
            "tabnet_model": tabnet_model,
            "threshold": threshold,
            "format": "gated_dict",
        },
        artifacts_path,
    )
    logger.info(f"Gated artifact saved: {artifacts_path}")

    # ---- Regression: reuse existing pipeline ----
    logger.info(f"Loading regression pipeline: {reg_pipeline_path}")
    reg_artifact = joblib.load(reg_pipeline_path)
    if isinstance(reg_artifact, dict) and reg_artifact.get("format") == "ensemble_dict":
        reg_model = reg_artifact["model"]
        reg_pre = reg_artifact["preprocessor"]
        X_test_reg = reg_pre.transform(X_test_raw)
        interest_rate = np.asarray(reg_model.predict(X_test_reg)).ravel()
    else:
        interest_rate = np.asarray(reg_artifact.predict(X_test_raw)).ravel()
    lo, hi = cfg.training.interest_rate_range
    interest_rate = np.clip(interest_rate, lo, hi).round(2)
    logger.info(f"InterestRate range: [{interest_rate.min():.2f}, {interest_rate.max():.2f}]")

    # ---- Write submission ----
    submission = pd.DataFrame({
        "Id": range(len(test_df)),
        "RiskTier": risk_tier,
        "InterestRate": interest_rate,
    })
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission written: {output_path}")
    logger.info(f"Rows: {len(submission)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.59)
    parser.add_argument(
        "--reg_pipeline",
        default="/Users/percy/MLFinal2026/outputs/models/regression_pipeline.joblib",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT / "outputs" / "predictions" / "submission_gated.csv"),
    )
    args = parser.parse_args()
    main(args.threshold, args.reg_pipeline, args.output)
