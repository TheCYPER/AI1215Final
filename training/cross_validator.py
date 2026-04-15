"""
K-fold cross-validation trainer.

Uses StratifiedKFold for classification, KFold for regression.
Builds a fresh preprocessor per fold to avoid data leakage.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from configs.config import Config, TaskType
from data_cleaning.column_types import infer_column_types
from feature_engineering.preprocessor import (
    build_preprocessor,
    get_categorical_feature_indices,
)
from modeling import model_factory
from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)

logger = logging.getLogger("training.cross_validator")


class CrossValidator:
    """K-fold cross-validation with per-fold preprocessing."""

    def __init__(self, config: Config):
        self.config = config

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data, separate features and target."""
        df = pd.read_csv(self.config.paths.train_csv)
        target_col = self.config.get_target()
        drop = [c for c in self.config.columns.targets if c in df.columns]
        X = df.drop(columns=drop)
        y = df[target_col]
        return X, y

    def _build_preprocessor(self, X: pd.DataFrame):
        """Build a fresh preprocessor for a fold."""
        num_cols, cat_cols = infer_column_types(
            X,
            targets=self.config.columns.targets,
            forced_categorical=self.config.columns.forced_categorical,
            drop_columns=self.config.columns.drop_columns,
        )
        return build_preprocessor(
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            freq_encoding_cols=self.config.features.freq_encoding_cols,
            log_transform_cols=self.config.features.log_transform_cols,
            enable_credit_features=self.config.features.enable_credit_features,
            target_encoding_cols=self.config.features.target_encoding_cols,
            native_categorical=self.config.features.native_categorical_for_lgbm,
        )

    def _get_splitter(self):
        """Return the appropriate CV splitter."""
        cfg = self.config.training
        if cfg.task_type == TaskType.CLASSIFICATION:
            return StratifiedKFold(
                n_splits=cfg.n_splits,
                shuffle=cfg.shuffle,
                random_state=cfg.random_state,
            )
        return KFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state,
        )

    def train_fold(
        self,
        fold_idx: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Train and evaluate a single fold."""
        # Fresh preprocessor per fold
        preprocessor = self._build_preprocessor(X_train)
        X_train_t = preprocessor.fit_transform(X_train, y_train)
        X_val_t = preprocessor.transform(X_val)
        cat_indices = get_categorical_feature_indices(preprocessor)

        # Fresh model per fold
        model = model_factory(self.config)

        # Sample weights
        sample_weight = None
        if (
            self.config.training.task_type == TaskType.CLASSIFICATION
            and self.config.training.use_class_weights
        ):
            sample_weight = compute_sample_weight("balanced", y_train)

        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if self.config.training.use_early_stopping:
            fit_kwargs["eval_set"] = [(X_val_t, y_val.values)]
        if cat_indices:
            fit_kwargs["categorical_feature"] = cat_indices

        model.fit(X_train_t, y_train.values, **fit_kwargs)

        val_pred = model.predict(X_val_t)

        if self.config.training.task_type == TaskType.CLASSIFICATION:
            metrics = compute_classification_metrics(y_val.values, val_pred)
        else:
            metrics = compute_regression_metrics(y_val.values, val_pred)

        return metrics

    def _aggregate_results(
        self, fold_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate metrics across folds."""
        if self.config.training.task_type == TaskType.CLASSIFICATION:
            key = "accuracy"
        else:
            key = "r2"

        scores = [r[key] for r in fold_results]
        return {
            f"{key}_mean": float(np.mean(scores)),
            f"{key}_std": float(np.std(scores)),
            f"{key}_per_fold": scores,
        }

    def run(self) -> Dict[str, Any]:
        """Run full cross-validation."""
        task = self.config.training.task_type.value
        logger.info(f"Starting {self.config.training.n_splits}-fold CV for {task}")

        X, y = self.load_data()
        splitter = self._get_splitter()

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            metrics = self.train_fold(fold_idx, X_train, y_train, X_val, y_val)
            fold_results.append(metrics)

            primary_key = "accuracy" if task == "classification" else "r2"
            logger.info(f"  Fold {fold_idx + 1}: {primary_key}={metrics[primary_key]:.4f}")

        summary = self._aggregate_results(fold_results)

        primary_key = "accuracy" if task == "classification" else "r2"
        logger.info(
            f"CV {primary_key}: {summary[f'{primary_key}_mean']:.4f} "
            f"+/- {summary[f'{primary_key}_std']:.4f}"
        )

        # Save summary
        summary_path = f"{self.config.paths.metrics_dir}/{task}_cv_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"CV summary saved to {summary_path}")

        return summary
