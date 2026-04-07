"""
Single train/validation split trainer.

Task-aware: uses config.training.task_type to select
stratification, metrics, and sample weights.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from configs.config import Config, TaskType
from data_cleaning.column_types import infer_column_types
from feature_engineering.preprocessor import build_preprocessor
from modeling import model_factory
from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)

logger = logging.getLogger("training.trainer")


class Trainer:
    """
    Orchestrates: load -> split -> preprocess -> train -> evaluate -> save.

    Completely config-driven. Swap model or task by changing config.
    """

    def __init__(self, config: Config):
        self.config = config
        self.preprocessor_ = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load train CSV, separate features and target."""
        df = pd.read_csv(self.config.paths.train_csv)
        target_col = self.config.get_target()
        # Always drop BOTH targets from features to prevent leakage
        drop = [c for c in self.config.columns.targets if c in df.columns]
        X = df.drop(columns=drop)
        y = df[target_col]
        return X, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split into train/val with stratification for classification."""
        stratify = y if self.config.training.task_type == TaskType.CLASSIFICATION else None
        return train_test_split(
            X, y,
            test_size=self.config.training.test_size,
            random_state=self.config.training.random_state,
            stratify=stratify,
        )

    def build_preprocessor(self, X: pd.DataFrame):
        """Build the preprocessing ColumnTransformer from config."""
        num_cols, cat_cols = infer_column_types(
            X,
            targets=self.config.columns.targets,
            forced_categorical=self.config.columns.forced_categorical,
            drop_columns=self.config.columns.drop_columns,
        )
        self.preprocessor_ = build_preprocessor(
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            freq_encoding_cols=self.config.features.freq_encoding_cols,
            log_transform_cols=self.config.features.log_transform_cols,
            enable_credit_features=self.config.features.enable_credit_features,
        )

    def compute_sample_weights(self, y: pd.Series) -> Optional[np.ndarray]:
        """Compute sample weights for imbalanced classification."""
        if (
            self.config.training.task_type == TaskType.CLASSIFICATION
            and self.config.training.use_class_weights
        ):
            return compute_sample_weight("balanced", y)
        return None

    def train(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train model and return metrics."""
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if self.config.training.use_early_stopping:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        model.fit(X_train, y_train, **fit_kwargs)

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        if self.config.training.task_type == TaskType.CLASSIFICATION:
            train_metrics = compute_classification_metrics(y_train, train_pred)
            val_metrics = compute_classification_metrics(y_val, val_pred)
        else:
            train_metrics = compute_regression_metrics(y_train, train_pred)
            val_metrics = compute_regression_metrics(y_val, val_pred)

        return {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def save_artifacts(
        self, model, results: Dict[str, Any]
    ):
        """Save preprocessor+model pipeline and metrics."""
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor_),
            ("model", model.model_),
        ])

        task = self.config.training.task_type.value
        pipeline_path = f"{self.config.paths.models_dir}/{task}_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)
        logger.info(f"Pipeline saved to {pipeline_path}")

        # Save metrics (exclude non-serializable items)
        metrics_to_save = {}
        for split in ("train_metrics", "val_metrics"):
            metrics_to_save[split] = {
                k: v for k, v in results[split].items()
                if not isinstance(v, str)  # skip classification_report string
            }

        metrics_path = f"{self.config.paths.metrics_dir}/{task}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

    def run(self) -> Dict[str, Any]:
        """Full training pipeline: load -> split -> preprocess -> train -> evaluate -> save."""
        task = self.config.training.task_type.value
        logger.info(f"Task: {task}")

        # Load
        X, y = self.load_data()
        logger.info(f"Loaded {len(X)} samples, target={self.config.get_target()}")

        # Split
        X_train, X_val, y_train, y_val = self.split_data(X, y)
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Preprocess
        self.build_preprocessor(X_train)
        X_train_t = self.preprocessor_.fit_transform(X_train, y_train)
        X_val_t = self.preprocessor_.transform(X_val)
        logger.info(f"Features after preprocessing: {X_train_t.shape[1]}")

        # Model
        model = model_factory(self.config)
        logger.info(f"Model: {self.config.get_model_type()}")

        # Weights
        sample_weight = self.compute_sample_weights(y_train)

        # Train
        results = self.train(
            model, X_train_t, y_train.values, X_val_t, y_val.values,
            sample_weight=sample_weight,
        )

        # Log primary metric
        if self.config.training.task_type == TaskType.CLASSIFICATION:
            metric_name, metric_val = "accuracy", results["val_metrics"]["accuracy"]
        else:
            metric_name, metric_val = "r2", results["val_metrics"]["r2"]
        logger.info(f"Val {metric_name}: {metric_val:.4f}")

        # Save
        self.save_artifacts(model, results)

        return results
