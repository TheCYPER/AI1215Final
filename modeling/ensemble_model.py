"""Ensemble of heterogeneous base models with pluggable combination modes.

Three modes:
- `"uniform"`: arithmetic mean of base predict_proba outputs (default, legacy).
- `"weighted"`: weighted mean using pre-supplied weights.
- `"stacking"`: train a meta-learner (LogisticRegression by default) on out-of-
  sample base predictions using an 80/20 holdout within each fold.

Design choices:
- Base models are pre-built and injected by `model_factory`, keeping
  the ensemble decoupled from the registry (no circular imports).
- `eval_set` / `sample_weight` / `categorical_feature` forward to every base
  model in uniform/weighted modes. In stacking mode, `eval_set` is dropped
  (bases train a fixed iteration budget; early stopping on the meta-holdout
  would double-use that split, and using the outer val fold would leak).
- Regression falls back to weighted mean of `predict` outputs. Stacking for
  regression uses Ridge as meta-learner.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import copy
import logging

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("modeling.ensemble")

from configs.config import TaskType
from modeling.base_model import BaseModel


class EnsembleModel(BaseModel):
    """Weighted blending or stacking ensemble."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        base_models: Optional[Sequence[BaseModel]] = None,
        weights: Optional[Sequence[float]] = None,
        mode: str = "uniform",
        meta_learner_type: str = "logreg",
        stack_holdout_size: float = 0.2,
        stack_random_state: int = 42,
        stack_method: str = "holdout",  # "holdout" or "oof"
        stack_inner_folds: int = 5,
    ):
        super().__init__(config)
        self.task_type = task_type
        self.base_models_ = list(base_models) if base_models else []
        self.weights_: Optional[List[float]] = (
            list(weights) if weights is not None else None
        )
        self.mode = mode
        self.meta_learner_type = meta_learner_type
        self.stack_holdout_size = stack_holdout_size
        self.stack_random_state = stack_random_state
        self.stack_method = stack_method
        self.stack_inner_folds = stack_inner_folds
        self._num_classes: int = 5
        self.meta_: Optional[Any] = None

    def build_model(self, num_classes: int = 5, **kwargs) -> "EnsembleModel":
        if not self.base_models_:
            raise ValueError(
                "EnsembleModel requires base_models at construction time"
            )
        self._num_classes = num_classes
        if self.weights_ is None:
            self.weights_ = [1.0] * len(self.base_models_)
        if len(self.weights_) != len(self.base_models_):
            raise ValueError(
                f"weights length {len(self.weights_)} != base_models length "
                f"{len(self.base_models_)}"
            )
        if self.mode not in ("uniform", "weighted", "stacking"):
            raise ValueError(f"Unknown ensemble mode: {self.mode}")
        self.model_ = self.base_models_
        return self

    # ------------------------------------------------------------------
    # fit dispatching
    # ------------------------------------------------------------------
    def fit(self, X, y, **kwargs):
        if self.mode == "stacking":
            if self.stack_method == "oof":
                return self._fit_stacking_oof(X, y, **kwargs)
            return self._fit_stacking(X, y, **kwargs)
        return self._fit_blend(X, y, **kwargs)

    def _fit_blend(self, X, y, **kwargs):
        """Uniform / weighted mode: just fit each base on the full data."""
        for m in self.base_models_:
            m.fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def _fit_stacking(
        self,
        X,
        y,
        eval_set=None,
        sample_weight=None,
        categorical_feature=None,
        **kwargs,
    ):
        """80/20 holdout: train bases on 80%, use their predictions on 20% to
        train the meta-learner. Drops eval_set to avoid double-use leakage.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)

        stratify = y if self.task_type == TaskType.CLASSIFICATION else None
        idx_train, idx_meta = train_test_split(
            np.arange(n),
            test_size=self.stack_holdout_size,
            stratify=stratify,
            random_state=self.stack_random_state,
        )

        X_tr, X_meta = X[idx_train], X[idx_meta]
        y_tr, y_meta = y[idx_train], y[idx_meta]
        sw_tr = sample_weight[idx_train] if sample_weight is not None else None

        # 1. Fit each base on the 80% partition. Intentionally no eval_set.
        for m in self.base_models_:
            m.fit(
                X_tr,
                y_tr,
                sample_weight=sw_tr,
                categorical_feature=categorical_feature,
            )

        # 2. Build meta-features from the 20% holdout.
        meta_features = self._stack_features(X_meta)

        # 3. Scale meta features (prevents LogReg overflow from correlated bases)
        #    and fit the meta-learner.
        self._meta_scaler = StandardScaler()
        meta_features = self._meta_scaler.fit_transform(meta_features)
        self.meta_ = self._build_meta_learner()
        self.meta_.fit(meta_features, y_meta)

        self.is_fitted_ = True
        return self

    def _fit_stacking_oof(
        self,
        X,
        y,
        eval_set=None,
        sample_weight=None,
        categorical_feature=None,
        **kwargs,
    ):
        """K-fold OOF stacking: uses ALL training data for both base and meta.

        1. K-inner-fold split on (X, y)
        2. For each inner fold, fit fresh copies of every base on inner_train,
           predict_proba on inner_val → fill OOF matrix
        3. Meta learner trains on full OOF matrix
        4. Refit all bases on full (X, y) for inference
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)
        n_bases = len(self.base_models_)

        # OOF matrix to fill
        oof = np.zeros((n, n_bases * self._num_classes), dtype=np.float64)

        if self.task_type == TaskType.CLASSIFICATION:
            splitter = StratifiedKFold(
                n_splits=self.stack_inner_folds, shuffle=True,
                random_state=self.stack_random_state,
            )
            split_iter = splitter.split(X, y)
        else:
            splitter = KFold(
                n_splits=self.stack_inner_folds, shuffle=True,
                random_state=self.stack_random_state,
            )
            split_iter = splitter.split(X)

        for fold_idx, (inner_train, inner_val) in enumerate(split_iter):
            logger.info(f"  OOF inner fold {fold_idx + 1}/{self.stack_inner_folds}")
            X_tr, X_val = X[inner_train], X[inner_val]
            y_tr = y[inner_train]
            sw_tr = sample_weight[inner_train] if sample_weight is not None else None

            for base_idx, base_model in enumerate(self.base_models_):
                # Deep copy to get fresh model for this fold
                fresh = copy.deepcopy(base_model)
                fresh.fit(
                    X_tr, y_tr,
                    sample_weight=sw_tr,
                    categorical_feature=categorical_feature,
                )
                proba = self._base_proba(fresh, X_val)
                col_start = base_idx * self._num_classes
                col_end = col_start + self._num_classes
                oof[inner_val, col_start:col_end] = proba

        # Scale + fit meta on full OOF predictions
        self._meta_scaler = StandardScaler()
        oof_scaled = self._meta_scaler.fit_transform(oof)
        self.meta_ = self._build_meta_learner()
        self.meta_.fit(oof_scaled, y)
        logger.info("  OOF meta learner fitted on full OOF predictions")

        # Refit all bases on full data for inference
        logger.info("  Refitting bases on full training data...")
        for m in self.base_models_:
            m.fit(
                X, y,
                sample_weight=sample_weight,
                categorical_feature=categorical_feature,
            )

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # predict dispatching
    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        if self.task_type == TaskType.CLASSIFICATION:
            return np.argmax(self.predict_proba(X), axis=1)
        # Regression
        if self.mode == "stacking":
            feats = self._stack_features(X)
            if hasattr(self, "_meta_scaler") and self._meta_scaler is not None:
                feats = self._meta_scaler.transform(feats)
            return np.asarray(self.meta_.predict(feats))
        total = sum(self.weights_)
        preds = np.zeros(len(X), dtype=np.float64)
        for m, w in zip(self.base_models_, self.weights_):
            preds += (w / total) * np.asarray(m.predict(X), dtype=np.float64).ravel()
        return preds

    def predict_proba(self, X) -> np.ndarray:
        if self.task_type != TaskType.CLASSIFICATION:
            raise NotImplementedError("predict_proba only for classification")
        if self.mode == "stacking":
            feats = self._stack_features(X)
            if hasattr(self, "_meta_scaler") and self._meta_scaler is not None:
                feats = self._meta_scaler.transform(feats)
            return self.meta_.predict_proba(feats)
        # uniform / weighted
        total = sum(self.weights_)
        out = np.zeros((len(X), self._num_classes), dtype=np.float64)
        for m, w in zip(self.base_models_, self.weights_):
            p = self._base_proba(m, X)
            out += (w / total) * p
        return out

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _stack_features(self, X) -> np.ndarray:
        """Concatenate each base's predict_proba into a (n, N_bases * n_classes) matrix."""
        parts = [self._base_proba(m, X) for m in self.base_models_]
        return np.hstack(parts)

    def _base_proba(self, model: BaseModel, X) -> np.ndarray:
        """Get base model predict_proba, normalizing shape quirks (CatBoost)."""
        p = np.asarray(model.predict_proba(X), dtype=np.float64)
        if p.ndim == 3:  # CatBoost MultiClass sometimes returns (n, 1, K)
            p = p.reshape(p.shape[0], -1)
        return p

    def _build_meta_learner(self):
        if self.task_type == TaskType.CLASSIFICATION:
            if self.meta_learner_type == "logreg":
                return LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
            if self.meta_learner_type == "lgbm_shallow":
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=50, max_depth=3, num_leaves=8,
                    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=-1, n_jobs=-1,
                )
            raise ValueError(
                f"Unsupported classification meta learner: {self.meta_learner_type}"
            )
        # Regression meta
        if self.meta_learner_type in ("logreg", "ridge"):
            return Ridge(alpha=1.0)
        raise ValueError(
            f"Unsupported regression meta learner: {self.meta_learner_type}"
        )
