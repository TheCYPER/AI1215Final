"""LightGBM model wrapper for classification and regression.

LightGBM often edges out XGBoost slightly on tabular data thanks to leaf-wise
tree growth (which reaches deeper interactions at fixed tree count) and
cheaper histogram-based splits on high-cardinality features. Uses native
early stopping via callback so the model picks its own round count rather
than running the full `n_estimators` budget.
"""

from typing import Any, Dict, Optional

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

from configs.config import TaskType
from modeling.base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM wrapper with early-stopping callback support."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type
        # Early stopping is applied via callback; not a constructor param in LGBM.
        self._early_stopping_rounds: Optional[int] = None

    def build_model(self, num_classes: int = 5, **kwargs) -> "LightGBMModel":
        params = dict(self.config)
        self._early_stopping_rounds = params.pop("early_stopping_rounds", None)

        if self.task_type == TaskType.CLASSIFICATION:
            params.setdefault("num_class", num_classes)
            self.model_ = LGBMClassifier(**params)
        else:
            params.pop("num_class", None)
            params.pop("objective", None)
            params.setdefault("objective", "regression")
            self.model_ = LGBMRegressor(**params)
        return self

    def fit(
        self,
        X,
        y,
        eval_set=None,
        sample_weight=None,
        categorical_feature=None,
        **kwargs,
    ):
        fit_params: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if categorical_feature is not None and len(categorical_feature) > 0:
            fit_params["categorical_feature"] = categorical_feature
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            callbacks = [log_evaluation(period=0)]
            if self._early_stopping_rounds:
                callbacks.append(
                    early_stopping(stopping_rounds=self._early_stopping_rounds, verbose=False)
                )
            fit_params["callbacks"] = callbacks
        self.model_.fit(X, y, **fit_params)
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(X)
