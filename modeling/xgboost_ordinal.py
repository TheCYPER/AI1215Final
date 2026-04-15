"""XGBoost ordinal-regression wrapper for ordered classification targets.

Trains XGBRegressor on integer class labels treated as continuous values,
then rounds and clips predictions back into the valid class range at
inference time. Well-suited for RiskTier-style ordinal targets where
adjacent-class errors dominate and softmax wastes capacity on nominal
class-identity signal.
"""

from typing import Any, Dict, Optional

import numpy as np
from xgboost import XGBRegressor

from configs.config import TaskType
from modeling.base_model import BaseModel


class XGBoostOrdinalModel(BaseModel):
    """XGBoost regressor that emits rounded integer class predictions."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type
        self.class_min_: Optional[int] = None
        self.class_max_: Optional[int] = None

    def build_model(self, num_classes: int = 5, **kwargs) -> "XGBoostOrdinalModel":
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError(
                "XGBoostOrdinalModel only makes sense for classification with "
                "ordered targets. For pure regression use XGBoostModel."
            )
        params = dict(self.config)
        params.pop("num_class", None)
        params.setdefault("objective", "reg:pseudohubererror")
        self.model_ = XGBRegressor(**params)
        # Default range; overridden from y in fit()
        self.class_min_ = 0
        self.class_max_ = num_classes - 1
        return self

    def fit(self, X, y, eval_set=None, sample_weight=None, categorical_feature=None, **kwargs):
        # categorical_feature intentionally ignored (XGBoost path).
        y = np.asarray(y)
        self.class_min_ = int(y.min())
        self.class_max_ = int(y.max())

        fit_params: Dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False
        self.model_.fit(X, y.astype(np.float32), **fit_params)
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        raw = self.model_.predict(X)
        rounded = np.rint(raw).astype(int)
        return np.clip(rounded, self.class_min_, self.class_max_)

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError(
            "Ordinal regression model emits point predictions, not probabilities."
        )
