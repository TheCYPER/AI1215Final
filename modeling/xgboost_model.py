"""XGBoost model wrapper supporting both classification and regression."""

from typing import Any, Dict, Optional

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from configs.config import TaskType
from modeling.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost wrapper.

    Builds XGBClassifier or XGBRegressor based on task_type.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type

    def build_model(self, num_classes: int = 5, **kwargs) -> "XGBoostModel":
        params = dict(self.config)
        if self.task_type == TaskType.CLASSIFICATION:
            params["num_class"] = num_classes
            self.model_ = XGBClassifier(**params)
        else:
            # Remove classification-specific keys if present
            params.pop("num_class", None)
            self.model_ = XGBRegressor(**params)
        return self

    def fit(self, X, y, eval_set=None, sample_weight=None, categorical_feature=None, **kwargs):
        # categorical_feature accepted but ignored: XGBoost doesn't support native
        # categorical handling the same way LightGBM does.
        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False
        self.model_.fit(X, y, **fit_params)
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(X)
