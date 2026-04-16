"""MLP wrapper using sklearn's MLPClassifier / MLPRegressor.

Primary role: a non-tree base learner that makes genuinely different errors
from CatBoost / LGBM / XGB, for ensemble diversity.

Design notes:
- Input comes from the ColumnTransformer already — numeric cols are
  standardized, categoricals are ordinal-encoded to small integer codes.
  sklearn MLP doesn't have an embedding layer; ordinal codes go in as
  floats. This is suboptimal vs. a proper embedding but adequate as a
  baseline before we reach for PyTorch.
- `categorical_feature` accepted but ignored (handled upstream / in data).
- `early_stopping=True` + `validation_fraction=0.1` lets MLP self-regularize.
- NaN / inf safeguard: MLP chokes on those; preprocessor should guarantee
  clean output but we clip extreme values as defence-in-depth.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from configs.config import TaskType
from modeling.base_model import BaseModel


class MLPModel(BaseModel):
    """sklearn MLP wrapper with built-in StandardScaler.

    The upstream preprocessor doesn't scale cat_native columns (ordinal codes
    0-50) — only numerics get StandardScaler. MLP is catastrophically sensitive
    to feature-scale mismatch, so we add our own scaler here at fit time.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type
        self._scaler = None

    def build_model(self, num_classes: int = 5, **kwargs) -> "MLPModel":
        from sklearn.preprocessing import StandardScaler
        params = dict(self.config)
        self._scaler = StandardScaler()
        if self.task_type == TaskType.CLASSIFICATION:
            self.model_ = MLPClassifier(**params)
        else:
            self.model_ = MLPRegressor(**params)
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
        X = self._clean(X)
        X = self._scaler.fit_transform(X)
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(self._scale(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(self._scale(X))

    def _scale(self, X) -> np.ndarray:
        return self._scaler.transform(self._clean(X))

    @staticmethod
    def _clean(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr
