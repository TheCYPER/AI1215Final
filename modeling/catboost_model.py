"""CatBoost model wrapper with native categorical-feature support.

CatBoost is the third base learner for ensemble diversity. Its distinguishing
trait is ordered-target-statistic encoding for categorical features: a
principled CV-safe alternative to one-hot / frequency / target encoding. On
datasets with strong categorical signal it often diverges enough from LGBM
and XGBoost to contribute real variance reduction in a blend, even when its
solo accuracy is comparable.

Key implementation notes:
- CatBoost's `cat_features` parameter wants integer or string columns;
  ColumnTransformer concatenation gives us float64, so `_to_cat_frame` casts
  the named columns back to int32 inside a pandas DataFrame before fit/predict.
- Early stopping requires `eval_set` to be a tuple `(X_val, y_val)` (not a
  list like XGBoost/LGBM accept).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from configs.config import TaskType
from modeling.base_model import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost wrapper for classification and regression."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type
        self._early_stopping_rounds: Optional[int] = None
        self._cat_idx_: Optional[List[int]] = None

    def build_model(self, num_classes: int = 5, **kwargs) -> "CatBoostModel":
        params = dict(self.config)
        self._early_stopping_rounds = params.pop("early_stopping_rounds", None)
        if self.task_type == TaskType.CLASSIFICATION:
            params.setdefault("loss_function", "MultiClass")
            params.setdefault("classes_count", num_classes)
            self.model_ = CatBoostClassifier(**params)
        else:
            params.pop("classes_count", None)
            params.setdefault("loss_function", "RMSE")
            self.model_ = CatBoostRegressor(**params)
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
        cat_idx = list(categorical_feature) if categorical_feature else None
        self._cat_idx_ = cat_idx

        X_df = self._to_cat_frame(X, cat_idx)
        fit_params: Dict[str, Any] = {"verbose": False}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if cat_idx:
            fit_params["cat_features"] = cat_idx
        if eval_set is not None:
            X_val, y_val = eval_set[0] if isinstance(eval_set, list) else eval_set
            X_val_df = self._to_cat_frame(X_val, cat_idx)
            fit_params["eval_set"] = (X_val_df, y_val)
            if self._early_stopping_rounds:
                fit_params["early_stopping_rounds"] = self._early_stopping_rounds

        self.model_.fit(X_df, y, **fit_params)
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        X_df = self._to_cat_frame(X, self._cat_idx_)
        pred = self.model_.predict(X_df)
        # Classification returns shape (n, 1); flatten.
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.ravel()
        return pred

    def predict_proba(self, X) -> np.ndarray:
        X_df = self._to_cat_frame(X, self._cat_idx_)
        return self.model_.predict_proba(X_df)

    @staticmethod
    def _to_cat_frame(X, cat_idx: Optional[List[int]]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        if cat_idx:
            cols = df.columns.tolist()
            for i in cat_idx:
                col = cols[i]
                df[col] = df[col].astype(np.int32)
        return df
