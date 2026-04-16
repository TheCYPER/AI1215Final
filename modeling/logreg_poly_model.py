"""Logistic regression with polynomial interaction features.

Role in the ensemble: a purely linear learner on top of engineered pairwise
interactions. Its error structure is fundamentally different from tree
ensembles — it captures smooth linear gradients while trees capture
axis-aligned partitions. For stacking diversity.

Pipeline: PolynomialFeatures(interaction_only=True, degree=2) →
StandardScaler (in case preproc wasn't scale-invariant on new cols) →
LogisticRegression (multinomial). Interaction-only keeps dimensionality
tractable — degree=2 on ~75 features yields ~75 + 75*74/2 = ~2850 features,
manageable for L2 LogReg.

For regression: PolynomialFeatures → StandardScaler → Ridge.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from configs.config import TaskType
from modeling.base_model import BaseModel


class LogRegPolyModel(BaseModel):
    """LogReg / Ridge with polynomial interaction features."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type

    def build_model(self, num_classes: int = 5, **kwargs) -> "LogRegPolyModel":
        params = dict(self.config)
        degree = int(params.pop("poly_degree", 2))
        interaction_only = bool(params.pop("poly_interaction_only", True))
        include_bias = bool(params.pop("poly_include_bias", False))

        steps = [
            (
                "poly",
                PolynomialFeatures(
                    degree=degree,
                    interaction_only=interaction_only,
                    include_bias=include_bias,
                ),
            ),
            ("scaler", StandardScaler(with_mean=False)),
        ]
        if self.task_type == TaskType.CLASSIFICATION:
            # Default: strong L2 to tame high-dim interaction features.
            params.setdefault("C", 0.1)
            params.setdefault("max_iter", 2000)
            params.setdefault("n_jobs", -1)
            steps.append(("clf", LogisticRegression(**params)))
        else:
            params.setdefault("alpha", 1.0)
            steps.append(("reg", Ridge(**params)))
        self.model_ = Pipeline(steps)
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
        if sample_weight is not None:
            fit_key = (
                "clf__sample_weight"
                if self.task_type == TaskType.CLASSIFICATION
                else "reg__sample_weight"
            )
            self.model_.fit(X, y, **{fit_key: sample_weight})
        else:
            self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(self._clean(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(self._clean(X))

    @staticmethod
    def _clean(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr
