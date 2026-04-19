"""XGBoost cumulative-logits ordinal classifier (CORAL-style).

Replaces the earlier "regressor + round/clip" approach that performed poorly
on CreditSense (see memory: project_ordinal_dead_end).

Approach:
- Train K-1 binary classifiers, where the k-th predicts P(y > k).
- At inference, enforce rank consistency (cum probs non-increasing in k)
  via monotone projection along axis=1, then differentiate to get P(y=k):
      P(y=0)   = 1 - P(y>0)
      P(y=k)   = P(y>k-1) - P(y>k)  for 0 < k < K-1
      P(y=K-1) = P(y>K-2)

Motivation:
- RiskTier errors are ~99% adjacent-class (ordinal). Softmax penalizes
  distance-1 and distance-4 mistakes equally; cumulative logits don't.
- Independent binary classifiers are simple, fast, and give TabNet a
  potentially uncorrelated error surface for the ensemble.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from xgboost import XGBClassifier

from configs.config import TaskType
from modeling.base_model import BaseModel


class XGBoostOrdinalModel(BaseModel):
    """K-1 binary classifiers on cumulative label thresholds."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type
        self.num_classes_: int = 5
        self.class_min_: int = 0
        # self.model_ holds a list[XGBClassifier] once build_model is called.

    def build_model(self, num_classes: int = 5, **kwargs) -> "XGBoostOrdinalModel":
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError(
                "XGBoostOrdinalModel requires ordered classification targets. "
                "For pure regression use XGBoostModel."
            )
        self.num_classes_ = num_classes
        params = dict(self.config)
        # Drop anything softmax-specific; we run binary heads.
        params.pop("num_class", None)
        params.pop("objective", None)
        params.setdefault("eval_metric", "logloss")

        self.model_: List[XGBClassifier] = [
            XGBClassifier(objective="binary:logistic", **params)
            for _ in range(num_classes - 1)
        ]
        self.class_min_ = 0
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
        y = np.asarray(y).astype(int)
        self.class_min_ = int(y.min())
        y_shifted = y - self.class_min_  # contiguous ints starting at 0

        fit_base: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_base["sample_weight"] = sample_weight

        for k, clf in enumerate(self.model_):
            y_bin = (y_shifted > k).astype(int)
            # Skip eval_set entirely — the ordinal head targets differ from the
            # softmax eval_set's y, so sklearn would raise on mismatched labels.
            clf.fit(X, y_bin, **fit_base)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = int(np.asarray(X).shape[0])
        K = self.num_classes_
        cum = np.zeros((n, K - 1), dtype=np.float64)
        for k, clf in enumerate(self.model_):
            cum[:, k] = clf.predict_proba(X)[:, 1]

        # Rank consistency: P(y>0) >= P(y>1) >= ... >= P(y>K-2)
        cum = np.minimum.accumulate(cum, axis=1)

        proba = np.zeros((n, K), dtype=np.float64)
        proba[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, K - 1):
            proba[:, k] = cum[:, k - 1] - cum[:, k]
        proba[:, K - 1] = cum[:, K - 2]

        proba = np.clip(proba, 0.0, 1.0)
        row_sum = proba.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return proba / row_sum

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1) + self.class_min_

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if not self.model_:
            return None
        imps = [
            m.feature_importances_
            for m in self.model_
            if hasattr(m, "feature_importances_")
        ]
        if not imps:
            return None
        return np.mean(imps, axis=0)

    def save(self, path: str):
        import joblib
        joblib.dump(
            {
                "model_": self.model_,
                "num_classes_": self.num_classes_,
                "class_min_": self.class_min_,
            },
            path,
        )

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        self.model_ = data["model_"]
        self.num_classes_ = data["num_classes_"]
        self.class_min_ = data["class_min_"]
        self.is_fitted_ = True
        return self
