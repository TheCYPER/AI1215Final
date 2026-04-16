"""TabNet wrapper (pytorch-tabnet).

TabNet is a tabular-specific architecture with sequential attention over
features — it learns which columns to pay attention to at each decision
step. Competitive with GBMs on many tabular benchmarks and produces
fundamentally different error patterns (attention vs. greedy splits),
making it a strong ensemble member.

Design notes:
- Accepts `cat_idxs` + `cat_dims` at construction so embedding layers fire
  for ordinal-encoded categoricals. We derive them from the input fit call
  when `categorical_feature` is supplied.
- `eval_set` is used for built-in early stopping.
- sklearn-style API via `TabNetClassifier.fit()` — accepts numpy arrays.
- No GPU on this machine; TabNet on CPU with ~35k rows takes ~20-40 min.
  We keep `n_a / n_d` modest to cap runtime.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from configs.config import TaskType
from modeling.base_model import BaseModel


class TabNetModel(BaseModel):
    """pytorch-tabnet wrapper with cat embedding support."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type
        self._max_epochs: int = 100
        self._patience: int = 15
        self._batch_size: int = 1024
        self._virtual_batch_size: int = 128
        self._cat_idxs: Optional[List[int]] = None
        self._cat_dims: Optional[List[int]] = None

    def build_model(self, num_classes: int = 5, **kwargs) -> "TabNetModel":
        # Stash the params; actual constructor is called via _build_constructor_params
        # so fit() can re-build with cat_idxs when known.
        ctor = self._build_constructor_params()
        if self.task_type == TaskType.CLASSIFICATION:
            self.model_ = TabNetClassifier(**ctor)
        else:
            self.model_ = TabNetRegressor(**ctor)
        return self

    def _build_constructor_params(self) -> Dict[str, Any]:
        # Limit threads to 1 to prevent fork-unsafe SEGFAULT when sklearn's
        # ColumnTransformer n_jobs=-1 spawns joblib workers alongside PyTorch.
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)

        params = dict(self.config)
        self._max_epochs = int(params.pop("max_epochs", 100))
        self._patience = int(params.pop("patience", 15))
        self._batch_size = int(params.pop("batch_size", 1024))
        self._virtual_batch_size = int(params.pop("virtual_batch_size", 128))
        params.setdefault("n_d", 16)
        params.setdefault("n_a", 16)
        params.setdefault("n_steps", 4)
        params.setdefault("gamma", 1.3)
        params.setdefault("lambda_sparse", 1e-3)
        params.setdefault("seed", 42)
        params.setdefault("verbose", 0)
        params.setdefault("device_name", "cpu")
        params.setdefault("optimizer_fn", torch.optim.Adam)
        params.setdefault("optimizer_params", {"lr": 2e-2})
        return params

    def fit(
        self,
        X,
        y,
        eval_set=None,
        sample_weight=None,
        categorical_feature=None,
        **kwargs,
    ):
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.asarray(y)

        # pytorch-tabnet requires cat_idxs / cat_dims at constructor time so
        # embedding layers are built with the correct sizes. Since we only know
        # cat_dims from data, we rebuild the model here with those values
        # before calling fit — this discards the fresh instance made by
        # build_model() but keeps the wrapper interface uniform.
        if categorical_feature:
            cat_idxs = list(categorical_feature)
            cat_dims = [int(X[:, i].max()) + 2 for i in cat_idxs]  # +2: slack for test-time unknowns
            constructor_params = self._build_constructor_params()
            constructor_params["cat_idxs"] = cat_idxs
            constructor_params["cat_dims"] = cat_dims
            constructor_params["cat_emb_dim"] = [min(50, (d + 1) // 2) for d in cat_dims]
            if self.task_type == TaskType.CLASSIFICATION:
                self.model_ = TabNetClassifier(**constructor_params)
            else:
                self.model_ = TabNetRegressor(**constructor_params)
            self._cat_idxs = cat_idxs
            self._cat_dims = cat_dims

        fit_kwargs: Dict[str, Any] = {
            "max_epochs": self._max_epochs,
            "patience": self._patience,
            "batch_size": self._batch_size,
            "virtual_batch_size": self._virtual_batch_size,
            # num_workers=0 → no DataLoader subprocesses. Anything else has
            # segfaulted under fork + MKL/OpenMP on this machine.
            "num_workers": 0,
        }
        if eval_set is not None:
            # pytorch-tabnet expects list of (X, y) tuples
            X_val, y_val = eval_set[0] if isinstance(eval_set, list) else eval_set
            X_val = np.asarray(X_val, dtype=np.float32)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
            y_val = np.asarray(y_val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_metric"] = (
                ["accuracy"] if self.task_type == TaskType.CLASSIFICATION else ["rmse"]
            )
        if sample_weight is not None:
            fit_kwargs["weights"] = np.asarray(sample_weight, dtype=np.float32)

        # TabNet regressor expects 2D y
        if self.task_type == TaskType.REGRESSION and y.ndim == 1:
            y = y.reshape(-1, 1)

        self.model_.fit(X, y, **fit_kwargs)
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        pred = self.model_.predict(X)
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.ravel()
        return pred

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.model_.predict_proba(X)
