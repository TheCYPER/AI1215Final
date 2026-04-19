"""FT-Transformer wrapper (rtdl_revisiting_models).

FT-Transformer (Gorishniy et al., 2021) is a transformer applied to tabular
data. Every feature (continuous or categorical) is embedded into the same
vector space, then a stack of transformer blocks processes the sequence with
a CLS token that carries the final prediction.

Why add it alongside TabNet:
- Dense attention across all features, not a sparse subset — complements
  TabNet's sparse feature selection (which uses only ~10 of 76 features).
- Different architecture family → potentially uncorrelated errors, which
  is what the stacking meta-learner feeds on.

Design notes:
- All features arrive as a single preprocessed numeric matrix from the
  pipeline. We pass them as continuous to FTTransformer (cat_cardinalities=[]).
- Minimal custom training loop (no Trainer dep): AdamW + CE, early stopping
  on val accuracy when eval_set is provided.
- CPU-only to stay consistent with TabNet; MPS on this machine has
  historically crashed pytorch-tabnet and isn't worth the debugging for
  a single base.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rtdl_revisiting_models import FTTransformer

from configs.config import TaskType
from modeling.base_model import BaseModel


class FTTransformerModel(BaseModel):
    """FT-Transformer wrapper with sklearn-like fit/predict API."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type
        self._num_classes: int = 5
        self._n_features: Optional[int] = None
        self._device = torch.device("cpu")

        params = dict(self.config)
        # Training-loop knobs (popped so they don't go into the backbone).
        self._max_epochs: int = int(params.pop("max_epochs", 100))
        self._patience: int = int(params.pop("patience", 15))
        self._batch_size: int = int(params.pop("batch_size", 1024))
        self._lr: float = float(params.pop("lr", 1e-4))
        self._weight_decay: float = float(params.pop("weight_decay", 1e-5))
        self._seed: int = int(params.pop("seed", 42))
        # Backbone knobs (fall through to FTTransformer)
        self._backbone_kwargs: Dict[str, Any] = params

    def build_model(self, num_classes: int = 5, **kwargs) -> "FTTransformerModel":
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        self._num_classes = num_classes
        # Backbone is built in fit() when we know n_features.
        return self

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _build_backbone(self, n_features: int) -> nn.Module:
        torch.manual_seed(self._seed)
        if self.task_type == TaskType.CLASSIFICATION:
            d_out = self._num_classes
        else:
            d_out = 1
        defaults = FTTransformer.get_default_kwargs(
            n_blocks=int(self._backbone_kwargs.pop("n_blocks", 3))
        )
        # User overrides layered on top of defaults
        merged = {**defaults, **self._backbone_kwargs}
        merged["_is_default"] = False
        merged["d_out"] = d_out
        model = FTTransformer(
            n_cont_features=n_features,
            cat_cardinalities=[],
            **merged,
        )
        return model.to(self._device)

    @staticmethod
    def _clean(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    def _iter_batches(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        shuffle: bool,
        rng: Optional[np.random.Generator] = None,
    ):
        n = len(X)
        idx = np.arange(n)
        if shuffle:
            if rng is None:
                rng = np.random.default_rng(self._seed)
            idx = rng.permutation(n)
        for start in range(0, n, self._batch_size):
            sel = idx[start : start + self._batch_size]
            x_batch = torch.from_numpy(X[sel]).to(self._device)
            y_batch = None if y is None else torch.from_numpy(y[sel]).to(self._device)
            yield x_batch, y_batch

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # rtdl FTTransformer expects (x_cont, x_cat). No cats → pass None.
        return self.model_(x, None)

    # ------------------------------------------------------------------
    # sklearn-style API
    # ------------------------------------------------------------------
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
        self._n_features = X.shape[1]

        if self.task_type == TaskType.CLASSIFICATION:
            y_arr = np.asarray(y).astype(np.int64)
        else:
            y_arr = np.asarray(y).astype(np.float32).reshape(-1)

        self.model_ = self._build_backbone(self._n_features)
        optimizer = torch.optim.AdamW(
            self.model_.make_parameter_groups(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        if self.task_type == TaskType.CLASSIFICATION:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.MSELoss()

        # Val split for early stopping
        X_val, y_val = None, None
        if eval_set is not None:
            X_val_raw, y_val_raw = (
                eval_set[0] if isinstance(eval_set, list) else eval_set
            )
            X_val = self._clean(X_val_raw)
            if self.task_type == TaskType.CLASSIFICATION:
                y_val = np.asarray(y_val_raw).astype(np.int64)
            else:
                y_val = np.asarray(y_val_raw).astype(np.float32).reshape(-1)

        best_metric: Optional[float] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_left = self._patience

        for epoch in range(self._max_epochs):
            self.model_.train()
            for x_batch, y_batch in self._iter_batches(X, y_arr, shuffle=True):
                optimizer.zero_grad()
                pred = self._forward(x_batch)
                if self.task_type == TaskType.REGRESSION:
                    pred = pred.squeeze(-1)
                    loss = loss_fn(pred, y_batch.float())
                else:
                    loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()

            # Early stopping
            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    preds = []
                    for x_b, _ in self._iter_batches(X_val, None, shuffle=False):
                        p = self._forward(x_b)
                        preds.append(p.cpu().numpy())
                preds = np.concatenate(preds, axis=0)
                if self.task_type == TaskType.CLASSIFICATION:
                    metric = (preds.argmax(axis=1) == y_val).mean()
                    higher_better = True
                else:
                    metric = -float(np.mean((preds.squeeze() - y_val) ** 2))
                    higher_better = True  # MSE negated
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_state = {
                        k: v.detach().clone() for k, v in self.model_.state_dict().items()
                    }
                    patience_left = self._patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.task_type != TaskType.CLASSIFICATION:
            raise NotImplementedError("predict_proba only for classification")
        X = self._clean(X)
        self.model_.eval()
        parts = []
        with torch.no_grad():
            for x_b, _ in self._iter_batches(X, None, shuffle=False):
                logits = self._forward(x_b)
                parts.append(F.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(parts, axis=0)

    def predict(self, X) -> np.ndarray:
        if self.task_type == TaskType.CLASSIFICATION:
            return self.predict_proba(X).argmax(axis=1)
        X = self._clean(X)
        self.model_.eval()
        parts = []
        with torch.no_grad():
            for x_b, _ in self._iter_batches(X, None, shuffle=False):
                p = self._forward(x_b).squeeze(-1)
                parts.append(p.cpu().numpy())
        return np.concatenate(parts, axis=0)
