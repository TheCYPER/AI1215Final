"""
TabularTransformer wrapper implementing our BaseModel interface.

Adapts a teammate's FT-Transformer-style backbone
(`modeling/_tabular_transformer.py`) to the fit / predict / predict_proba
contract used by `CrossValidator` and `EnsembleModel`.

Design decisions vs. the teammate's pipeline:
- **Device**: CUDA > MPS > CPU auto-selection, with amp/GradScaler enabled
  only on CUDA. The teammate required CUDA; we downgrade gracefully so
  smoke tests and ensembles can run on macOS.
- **cat_idxs / cat_dims**: inferred from `categorical_feature` kwarg if
  provided (our CV passes it from `get_categorical_feature_indices`). Dims
  are computed as `max(X[:, idx]) + 2`, matching our TabNet wrapper's
  strategy for absorbing test-time unknowns.
- **NaN / inf guards**: `np.nan_to_num` on both X and y (teammate used
  median imputation upstream; our ColumnTransformer already fills NaNs
  but a dense cast still leaks the odd inf through log-transforms).
- **Regression target**: standardized inside fit() from train rows only;
  `predict()` inverts the standardization so the caller sees raw R²-ready
  values, mirroring the teammate's `pre.transform_target / inverse_target`.
- **CORN ordinal head**: optional. When `use_ordinal=True` the classification
  forward emits `K-1` logits and the loss uses `corn_loss`; `predict_proba`
  converts back to per-class probabilities so ensemble stacking stays uniform.

The search-space / tune-time param shape follows our other NN wrappers
(CoralMLPModel, TabNetModel): a flat dict of scalars, with `max_epochs`,
`patience`, `batch_size`, and optimizer-level keys extracted via `params.pop`
before the backbone is constructed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import TaskType
from modeling._ordinal_heads import corn_class_probs, corn_decode, corn_loss
from modeling._tabular_transformer import TabularTransformer
from modeling.base_model import BaseModel


def _pick_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the strongest available device, unless caller forces one.

    MPS is intentionally skipped in auto-pick mode — our transformer forward
    pass hits a PyTorch MPS kernel path that produces NaN probabilities
    intermittently (observed on torch 2.11, macOS 24.6). Callers can still
    opt-in with `device="mps"` if they want to experiment.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _amp_autocast(device: torch.device):
    """Return a context manager for mixed precision. CPU/MPS: no-op."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    import contextlib

    return contextlib.nullcontext()


class _EarlyStopper:
    """Max-mode early stopping with best-state retention (CPU-side tensors)."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.stopped = False

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {
                k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped


class TabularTransformerModel(BaseModel):
    """FT-Transformer-style backbone wrapped as a BaseModel.

    Classification: CE loss (default) or CORN ordinal loss (`use_ordinal=True`).
    Regression: Huber loss on a standardized target; predict() returns raw values.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        self.task_type = task_type

        params = dict(self.config)
        # Training-loop knobs — popped out before the backbone sees params.
        self._max_epochs: int = int(params.pop("max_epochs", 60))
        self._patience: int = int(params.pop("patience", 10))
        self._batch_size: int = int(params.pop("batch_size", 256))
        self._lr: float = float(params.pop("lr", 7e-4))
        self._weight_decay: float = float(params.pop("weight_decay", 1e-4))
        self._grad_clip: float = float(params.pop("grad_clip", 1.0))
        self._huber_delta: float = float(params.pop("huber_delta", 1.0))
        self._seed: int = int(params.pop("seed", 42))
        self._device_str: Optional[str] = params.pop("device", None)
        self._num_workers: int = int(params.pop("num_workers", 0))

        # Backbone knobs — consumed by TabularTransformer.
        self._d_model: int = int(params.pop("d_model", 128))
        self._n_heads: int = int(params.pop("n_heads", 4))
        self._n_layers: int = int(params.pop("n_layers", 2))
        self._d_ff: int = int(params.pop("d_ff", 256))
        self._dropout: float = float(params.pop("dropout", 0.15))
        self._num_embedding_type: str = str(params.pop("num_embedding_type", "numerical"))
        self._use_cls_token: bool = bool(params.pop("use_cls_token", True))
        self._pooling: str = str(params.pop("pooling", "cls"))
        self._pre_norm: bool = bool(params.pop("pre_norm", True))
        self._use_column_embedding: bool = bool(params.pop("use_column_embedding", False))
        self._use_ordinal: bool = bool(params.pop("use_ordinal", False))

        if params:
            # Leftover keys would silently get ignored — surface them.
            raise ValueError(
                f"Unknown TabularTransformerModel params: {sorted(params.keys())}"
            )

        self._num_classes: int = 5
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None
        # Scaler stats cover only the *non-categorical* column indices.
        # Categoricals keep their integer codes intact so the embedding lookup works.
        self._scaler_mean_: Optional[np.ndarray] = None
        self._scaler_std_: Optional[np.ndarray] = None
        self._cat_idxs_: List[int] = []
        self._cat_dims_: List[int] = []
        self._num_idxs_: List[int] = []
        self._n_features_: Optional[int] = None

    # --------- BaseModel API ---------

    def build_model(self, num_classes: int = 5, **kwargs) -> "TabularTransformerModel":
        # Thread caps identical to CoralMLP / TabNet — joblib + OMP segfaults on macOS.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        self._num_classes = num_classes
        # Backbone is built lazily in fit() once we've seen X.shape and cat info.
        return self

    def fit(
        self,
        X,
        y,
        eval_set=None,
        sample_weight=None,
        categorical_feature: Optional[List[int]] = None,
        **kwargs,
    ):
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        device = _pick_device(self._device_str)

        # Resolve cat / num splits first so standardization can skip cat columns.
        X_raw = np.asarray(X, dtype=np.float32)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        self._n_features_ = X_raw.shape[1]

        cat_idxs = sorted(int(i) for i in (categorical_feature or []))
        num_idxs = [i for i in range(self._n_features_) if i not in set(cat_idxs)]
        self._cat_idxs_ = cat_idxs
        self._num_idxs_ = num_idxs

        cat_dims: List[int] = []
        for i in cat_idxs:
            # +2 slack so val / test rows with unseen codes can still be looked up
            # (mirrors our TabNet wrapper's strategy).
            max_code = int(np.rint(X_raw[:, i]).max()) if X_raw.shape[0] else 0
            cat_dims.append(max_code + 2)
        self._cat_dims_ = cat_dims

        # Per-feature standardization for numeric columns only.
        self._scaler_mean_ = np.zeros((1, self._n_features_), dtype=np.float32)
        self._scaler_std_ = np.ones((1, self._n_features_), dtype=np.float32)
        if num_idxs:
            num_cols = X_raw[:, num_idxs]
            self._scaler_mean_[0, num_idxs] = num_cols.mean(axis=0)
            std = num_cols.std(axis=0)
            std[std < 1e-6] = 1.0
            self._scaler_std_[0, num_idxs] = std

        X_np = self._apply_scale(X_raw)
        y_np = self._prepare_y(y)

        # Auto-fix n_heads divisibility instead of crashing — trivial rescue
        # when a bad config slips through tune / search spaces.
        n_heads = self._n_heads
        if self._d_model % n_heads != 0:
            for candidate in (4, 8, 2, 1):
                if self._d_model % candidate == 0:
                    n_heads = candidate
                    break

        backbone = TabularTransformer(
            n_features=self._n_features_,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            n_classes=self._num_classes,
            d_model=self._d_model,
            n_heads=n_heads,
            n_layers=self._n_layers,
            d_ff=self._d_ff,
            dropout=self._dropout,
            use_cls_token=self._use_cls_token,
            pooling=self._pooling,
            pre_norm=self._pre_norm,
            num_embedding_type=self._num_embedding_type,
            task=("classification" if self.task_type == TaskType.CLASSIFICATION else "regression"),
            use_column_embedding=self._use_column_embedding,
            use_ordinal=self._use_ordinal,
        ).to(device)
        self.model_ = backbone
        self._device_ = device

        # Standardize regression targets using train-only stats.
        if self.task_type == TaskType.REGRESSION:
            self._target_mean = float(y_np.mean())
            std = float(y_np.std())
            self._target_std = std if std > 1e-8 else 1.0
            y_train = (y_np - self._target_mean) / self._target_std
        else:
            y_train = y_np

        # Build eval tensors up-front so we can early-stop on val accuracy / R².
        X_val_np: Optional[np.ndarray] = None
        y_val_raw: Optional[np.ndarray] = None
        if eval_set is not None:
            X_val_raw, y_val_raw_ = eval_set[0] if isinstance(eval_set, list) else eval_set
            X_val_np = self._apply_scale(
                np.nan_to_num(
                    np.asarray(X_val_raw, dtype=np.float32),
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                )
            )
            y_val_raw = np.asarray(y_val_raw_)

        optimizer = torch.optim.AdamW(
            backbone.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=max(1, self._patience // 3)
        )
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        stopper = _EarlyStopper(self._patience)

        n = X_np.shape[0]
        rng = np.random.default_rng(self._seed)

        for epoch in range(self._max_epochs):
            backbone.train()
            perm = rng.permutation(n)
            for start in range(0, n, self._batch_size):
                idx = perm[start : start + self._batch_size]
                xb = torch.from_numpy(X_np[idx]).to(device, non_blocking=True)
                yb = torch.from_numpy(y_train[idx]).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with _amp_autocast(device):
                    out = backbone(xb)
                    loss = self._loss(out, yb)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if self._grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            backbone.parameters(), self._grad_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self._grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            backbone.parameters(), self._grad_clip
                        )
                    optimizer.step()

            if X_val_np is not None and y_val_raw is not None:
                val_score = self._score_val(backbone, X_val_np, y_val_raw, device)
                scheduler.step(val_score)
                if stopper.step(val_score, backbone):
                    break

        if stopper.best_state is not None:
            backbone.load_state_dict(
                {k: v.to(device) for k, v in stopper.best_state.items()}
            )

        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        logits = self._forward_logits(X)
        if self.task_type == TaskType.CLASSIFICATION:
            if self._use_ordinal:
                return corn_decode(torch.from_numpy(logits)).numpy()
            return logits.argmax(axis=1)
        # Regression: un-standardize before returning.
        z = logits.reshape(-1)
        return z * (self._target_std or 1.0) + (self._target_mean or 0.0)

    def predict_proba(self, X) -> np.ndarray:
        if self.task_type != TaskType.CLASSIFICATION:
            raise NotImplementedError(
                "predict_proba is classification-only on TabularTransformerModel"
            )
        logits = self._forward_logits(X)
        if self._use_ordinal:
            probs = corn_class_probs(torch.from_numpy(logits), self._num_classes).numpy()
            return probs
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    # --------- Internals ---------

    def _apply_scale(self, X_raw: np.ndarray) -> np.ndarray:
        """Standardize numeric columns only; preserve categorical integer codes.

        Also clips cat values to `[0, cat_dim - 1]` so an unseen test-time code
        maps to the reserved slack index rather than indexing out of range.
        """
        X_np = (X_raw - self._scaler_mean_) / self._scaler_std_
        if self._cat_idxs_:
            for i, dim in zip(self._cat_idxs_, self._cat_dims_):
                col = np.rint(X_raw[:, i]).astype(np.int64)
                col = np.clip(col, 0, dim - 1)
                X_np[:, i] = col.astype(np.float32)
        return X_np

    def _prepare_y(self, y) -> np.ndarray:
        if y is None:
            return None
        if self.task_type == TaskType.CLASSIFICATION:
            return np.asarray(y).astype(np.int64)
        y_np = np.asarray(y, dtype=np.float32)
        return np.nan_to_num(y_np, nan=0.0, posinf=1e6, neginf=-1e6)

    def _loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.task_type == TaskType.CLASSIFICATION:
            if self._use_ordinal:
                return corn_loss(out, y, self._num_classes)
            return F.cross_entropy(out, y.long())
        return F.huber_loss(out, y.float(), delta=self._huber_delta)

    def _forward_logits(self, X) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fit before predict / predict_proba")
        X_raw = np.asarray(X, dtype=np.float32)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        X_np = self._apply_scale(X_raw)
        device = self._device_
        self.model_.eval()
        outs = []
        with torch.no_grad(), _amp_autocast(device):
            for start in range(0, X_np.shape[0], self._batch_size):
                xb = torch.from_numpy(X_np[start : start + self._batch_size]).to(
                    device, non_blocking=True
                )
                out = self.model_(xb)
                outs.append(out.float().cpu())
        return torch.cat(outs, dim=0).numpy()

    def _score_val(
        self,
        backbone: nn.Module,
        X_val: np.ndarray,
        y_val_raw: np.ndarray,
        device: torch.device,
    ) -> float:
        backbone.eval()
        outs = []
        with torch.no_grad(), _amp_autocast(device):
            for start in range(0, X_val.shape[0], self._batch_size):
                xb = torch.from_numpy(X_val[start : start + self._batch_size]).to(
                    device, non_blocking=True
                )
                outs.append(backbone(xb).float().cpu())
        out_np = torch.cat(outs, dim=0).numpy()

        if self.task_type == TaskType.CLASSIFICATION:
            if self._use_ordinal:
                pred = corn_decode(torch.from_numpy(out_np)).numpy()
            else:
                pred = out_np.argmax(axis=1)
            return float((pred == y_val_raw).mean())

        # Regression: maximize R² on the raw (un-standardized) scale.
        y_pred = out_np.reshape(-1) * (self._target_std or 1.0) + (self._target_mean or 0.0)
        ss_res = float(((y_val_raw - y_pred) ** 2).sum())
        ss_tot = float(((y_val_raw - y_val_raw.mean()) ** 2).sum())
        if ss_tot < 1e-12:
            return 0.0
        return 1.0 - ss_res / ss_tot
