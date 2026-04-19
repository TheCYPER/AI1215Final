"""CORN-MLP ordinal classifier.

Neural network with CORN (Conditional Ordinal Regression for Neural Networks)
output head. Unlike softmax MLP, the head emits K-1 logits and the loss
enforces the ordinal constraint, so the model's errors cluster on adjacent
classes — matching the structure of the RiskTier dataset.

Why CORN over CORAL:
- CORN handles the rank-consistency issue implicitly in the loss
  (uses conditional probabilities P(y>k | y>=k) whose product gives
  rank-consistent probabilities by construction).
- CORAL needs `levels_from_labelbatch` preprocessing.
- Both are from the same authors (Raschka & co.); CORN is a refinement.

Why MLP backbone (not TabNet):
- pytorch-tabnet's internals are hard to swap the loss + head cleanly.
- MLP is a different architecture from TabNet AND from GBMs, so it
  brings genuinely new error geometry to the ensemble.

Reference: Xintong Shi, Wenzhi Cao, Sebastian Raschka. Deep Neural Networks
for Rank-Consistent Ordinal Regression Based On Conditional Probabilities.
arXiv:2111.08851 (2021).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coral_pytorch.losses import corn_loss

from configs.config import TaskType
from modeling.base_model import BaseModel


class _MLPBackbone(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_sizes: List[int],
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)
        # CORN head: K-1 output logits
        self.head = nn.Linear(prev, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class CoralMLPModel(BaseModel):
    """MLP with CORN ordinal output head."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        if task_type != TaskType.CLASSIFICATION:
            raise ValueError("CoralMLPModel is classification-only (ordinal targets).")
        self.task_type = task_type
        self._num_classes: int = 5
        self._device = torch.device("cpu")

        params = dict(self.config)
        self._hidden_sizes: List[int] = list(
            params.pop("hidden_sizes", [256, 128, 64])
        )
        self._dropout: float = float(params.pop("dropout", 0.2))
        self._max_epochs: int = int(params.pop("max_epochs", 80))
        self._patience: int = int(params.pop("patience", 12))
        self._batch_size: int = int(params.pop("batch_size", 1024))
        self._lr: float = float(params.pop("lr", 1e-3))
        self._weight_decay: float = float(params.pop("weight_decay", 1e-4))
        self._seed: int = int(params.pop("seed", 42))

    def build_model(self, num_classes: int = 5, **kwargs) -> "CoralMLPModel":
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        self._num_classes = num_classes
        return self

    @staticmethod
    def _clean(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    def _iter_batches(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        shuffle: bool,
    ):
        n = len(X)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(self._seed)
            rng.shuffle(idx)
        for start in range(0, n, self._batch_size):
            sel = idx[start : start + self._batch_size]
            x_batch = torch.from_numpy(X[sel]).to(self._device)
            y_batch = None if y is None else torch.from_numpy(y[sel]).to(self._device)
            yield x_batch, y_batch

    def fit(
        self,
        X,
        y,
        eval_set=None,
        sample_weight=None,
        categorical_feature=None,
        **kwargs,
    ):
        torch.manual_seed(self._seed)
        X = self._clean(X)
        # Internal standardization — credit_features + target-encoded cols
        # + ordinal-categoricals bypass the upstream StandardScaler, leaving
        # the matrix with mixed scales (std ~33000 observed). CORN-MLP with
        # lr=1e-3 blows up without this; CatBoost / XGBoost are scale-invariant
        # so the shared preprocessor doesn't bother.
        self._scaler_mean_ = X.mean(axis=0, keepdims=True).astype(np.float32)
        self._scaler_std_ = X.std(axis=0, keepdims=True).astype(np.float32)
        self._scaler_std_[self._scaler_std_ < 1e-6] = 1.0
        X = (X - self._scaler_mean_) / self._scaler_std_
        y_arr = np.asarray(y).astype(np.int64)
        n_features = X.shape[1]

        self.model_ = _MLPBackbone(
            n_features=n_features,
            hidden_sizes=self._hidden_sizes,
            dropout=self._dropout,
            num_classes=self._num_classes,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        X_val, y_val = None, None
        if eval_set is not None:
            X_val_raw, y_val_raw = (
                eval_set[0] if isinstance(eval_set, list) else eval_set
            )
            X_val = self._clean(X_val_raw)
            X_val = (X_val - self._scaler_mean_) / self._scaler_std_
            y_val = np.asarray(y_val_raw).astype(np.int64)

        best_acc: Optional[float] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_left = self._patience
        train_rng = np.random.default_rng(self._seed)

        for epoch in range(self._max_epochs):
            self.model_.train()
            # Re-shuffle every epoch (distinct permutation from train_rng)
            perm = train_rng.permutation(len(X))
            n = len(X)
            for start in range(0, n, self._batch_size):
                sel = perm[start : start + self._batch_size]
                x_batch = torch.from_numpy(X[sel]).to(self._device)
                y_batch = torch.from_numpy(y_arr[sel]).to(self._device)
                optimizer.zero_grad()
                logits = self.model_(x_batch)
                loss = corn_loss(logits, y_batch, self._num_classes)
                loss.backward()
                optimizer.step()

            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = []
                    for x_b, _ in self._iter_batches(X_val, None, shuffle=False):
                        val_logits.append(self.model_(x_b).cpu().numpy())
                val_logits = np.concatenate(val_logits, axis=0)
                val_pred = self._logits_to_labels(val_logits)
                acc = (val_pred == y_val).mean()
                if best_acc is None or acc > best_acc:
                    best_acc = acc
                    best_state = {
                        k: v.detach().clone()
                        for k, v in self.model_.state_dict().items()
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

    @staticmethod
    def _logits_to_probs(logits: np.ndarray, num_classes: int) -> np.ndarray:
        """CORN: conditional probabilities P(y>k | y>=k) = sigmoid(logit_k).

        P(y>k) = prod_{j<=k} sigmoid(logit_j)     (cumulative product)
        P(y=0) = 1 - P(y>0)
        P(y=k) = P(y>k-1) - P(y>k)  for 0 < k < K-1
        P(y=K-1) = P(y>K-2)
        """
        # sigmoid of each conditional logit
        cond = 1.0 / (1.0 + np.exp(-logits))            # shape (n, K-1)
        # cumulative product => P(y > k)
        cum = np.cumprod(cond, axis=1)                   # rank-consistent by construction
        n = logits.shape[0]
        probs = np.zeros((n, num_classes), dtype=np.float64)
        probs[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, num_classes - 1):
            probs[:, k] = cum[:, k - 1] - cum[:, k]
        probs[:, num_classes - 1] = cum[:, num_classes - 2]
        probs = np.clip(probs, 0.0, 1.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return probs / row_sum

    def _logits_to_labels(self, logits: np.ndarray) -> np.ndarray:
        probs = self._logits_to_probs(logits, self._num_classes)
        return probs.argmax(axis=1)

    def predict_proba(self, X) -> np.ndarray:
        X = self._clean(X)
        X = (X - self._scaler_mean_) / self._scaler_std_
        self.model_.eval()
        parts = []
        with torch.no_grad():
            for x_b, _ in self._iter_batches(X, None, shuffle=False):
                parts.append(self.model_(x_b).cpu().numpy())
        logits = np.concatenate(parts, axis=0)
        return self._logits_to_probs(logits, self._num_classes)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
