"""
CORN ordinal-regression helpers.

Ported from the teammate's `src/common/ordinal.py` so the Transformer wrapper
doesn't depend on `coral_pytorch` at the loss/decoding layer. Our CoralMLPModel
still uses `coral_pytorch.losses.corn_loss`; they behave identically. Having a
local copy here lets the Transformer stay importable even if coral-pytorch is
missing or version-pinned differently.

Reference: Shi, Cao, Raschka — "Deep Neural Networks for Rank-Consistent Ordinal
Regression Based On Conditional Probabilities" (arXiv:2111.08851).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def corn_loss(logits: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Mean over K-1 conditional binary cross-entropy terms.

    The k-th head is trained only on samples where `y >= k` — those that
    "reached" the k-th decision point. This conditional training is what gives
    CORN rank-consistent probabilities at inference time.
    """
    if logits.size(1) != n_classes - 1:
        raise ValueError(
            f"corn_loss expected logits with {n_classes - 1} columns, got {logits.size(1)}"
        )
    y = y.long()
    losses = []
    for k in range(n_classes - 1):
        mask = y >= k
        if mask.sum() == 0:
            continue
        target_k = (y[mask] > k).to(logits.dtype)
        loss_k = F.binary_cross_entropy_with_logits(logits[mask, k], target_k)
        losses.append(loss_k)
    # y >= 0 always holds for a valid label batch, so `losses` is non-empty.
    return torch.stack(losses).mean()


def corn_cum_probs(logits: torch.Tensor) -> torch.Tensor:
    """Chain conditional sigmoids into P(y > k) via cumulative product."""
    cond = torch.sigmoid(logits.float())
    return torch.cumprod(cond, dim=1)


def corn_decode(logits: torch.Tensor) -> torch.Tensor:
    """Integer class prediction from CORN logits via 0.5-threshold on P(y > k)."""
    return (corn_cum_probs(logits) > 0.5).sum(dim=1).long()


def corn_class_probs(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert CORN cumulative P(y>k) into per-class P(y=k). Rows sum to 1."""
    cum = corn_cum_probs(logits)
    batch = logits.size(0)
    device = logits.device
    probs = torch.zeros(batch, n_classes, device=device, dtype=cum.dtype)
    probs[:, 0] = 1.0 - cum[:, 0]
    for k in range(1, n_classes - 1):
        probs[:, k] = cum[:, k - 1] - cum[:, k]
    probs[:, n_classes - 1] = cum[:, n_classes - 2]
    probs = probs.clamp_min(0.0)
    row_sum = probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs / row_sum
