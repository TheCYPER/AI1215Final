"""
TabularTransformer backbone (FT-Transformer-style).

Ported from a teammate's `src/common/transformer_model.py`. This module holds
only the pure `nn.Module` classes — no training loops, no device logic, no
CUDA-only dependencies. The sklearn-style wrapper that plugs into our
CrossValidator lives in `modeling/tabular_transformer_model.py`.

Architecture notes:
- Feature-token embeddings: categoricals via lookup, numerics via one of
  {linear, numerical, periodic, PLE}. The `periodic` and `ple` variants come
  from Gorishniy et al. 2022 (NeurIPS).
- Transformer blocks: pre-norm MHSA + GELU FFN, standard xavier init with
  reduced gain on the output projections for training stability.
- Pooling: either a dedicated CLS token or mean-pool over feature tokens.
- Two heads: either `n_classes` softmax logits (classification) or a single
  scalar (regression). CORN ordinal head (`use_ordinal=True`) emits `K-1`
  logits instead, consumed by `_ordinal_heads.corn_loss`.

References:
- Vaswani et al., "Attention Is All You Need" (2017)
- Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data"
  (FT-Transformer, 2021)
- Gorishniy et al., "On Embeddings for Numerical Features in Tabular DL"
  (NeurIPS 2022) — periodic + PLE embeddings
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicEmbedding(nn.Module):
    """FT-Transformer periodic embedding for numerical features."""

    def __init__(self, d_model: int, n_frequencies: int = 48, sigma: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * sigma)
        self.projection = nn.Linear(2 * n_frequencies, d_model)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        angles = 2 * math.pi * x * self.frequencies
        periodic = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.projection(periodic)


class PiecewiseLinearEmbedding(nn.Module):
    """Soft-binned numerical embedding with learnable boundaries."""

    def __init__(self, n_features: int, d_model: int, n_bins: int = 32):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.d_model = d_model

        boundaries = torch.linspace(-3, 3, n_bins).unsqueeze(0).expand(n_features, -1)
        self.boundaries = nn.Parameter(boundaries.clone())
        self.bin_embeddings = nn.Parameter(
            torch.randn(n_features, n_bins + 1, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        boundaries, _ = torch.sort(self.boundaries, dim=-1)

        x_expanded = x.unsqueeze(-1)
        bin_weights = torch.sigmoid((x_expanded - boundaries) * 5.0)

        ones = torch.ones(
            batch_size, self.n_features, 1, device=x.device, dtype=bin_weights.dtype
        )
        zeros = torch.zeros(
            batch_size, self.n_features, 1, device=x.device, dtype=bin_weights.dtype
        )
        bin_weights = torch.cat([ones, bin_weights, zeros], dim=-1)
        weights = bin_weights[:, :, :-1] - bin_weights[:, :, 1:]

        if (
            weights.ndim != 3
            or self.bin_embeddings.ndim != 3
            or weights.shape[1] != self.bin_embeddings.shape[0]
            or weights.shape[2] != self.bin_embeddings.shape[1]
        ):
            raise RuntimeError(
                f"PLE shape mismatch: weights={tuple(weights.shape)}, "
                f"bin_embeddings={tuple(self.bin_embeddings.shape)}"
            )
        return torch.einsum("bfk,fkd->bfd", weights, self.bin_embeddings)


class NumericalEmbedding(nn.Module):
    """Per-feature learnable linear projection: x * weight + bias."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_features, d_model))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        return x * self.weight + self.bias


class FeatureEmbedding(nn.Module):
    """Mixed categorical (learned lookup) + numerical (configurable) embedding."""

    def __init__(
        self,
        n_features: int,
        cat_idxs: List[int],
        cat_dims: List[int],
        d_model: int,
        num_embedding_type: str = "numerical",
        n_frequencies: int = 48,
        n_bins: int = 32,
        use_column_embedding: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.num_embedding_type = num_embedding_type
        self.use_column_embedding = use_column_embedding

        self.cat_idxs = list(cat_idxs)
        self.num_idxs = [i for i in range(n_features) if i not in set(self.cat_idxs)]
        self.n_num = len(self.num_idxs)

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(dim, d_model) for dim in cat_dims]
        )

        if self.n_num > 0:
            if num_embedding_type == "numerical":
                self.num_embedding = NumericalEmbedding(self.n_num, d_model)
            elif num_embedding_type == "periodic":
                self.num_embedding = PeriodicEmbedding(d_model, n_frequencies)
            elif num_embedding_type == "ple":
                self.num_embedding = PiecewiseLinearEmbedding(
                    self.n_num, d_model, n_bins
                )
            elif num_embedding_type == "linear":
                self.num_embedding = nn.Linear(1, d_model)
            else:
                raise ValueError(f"Unknown num_embedding_type: {num_embedding_type}")

        if use_column_embedding:
            self.col_embedding = nn.Embedding(n_features, d_model)
            self.type_embedding = nn.Embedding(2, d_model)
            nn.init.normal_(self.col_embedding.weight, std=0.02)
            nn.init.normal_(self.type_embedding.weight, std=0.02)
            type_ids = torch.zeros(n_features, dtype=torch.long)
            for i in self.cat_idxs:
                type_ids[i] = 1
            self.register_buffer("type_ids", type_ids, persistent=True)
            self.register_buffer(
                "col_ids", torch.arange(n_features, dtype=torch.long), persistent=True
            )

        self.embed_norm = nn.LayerNorm(d_model)
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        embed_dtype = self.embed_norm.weight.dtype
        embeddings = torch.zeros(
            batch_size, self.n_features, self.d_model, device=device, dtype=embed_dtype
        )

        for feat_idx, emb_layer in zip(self.cat_idxs, self.cat_embeddings):
            cat_emb = emb_layer(x[:, feat_idx].long()).to(dtype=embed_dtype)
            embeddings[:, feat_idx, :] = cat_emb

        if self.n_num > 0:
            num_vals = x[:, self.num_idxs]
            if self.num_embedding_type == "linear":
                num_embedded = self.num_embedding(num_vals.unsqueeze(-1))
            else:
                num_embedded = self.num_embedding(num_vals)
            num_embedded = num_embedded.to(dtype=embed_dtype)
            embeddings[:, self.num_idxs, :] = num_embedded

        if self.use_column_embedding:
            col_emb = self.col_embedding(self.col_ids).to(embed_dtype)
            type_emb = self.type_embedding(self.type_ids).to(embed_dtype)
            embeddings = embeddings + (col_emb + type_emb).unsqueeze(0)

        return self.embed_norm(embeddings)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        for module in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.W_o.weight, gain=0.5)
        nn.init.zeros_(self.W_o.bias)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        Q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = (
            torch.matmul(attn, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.5)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.dropout1(self.attention(self.norm1(x)))
            x = x + self.dropout2(self.ff(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self.attention(x)))
            x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class TabularTransformer(nn.Module):
    """
    Transformer encoder for tabular data with a task-parametric output head.

    task="classification" — forward returns `(B, n_classes)` logits (use
        `cross_entropy`) unless `use_ordinal=True`, in which case it returns
        `(B, n_classes - 1)` CORN logits (use `_ordinal_heads.corn_loss`).
    task="regression"     — forward returns `(B,)` raw scalar (use `MSE` or
        `Huber`; the wrapper inverse-standardizes for metric reporting).
    """

    def __init__(
        self,
        n_features: int,
        cat_idxs: List[int],
        cat_dims: List[int],
        n_classes: Optional[int] = None,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        pooling: str = "cls",
        pre_norm: bool = True,
        num_embedding_type: str = "numerical",
        task: str = "classification",
        use_column_embedding: bool = False,
        use_ordinal: bool = False,
    ):
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be classification or regression, got {task}"
            )
        if task == "classification" and not n_classes:
            raise ValueError("n_classes must be set for classification")
        if use_ordinal and task != "classification":
            raise ValueError("use_ordinal is only valid with task='classification'")

        self.task = task
        self.n_features = n_features
        self.use_cls_token = use_cls_token
        self.pooling = pooling
        self.d_model = d_model
        self.use_column_embedding = use_column_embedding
        self.use_ordinal = use_ordinal
        self.n_classes = n_classes

        self.embedding = FeatureEmbedding(
            n_features=n_features,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            d_model=d_model,
            num_embedding_type=num_embedding_type,
            use_column_embedding=use_column_embedding,
        )

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, pre_norm)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

        if task == "classification":
            out_dim = (n_classes - 1) if use_ordinal else n_classes
        else:
            out_dim = 1
        self.head = self._build_head(d_model, out_dim, dropout)

    @staticmethod
    def _build_head(d_model: int, out_dim: int, dropout: float) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )
        for module in head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        return head

    def encode(
        self, x: torch.Tensor, return_tokens: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)
        x = self.embedding(x)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        if self.pooling == "cls" and self.use_cls_token:
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)

        if not return_tokens:
            return pooled

        token_start = 1 if self.use_cls_token else 0
        feature_tokens = x[:, token_start:, :]
        return pooled, feature_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(x, return_tokens=False)
        out = self.head(pooled)
        if self.task == "regression":
            out = out.squeeze(-1)
        return out
