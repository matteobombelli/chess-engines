"""The pre-LN decoder-only transformer that predicts the next move token."""

from __future__ import annotations

import math
from typing import Any

from .errors import DependencyUnavailable

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # Operations that only inspect manifests must remain usable.
    torch = None
    Tensor = Any
    nn = None
    F = None


def require_torch() -> Any:
    if torch is None:
        raise DependencyUnavailable("training requires PyTorch; run `uv sync --extra train`")
    return torch


if nn is not None:

    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int, dropout: float):
            super().__init__()
            if d_model % n_heads:
                raise ValueError("d_model must be divisible by n_heads")
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.dropout = dropout
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.projection = nn.Linear(d_model, d_model, bias=False)
            self.residual_dropout = nn.Dropout(dropout)

        def forward(self, value: Tensor) -> Tensor:
            batch, length = value.shape[0], value.shape[1]
            # Split on the constant width, not the traced one, so the exported
            # graph carries no shape-derived constant.
            query, key, values = self.qkv(value).split(self.d_model, dim=2)
            shape = (batch, length, self.n_heads, self.head_dim)
            query = query.view(shape).transpose(1, 2)
            key = key.view(shape).transpose(1, 2)
            values = values.view(shape).transpose(1, 2)
            # is_causal builds the mask from the runtime length; the ONNX exporter
            # keeps that dynamic, so no explicit tril constant is needed.
            attention = F.scaled_dot_product_attention(
                query,
                key,
                values,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            attention = attention.transpose(1, 2).reshape(batch, length, self.d_model)
            return self.residual_dropout(self.projection(attention))

    class FeedForward(nn.Module):
        def __init__(self, d_model: int, d_ff: int, dropout: float):
            super().__init__()
            self.up = nn.Linear(d_model, d_ff, bias=False)
            self.down = nn.Linear(d_ff, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, value: Tensor) -> Tensor:
            return self.dropout(self.down(F.gelu(self.up(value))))

    class Block(nn.Module):
        def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attention = CausalSelfAttention(d_model, n_heads, dropout)
            self.ln2 = nn.LayerNorm(d_model)
            self.feed_forward = FeedForward(d_model, d_ff, dropout)

        def forward(self, value: Tensor) -> Tensor:
            value = value + self.attention(self.ln1(value))
            return value + self.feed_forward(self.ln2(value))

    class MiniGpt(nn.Module):
        """Tied-embedding GPT over the 4736-token move vocabulary."""

        def __init__(
            self,
            *,
            vocab: int = 4736,
            ctx: int = 256,
            d_model: int = 512,
            n_layers: int = 12,
            n_heads: int = 8,
            d_ff: int = 2048,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.vocab = vocab
            self.ctx = ctx
            self.token_embedding = nn.Embedding(vocab, d_model)
            self.position_embedding = nn.Embedding(ctx, d_model)
            self.embedding_dropout = nn.Dropout(dropout)
            self.blocks = nn.ModuleList(
                Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
            )
            self.ln_final = nn.LayerNorm(d_model)
            self.apply(self._initialize)
            # GPT-2's residual scaling: keep the variance of the residual stream
            # constant as depth grows.
            scaled = 0.02 / math.sqrt(2 * n_layers)
            for name, parameter in self.named_parameters():
                if name.endswith(("attention.projection.weight", "feed_forward.down.weight")):
                    nn.init.normal_(parameter, mean=0.0, std=scaled)

        @staticmethod
        def _initialize(module: Any) -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, tokens: Tensor) -> Tensor:
            length = tokens.shape[1]
            positions = torch.arange(length, device=tokens.device)
            value = self.token_embedding(tokens) + self.position_embedding(positions)
            value = self.embedding_dropout(value)
            for block in self.blocks:
                value = block(value)
            # The output head is the token embedding transposed.
            return F.linear(self.ln_final(value), self.token_embedding.weight)

else:

    class MiniGpt:  # pragma: no cover - only instantiated after require_torch
        def __init__(self, *_: Any, **__: Any):
            require_torch()


def build_model(config: Any) -> Any:
    require_torch()
    model = config.values["model"] if hasattr(config, "values") else config["model"]
    return MiniGpt(
        vocab=int(model["vocab"]),
        ctx=int(model["ctx"]),
        d_model=int(model["d_model"]),
        n_layers=int(model["n_layers"]),
        n_heads=int(model["n_heads"]),
        d_ff=int(model["d_ff"]),
        dropout=float(model["dropout"]),
    )


def parameter_count(model: Any) -> int:
    require_torch()
    return sum(parameter.numel() for parameter in model.parameters())
