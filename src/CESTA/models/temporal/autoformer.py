"""Autoformer model for many-to-many fault classification.

This module wraps Autoformer encoder layers from the HuggingFace
transformers library with a linear classification head to perform
per-timestep fault diagnosis. Only the encoder is used — the
decoder (designed for forecasting) is not needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from CESTA.models.base import BaseModel
from CESTA.models.temporal.positional import PositionalEncoding


class AutoformerClassifier(BaseModel):
    """Autoformer model for many-to-many sequence classification.

    Uses AutoformerEncoderLayer blocks (with auto-correlation attention
    and series decomposition) from HuggingFace, preceded by a linear
    input projection and followed by a classification head.

    Architecture:
        Input -> Linear(input_size, d_model) -> N x AutoformerEncoderLayer
        -> Dropout -> Linear(d_model, num_classes) -> Output

    Args:
        input_size: Number of input features per timestep.
        d_model: Dimension of the transformer hidden states.
        num_layers: Number of encoder layers.
        num_classes: Number of output classes (fault types).
        n_heads: Number of attention heads.
        d_ff: Dimension of the feed-forward layers.
        max_len: Maximum input sequence length (for positional encoding).
        dropout: Dropout probability.
        moving_average: Window size for the series decomposition.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        num_layers: int = 1,
        num_classes: int = 4,
        n_heads: int = 4,
        d_ff: int = 64,
        max_len: int = 60,
        dropout: float = 0.1,
        moving_average: int = 5,
    ) -> None:
        super().__init__()

        from transformers import AutoformerConfig
        from transformers.models.autoformer.modeling_autoformer import (
            AutoformerEncoderLayer,
        )

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout_prob = dropout
        self.moving_average = moving_average

        hf_config = AutoformerConfig(
            d_model=d_model,  # pyright: ignore[reportCallIssue]
            encoder_attention_heads=n_heads,  # pyright: ignore[reportCallIssue]
            encoder_ffn_dim=d_ff,  # pyright: ignore[reportCallIssue]
            dropout=dropout,  # pyright: ignore[reportCallIssue]
            activation_dropout=dropout,  # pyright: ignore[reportCallIssue]
            attention_dropout=dropout,  # pyright: ignore[reportCallIssue]
            moving_average=moving_average,  # pyright: ignore[reportCallIssue]
        )

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.layers = nn.ModuleList(
            [AutoformerEncoderLayer(hf_config) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    @property
    def name(self) -> str:
        return "autoformer"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for many-to-many classification.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits tensor of shape (batch, seq_len, num_classes).
        """
        hidden = self.input_proj(x)
        hidden = self.pos_encoding(hidden)
        for layer in self.layers:
            layer_out = layer(hidden, attention_mask=None)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        hidden = self.layer_norm(hidden)
        hidden = self.dropout_layer(hidden)
        logits = self.fc(hidden)
        return logits

    def get_config(self) -> dict[str, object]:
        """Return model configuration for serialization."""
        return {
            "input_size": self.input_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout_prob,
            "moving_average": self.moving_average,
        }

