"""Vanilla Transformer model for many-to-many fault classification.

This module implements a standard Transformer encoder architecture with
positional encoding for per-timestep fault diagnosis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from CESTA.models.base import BaseModel
from CESTA.models.temporal.positional import PositionalEncoding


class TransformerClassifier(BaseModel):
    """Vanilla Transformer encoder for many-to-many sequence classification.

    Architecture:
        Input -> Linear(input_size, d_model) -> PositionalEncoding
        -> N x TransformerEncoderLayer -> LayerNorm -> Dropout
        -> Linear(d_model, num_classes) -> Output

    Args:
        input_size: Number of input features per timestep.
        d_model: Dimension of the transformer hidden states.
        num_layers: Number of encoder layers.
        num_classes: Number of output classes (fault types).
        n_heads: Number of attention heads.
        d_ff: Dimension of the feed-forward layers.
        max_len: Maximum input sequence length (for positional encoding).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        n_heads: int = 4,
        d_ff: int = 128,
        max_len: int = 60,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_prob = dropout

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    @property
    def name(self) -> str:
        return "transformer"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for many-to-many classification.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits tensor of shape (batch, seq_len, num_classes).
        """
        hidden = self.input_proj(x)
        hidden = self.pos_encoding(hidden)
        hidden = self.encoder(hidden)
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
        }

