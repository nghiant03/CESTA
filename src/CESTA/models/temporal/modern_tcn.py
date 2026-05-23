"""ModernTCN model for many-to-many fault classification.

This module implements a ModernTCN-style temporal convolutional network
with depthwise temporal convolutions, pointwise channel mixing, LayerNorm,
and residual connections for per-timestep fault diagnosis.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from CESTA.models.base import BaseModel


class TimeChannelLayerNorm(nn.Module):
    """Apply LayerNorm over channels for Conv1d feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a ``(batch, channels, seq_len)`` tensor over channels."""
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ModernTCNBlock(nn.Module):
    """Residual ModernTCN block with depthwise temporal mixing and MLP."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        expansion_ratio: float,
        dropout: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        hidden_channels = max(channels, int(channels * expansion_ratio))
        padding = ((kernel_size - 1) // 2) * dilation

        self.norm = TimeChannelLayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise_in = nn.Conv1d(channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pointwise_out = nn.Conv1d(hidden_channels, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.full((channels,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single ModernTCN block."""
        residual = x
        hidden = self.norm(x)
        hidden = self.depthwise(hidden)
        hidden = self.pointwise_in(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.pointwise_out(hidden)
        hidden = self.dropout(hidden)
        hidden = hidden * self.layer_scale.view(1, -1, 1)
        return residual + hidden


class ModernTCNClassifier(BaseModel):
    """ModernTCN for many-to-many sequence classification.

    Architecture:
        Input -> Linear(input_size, hidden_size)
        -> N x [LayerNorm -> depthwise Conv1d -> pointwise MLP -> residual]
        -> Dropout -> Linear(hidden_size, num_classes) -> Output

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Hidden channel width for temporal blocks.
        num_blocks: Number of stacked ModernTCN blocks.
        kernel_size: Temporal kernel size for depthwise convolutions.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        expansion_ratio: Expansion ratio for the pointwise MLP.
        dilation_base: Exponential base for block dilation growth.
        layer_scale_init: Initial residual layer-scale value.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 15,
        num_classes: int = 4,
        dropout: float = 0.1,
        expansion_ratio: float = 2.0,
        dilation_base: int = 2,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            msg = "kernel_size must be a positive odd integer"
            raise ValueError(msg)
        if dilation_base < 1:
            msg = "dilation_base must be >= 1"
            raise ValueError(msg)
        if expansion_ratio < 1.0:
            msg = "expansion_ratio must be >= 1.0"
            raise ValueError(msg)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.expansion_ratio = expansion_ratio
        self.dilation_base = dilation_base
        self.layer_scale_init = layer_scale_init

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(
                    hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation_base**index,
                    expansion_ratio=expansion_ratio,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                )
                for index in range(num_blocks)
            ]
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    @property
    def name(self) -> str:
        return "modern_tcn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for many-to-many classification.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Logits tensor of shape ``(batch, seq_len, num_classes)``.
        """
        hidden = self.input_proj(x)
        hidden = hidden.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden.transpose(1, 2)
        hidden = self.dropout_layer(hidden)
        return self.fc(hidden)

    def get_config(self) -> dict[str, object]:
        """Return model configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_blocks": self.num_blocks,
            "kernel_size": self.kernel_size,
            "num_classes": self.num_classes,
            "dropout": self.dropout_prob,
            "expansion_ratio": self.expansion_ratio,
            "dilation_base": self.dilation_base,
            "layer_scale_init": self.layer_scale_init,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "ModernTCNClassifier":
        from CESTA.artifacts import load_checkpoint

        model = load_checkpoint(path)
        assert isinstance(model, cls)
        return model
