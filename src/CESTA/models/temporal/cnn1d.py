"""1D CNN model for many-to-many fault classification.

This module implements a 1D Convolutional Neural Network for
sequence-to-sequence fault diagnosis, predicting a fault label at each
timestep. Causal (left) padding is used so that each output timestep
depends only on current and past inputs, preserving temporal causality.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from CESTA.models.base import BaseModel


class CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding.

    Ensures that the output at timestep *t* depends only on inputs at
    timesteps ≤ *t*, which is appropriate for online fault diagnosis.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        dilation: Spacing between kernel elements.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, channels, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, seq_len)``.
        """
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CNN1DBlock(nn.Module):
    """Single 1D CNN block: CausalConv1d -> BatchNorm -> ReLU -> Dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        dilation: Spacing between kernel elements.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, in_channels, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, seq_len)``.
        """
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        return self.dropout(out)


class CNN1DClassifier(BaseModel):
    """1D CNN model for many-to-many sequence classification.

    Architecture:
        Input -> [CausalConv1d -> BatchNorm -> ReLU -> Dropout] x N -> Linear -> Output

    Stacks multiple 1D convolutional blocks with increasing dilation to
    capture both local and long-range temporal patterns while preserving
    the sequence length for per-timestep classification.

    Args:
        input_size: Number of input features per timestep.
        num_channels: Number of channels (filters) per conv block.
        num_blocks: Number of stacked convolutional blocks.
        kernel_size: Kernel size for each convolutional layer.
        num_classes: Number of output classes (fault types).
        dropout: Dropout probability.
        dilation_base: Base for exponential dilation growth. Block *i*
            uses dilation ``dilation_base ** i``.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        num_classes: int = 4,
        dropout: float = 0.2,
        dilation_base: int = 2,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.dilation_base = dilation_base

        blocks: list[nn.Module] = []
        in_ch = input_size
        for i in range(num_blocks):
            dilation = dilation_base**i
            blocks.append(
                CNN1DBlock(in_ch, num_channels, kernel_size, dilation, dropout)
            )
            in_ch = num_channels
        self.blocks = nn.Sequential(*blocks)

        self.fc = nn.Linear(num_channels, num_classes)

    @property
    def name(self) -> str:
        return "cnn1d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for many-to-many classification.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Logits tensor of shape ``(batch, seq_len, num_classes)``.
        """
        # Conv1d expects (batch, channels, seq_len)
        out = x.transpose(1, 2)
        out = self.blocks(out)
        # Back to (batch, seq_len, channels)
        out = out.transpose(1, 2)
        logits = self.fc(out)
        return logits

    def get_config(self) -> dict[str, object]:
        """Return model configuration for serialization."""
        return {
            "input_size": self.input_size,
            "num_channels": self.num_channels,
            "num_blocks": self.num_blocks,
            "kernel_size": self.kernel_size,
            "num_classes": self.num_classes,
            "dropout": self.dropout_prob,
            "dilation_base": self.dilation_base,
        }

