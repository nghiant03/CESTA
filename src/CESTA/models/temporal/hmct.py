"""HMCT hybrid multi-scale convolution and Transformer architecture."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn as nn

from CESTA.batch import GraphWindowBatch
from CESTA.models.base import BaseModel


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding: torch.Tensor = self.encoding  # type: ignore[assignment]
        if x.shape[1] > encoding.shape[1]:
            raise ValueError(f"sequence length {x.shape[1]} exceeds max_len {encoding.shape[1]}")
        return self.dropout(x + encoding[:, : x.shape[1]])


class MultiScaleConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...],
        dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have equal length")
        if not kernel_sizes:
            raise ValueError("kernel_sizes cannot be empty")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size // 2) * dilation, dilation=dilation),
                    nn.GELU(),
                    nn.BatchNorm1d(channels),
                )
                for kernel_size, dilation in zip(kernel_sizes, dilations, strict=True)
            ]
        )
        self.merge = nn.Sequential(
            nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.merge(torch.cat([branch(x) for branch in self.branches], dim=1))


class LargeDilationBlock(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("large_dilations cannot be empty")
        self.branches = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation) for dilation in dilations]
        )
        self.merge = nn.Sequential(
            nn.BatchNorm1d(channels * len(dilations)),
            nn.GELU(),
            nn.Conv1d(channels * len(dilations), channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.merge(torch.cat([branch(x) for branch in self.branches], dim=1))


class HMCTClassifier(BaseModel):
    """Hybrid multi-scale CNN and Transformer for independent per-node diagnosis."""

    required_metadata: ClassVar[set[str]] = {"graph"}

    def __init__(
        self,
        input_size: int,
        num_nodes: int,
        d_model: int = 128,
        cnn_channels: int = 128,
        num_cnn_blocks: int = 2,
        transformer_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        kernel_sizes: tuple[int, ...] | list[int] = (3, 5, 7),
        dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        large_dilations: tuple[int, ...] | list[int] = (8, 16),
        max_len: int = 10000,
        causal: bool = False,
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_size % num_nodes != 0:
            raise ValueError("input_size must be divisible by num_nodes")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if num_cnn_blocks < 1:
            raise ValueError("num_cnn_blocks must be at least 1")
        if transformer_layers < 1:
            raise ValueError("transformer_layers must be at least 1")

        self.input_size = input_size
        self.num_nodes = num_nodes
        self.features_per_node = input_size // num_nodes
        self.d_model = d_model
        self.cnn_channels = cnn_channels
        self.num_cnn_blocks = num_cnn_blocks
        self.transformer_layers = transformer_layers
        self.n_heads = n_heads
        self.ff_mult = ff_mult
        self.kernel_sizes = tuple(kernel_sizes)
        self.dilations = tuple(dilations)
        self.large_dilations = tuple(large_dilations)
        self.max_len = max_len
        self.causal = causal
        self.num_classes = num_classes
        self.dropout_probability = dropout

        self.input_projection = nn.Linear(self.features_per_node * 2, cnn_channels)
        self.cnn = nn.Sequential(
            *[
                MultiScaleConvBlock(cnn_channels, self.kernel_sizes, self.dilations, dropout)
                for _ in range(num_cnn_blocks)
            ],
            LargeDilationBlock(cnn_channels, self.large_dilations, dropout),
        )
        self.model_projection = nn.Conv1d(cnn_channels, d_model, kernel_size=1)
        self.position_encoding = SinusoidalPositionEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    @property
    def name(self) -> str:
        return "hmct"

    def forward(self, x: torch.Tensor | GraphWindowBatch) -> torch.Tensor:
        if isinstance(x, GraphWindowBatch):
            x = x.x
        if x.ndim == 3:
            batch_size, sequence_length, _ = x.shape
            node_inputs = x.view(batch_size, sequence_length, self.num_nodes, self.features_per_node)
        elif x.ndim == 4:
            batch_size, sequence_length, _, _ = x.shape
            node_inputs = x
        else:
            raise ValueError("input must have shape (batch, time, features) or (batch, time, nodes, features_per_node)")

        node_sequences = node_inputs.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_nodes,
            sequence_length,
            self.features_per_node,
        )
        differences = torch.diff(node_sequences, dim=1, prepend=node_sequences[:, :1])
        features = self.input_projection(torch.cat([node_sequences, differences], dim=-1))
        features = self.cnn(features.transpose(1, 2))
        features = self.model_projection(features).transpose(1, 2)
        features = self.position_encoding(features)
        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(sequence_length, device=features.device)
        features = self.transformer(features, mask=mask)
        logits = self.classifier(features)
        return logits.view(batch_size, self.num_nodes, sequence_length, self.num_classes).permute(0, 2, 1, 3)

    def get_config(self) -> dict[str, object]:
        return {
            "input_size": self.input_size,
            "num_nodes": self.num_nodes,
            "d_model": self.d_model,
            "cnn_channels": self.cnn_channels,
            "num_cnn_blocks": self.num_cnn_blocks,
            "transformer_layers": self.transformer_layers,
            "n_heads": self.n_heads,
            "ff_mult": self.ff_mult,
            "kernel_sizes": list(self.kernel_sizes),
            "dilations": list(self.dilations),
            "large_dilations": list(self.large_dilations),
            "max_len": self.max_len,
            "causal": self.causal,
            "num_classes": self.num_classes,
            "dropout": self.dropout_probability,
        }
