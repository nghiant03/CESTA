"""PatchTST model for many-to-many fault classification.

This module wraps the PatchTST encoder from the HuggingFace transformers
library with a linear classification head to perform per-timestep fault
diagnosis. PatchTST segments the input time series into patches, processes
them with a Transformer encoder, then interpolates back to the original
sequence length for per-timestep classification.

PatchTST treats each input channel independently (channel-independence)
by default, which improves generalization for multivariate time series.

Reference:
    Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
    with Transformers", ICLR 2023.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from CESTA.models.base import BaseModel


class PatchTSTClassifier(BaseModel):
    """PatchTST model for many-to-many sequence classification.

    Uses the HuggingFace PatchTSTModel encoder with channel-independent
    patching, followed by interpolation and a classification head to
    produce per-timestep predictions.

    Architecture:
        Input (batch, seq_len, features)
        -> PatchTSTModel (patching + Transformer encoder)
        -> Mean across channels
        -> Interpolate to original seq_len
        -> LayerNorm -> Dropout -> Linear(d_model, num_classes)
        -> Output (batch, seq_len, num_classes)

    Args:
        input_size: Number of input features per timestep.
        d_model: Dimension of the transformer hidden states.
        num_layers: Number of encoder layers.
        num_classes: Number of output classes (fault types).
        n_heads: Number of attention heads.
        d_ff: Dimension of the feed-forward layers.
        patch_length: Length of each patch.
        patch_stride: Stride between patches.
        dropout: Dropout probability.
        max_len: Maximum input sequence length (context_length).
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        num_layers: int = 2,
        num_classes: int = 4,
        n_heads: int = 4,
        d_ff: int = 64,
        patch_length: int = 8,
        patch_stride: int = 1,
        max_len: int = 60,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        from transformers import PatchTSTConfig
        from transformers.models.patchtst.modeling_patchtst import PatchTSTModel

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.max_len = max_len
        self.dropout_prob = dropout

        hf_config = PatchTSTConfig(
            num_input_channels=input_size,  # pyright: ignore[reportCallIssue]
            context_length=max_len,  # pyright: ignore[reportCallIssue]
            patch_length=patch_length,  # pyright: ignore[reportCallIssue]
            patch_stride=patch_stride,  # pyright: ignore[reportCallIssue]
            d_model=d_model,  # pyright: ignore[reportCallIssue]
            num_hidden_layers=num_layers,  # pyright: ignore[reportCallIssue]
            num_attention_heads=n_heads,  # pyright: ignore[reportCallIssue]
            ffn_dim=d_ff,  # pyright: ignore[reportCallIssue]
            attention_dropout=dropout,  # pyright: ignore[reportCallIssue]
            ff_dropout=dropout,  # pyright: ignore[reportCallIssue]
            positional_dropout=dropout,  # pyright: ignore[reportCallIssue]
            norm_type="batchnorm",  # pyright: ignore[reportCallIssue]
        )

        self.encoder = PatchTSTModel(hf_config)
        self.layer_norm = nn.LayerNorm(input_size * d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size * d_model, num_classes)

    @property
    def name(self) -> str:
        return "patchtst"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for many-to-many classification.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits tensor of shape (batch, seq_len, num_classes).
        """
        batch_size, seq_len, n_vars = x.shape
        
        out = self.encoder(past_values=x)
        hidden = out.last_hidden_state 
        num_patches = hidden.shape[2]

        hidden = hidden.permute(0, 2, 1, 3).contiguous()
        hidden = hidden.view(batch_size, num_patches, n_vars * self.d_model)

        if num_patches != seq_len:
            hidden = hidden.permute(0, 2, 1) 
            hidden = F.interpolate(hidden, size=seq_len, mode="linear", align_corners=False)
            hidden = hidden.permute(0, 2, 1)

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
            "patch_length": self.patch_length,
            "patch_stride": self.patch_stride,
            "max_len": self.max_len,
            "dropout": self.dropout_prob,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> PatchTSTClassifier:
        from CESTA.artifacts import load_checkpoint

        model = load_checkpoint(path)
        assert isinstance(model, cls)
        return model
