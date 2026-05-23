"""Base class for deep learning models.

All model implementations should inherit from BaseModel and implement
the required abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for fault diagnosis models.

    All models must implement:
        - name: Property returning the model's registered name
        - forward: Standard PyTorch forward pass
        - get_config: Return architecture config dict for serialization

    Subclasses that need extra data (e.g. graph topology) should list
    required metadata keys in ``required_metadata``.  The model registry
    validates these before construction.

    The base class provides common utilities for model management.
    """

    required_metadata: ClassVar[set[str]] = set()
    optional_metadata: ClassVar[set[str]] = set()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the registered name of this model."""
        ...

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, seq_len, features).

        Returns:
            Output tensor of shape (batch, seq_len, num_classes) for
            many-to-many classification.
        """
        ...

    @abstractmethod
    def get_config(self) -> dict[str, object]:
        """Return model architecture configuration for serialization."""
        ...

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(
        self,
        path: str | Path,
        config_dict: dict[str, object] | None = None,
    ) -> None:
        """Save model to a directory with weight.pt and config.json.

        Args:
            path: Directory path to save the model into.
            config_dict: Optional training config to embed alongside model config.
        """
        from CESTA.artifacts import save_checkpoint

        save_checkpoint(self, path, config_dict=config_dict)

    @staticmethod
    def load_config(path: str | Path) -> dict[str, object] | None:
        """Load training config from a saved model directory.

        Args:
            path: Path to the model directory.

        Returns:
            Training config dictionary if present, None otherwise.
        """
        from CESTA.artifacts import checkpoint_metadata_path, load_checkpoint_train_config

        if not checkpoint_metadata_path(path).exists():
            return None
        return load_checkpoint_train_config(path)

    @staticmethod
    def load_metadata(path: str | Path) -> dict[str, object]:
        """Load full metadata (model_name, model_config, train_config) from directory.

        Args:
            path: Path to the model directory.

        Returns:
            Full metadata dictionary.
        """
        from CESTA.artifacts import load_checkpoint_metadata

        return load_checkpoint_metadata(path)
