"""Base class for deep learning models.

All model implementations should inherit from BaseModel and implement
the required abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
        """Count active trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Count all model parameters, including inactive parameters."""
        return sum(p.numel() for p in self.parameters())

