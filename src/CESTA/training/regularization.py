from __future__ import annotations

from dataclasses import replace

import torch

from CESTA.batch import GraphWindowBatch, TemporalWindowBatch
from CESTA.schema.config import TrainConfig


def augment_training_input(
    batch: torch.Tensor | GraphWindowBatch | TemporalWindowBatch,
    config: TrainConfig,
) -> torch.Tensor | GraphWindowBatch | TemporalWindowBatch:
    if config.augmentation_scale == 0.0 and config.augmentation_offset == 0.0:
        return batch
    features = batch if isinstance(batch, torch.Tensor) else batch.x
    shape = (features.shape[0], 1, *features.shape[2:])
    scale = 1.0 + config.augmentation_scale * (2.0 * torch.rand(shape, device=features.device, dtype=features.dtype) - 1.0)
    offset = config.augmentation_offset * (2.0 * torch.rand(shape, device=features.device, dtype=features.dtype) - 1.0)
    augmented = features * scale + offset
    if isinstance(batch, GraphWindowBatch):
        nodes = batch.node_mask.shape[-1]
        channels = features.shape[-1] if features.ndim == 4 else features.shape[-1] // nodes
        valid = batch.node_mask.unsqueeze(-1).expand(*batch.node_mask.shape, channels).reshape_as(features)
        augmented = torch.where(valid, augmented, features)
    if isinstance(batch, torch.Tensor):
        return augmented
    return replace(batch, x=augmented)


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    optimizer_class = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
    return optimizer_class(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
