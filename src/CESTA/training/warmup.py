"""Training-only local BiGRU preparation, without changing inference architecture."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from CESTA.batch import GraphWindowBatch
from CESTA.logging import logger
from CESTA.models.base import BaseModel
from CESTA.schema import TrainConfig
from CESTA.training.batch_utils import prepare_batch
from CESTA.training.distillation import distillation_loss
from CESTA.training.objectives import masked_loss


def warmup_local_gru(
    model: BaseModel, loader: DataLoader, config: TrainConfig, device: torch.device, teacher: nn.Module | None,
    optimizer: torch.optim.Optimizer | None = None, criterion: nn.Module | None = None,
) -> None:
    from CESTA.models.spatial.cesta.model import CESTAClassifier

    if not isinstance(model, CESTAClassifier) or model.temporal_backend != "gru":
        raise ValueError("Local warmup requires the original CESTA GRU")
    parameters = [*model.temporal_encoder.parameters(), *model.classifier.parameters()]
    if optimizer is None:
        optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    model.train()
    for epoch in range(config.local_warmup_epochs):
        total = 0.0
        for batch in loader:
            inputs, targets, mask, _ = prepare_batch(batch, device)
            if not isinstance(inputs, GraphWindowBatch):
                raise ValueError("Local warmup requires graph batches")
            b, t, n = targets.shape
            x = inputs.x.reshape(b, t, n, -1).permute(0, 2, 1, 3).reshape(b*n, t, -1)
            local_mask = inputs.node_mask.permute(0, 2, 1).reshape(b*n, t)
            hidden, _ = model.temporal_encoder(model.prepare_temporal_input(x, local_mask))
            logits = model.classifier(hidden).reshape(b, n, t, -1).permute(0, 2, 1, 3)
            loss = masked_loss(criterion if criterion is not None else nn.CrossEntropyLoss(), logits, targets, mask)
            if teacher is not None:
                loss = loss + config.distillation_local_weight * distillation_loss(
                    teacher, inputs, logits, targets, config.distillation_temperature, config.distillation_drift_weight,
                )
            optimizer.zero_grad()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite local warmup loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("Nonfinite local warmup gradients")
            optimizer.step()
            total += loss.item()
        logger.info("Local-only training warmup epoch {} / {} loss {}", epoch+1, config.local_warmup_epochs, total/max(len(loader), 1))
    # No validation checkpoint selection in warmup: always use the terminal state.
    # Optional shared optimizer preserves local moments; graph parameters had no gradients.
