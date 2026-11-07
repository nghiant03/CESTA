"""Training-only frozen temporal teacher on aligned, complete node windows."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from CESTA.batch import GraphWindowBatch
from CESTA.schema import TrainConfig


def load_teacher(config: TrainConfig, device: torch.device) -> nn.Module | None:
    if config.distillation_weight == 0 and config.distillation_local_weight == 0:
        return None
    from CESTA.artifacts import load_checkpoint

    paths = config.distillation_teachers or [config.distillation_teacher or ""]
    teachers = []
    for name in paths:
        path = Path(name)
        meta = json.loads((path / "config.json").read_text())
        teacher_config = TrainConfig.model_validate(meta["train_config"])
        if teacher_config.model != "patchtst" or teacher_config.features != config.features or teacher_config.data != config.data:
            raise ValueError("Teacher must be PatchTST with identical features, windows and split recipe")
        teacher = load_checkpoint(path, map_location=device)
        if not isinstance(teacher, nn.Module):
            raise TypeError("Teacher must be a torch module")
        teachers.append(teacher.to(device).eval().requires_grad_(False))
    return nn.ModuleList(teachers).eval()


def distillation_loss(
    teacher: nn.Module, model_input: object,
    logits: torch.Tensor, targets: torch.Tensor, temperature: float, drift_weight: float = 1.0,
    target_cache: dict[str, torch.Tensor] | None = None, correct_only: bool = False,
) -> torch.Tensor:
    """Never called on validation. Batch-local reshaping preserves shuffle/oversampling alignment."""
    if not isinstance(model_input, GraphWindowBatch) or logits.ndim != 4:
        raise ValueError("Distillation requires graph student batches")
    batch, time, nodes, classes = logits.shape
    x = model_input.x.reshape(batch, time, nodes, -1).permute(0, 2, 1, 3).reshape(batch * nodes, time, -1)
    valid = targets >= 0
    if model_input.node_mask is not None:
        valid = valid & model_input.node_mask
    complete = valid.all(dim=1).reshape(-1)
    if not bool(complete.any()):
        return logits.sum() * 0
    # Match the standalone temporal cohort: incomplete node windows are excluded,
    # not interpolated or filled, including their otherwise valid timesteps.
    student = logits.permute(0, 2, 1, 3).reshape(batch * nodes, time, classes)[complete]
    with torch.no_grad():
        members = teacher if isinstance(teacher, nn.ModuleList) else [teacher]
        probabilities = target_cache.get("probabilities") if target_cache is not None else None
        if probabilities is None:
            probabilities = torch.stack([(member(x[complete]) / temperature).softmax(-1) for member in members]).mean(0)
            if target_cache is not None:
                target_cache["probabilities"] = probabilities
        if probabilities.shape != student.shape:
            raise ValueError("Teacher/student temporal or class alignment mismatch")
    terms = F.kl_div((student / temperature).log_softmax(dim=-1), probabilities, reduction="none").sum(-1)
    labels = targets.permute(0, 2, 1).reshape(batch * nodes, time)[complete]
    weights = torch.where(labels == 2, drift_weight, 1.0)
    if correct_only:
        accepted = probabilities.argmax(-1) == labels
        weights = weights * accepted
        if target_cache is not None:
            target_cache["retention_counts"] = torch.stack([
                torch.stack([((labels == c) & accepted).sum(), (labels == c).sum()]) for c in range(classes)
            ]).detach()
    return (terms * weights).sum() / weights.sum().clamp_min(1.0) * temperature**2
