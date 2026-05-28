"""Shared objective and prediction helpers for training and evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from CESTA.models.base import BaseModel
from CESTA.schema import TrainConfig


def masked_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    if node_mask is None:
        return criterion(flat_logits, flat_targets)
    valid = node_mask.reshape(-1) & (flat_targets >= 0)
    if not bool(valid.any()):
        return flat_logits.sum() * 0.0
    return criterion(flat_logits[valid], flat_targets[valid])


def apply_training_objective(
    config: TrainConfig,
    model: BaseModel,
    loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> torch.Tensor:
    loss = add_auxiliary_loss(config, model, loss)
    loss = add_voi_gate_loss(config, model, loss, targets, node_mask)
    loss = add_counterfactual_voi_loss(config, model, loss, targets, node_mask)
    loss = add_boundary_loss(config, model, loss, targets, node_mask)
    loss = add_crf_loss(config, model, loss, logits, targets, node_mask)
    return add_persistent_transition_loss(config, model, loss)


def add_auxiliary_loss(
    config: TrainConfig,
    model: BaseModel,
    loss: torch.Tensor,
) -> torch.Tensor:
    weight = config.communication_penalty_weight
    if weight > 0.0:
        comm_loss = getattr(model, "communication_loss", None)
        if comm_loss is not None and isinstance(comm_loss, torch.Tensor):
            if config.communication_penalty_mode == "budget_hinge":
                excess = torch.relu(comm_loss - config.target_request_ratio)
                loss = loss + weight * (excess**2)
            else:
                loss = loss + weight * comm_loss

    entropy_weight = config.gate_entropy_weight
    if entropy_weight > 0.0:
        gate_entropy = getattr(model, "gate_entropy", None)
        if gate_entropy is not None and isinstance(gate_entropy, torch.Tensor):
            loss = loss - entropy_weight * gate_entropy

    return loss


def add_voi_gate_loss(
    config: TrainConfig,
    model: BaseModel,
    loss: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> torch.Tensor:
    weight = config.voi_gate_loss_weight
    if weight <= 0.0:
        return loss
    local_logits = getattr(model, "local_logits", None)
    communication_logits = getattr(model, "communication_logits", None)
    communication_loss = getattr(model, "communication_loss", None)
    if not isinstance(local_logits, torch.Tensor) or not isinstance(communication_logits, torch.Tensor):
        return loss
    if not isinstance(communication_loss, torch.Tensor):
        return loss
    valid = targets >= 0
    if node_mask is not None:
        valid = valid & node_mask
    if not bool(valid.any()):
        return loss
    safe_targets = targets.clamp_min(0).unsqueeze(-1)
    local_correct_logits = local_logits.gather(-1, safe_targets).squeeze(-1)
    communication_correct_logits = communication_logits.gather(-1, safe_targets).squeeze(-1)
    improvement = communication_correct_logits - local_correct_logits
    no_improvement = torch.relu(-improvement[valid]).mean()
    return loss + weight * communication_loss * no_improvement


def add_counterfactual_voi_loss(
    config: TrainConfig,
    model: BaseModel,
    loss: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> torch.Tensor:
    weight = config.counterfactual_voi_loss_weight
    if weight <= 0.0:
        return loss
    local_logits = getattr(model, "local_logits", None)
    communication_logits = getattr(model, "communication_logits", None)
    communication_activity = getattr(model, "communication_activity", None)
    if not isinstance(local_logits, torch.Tensor) or not isinstance(communication_logits, torch.Tensor):
        return loss
    if not isinstance(communication_activity, torch.Tensor):
        return loss
    valid = targets >= 0
    if node_mask is not None:
        valid = valid & node_mask
    if not bool(valid.any()):
        return loss
    safe_targets = targets.clamp_min(0).unsqueeze(-1)
    local_correct_logits = local_logits.gather(-1, safe_targets).squeeze(-1)
    communication_correct_logits = communication_logits.gather(-1, safe_targets).squeeze(-1)
    improvement = communication_correct_logits - local_correct_logits
    harmful = torch.relu(-improvement)
    useful = torch.relu(improvement).detach()
    active = communication_activity.clamp(0.0, 1.0)
    penalty = active * (harmful * config.counterfactual_voi_penalty_weight - useful)
    return loss + weight * penalty[valid].mean()


def add_boundary_loss(
    config: TrainConfig,
    model: BaseModel,
    loss: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> torch.Tensor:
    weight = config.boundary_loss_weight
    if weight <= 0.0:
        return loss
    boundary_logits = getattr(model, "last_boundary_logits", None)
    if not isinstance(boundary_logits, torch.Tensor):
        return loss
    boundary_targets, boundary_mask = boundary_targets_from_labels(targets, node_mask, config.boundary_dilation)
    flat_logits = boundary_logits.reshape(-1)
    flat_targets = boundary_targets.reshape(-1)
    flat_mask = boundary_mask.reshape(-1)
    if not bool(flat_mask.any()):
        return loss
    selected_logits = flat_logits[flat_mask]
    selected_targets = flat_targets[flat_mask]
    pos_weight = None
    if config.boundary_positive_weight is not None:
        pos_weight = torch.tensor(config.boundary_positive_weight, dtype=selected_logits.dtype, device=selected_logits.device)
    boundary_loss = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, pos_weight=pos_weight, reduction="none")
    gamma = config.boundary_focal_gamma
    if gamma > 0.0:
        probs = torch.sigmoid(selected_logits)
        p_t = torch.where(selected_targets > 0.5, probs, 1.0 - probs)
        boundary_loss = ((1.0 - p_t).clamp_min(0.0) ** gamma) * boundary_loss
    return loss + weight * boundary_loss.mean()


def add_crf_loss(
    config: TrainConfig,
    model: BaseModel,
    loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> torch.Tensor:
    weight = config.crf_loss_weight
    if weight <= 0.0:
        return loss
    crf_nll = getattr(model, "crf_negative_log_likelihood", None)
    if not callable(crf_nll):
        return loss
    crf_loss = crf_nll(logits, targets, node_mask)
    if not isinstance(crf_loss, torch.Tensor):
        return loss
    return loss + weight * crf_loss


def add_persistent_transition_loss(
    config: TrainConfig,
    model: BaseModel,
    loss: torch.Tensor,
) -> torch.Tensor:
    weight = config.persistent_transition_loss_weight
    if weight <= 0.0:
        return loss
    transitions = getattr(model, "crf_transitions", None)
    if not isinstance(transitions, torch.Tensor) or transitions.ndim != 2:
        return loss
    penalties: list[torch.Tensor] = []
    if config.persistent_classes:
        persistent_classes = torch.tensor(config.persistent_classes, dtype=torch.long, device=transitions.device)
        valid_persistent = persistent_classes[(persistent_classes >= 0) & (persistent_classes < transitions.size(0))]
        if valid_persistent.numel() > 0:
            stay_scores = transitions[valid_persistent, valid_persistent]
            exit_to_normal_scores = transitions[valid_persistent, 0]
            penalties.append(torch.relu(config.persistent_transition_margin - (stay_scores - exit_to_normal_scores)).mean())
    if config.transient_classes:
        transient_classes = torch.tensor(config.transient_classes, dtype=torch.long, device=transitions.device)
        valid_transient = transient_classes[(transient_classes >= 0) & (transient_classes < transitions.size(0))]
        if valid_transient.numel() > 0:
            stay_scores = transitions[valid_transient, valid_transient]
            exit_to_normal_scores = transitions[valid_transient, 0]
            penalties.append(torch.relu(config.transient_transition_margin + stay_scores - exit_to_normal_scores).mean())
    if not penalties:
        return loss
    transition_loss = torch.stack(penalties).mean()
    return loss + weight * transition_loss


def boundary_targets_from_labels(
    targets: torch.Tensor,
    node_mask: torch.Tensor | None,
    dilation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = targets >= 0
    if node_mask is not None:
        valid = valid & node_mask
    boundary = torch.zeros_like(targets, dtype=torch.float32)
    transition_valid = valid[:, 1:] & valid[:, :-1]
    raw_boundary = ((targets[:, 1:] != targets[:, :-1]) & transition_valid).to(torch.float32)
    boundary[:, 1:] = raw_boundary
    boundary_mask = valid.clone()
    boundary_mask[:, 0] = False
    if dilation > 0:
        dilated = boundary.clone()
        for shift in range(1, dilation + 1):
            dilated[:, shift:] = torch.maximum(dilated[:, shift:], boundary[:, :-shift])
            dilated[:, :-shift] = torch.maximum(dilated[:, :-shift], boundary[:, shift:])
        boundary = dilated * boundary_mask.to(torch.float32)
    return boundary, boundary_mask


def decode_predictions(
    model: BaseModel,
    logits: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> torch.Tensor:
    decoder = getattr(model, "crf_decode", None)
    if callable(decoder) and bool(getattr(model, "use_crf", False)):
        decoded = decoder(logits, node_mask)
        if isinstance(decoded, torch.Tensor):
            return decoded
    return logits.argmax(dim=-1)


def valid_predictions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_preds = preds.reshape(-1)
    flat_targets = targets.reshape(-1)
    if node_mask is None:
        return flat_preds, flat_targets
    valid = node_mask.reshape(-1) & (flat_targets >= 0)
    return flat_preds[valid], flat_targets[valid]


def valid_outputs(
    preds: torch.Tensor,
    targets: torch.Tensor,
    probs: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_preds = preds.reshape(-1)
    flat_targets = targets.reshape(-1)
    flat_probs = probs.reshape(-1, probs.size(-1))
    if node_mask is None:
        return flat_preds, flat_targets, flat_probs
    valid = node_mask.reshape(-1) & (flat_targets >= 0)
    return flat_preds[valid], flat_targets[valid], flat_probs[valid]
