"""Focal loss for imbalanced classification.

Focal loss down-weights well-classified examples and focuses on hard,
misclassified ones.  This is particularly useful for fault diagnosis
where the NORMAL class dominates.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter.  gamma=0 recovers standard cross-entropy.
            Higher values increase focus on hard examples.
        alpha: Per-class weights as a 1-D tensor of length ``num_classes``.
            ``None`` means uniform weighting.
        reduction: How to reduce the per-sample loss (``"mean"`` | ``"sum"`` | ``"none"``).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits of shape ``(N, C)`` where *C* is the number of classes.
            targets: Ground-truth labels of shape ``(N,)`` with integer class indices.

        Returns:
            Scalar loss (or per-sample if ``reduction="none"``).
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        log_p_t = log_probs.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
        complement = (-log_p_t.expm1()).clamp_min(torch.finfo(log_p_t.dtype).tiny)
        focal_weight = complement ** self.gamma
        loss = -focal_weight * log_p_t

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.long())
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
