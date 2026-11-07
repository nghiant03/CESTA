"""One-sided auxiliary-gradient projection; no inference changes."""
from __future__ import annotations

import torch
from torch import nn


def protected_backward(task: torch.Tensor, auxiliary: torch.Tensor, parameters: list[nn.Parameter]) -> dict[str, float]:
    """Protect the concatenated task gradient on specified shared parameters.

    u' = u - min(<u,g>,0)/||g||² * g when ||g||² > 0.
    Other parameters receive ordinary task+auxiliary gradients unchanged.
    """
    shared = [p for p in parameters if p.requires_grad]
    task_grad = torch.autograd.grad(task, shared, retain_graph=True, allow_unused=True)
    auxiliary_grad = torch.autograd.grad(auxiliary, shared, retain_graph=True, allow_unused=True)
    zero = task.detach().new_zeros(())
    dot = sum((g.detach().mul(u.detach()).sum() for g, u in zip(task_grad, auxiliary_grad) if g is not None and u is not None), zero)
    norm = sum((g.detach().square().sum() for g in task_grad if g is not None), zero)
    if not bool(torch.isfinite(dot) & torch.isfinite(norm)):
        raise FloatingPointError("Nonfinite task-protection gradient statistics")
    coefficient = torch.where(norm > 1e-12, dot.clamp_max(0) / norm.clamp_min(1e-12), zero)
    (task + auxiliary).backward()
    for p, g in zip(shared, task_grad):
        if g is not None and p.grad is not None:
            p.grad.sub_(coefficient * g.detach())
    auxiliary_norm = sum((u.detach().square().sum() for u in auxiliary_grad if u is not None), zero)
    return {"active": float(coefficient < 0), "dot": float(dot), "task_norm": float(norm.sqrt()),
            "kd_norm": float(auxiliary_norm.sqrt()),
            "cosine": float(dot / (norm * auxiliary_norm).sqrt().clamp_min(1e-12))}
