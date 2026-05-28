"""Backward-compatible metric exports."""

from __future__ import annotations

from CESTA.metrics import ClassMetrics, compute_class_metrics, confusion_matrix, macro_f1

__all__ = [
    "ClassMetrics",
    "compute_class_metrics",
    "confusion_matrix",
    "macro_f1",
]
