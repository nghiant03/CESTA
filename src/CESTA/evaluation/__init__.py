"""Evaluation module for fault diagnosis models.

Provides the Evaluator class and metric computation utilities.
"""

from CESTA.evaluation.communication import (
    aggregate_communication_stats,
    save_communication_metrics,
)
from CESTA.evaluation.energy import RadioEnergyConfig, compute_radio_energy_metrics
from CESTA.evaluation.evaluator import Evaluator
from CESTA.evaluation.result import EvalResult
from CESTA.metrics import (
    ClassMetrics,
    compute_class_metrics,
    confusion_matrix,
    macro_f1,
)

__all__ = [
    "ClassMetrics",
    "EvalResult",
    "Evaluator",
    "RadioEnergyConfig",
    "aggregate_communication_stats",
    "compute_class_metrics",
    "compute_radio_energy_metrics",
    "confusion_matrix",
    "macro_f1",
    "save_communication_metrics",
]
