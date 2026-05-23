"""Training callbacks for monitoring and controlling training.

Callbacks are invoked by the Trainer at specific points during the
training loop to provide extensible hooks for logging, early stopping,
checkpointing, and other side effects.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

from CESTA.evaluation.metrics import ClassMetrics
from CESTA.logging import logger
from CESTA.models.base import BaseModel


@dataclass
class TrainMetrics:
    """Metrics collected during a single epoch.

    Attributes:
        epoch: Current epoch number (1-indexed).
        train_loss: Average training loss for the epoch.
        val_loss: Average validation loss for the epoch (None if no val set).
        train_acc: Training accuracy for the epoch.
        val_acc: Validation accuracy for the epoch (None if no val set).
        train_macro_f1: Macro-averaged F1 on training set.
        val_macro_f1: Macro-averaged F1 on validation set (None if no val set).
        train_class_metrics: Per-class metrics on training set (None if not computed).
        val_class_metrics: Per-class metrics on validation set (None if not computed).
    """

    epoch: int
    train_loss: float
    val_loss: float | None = None
    train_acc: float | None = None
    val_acc: float | None = None
    train_macro_f1: float | None = None
    val_macro_f1: float | None = None
    train_class_metrics: ClassMetrics | None = None
    val_class_metrics: ClassMetrics | None = None


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""

    @abstractmethod
    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        """Called at the end of each epoch.

        Args:
            metrics: Collected metrics for the epoch.
            model: The model being trained.

        Returns:
            True to continue training, False to stop early.
        """
        ...


class LoggingCallback(TrainingCallback):
    """Logs training metrics at each epoch."""

    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        parts = [
            f"Epoch {metrics.epoch}",
            f"train_loss={metrics.train_loss:.4f}",
        ]
        if metrics.train_acc is not None:
            parts.append(f"train_acc={metrics.train_acc:.4f}")
        if metrics.train_macro_f1 is not None:
            parts.append(f"train_f1={metrics.train_macro_f1:.4f}")
        if metrics.val_loss is not None:
            parts.append(f"val_loss={metrics.val_loss:.4f}")
        if metrics.val_acc is not None:
            parts.append(f"val_acc={metrics.val_acc:.4f}")
        if metrics.val_macro_f1 is not None:
            parts.append(f"val_f1={metrics.val_macro_f1:.4f}")

        logger.info(" | ".join(parts))
        return True


@dataclass
class EarlyStoppingCallback(TrainingCallback):
    """Stops training when a monitored metric stops improving.

    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        monitor: Metric name from ``TrainMetrics`` to watch
            (``val_loss``, ``val_macro_f1``, or ``val_acc``).
            ``val_macro_f1`` and ``val_acc`` are maximized; ``val_loss`` is minimized.
    """

    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    _best: float = field(default=float("inf"), init=False, repr=False)
    _counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._direction_maximize = self.monitor != "val_loss"
        if self._direction_maximize:
            self._best = float("-inf")

    def _metric_value(self, metrics: TrainMetrics) -> float | None:
        if self.monitor == "val_loss":
            return metrics.val_loss
        if self.monitor == "val_macro_f1":
            return metrics.val_macro_f1
        return metrics.val_acc

    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        value = self._metric_value(metrics)
        if value is None:
            return True

        improved = (
            value > self._best + self.min_delta
            if self._direction_maximize
            else value < self._best - self.min_delta
        )
        if improved:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logger.info(
                    "Early stopping triggered after {} epochs without improvement ({}={:.4f})",
                    self.patience,
                    self.monitor,
                    self._best,
                )
                return False
        return True


@dataclass
class CheckpointCallback(TrainingCallback):
    """Saves model checkpoint when a monitored metric improves.

    Saves to a directory with weight.pt and config.json.

    Attributes:
        save_path: Directory path to save the best model checkpoint.
        config_dict: Optional config dictionary to include in checkpoint.
        monitor: Metric name from ``TrainMetrics`` to watch
            (``val_loss``, ``val_macro_f1``, or ``val_acc``).
            ``val_macro_f1`` and ``val_acc`` are maximized; ``val_loss`` is minimized.
    """

    save_path: str | Path = "best_model"
    config_dict: dict[str, object] | None = None
    monitor: str = "val_loss"
    _best: float = field(default=float("inf"), init=False, repr=False)

    def __post_init__(self) -> None:
        self._direction_maximize = self.monitor != "val_loss"
        if self._direction_maximize:
            self._best = float("-inf")

    def _metric_value(self, metrics: TrainMetrics) -> float:
        if self.monitor == "val_loss":
            return metrics.val_loss if metrics.val_loss is not None else metrics.train_loss
        if self.monitor == "val_macro_f1":
            return (
                metrics.val_macro_f1
                if metrics.val_macro_f1 is not None
                else metrics.train_macro_f1 if metrics.train_macro_f1 is not None else 0.0
            )
        return (
            metrics.val_acc
            if metrics.val_acc is not None
            else metrics.train_acc if metrics.train_acc is not None else 0.0
        )

    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        from CESTA.artifacts import save_checkpoint

        value = self._metric_value(metrics)

        improved = value > self._best if self._direction_maximize else value < self._best
        if improved:
            self._best = value
            save_checkpoint(model, self.save_path, config_dict=self.config_dict)
            logger.info("Saved checkpoint to {} ({}={:.4f})", self.save_path, self.monitor, value)
        return True


@dataclass
class HistoryCallback(TrainingCallback):
    """Persists per-epoch ``TrainMetrics`` as JSONL.

    Each epoch appends one line to ``history.jsonl`` so training curves,
    early-stopping points, and per-class metrics can be recovered after
    the run completes.

    Attributes:
        save_path: Directory where ``history.jsonl`` is written.
    """

    save_path: str | Path = "."
    _file_path: Path = field(init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._file_path = Path(self.save_path) / "history.jsonl"

    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        if not self._initialized:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate any previous run's history in this directory.
            self._file_path.write_text("")
            self._initialized = True

        payload = asdict(metrics)
        with self._file_path.open("a") as fh:
            fh.write(json.dumps(payload) + "\n")
        return True
