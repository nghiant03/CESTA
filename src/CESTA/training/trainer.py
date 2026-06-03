"""Trainer for fault diagnosis models.

Handles the full training loop including:
- Optional oversampling of minority classes
- Configurable loss function (cross-entropy or focal loss)
- Callback-driven logging, early stopping, and checkpointing
- Per-class precision, recall, F1 metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from CESTA.datasets.windowed import WindowedSplit
from CESTA.logging import logger
from CESTA.metrics import ClassMetrics, compute_class_metrics, macro_f1
from CESTA.models.base import BaseModel
from CESTA.schema import TrainConfig
from CESTA.seed import seed_everything
from CESTA.training.batch_utils import infer_num_classes, make_window_loader, prepare_batch
from CESTA.training.callbacks import (
    LoggingCallback,
    TrainingCallback,
    TrainMetrics,
)
from CESTA.training.loss import FocalLoss
from CESTA.training.objectives import (
    add_auxiliary_loss,
    apply_training_objective,
    decode_predictions,
    masked_loss,
    valid_predictions,
)
from CESTA.training.oversampling import oversample_split


def build_loss(
    config: TrainConfig,
    device: torch.device,
) -> nn.Module:
    """Build the loss function from config.

    Args:
        config: Training configuration.
        device: Target device for tensors.

    Returns:
        Loss module ready for ``(logits, targets)`` inputs.
    """
    if config.use_focal_loss:
        alpha = (
            torch.tensor(config.focal_alpha, dtype=torch.float32).to(device)
            if config.focal_alpha is not None
            else None
        )
        logger.debug(
            "Using FocalLoss with gamma={}, alpha={}",
            config.focal_gamma,
            config.focal_alpha,
        )
        return FocalLoss(gamma=config.focal_gamma, alpha=alpha)
    logger.debug("Using CrossEntropyLoss")
    return nn.CrossEntropyLoss()


def _prepare_data(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    config: TrainConfig,
    node_mask: NDArray[np.bool_] | None = None,
    edge_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_] | None, NDArray[np.bool_] | None]:
    """Apply oversampling if enabled.

    Args:
        X: Feature array ``(N, seq_len, features)``.
        y: Label array ``(N, seq_len)``.
        config: Training configuration.

    Returns:
        Possibly oversampled ``(X, y)`` tuple.
    """
    if config.oversample:
        logger.debug(
            "Oversampling minority classes with ratio={}, seed={}",
            config.oversample_ratio,
            config.seed,
        )
        split = oversample_split(
            WindowedSplit(X=X, y=y, node_mask=node_mask, edge_mask=edge_mask),
            ratio=config.oversample_ratio,
            seed=config.seed,
        )
        logger.info(
            "Oversampled: {} -> {} windows",
            len(X),
            len(split.X),
        )
        return split.X, split.y, split.node_mask, split.edge_mask
    return X, y, node_mask, edge_mask


@dataclass
class TrainResult:
    """Result container returned after training completes.

    Attributes:
        history: Per-epoch metrics collected during training.
        best_val_loss: Lowest validation loss seen (``None`` if no val data).
        stopped_epoch: Epoch at which training stopped (may be < total if early stopped).
    """

    history: list[TrainMetrics] = field(default_factory=list)
    best_val_loss: float | None = None
    stopped_epoch: int = 0


class Trainer:
    """Trains a fault-diagnosis model.

    Args:
        config: Training configuration (loss, oversampling, hyperparams).
        callbacks: Optional sequence of callbacks. If ``None``, a
            :class:`LoggingCallback` is used by default.
        device: PyTorch device string. ``None`` auto-selects CUDA if available.
    """

    def __init__(
        self,
        config: TrainConfig,
        callbacks: Sequence[TrainingCallback] | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.callbacks: list[TrainingCallback] = (
            list(callbacks) if callbacks is not None else [LoggingCallback()]
        )
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def fit(
        self,
        model: BaseModel,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
        metadata: dict[str, object] | None = None,
        node_mask_train: NDArray[np.bool_] | None = None,
        edge_mask_train: NDArray[np.bool_] | None = None,
        node_mask_val: NDArray[np.bool_] | None = None,
        edge_mask_val: NDArray[np.bool_] | None = None,
    ) -> TrainResult:
        """Train the model.

        Args:
            model: Model instance to train (modified in-place).
            X_train: Training features ``(N, seq_len, features)``.
            y_train: Training labels ``(N, seq_len)``.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            :class:`TrainResult` with full training history.
        """
        seed_everything(self.config.seed)

        logger.debug(
            "Training data shape: X={}, y={}",
            X_train.shape,
            y_train.shape,
        )
        X_train, y_train, node_mask_train, edge_mask_train = _prepare_data(
            X_train,
            y_train,
            self.config,
            node_mask_train,
            edge_mask_train,
        )

        if X_val is not None and y_val is not None:
            logger.info(
                "Using provided validation data: train={}, val={}",
                len(X_train),
                len(X_val),
            )

        train_loader = self._make_loader(
            X_train,
            y_train,
            shuffle=True,
            metadata=metadata,
            node_mask=node_mask_train,
            edge_mask=edge_mask_train,
        )
        val_loader = (
            self._make_loader(
                X_val,
                y_val,
                shuffle=False,
                metadata=metadata,
                node_mask=node_mask_val,
                edge_mask=edge_mask_val,
            )
            if X_val is not None and y_val is not None
            else None
        )
        logger.debug(
            "Train batches: {}, Val batches: {}",
            len(train_loader),
            len(val_loader) if val_loader is not None else 0,
        )

        model = model.to(self.device)

        num_classes = infer_num_classes(model, X_train, metadata, self.device)
        logger.info("Using device: {}", self.device)
        criterion = build_loss(self.config, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        logger.debug("Optimizer: Adam(lr={})", self.config.learning_rate)

        result = TrainResult()

        for epoch in range(1, self.config.epochs + 1):
            self._maybe_anneal_gumbel_temperature(model, epoch)

            train_loss, train_acc, train_cm = self._train_epoch(
                model, train_loader, criterion, optimizer
            )
            train_class_metrics = compute_class_metrics(
                train_cm[0], train_cm[1], num_classes
            )
            train_macro_f1 = macro_f1(train_class_metrics)

            val_loss: float | None = None
            val_acc: float | None = None
            val_macro_f1: float | None = None
            val_class_metrics: ClassMetrics | None = None

            if val_loader is not None:
                val_loss, val_acc, val_cm = self._eval_epoch(
                    model, val_loader, criterion
                )
                val_class_metrics = compute_class_metrics(
                    val_cm[0], val_cm[1], num_classes
                )
                val_macro_f1 = macro_f1(val_class_metrics)

            metrics = TrainMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                train_macro_f1=train_macro_f1,
                val_macro_f1=val_macro_f1,
                train_class_metrics=train_class_metrics,
                val_class_metrics=val_class_metrics,
            )
            result.history.append(metrics)

            if val_loss is not None and (
                result.best_val_loss is None or val_loss < result.best_val_loss
            ):
                result.best_val_loss = val_loss

            should_continue = all(
                cb.on_epoch_end(metrics, model) for cb in self.callbacks
            )
            if not should_continue:
                result.stopped_epoch = epoch
                logger.info("Training stopped early at epoch {}", epoch)
                break
        else:
            result.stopped_epoch = self.config.epochs
            logger.info("Training completed all {} epochs", self.config.epochs)

        self._log_final_metrics(result)
        return result

    def _log_final_metrics(self, result: TrainResult) -> None:
        """Log a summary of final training metrics."""
        if not result.history:
            return

        last = result.history[-1]
        logger.info("--- Training Summary ---")
        logger.info(
            "Stopped at epoch {} | train_loss={:.4f} | train_acc={:.4f} | train_f1={:.4f}",
            result.stopped_epoch,
            last.train_loss,
            last.train_acc if last.train_acc is not None else 0.0,
            last.train_macro_f1 if last.train_macro_f1 is not None else 0.0,
        )
        if last.val_loss is not None:
            logger.info(
                "val_loss={:.4f} | val_acc={:.4f} | val_f1={:.4f} | best_val_loss={:.4f}",
                last.val_loss,
                last.val_acc if last.val_acc is not None else 0.0,
                last.val_macro_f1 if last.val_macro_f1 is not None else 0.0,
                result.best_val_loss
                if result.best_val_loss is not None
                else float("nan"),
            )
        if last.val_class_metrics is not None:
            self._log_class_metrics("Validation", last.val_class_metrics)
        elif last.train_class_metrics is not None:
            self._log_class_metrics("Training", last.train_class_metrics)

    @staticmethod
    def _log_class_metrics(split_name: str, cm: ClassMetrics) -> None:
        """Log per-class metrics table."""
        from CESTA.schema.fault import FaultType

        names = FaultType.names()
        logger.info("--- {} Per-Class Metrics ---", split_name)
        logger.info(
            "{:<10s}  {:>9s}  {:>9s}  {:>9s}  {:>9s}",
            "Class",
            "Precision",
            "Recall",
            "F1",
            "Support",
        )
        for i, name in enumerate(names):
            if i < len(cm.precision):
                logger.info(
                    "{:<10s}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9d}",
                    name,
                    cm.precision[i],
                    cm.recall[i],
                    cm.f1[i],
                    cm.support[i],
                )

    def _maybe_anneal_gumbel_temperature(
        self,
        model: BaseModel,
        epoch: int,
    ) -> None:
        config = self.config
        if (
            config.gumbel_tau_anneal_epochs < 1
            or config.gumbel_tau_start == config.gumbel_tau_end
        ):
            return
        setter = getattr(model, "set_gumbel_temperature", None)
        if setter is None:
            return
        progress = min(float(epoch - 1) / config.gumbel_tau_anneal_epochs, 1.0)
        tau = config.gumbel_tau_start + progress * (
            config.gumbel_tau_end - config.gumbel_tau_start
        )
        setter(tau)

    def _make_loader(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        shuffle: bool,
        metadata: dict[str, object] | None = None,
        node_mask: NDArray[np.bool_] | None = None,
        edge_mask: NDArray[np.bool_] | None = None,
    ) -> DataLoader[object]:
        """Create a DataLoader from numpy arrays."""
        return make_window_loader(
            X,
            y,
            self.config.batch_size,
            shuffle=shuffle,
            metadata=metadata,
            node_mask=node_mask,
            edge_mask=edge_mask,
            seed=self.config.seed,
            node_identity_split="train" if shuffle else "val",
        )

    def _infer_num_classes(
        self,
        model: BaseModel,
        X_train: NDArray[np.float32],
        metadata: dict[str, object] | None,
    ) -> int:
        return infer_num_classes(model, X_train, metadata, self.device)

    def _train_epoch(
        self,
        model: BaseModel,
        loader: DataLoader[object],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Run one training epoch.

        Returns:
            ``(avg_loss, accuracy, (all_preds, all_targets))`` over the epoch.
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in loader:
            model_input, y_batch, node_mask, batch_size = prepare_batch(batch, self.device)

            optimizer.zero_grad()
            logits = model(model_input)

            loss = masked_loss(criterion, logits, y_batch, node_mask)
            loss = apply_training_objective(self.config, model, loss, logits, y_batch, node_mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            preds = decode_predictions(model, logits, node_mask)
            valid_preds, valid_targets = valid_predictions(preds, y_batch, node_mask)
            correct += (valid_preds == valid_targets).sum().item()
            total += valid_targets.numel()

            all_preds.append(valid_preds.detach().cpu())
            all_targets.append(valid_targets.detach().cpu())

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy, (all_preds, all_targets)

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: BaseModel,
        loader: DataLoader[object],
        criterion: nn.Module,
    ) -> tuple[float, float, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Run one evaluation epoch.

        Returns:
            ``(avg_loss, accuracy, (all_preds, all_targets))`` over the dataset.
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in loader:
            model_input, y_batch, node_mask, batch_size = prepare_batch(batch, self.device)

            logits = model(model_input)
            loss = masked_loss(criterion, logits, y_batch, node_mask)
            loss = add_auxiliary_loss(self.config, model, loss)

            total_loss += loss.item() * batch_size
            preds = decode_predictions(model, logits, node_mask)
            valid_preds, valid_targets = valid_predictions(preds, y_batch, node_mask)
            correct += (valid_preds == valid_targets).sum().item()
            total += valid_targets.numel()

            all_preds.append(valid_preds.detach().cpu())
            all_targets.append(valid_targets.detach().cpu())

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy, (all_preds, all_targets)
