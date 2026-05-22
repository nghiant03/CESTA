"""Optuna-based hyperparameter optimizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna

from CESTA.datasets import load_dataset
from CESTA.logging import logger
from CESTA.models import create_model, get_model_class
from CESTA.models.base import BaseModel
from CESTA.optimization.search_spaces import (
    get_search_space,
    suggest_train_hyperparams,
)
from CESTA.schema import OptimizeConfig, TrainConfig
from CESTA.schema.fault import FaultType
from CESTA.seed import seed_everything
from CESTA.training import LoggingCallback, Trainer, TrainingCallback
from CESTA.training.callbacks import TrainMetrics


class _OptunaPruneCallback(TrainingCallback):
    """Reports the optimization metric to Optuna and triggers pruning.

    Returns ``False`` from :meth:`on_epoch_end` when Optuna asks the trial
    to be pruned, so that the :class:`Trainer` exits early.
    """

    def __init__(self, trial: optuna.trial.Trial, metric: str) -> None:
        self.trial = trial
        self.metric = metric
        self.pruned = False

    def on_epoch_end(self, metrics: TrainMetrics, model: BaseModel) -> bool:
        value = self._read_metric(metrics)
        if value is None:
            return True
        self.trial.report(value, metrics.epoch)
        if self.trial.should_prune():
            logger.info("Trial {} pruned at epoch {}", self.trial.number, metrics.epoch)
            self.pruned = True
            return False
        return True

    def _read_metric(self, metrics: TrainMetrics) -> float | None:
        if self.metric == "val_loss":
            return metrics.val_loss
        if self.metric == "val_macro_f1":
            return metrics.val_macro_f1
        if self.metric == "val_acc":
            return metrics.val_acc
        return None


class Optimizer:
    """Drives an Optuna study for hyperparameter search.

    Each trial loads the dataset once (cached on the optimizer instance),
    samples both training-loop and model-architecture hyperparameters,
    builds a fresh model, and runs :class:`Trainer` for
    ``config.epochs`` epochs. The chosen validation metric is reported
    back to Optuna and used for pruning.
    """

    def __init__(self, config: OptimizeConfig, data_path: str | Path) -> None:
        self.config = config
        self.data_path = Path(data_path)
        self._prepared: Any | None = None
        self._dataset: Any | None = None

    def run(self) -> optuna.Study:
        """Create or resume the study and execute ``n_trials`` trials."""
        study = self._build_study()
        logger.info(
            "Optimizing {} | study={} | metric={} | direction={} | trials={}",
            self.config.model,
            self.config.resolved_study_name(),
            self.config.metric,
            study.direction.name.lower(),
            self.config.n_trials,
        )
        study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            gc_after_trial=True,
        )
        self._log_study_summary(study)
        return study

    def _build_study(self) -> optuna.Study:
        sampler: optuna.samplers.BaseSampler
        if self.config.sampler == "random":
            sampler = optuna.samplers.RandomSampler(seed=self.config.seed)
        else:
            sampler = optuna.samplers.TPESampler(
                seed=self.config.seed,
                n_startup_trials=self.config.startup_trials,
            )

        pruner: optuna.pruners.BasePruner
        if self.config.pruner == "none":
            pruner = optuna.pruners.NopPruner()
        else:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.config.startup_trials,
                n_warmup_steps=1,
            )

        direction = self.config.resolved_direction()
        return optuna.create_study(
            study_name=self.config.resolved_study_name(),
            storage=self.config.storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=self.config.load_if_exists,
        )

    def _ensure_dataset(self, model_name: str) -> Any:
        """Load dataset and prepare windowed splits once per optimizer run."""
        if self._prepared is not None:
            return self._prepared

        logger.info("Loading dataset from: {}", self.data_path)
        dataset = load_dataset(self.data_path)
        dataset.print_summary()
        model_cls = get_model_class(model_name)
        prepared = dataset.prepare(
            window_config=self.config.data.window,
            split_config=self.config.data.split,
            features=self.config.features,
            required_metadata=model_cls.required_metadata,
        )
        if not prepared.has_val:
            raise RuntimeError(
                "Optimization requires a validation split. "
                "Use a data split config with val_ratio > 0."
            )
        self._dataset = dataset
        self._prepared = prepared
        return prepared

    def _objective(self, trial: optuna.trial.Trial) -> float:
        prepared = self._ensure_dataset(self.config.model)

        train_overrides = suggest_train_hyperparams(trial)
        model_kwargs = get_search_space(self.config.model)(trial)

        train_cfg = TrainConfig(
            model=self.config.model,
            epochs=self.config.epochs,
            seed=self.config.seed,
            features=self.config.features,
            data=self.config.data,
            model_kwargs=model_kwargs,
            **train_overrides,
        )
        seed_everything(train_cfg.seed)

        net = create_model(
            self.config.model,
            input_size=prepared.input_size,
            num_classes=FaultType.count(),
            metadata=prepared.metadata,
            **train_cfg.model_kwargs,
        )

        prune_cb = _OptunaPruneCallback(trial, self.config.metric)
        trainer = Trainer(
            config=train_cfg,
            callbacks=[LoggingCallback(), prune_cb],
        )
        result = trainer.fit(
            model=net,
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_val=prepared.X_val,
            y_val=prepared.y_val,
            metadata=prepared.metadata,
            node_mask_train=prepared.node_mask_train,
            edge_mask_train=prepared.edge_mask_train,
            node_mask_val=prepared.node_mask_val,
            edge_mask_val=prepared.edge_mask_val,
        )

        if prune_cb.pruned:
            raise optuna.TrialPruned()

        value = self._final_metric(result)
        if value is None:
            raise optuna.TrialPruned("Metric unavailable for trial")
        return value

    def _final_metric(self, result: Any) -> float | None:
        """Pull the configured metric out of the training history."""
        if not result.history:
            return None
        if self.config.metric == "val_loss":
            return result.best_val_loss
        last = result.history[-1]
        if self.config.metric == "val_macro_f1":
            return last.val_macro_f1
        if self.config.metric == "val_acc":
            return last.val_acc
        return None

    @staticmethod
    def _log_study_summary(study: optuna.Study) -> None:
        try:
            best = study.best_trial
        except ValueError:
            logger.warning("No completed trials in study")
            return
        logger.info(
            "Best trial #{}: value={:.6f}", best.number, best.value if best.value is not None else float("nan")
        )
        for k, v in best.params.items():
            logger.info("  {} = {}", k, v)
