"""Reusable training workflow orchestration."""

from __future__ import annotations

import time
from pathlib import Path

from CESTA.artifacts import create_run_dir, load_checkpoint_weights, write_manifest
from CESTA.datasets import load_dataset
from CESTA.evaluation import Evaluator
from CESTA.logging import logger
from CESTA.models import create_model, get_model_class
from CESTA.schema import (
    EvaluateConfig,
    RunManifest,
    Timing,
    TrainConfig,
)
from CESTA.schema.fault import FaultType
from CESTA.seed import seed_everything
from CESTA.training import (
    CheckpointCallback,
    EarlyStoppingCallback,
    HistoryCallback,
    LoggingCallback,
    Trainer,
    build_loss,
)
from CESTA.utils import (
    collect_env_info,
    collect_git_info,
    utc_now_iso,
)


def run_training(
    config: TrainConfig,
    data: Path,
    output: Path | None = None,
    early_stopping: bool = False,
    test_evaluation: bool = True,
) -> None:
    logger.debug("TrainConfig: {}", config.model_dump(mode="json"))

    logger.info("Loading data from: {}", data)
    dataset = load_dataset(data)
    dataset.print_summary()

    model_cls = get_model_class(config.model)
    requested_metadata = set(model_cls.required_metadata)
    if config.model_kwargs.get("node_embedding_dim", 0):
        requested_metadata.add("node_identity")
    prepared = dataset.prepare(
        window_config=config.data.window,
        split_config=config.data.split,
        features=config.features,
        required_metadata=requested_metadata,
    )

    logger.debug(
        "Windowed shapes: X_train={}, y_train={}, X_val={}, y_val={}, X_test={}, y_test={}",
        prepared.X_train.shape,
        prepared.y_train.shape,
        prepared.X_val.shape,
        prepared.y_val.shape,
        prepared.X_test.shape,
        prepared.y_test.shape,
    )

    input_size = prepared.input_size
    num_classes = FaultType.count()
    seed_everything(config.seed)
    logger.debug(
        "Creating model: arch={}, input_size={}, num_classes={}",
        config.model,
        input_size,
        num_classes,
    )

    net = create_model(
        config.model,
        input_size=input_size,
        num_classes=num_classes,
        metadata=prepared.metadata,
        **config.model_kwargs,
    )
    logger.info("Model: {} ({:,} parameters)", net.name, net.count_parameters())

    output_root = output if output is not None else Path(f"runs/{config.model}")
    git = collect_git_info()
    run_dir = create_run_dir(output_root, model=config.model, seed=config.seed, git=git)
    run_id = run_dir.name
    logger.info("Run dir: {}", run_dir)

    energy_constrained = config.communication_penalty_mode == "energy_budget_hinge"
    checkpoint_callback = CheckpointCallback(
        save_path=run_dir,
        config_dict=config.model_dump(mode="json"),
        monitor=config.checkpoint_monitor,
        maximum_val_energy_ratio=config.target_energy_ratio if energy_constrained else None,
    )
    callbacks = [
        LoggingCallback(),
        checkpoint_callback,
        HistoryCallback(save_path=run_dir),
    ]

    if early_stopping:
        callbacks.append(EarlyStoppingCallback(patience=10, monitor=config.early_stopping_monitor))

    trainer = Trainer(config=config, callbacks=callbacks)

    env = collect_env_info(trainer.device)
    dataset_info = dataset.describe(data)

    logger.info(
        "Training for {} epochs | batch_size={} | lr={} | focal_loss={} | oversample={}",
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.use_focal_loss,
        config.oversample,
    )

    started_at = utc_now_iso()
    t0 = time.perf_counter()
    result = trainer.fit(
        model=net,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_val=prepared.X_val if prepared.has_val else None,
        y_val=prepared.y_val if prepared.has_val else None,
        metadata=prepared.metadata,
        node_mask_train=prepared.node_mask_train,
        edge_mask_train=prepared.edge_mask_train,
        node_mask_val=prepared.node_mask_val if prepared.has_val else None,
        edge_mask_val=prepared.edge_mask_val if prepared.has_val else None,
    )
    duration = time.perf_counter() - t0

    logger.info(
        "Training complete at epoch {} | best_val_loss={:.4f}",
        result.stopped_epoch,
        result.best_val_loss if result.best_val_loss is not None else float("nan"),
    )
    if not checkpoint_callback.has_eligible_checkpoint:
        raise RuntimeError("training produced no budget-feasible validation checkpoint")
    logger.info("Model saved to: {}", run_dir)

    if test_evaluation and prepared.has_test:
        logger.info("--- Final Test Evaluation ---")
        weight_path = run_dir / "weight.pt"
        if weight_path.exists():
            load_checkpoint_weights(net, run_dir, map_location=trainer.device)
            logger.info("Reloaded best checkpoint from {} for test evaluation", weight_path)
        else:
            logger.warning("No checkpoint at {}; evaluating final-epoch weights", weight_path)
        evaluator = Evaluator(
            config=EvaluateConfig(batch_size=config.batch_size),
            device=str(trainer.device),
        )
        criterion = build_loss(config, trainer.device)
        eval_result = evaluator.evaluate(
            net,
            prepared.X_test,
            prepared.y_test,
            criterion=criterion,
            metadata=prepared.metadata,
            node_mask=prepared.node_mask_test,
            edge_mask=prepared.edge_mask_test,
            split="test",
        )
        evaluator.log_results(eval_result)

        eval_result.save(
            run_dir,
            train_config=config.model_dump(mode="json"),
            injection_config=dataset.config.model_dump(mode="json"),
        )
        logger.info("Results saved to: {}", run_dir)

    ended_at = utc_now_iso()
    manifest = RunManifest(
        run_id=run_id,
        kind="train",
        seed=config.seed,
        model=config.model,
        num_parameters=net.count_parameters(),
        total_parameters=net.count_all_parameters(),
        git=git,
        env=env,
        dataset=dataset_info,
        timing=Timing(
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            epochs_run=result.stopped_epoch,
        ),
        train_config=config.model_dump(mode="json"),
        injection_config=dataset.config.model_dump(mode="json"),
    )
    manifest_path = write_manifest(run_dir, manifest)
    logger.info("Manifest written to: {}", manifest_path)
