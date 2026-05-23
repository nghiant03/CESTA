"""Reusable evaluation workflow orchestration."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from CESTA.artifacts import create_run_dir, load_checkpoint_metadata, load_checkpoint_weights, write_manifest
from CESTA.datasets import load_dataset
from CESTA.evaluation import Evaluator
from CESTA.logging import logger
from CESTA.models import create_model, get_model_class
from CESTA.schema import (
    EvaluateConfig,
    RunManifest,
    Timing,
)
from CESTA.schema.fault import FaultType
from CESTA.utils import (
    collect_env_info,
    collect_git_info,
    utc_now_iso,
)


def run_evaluation(model: Path, data: Path, config: EvaluateConfig, output: Path | None = None) -> None:
    from CESTA.models.base import BaseModel
    from CESTA.schema.window import DataConfig

    logger.info("Loading data from: {}", data)
    dataset = load_dataset(data)
    dataset.print_summary()

    logger.info("Loading model from: {}", model)
    meta = load_checkpoint_metadata(model)
    model_name = str(meta.get("model_name", "lstm"))
    model_config = meta.get("model_config", {})
    assert isinstance(model_config, dict)

    train_cfg = meta.get("train_config")
    saved_features: list[str] | None = None
    data_config = None
    if isinstance(train_cfg, dict):
        saved_features = train_cfg.get("features")
        data_config = train_cfg.get("data")

    model_cls = get_model_class(model_name)
    if isinstance(data_config, dict):
        resolved_data_config = DataConfig.model_validate(data_config)
    else:
        resolved_data_config = DataConfig()
    prepared = dataset.prepare(
        window_config=resolved_data_config.window,
        split_config=resolved_data_config.split,
        features=saved_features,
        required_metadata=model_cls.required_metadata,
    )

    if not prepared.has_test:
        logger.error("No test data available in dataset")
        raise typer.Exit(code=1)

    input_size = prepared.input_size
    num_classes = FaultType.count()
    model_kwargs = {}
    if isinstance(train_cfg, dict):
        saved_model_kwargs = train_cfg.get("model_kwargs", {})
        if isinstance(saved_model_kwargs, dict):
            model_kwargs = saved_model_kwargs
    net = create_model(
        model_name,
        input_size=input_size,
        num_classes=num_classes,
        metadata=prepared.metadata,
        **model_kwargs,
    )
    assert isinstance(net, BaseModel)
    load_checkpoint_weights(net, model)
    logger.info("Model: {} ({:,} parameters)", net.name, net.count_parameters())

    evaluator = Evaluator(config=config)
    logger.info("Evaluating with batch_size={}", config.batch_size)

    started_at = utc_now_iso()
    t0 = time.perf_counter()
    result = evaluator.evaluate(
        net,
        prepared.X_test,
        prepared.y_test,
        metadata=prepared.metadata,
        node_mask=prepared.node_mask_test,
        edge_mask=prepared.edge_mask_test,
    )
    duration = time.perf_counter() - t0
    ended_at = utc_now_iso()

    evaluator.log_results(result)

    if output is not None:
        git = collect_git_info()
        seed = int(train_cfg.get("seed", 0)) if isinstance(train_cfg, dict) else 0
        save_dir = create_run_dir(output, model=model_name, seed=seed, git=git)
    else:
        git = collect_git_info()
        seed = int(train_cfg.get("seed", 0)) if isinstance(train_cfg, dict) else 0
        save_dir = model
        save_dir.mkdir(parents=True, exist_ok=True)

    result.save(
        save_dir,
        train_config=train_cfg,  # type: ignore[arg-type]
        injection_config=dataset.config.model_dump(mode="json"),
    )
    logger.info("Results saved to: {}", save_dir)

    if output is not None:
        env = collect_env_info(evaluator.device)
        dataset_info = dataset.describe(data)
        manifest = RunManifest(
            run_id=save_dir.name,
            kind="evaluate",
            seed=seed,
            model=model_name,
            num_parameters=net.count_parameters(),
            git=git,
            env=env,
            dataset=dataset_info,
            timing=Timing(
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=duration,
            ),
            train_config=train_cfg if isinstance(train_cfg, dict) else None,
            eval_config=config.model_dump(mode="json"),
            injection_config=dataset.config.model_dump(mode="json"),
        )
        manifest_path = write_manifest(save_dir, manifest)
        logger.info("Manifest written to: {}", manifest_path)
