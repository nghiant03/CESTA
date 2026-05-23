"""CLI subcommand for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from CESTA.schema import TrainConfig
from CESTA.schema.config import load_config_file
from CESTA.workflows import run_training


def train(
    config_file: Annotated[
        Path,
        typer.Argument(help="Path to YAML/JSON training config file"),
    ],
    data: Annotated[
        Path,
        typer.Argument(help="Path to injected dataset directory"),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Parent directory for runs (default: runs/<model>). A new run subdirectory is created per invocation.",
        ),
    ] = None,
    early_stopping: Annotated[
        bool,
        typer.Option("--early-stopping/--no-early-stopping", help="Enable early stopping"),
    ] = False,
) -> None:
    """Train a fault diagnosis model."""
    config = TrainConfig.model_validate(load_config_file(config_file))
    run_training(config=config, data=data, output=output, early_stopping=early_stopping)
