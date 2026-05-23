"""CLI subcommand for model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from CESTA.schema import EvaluateConfig
from CESTA.workflows import run_evaluation


def evaluate(
    model: Annotated[
        Path,
        typer.Option("--model", "-m", help="Path to trained model directory"),
    ],
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to injected dataset directory"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for evaluation results"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Evaluation batch size",
        ),
    ] = 64,
) -> None:
    """Evaluate a trained model on test data."""
    config = EvaluateConfig(batch_size=batch_size)
    run_evaluation(model=model, data=data, output=output, config=config)
