"""CLI subcommand for canonical dataset transformation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from CESTA.schema import TransformConfig
from CESTA.schema.config import load_config_file
from CESTA.transform import run_transform


def transform(
    dataset: Annotated[
        str,
        typer.Argument(help="Dataset to use"),
    ],
    raw_path: Annotated[
        Path,
        typer.Argument(help="Path to raw dataset file or directory"),
    ],
    output: Annotated[
        Path,
        typer.Argument(help="Output path for canonical CESTA dataset"),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to YAML/JSON transform config file"),
    ] = None,
) -> None:
    """Transform raw data into a canonical CESTA dataset artifact."""
    transform_config = TransformConfig.model_validate(load_config_file(config)) if config is not None else TransformConfig()
    result = run_transform(dataset_name=dataset, raw_path=raw_path, output=output, config=transform_config)
    result.print_summary()
