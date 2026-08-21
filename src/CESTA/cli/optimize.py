"""CLI subcommand for hyperparameter optimization with Optuna."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from CESTA.logging import logger
from CESTA.schema import OptimizeConfig

app = typer.Typer(no_args_is_help=True, invoke_without_command=True)


@app.callback()
def optimize(
    ctx: typer.Context,
    data: Annotated[
        Optional[Path],
        typer.Option("--data", "-d", help="Path to canonical CESTA dataset directory"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model architecture"),
    ] = "cnn1d",
    n_trials: Annotated[
        int,
        typer.Option("--n-trials", "-n", help="Number of trials"),
    ] = 20,
    timeout: Annotated[
        Optional[int],
        typer.Option("--timeout", "-t", help="Optimization timeout in seconds"),
    ] = None,
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Epochs per trial"),
    ] = 20,
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            help="Metric to optimize: val_loss|val_macro_f1|val_acc",
        ),
    ] = "val_loss",
    sampler: Annotated[
        str,
        typer.Option("--sampler", help="Sampler: tpe|random"),
    ] = "tpe",
    pruner: Annotated[
        str,
        typer.Option("--pruner", help="Pruner: median|none"),
    ] = "median",
    study_name: Annotated[
        Optional[str],
        typer.Option("--study-name", help="Optuna study name (default: cesta-<model>)"),
    ] = None,
    storage: Annotated[
        str,
        typer.Option("--storage", help="Optuna storage URL"),
    ] = "sqlite:///optuna.db",
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="Random seed"),
    ] = 42,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Path to write best params as JSON"),
    ] = None,
    features: Annotated[
        Optional[list[str]],
        typer.Option("--features", "-f", help="Subset of features to use (default: all)"),
    ] = None,
) -> None:
    """Run hyperparameter optimization with Optuna."""
    if ctx.invoked_subcommand is not None:
        return
    if data is None:
        raise typer.BadParameter("Missing option '--data'.")

    config = OptimizeConfig(
        model=model,
        n_trials=n_trials,
        timeout=timeout,
        epochs=epochs,
        metric=metric,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        seed=seed,
        features=features,
    )
    logger.debug("OptimizeConfig: {}", config.model_dump(mode="json"))

    from CESTA.optimization import Optimizer

    optimizer = Optimizer(config=config, data_path=data)
    study = optimizer.run()

    try:
        best = study.best_trial
    except ValueError:
        logger.warning("No completed trials; nothing to save.")
        return

    if output is not None:
        payload = {
            "study_name": config.resolved_study_name(),
            "metric": config.metric,
            "direction": study.direction.name.lower(),
            "best_value": best.value,
            "best_trial_number": best.number,
            "best_params": best.params,
            "optimize_config": config.model_dump(mode="json"),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        logger.info("Best params written to: {}", output)


@app.command("show")
def optimize_show(
    study_name: Annotated[
        str,
        typer.Argument(help="Name of the study to show"),
    ],
    storage: Annotated[
        str,
        typer.Option("--storage", help="Optuna storage URL"),
    ] = "sqlite:///optuna.db",
    top: Annotated[
        int,
        typer.Option("--top", "-k", help="Number of top trials to display"),
    ] = 10,
) -> None:
    """Show results from an existing Optuna study."""
    import optuna
    from rich.console import Console
    from rich.table import Table

    study = optuna.load_study(study_name=study_name, storage=storage)

    console = Console()
    direction = study.direction.name.lower()
    console.print(f"[bold]Study:[/bold] {study_name}  [bold]direction:[/bold] {direction}  [bold]trials:[/bold] {len(study.trials)}")

    completed = [t for t in study.trials if t.value is not None]
    reverse = direction == "maximize"
    completed.sort(key=lambda t: t.value if t.value is not None else 0.0, reverse=reverse)

    table = Table(title=f"Top {min(top, len(completed))} trials", show_header=True)
    table.add_column("#", style="cyan")
    table.add_column("value", style="green")
    table.add_column("state")
    table.add_column("params")
    for trial in completed[:top]:
        table.add_row(
            str(trial.number),
            f"{trial.value:.6f}" if trial.value is not None else "-",
            trial.state.name,
            ", ".join(f"{k}={v}" for k, v in trial.params.items()),
        )
    console.print(table)

    try:
        best = study.best_trial
        console.print(f"\n[bold]Best trial:[/bold] #{best.number}  value={best.value}")
    except ValueError:
        console.print("[yellow]No completed trials yet.[/yellow]")
