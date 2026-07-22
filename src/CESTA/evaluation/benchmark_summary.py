"""Deterministic aggregation and paired uncertainty for audited benchmark records."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

_DEFAULT_BOOTSTRAP_SEED = 20260723
_DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class ClassificationRecord:
    variant: str
    dataset: str
    seed: int
    macro_f1: float
    accuracy: float
    normal_f1: float
    spike_f1: float
    drift_f1: float
    stuck_f1: float


@dataclass(frozen=True)
class ValidationRecord:
    variant: str
    dataset: str
    seed: int
    validation_macro_f1: float
    num_parameters: int


@dataclass(frozen=True)
class ComparatorRanking:
    rank: int
    variant: str
    cells: int
    validation_macro_f1_mean: float
    validation_macro_f1_std: float
    num_parameters: int


@dataclass(frozen=True)
class AggregateSummary:
    variant: str
    dataset: str
    runs: int
    macro_f1_mean: float
    macro_f1_std: float
    accuracy_mean: float
    accuracy_std: float
    normal_f1_mean: float
    spike_f1_mean: float
    drift_f1_mean: float
    stuck_f1_mean: float


@dataclass(frozen=True)
class PairedDifference:
    dataset: str
    seed: int
    value: float


@dataclass(frozen=True)
class PairedSummary:
    variant: str
    reference: str
    metric: str
    pairs: int
    mean_difference: float
    difference_std: float
    confidence_level: float
    confidence_low: float
    confidence_high: float
    bootstrap_resamples: int
    bootstrap_seed: int
    differences: tuple[PairedDifference, ...]


def load_classification_records(path: str | Path) -> list[ClassificationRecord]:
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    records = []
    for row in rows:
        records.append(
            ClassificationRecord(
                variant=_required_text(row, "variant"),
                dataset=_required_text(row, "dataset"),
                seed=_required_integer(row, "seed"),
                macro_f1=_required_finite(row, "macro_f1"),
                accuracy=_required_finite(row, "accuracy"),
                normal_f1=_required_finite(row, "normal_f1"),
                spike_f1=_required_finite(row, "spike_f1"),
                drift_f1=_required_finite(row, "drift_f1"),
                stuck_f1=_required_finite(row, "stuck_f1"),
            )
        )
    _require_unique_cells(records)
    return records


def rank_validation_comparators(records: Iterable[ValidationRecord], *, expected_cells: int) -> list[ComparatorRanking]:
    if expected_cells <= 0:
        raise ValueError("expected_cells must be positive")
    grouped: dict[str, list[ValidationRecord]] = {}
    cells: set[tuple[str, str, int]] = set()
    for record in records:
        cell = (record.variant, record.dataset, record.seed)
        if cell in cells:
            raise ValueError("Validation records contain duplicate variant, dataset, and seed cells")
        cells.add(cell)
        grouped.setdefault(record.variant, []).append(record)
    incomplete = {variant: len(items) for variant, items in grouped.items() if len(items) != expected_cells}
    if incomplete:
        raise ValueError(f"Validation comparator coverage is incomplete: {incomplete}")
    summaries = []
    for variant, items in grouped.items():
        values = [record.validation_macro_f1 for record in items]
        parameter_counts = {record.num_parameters for record in items}
        if len(parameter_counts) != 1:
            raise ValueError(f"Parameter count differs across {variant} runs")
        summaries.append(
            (
                variant,
                statistics.fmean(values),
                statistics.stdev(values) if len(values) > 1 else 0.0,
                parameter_counts.pop(),
                len(items),
            )
        )
    ordered = sorted(summaries, key=lambda item: (-item[1], item[2], item[3], item[0]))
    return [
        ComparatorRanking(
            rank=rank,
            variant=variant,
            cells=count,
            validation_macro_f1_mean=mean,
            validation_macro_f1_std=standard_deviation,
            num_parameters=parameters,
        )
        for rank, (variant, mean, standard_deviation, parameters, count) in enumerate(ordered, start=1)
    ]


def aggregate_classification(records: Iterable[ClassificationRecord]) -> list[AggregateSummary]:
    grouped: dict[tuple[str, str], list[ClassificationRecord]] = {}
    materialized = list(records)
    _require_unique_cells(materialized)
    for record in materialized:
        grouped.setdefault((record.variant, record.dataset), []).append(record)
        grouped.setdefault((record.variant, "ALL"), []).append(record)
    return [_aggregate_group(variant, dataset, grouped[(variant, dataset)]) for variant, dataset in sorted(grouped)]


def paired_summary(
    records: Iterable[ClassificationRecord],
    *,
    variant: str,
    reference: str,
    metric: str = "macro_f1",
    bootstrap_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> PairedSummary:
    if metric not in ClassificationRecord.__dataclass_fields__ or metric in {"variant", "dataset", "seed"}:
        raise ValueError(f"Unsupported paired metric: {metric}")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    materialized = list(records)
    _require_unique_cells(materialized)
    variant_values = _values_by_cell(materialized, variant, metric)
    reference_values = _values_by_cell(materialized, reference, metric)
    if variant_values.keys() != reference_values.keys():
        missing_variant = sorted(reference_values.keys() - variant_values.keys())
        missing_reference = sorted(variant_values.keys() - reference_values.keys())
        raise ValueError(f"Paired cells differ; missing {variant}: {missing_variant}; missing {reference}: {missing_reference}")
    if not variant_values:
        raise ValueError("No paired cells found")
    differences = tuple(
        PairedDifference(dataset=dataset, seed=seed, value=variant_values[(dataset, seed)] - reference_values[(dataset, seed)])
        for dataset, seed in sorted(variant_values)
    )
    values = [difference.value for difference in differences]
    confidence_low, confidence_high = _bootstrap_interval(
        values,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return PairedSummary(
        variant=variant,
        reference=reference,
        metric=metric,
        pairs=len(values),
        mean_difference=statistics.fmean(values),
        difference_std=statistics.stdev(values) if len(values) > 1 else 0.0,
        confidence_level=confidence_level,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        differences=differences,
    )


def write_benchmark_summary(
    aggregates: Iterable[AggregateSummary],
    comparisons: Iterable[PairedSummary],
    output: str | Path,
) -> None:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    aggregate_rows = [asdict(summary) for summary in aggregates]
    comparison_rows = [_paired_payload(summary) for summary in comparisons]
    _write_csv(directory / "aggregates.csv", aggregate_rows, list(AggregateSummary.__dataclass_fields__))
    (directory / "paired_comparisons.json").write_text(json.dumps(comparison_rows, indent=2, sort_keys=True) + "\n")


def _aggregate_group(variant: str, dataset: str, records: list[ClassificationRecord]) -> AggregateSummary:
    ordered = sorted(records, key=lambda record: (record.dataset, record.seed))
    macro_f1 = [record.macro_f1 for record in ordered]
    accuracy = [record.accuracy for record in ordered]
    return AggregateSummary(
        variant=variant,
        dataset=dataset,
        runs=len(ordered),
        macro_f1_mean=statistics.fmean(macro_f1),
        macro_f1_std=statistics.stdev(macro_f1) if len(macro_f1) > 1 else 0.0,
        accuracy_mean=statistics.fmean(accuracy),
        accuracy_std=statistics.stdev(accuracy) if len(accuracy) > 1 else 0.0,
        normal_f1_mean=statistics.fmean(record.normal_f1 for record in ordered),
        spike_f1_mean=statistics.fmean(record.spike_f1 for record in ordered),
        drift_f1_mean=statistics.fmean(record.drift_f1 for record in ordered),
        stuck_f1_mean=statistics.fmean(record.stuck_f1 for record in ordered),
    )


def _values_by_cell(records: list[ClassificationRecord], variant: str, metric: str) -> dict[tuple[str, int], float]:
    return {(record.dataset, record.seed): float(getattr(record, metric)) for record in records if record.variant == variant}


def _bootstrap_interval(values: list[float], *, resamples: int, seed: int, confidence_level: float) -> tuple[float, float]:
    generator = random.Random(seed)
    sample_size = len(values)
    means = sorted(statistics.fmean(values[generator.randrange(sample_size)] for _ in range(sample_size)) for _ in range(resamples))
    alpha = (1.0 - confidence_level) / 2.0
    return _quantile(means, alpha), _quantile(means, 1.0 - alpha)


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _require_unique_cells(records: list[ClassificationRecord]) -> None:
    cells = [(record.variant, record.dataset, record.seed) for record in records]
    if len(cells) != len(set(cells)):
        raise ValueError("Classification records contain duplicate variant, dataset, and seed cells")


def _paired_payload(summary: PairedSummary) -> dict[str, Any]:
    payload = asdict(summary)
    payload["differences"] = [asdict(difference) for difference in summary.differences]
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _required_text(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_integer(row: dict[str, str], key: str) -> int:
    try:
        return int(_required_text(row, key))
    except ValueError as error:
        raise ValueError(f"{key} must be an integer") from error


def _required_finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(_required_text(row, key))
    except ValueError as error:
        raise ValueError(f"{key} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value
