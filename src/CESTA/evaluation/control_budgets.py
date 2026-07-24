"""Validation-only budget derivation and deterministic control selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class BudgetSourceRecord:
    source_variant: str
    dataset: str
    seed: int
    artifact_path: str
    git_commit: str
    data_sha256: str
    meta_sha256: str
    request_ratio: float
    requested_messages: float
    possible_messages: float
    transmitted_bits: float
    tx_energy_j: float
    rx_energy_j: float
    total_energy_j: float
    dense_total_energy_j: float
    comparison_signature_json: str


@dataclass(frozen=True)
class ControlCandidate:
    controller: str
    variant: str
    dataset: str
    seeds: tuple[int, ...]
    validation_macro_f1_mean: float
    validation_macro_f1_std: float
    request_ratio_mean: float
    total_energy_j_mean: float
    energy_target_j: float
    energy_ratio: float
    energy_mismatch_ratio: float
    status: str
    num_parameters: int
    total_parameters: int
    communication_config: dict[str, Any]
    artifact_paths: tuple[str, ...]


def derive_control_budgets(
    runs_csv: str | Path,
    *,
    source_variants: Iterable[str],
    expected_seeds: Iterable[int],
    tolerance_ratio: float = 0.02,
) -> dict[str, Any]:
    if not 0.0 <= tolerance_ratio < 1.0:
        raise ValueError("tolerance_ratio must be in [0, 1)")
    variants = tuple(source_variants)
    seeds = tuple(expected_seeds)
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("source_variants must be non-empty and unique")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("expected_seeds must be non-empty and unique")

    rows = _read_csv(runs_csv)
    if any(row.get("split") != "val" for row in rows):
        raise ValueError("Budget derivation accepts validation records only")
    selected = [_budget_source_record(row) for row in rows if row.get("variant") in variants]
    if not selected:
        raise ValueError("No source-variant records found")
    unexpected = sorted({record.source_variant for record in selected} - set(variants))
    if unexpected:
        raise ValueError(f"Unexpected source variants: {unexpected}")

    grouped: dict[tuple[str, str], list[BudgetSourceRecord]] = {}
    seen: set[tuple[str, str, int]] = set()
    for record in selected:
        key = (record.source_variant, record.dataset, record.seed)
        if key in seen:
            raise ValueError(f"Duplicate budget source cell: {key}")
        seen.add(key)
        grouped.setdefault((record.source_variant, record.dataset), []).append(record)

    budgets = []
    for source_variant in variants:
        datasets = sorted(dataset for variant, dataset in grouped if variant == source_variant)
        if not datasets:
            raise ValueError(f"Source variant has no records: {source_variant}")
        for dataset in datasets:
            records = sorted(grouped[(source_variant, dataset)], key=lambda item: item.seed)
            actual_seeds = tuple(record.seed for record in records)
            if actual_seeds != tuple(sorted(seeds)):
                raise ValueError(
                    f"Budget source coverage for {source_variant}/{dataset} is {actual_seeds}, expected {tuple(sorted(seeds))}"
                )
            signatures = {record.comparison_signature_json for record in records}
            hashes = {(record.data_sha256, record.meta_sha256) for record in records}
            if len(signatures) != 1 or len(hashes) != 1:
                raise ValueError(f"Budget source provenance differs across seeds for {source_variant}/{dataset}")
            budgets.append(_aggregate_budget(source_variant, dataset, records, tolerance_ratio))

    payload: dict[str, Any] = {
        "version": 1,
        "selection_split": "val",
        "aggregation": "dataset_mean_across_seeds",
        "expected_seeds": list(sorted(seeds)),
        "matching": {
            "primary_metric": "tx_plus_rx_energy_j",
            "minimum_target_ratio": 1.0 - tolerance_ratio,
            "maximum_target_ratio": 1.0,
            "matched_selection": "highest_validation_macro_f1",
            "fallback_selection": "nearest_below_target",
            "tie_breakers": [
                "smaller_energy_mismatch",
                "lower_validation_macro_f1_std",
                "fewer_active_parameters",
                "lexical_variant",
            ],
        },
        "budgets": budgets,
    }
    payload["content_sha256"] = _payload_hash(payload)
    return payload


def select_control_policies(
    budget_payload: dict[str, Any],
    validation_runs_csv: str | Path,
    controller_variants: dict[str, list[str]],
) -> dict[str, Any]:
    if budget_payload.get("selection_split") != "val":
        raise ValueError("Control budgets must be derived from validation records")
    rows = _read_csv(validation_runs_csv)
    if any(row.get("split") != "val" for row in rows):
        raise ValueError("Control selection accepts validation records only")
    expected_seeds = tuple(_integer(value, "expected_seeds") for value in _list(budget_payload, "expected_seeds"))
    matching = _mapping(budget_payload.get("matching"), "matching")
    minimum_ratio = _finite(matching.get("minimum_target_ratio"), "minimum_target_ratio")
    maximum_ratio = _finite(matching.get("maximum_target_ratio"), "maximum_target_ratio")
    budgets = _list(budget_payload, "budgets")
    known_variants = {variant for variants in controller_variants.values() for variant in variants}
    candidate_rows = [row for row in rows if row.get("variant") in known_variants]

    locks = []
    for budget_raw in budgets:
        budget = _mapping(budget_raw, "budget")
        dataset = _text(budget.get("dataset"), "budget.dataset")
        source_variant = _text(budget.get("source_variant"), "budget.source_variant")
        target_energy = _finite(budget.get("target_total_energy_j"), "budget.target_total_energy_j")
        for controller, variants in sorted(controller_variants.items()):
            candidates = [
                _aggregate_candidate(
                    controller,
                    variant,
                    dataset,
                    source_variant,
                    target_energy,
                    expected_seeds,
                    candidate_rows,
                    minimum_ratio,
                    maximum_ratio,
                )
                for variant in variants
            ]
            eligible = [candidate for candidate in candidates if candidate.status == "matched"]
            if eligible:
                selected = min(
                    eligible,
                    key=lambda item: (
                        -item.validation_macro_f1_mean,
                        item.energy_mismatch_ratio,
                        item.validation_macro_f1_std,
                        item.num_parameters,
                        item.variant,
                    ),
                )
            else:
                under_budget = [candidate for candidate in candidates if candidate.energy_ratio <= maximum_ratio]
                if not under_budget:
                    raise ValueError(f"All {controller} candidates exceed {source_variant}/{dataset} energy target")
                selected = min(
                    under_budget,
                    key=lambda item: (
                        item.energy_mismatch_ratio,
                        -item.validation_macro_f1_mean,
                        item.validation_macro_f1_std,
                        item.num_parameters,
                        item.variant,
                    ),
                )
            locks.append(
                {
                    "budget_id": _text(budget.get("id"), "budget.id"),
                    "source_variant": source_variant,
                    "dataset": dataset,
                    "controller": controller,
                    "selection_status": selected.status,
                    "selected_variant": selected.variant,
                    "selected_communication_config": selected.communication_config,
                    "target_total_energy_j": target_energy,
                    "achieved_validation_total_energy_j": selected.total_energy_j_mean,
                    "validation_energy_ratio": selected.energy_ratio,
                    "validation_energy_mismatch_ratio": selected.energy_mismatch_ratio,
                    "validation_macro_f1_mean": selected.validation_macro_f1_mean,
                    "validation_macro_f1_std": selected.validation_macro_f1_std,
                    "num_parameters": selected.num_parameters,
                    "total_parameters": selected.total_parameters,
                    "source_artifacts": list(selected.artifact_paths),
                    "candidates": [asdict(candidate) for candidate in sorted(candidates, key=lambda item: item.variant)],
                }
            )

    payload = {
        "version": 1,
        "selection_split": "val",
        "budget_content_sha256": budget_payload.get("content_sha256"),
        "expected_seeds": list(expected_seeds),
        "matching": matching,
        "locks": locks,
    }
    payload["content_sha256"] = _payload_hash(payload)
    return payload


def write_yaml_payload(payload: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    temporary.replace(path)


def load_yaml_payload(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    payload = _mapping(raw, str(path))
    expected_hash = payload.get("content_sha256")
    without_hash = dict(payload)
    without_hash.pop("content_sha256", None)
    if expected_hash != _payload_hash(without_hash):
        raise ValueError(f"Content hash does not match: {path}")
    return payload


def _budget_source_record(row: dict[str, str]) -> BudgetSourceRecord:
    if row.get("split") != "val":
        raise ValueError("Budget derivation accepts validation records only")
    return BudgetSourceRecord(
        source_variant=_required_row(row, "variant"),
        dataset=_required_row(row, "dataset"),
        seed=_integer(_required_row(row, "seed"), "seed"),
        artifact_path=_required_row(row, "artifact_path"),
        git_commit=_required_row(row, "git_commit"),
        data_sha256=_required_row(row, "data_sha256"),
        meta_sha256=_required_row(row, "meta_sha256"),
        request_ratio=_finite(_required_row(row, "request_ratio"), "request_ratio"),
        requested_messages=_finite(_required_row(row, "requested_messages"), "requested_messages"),
        possible_messages=_finite(_required_row(row, "possible_messages"), "possible_messages"),
        transmitted_bits=_finite(_required_row(row, "transmitted_bits"), "transmitted_bits"),
        tx_energy_j=_finite(_required_row(row, "tx_energy_j"), "tx_energy_j"),
        rx_energy_j=_finite(_required_row(row, "rx_energy_j"), "rx_energy_j"),
        total_energy_j=_finite(_required_row(row, "total_energy_j"), "total_energy_j"),
        dense_total_energy_j=_finite(_required_row(row, "dense_total_energy_j"), "dense_total_energy_j"),
        comparison_signature_json=_required_row(row, "comparison_signature_json"),
    )


def _aggregate_budget(
    source_variant: str,
    dataset: str,
    records: list[BudgetSourceRecord],
    tolerance_ratio: float,
) -> dict[str, Any]:
    data_sha256, meta_sha256 = next(iter({(record.data_sha256, record.meta_sha256) for record in records}))
    return {
        "id": f"{source_variant}__{dataset}",
        "source_variant": source_variant,
        "dataset": dataset,
        "seeds": [record.seed for record in records],
        "target_request_ratio": statistics.fmean(record.request_ratio for record in records),
        "target_requested_messages": statistics.fmean(record.requested_messages for record in records),
        "target_possible_messages": statistics.fmean(record.possible_messages for record in records),
        "target_transmitted_bits": statistics.fmean(record.transmitted_bits for record in records),
        "target_tx_energy_j": statistics.fmean(record.tx_energy_j for record in records),
        "target_rx_energy_j": statistics.fmean(record.rx_energy_j for record in records),
        "target_total_energy_j": statistics.fmean(record.total_energy_j for record in records),
        "dense_reference_total_energy_j": statistics.fmean(record.dense_total_energy_j for record in records),
        "allowed_total_energy_j": statistics.fmean(record.total_energy_j for record in records),
        "minimum_matched_total_energy_j": (1.0 - tolerance_ratio) * statistics.fmean(record.total_energy_j for record in records),
        "data_sha256": data_sha256,
        "meta_sha256": meta_sha256,
        "comparison_signature_json": records[0].comparison_signature_json,
        "source_records": [asdict(record) for record in records],
    }


def _aggregate_candidate(
    controller: str,
    variant: str,
    dataset: str,
    source_variant: str,
    target_energy: float,
    expected_seeds: tuple[int, ...],
    rows: list[dict[str, str]],
    minimum_ratio: float,
    maximum_ratio: float,
) -> ControlCandidate:
    selected = sorted(
        (row for row in rows if row.get("variant") == variant and row.get("dataset") == dataset),
        key=lambda row: _integer(_required_row(row, "seed"), "seed"),
    )
    seeds = tuple(_integer(_required_row(row, "seed"), "seed") for row in selected)
    if seeds != tuple(sorted(expected_seeds)):
        raise ValueError(f"Candidate coverage for {variant}/{dataset} is {seeds}, expected {tuple(sorted(expected_seeds))}")
    configs = {_required_row(row, "communication_config_json") for row in selected}
    active_parameters = {_integer(_required_row(row, "num_parameters"), "num_parameters") for row in selected}
    total_parameters = {_integer(_required_row(row, "total_parameters"), "total_parameters") for row in selected}
    if len(configs) != 1 or len(active_parameters) != 1 or len(total_parameters) != 1:
        raise ValueError(f"Candidate configuration or parameter counts differ across {variant}/{dataset}")
    energy = statistics.fmean(_finite(_required_row(row, "total_energy_j"), "total_energy_j") for row in selected)
    energy_ratio = energy / target_energy if target_energy > 0.0 else 0.0
    if minimum_ratio <= energy_ratio <= maximum_ratio:
        status = "matched"
    elif energy_ratio < minimum_ratio:
        status = "under_budget_unmatched"
    else:
        status = "over_budget"
    f1_values = [_finite(_required_row(row, "macro_f1"), "macro_f1") for row in selected]
    return ControlCandidate(
        controller=controller,
        variant=variant,
        dataset=dataset,
        seeds=seeds,
        validation_macro_f1_mean=statistics.fmean(f1_values),
        validation_macro_f1_std=statistics.stdev(f1_values) if len(f1_values) > 1 else 0.0,
        request_ratio_mean=statistics.fmean(_finite(_required_row(row, "request_ratio"), "request_ratio") for row in selected),
        total_energy_j_mean=energy,
        energy_target_j=target_energy,
        energy_ratio=energy_ratio,
        energy_mismatch_ratio=abs(energy_ratio - 1.0),
        status=status,
        num_parameters=active_parameters.pop(),
        total_parameters=total_parameters.pop(),
        communication_config=json.loads(configs.pop()),
        artifact_paths=tuple(_required_row(row, "artifact_path") for row in selected),
    )


def _payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def _required_row(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"CSV row missing {key}")
    return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number
