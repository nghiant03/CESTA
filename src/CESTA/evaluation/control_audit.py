"""Audit locked matched-control test records against validation selections."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from CESTA.evaluation.control_budgets import load_yaml_payload


@dataclass(frozen=True)
class LockedControlAuditRecord:
    budget_id: str
    source_variant: str
    controller: str
    selected_variant: str
    dataset: str
    seed: int
    validation_selection_status: str
    validation_energy_ratio: float
    test_energy_j: float
    test_target_energy_j: float
    test_energy_ratio: float
    test_budget_status: str
    macro_f1: float
    request_ratio: float
    artifact_path: str


def audit_locked_control_runs(lock_path: str | Path, test_runs_csv: str | Path) -> dict[str, Any]:
    lock = load_yaml_payload(lock_path)
    if lock.get("selection_split") != "val":
        raise ValueError("Policy lock was not selected on validation data")
    expected_seeds = tuple(_integer(value, "expected_seeds") for value in _list(lock, "expected_seeds"))
    matching = _mapping(lock.get("matching"), "matching")
    minimum_ratio = _number(matching.get("minimum_target_ratio"), "minimum_target_ratio")
    maximum_ratio = _number(matching.get("maximum_target_ratio"), "maximum_target_ratio")
    rows = _read_csv(test_runs_csv)
    if any(row.get("split") != "test" for row in rows):
        raise ValueError("Locked-control audit accepts test records only")

    expected: list[str] = []
    missing: list[str] = []
    duplicate: dict[str, list[str]] = {}
    invalid: dict[str, list[str]] = {}
    records: list[LockedControlAuditRecord] = []
    for raw_lock in _list(lock, "locks"):
        policy = _mapping(raw_lock, "lock")
        dataset = _text(policy.get("dataset"), "lock.dataset")
        variant = _text(policy.get("selected_variant"), "lock.selected_variant")
        expected_config = policy.get("selected_communication_config")
        target_energy = _number(policy.get("target_total_energy_j"), "target_total_energy_j")
        for seed in expected_seeds:
            key = f"{variant}/{dataset}/seed{seed}"
            expected.append(key)
            matches = [row for row in rows if row.get("variant") == variant and row.get("dataset") == dataset and _row_seed(row) == seed]
            if not matches:
                missing.append(key)
                continue
            if len(matches) > 1:
                duplicate[key] = [_required(row, "artifact_path") for row in matches]
                continue
            row = matches[0]
            errors = _validate_locked_row(row, expected_config)
            if errors:
                invalid[key] = errors
                continue
            energy = _number(_required(row, "total_energy_j"), "total_energy_j")
            ratio = energy / target_energy if target_energy > 0.0 else 0.0
            if minimum_ratio <= ratio <= maximum_ratio:
                budget_status = "matched"
            elif ratio < minimum_ratio:
                budget_status = "under_budget_unmatched"
            else:
                budget_status = "over_budget_drift"
            records.append(
                LockedControlAuditRecord(
                    budget_id=_text(policy.get("budget_id"), "lock.budget_id"),
                    source_variant=_text(policy.get("source_variant"), "lock.source_variant"),
                    controller=_text(policy.get("controller"), "lock.controller"),
                    selected_variant=variant,
                    dataset=dataset,
                    seed=seed,
                    validation_selection_status=_text(policy.get("selection_status"), "lock.selection_status"),
                    validation_energy_ratio=_number(policy.get("validation_energy_ratio"), "validation_energy_ratio"),
                    test_energy_j=energy,
                    test_target_energy_j=target_energy,
                    test_energy_ratio=ratio,
                    test_budget_status=budget_status,
                    macro_f1=_number(_required(row, "macro_f1"), "macro_f1"),
                    request_ratio=_number(_required(row, "request_ratio"), "request_ratio"),
                    artifact_path=_required(row, "artifact_path"),
                )
            )

    return {
        "complete": not missing and not duplicate and not invalid,
        "lock_content_sha256": lock.get("content_sha256"),
        "expected": expected,
        "valid": [f"{record.selected_variant}/{record.dataset}/seed{record.seed}" for record in records],
        "missing": missing,
        "duplicate": duplicate,
        "invalid": invalid,
        "records": [asdict(record) for record in records],
    }


def write_locked_control_audit(audit: dict[str, Any], output: str | Path) -> None:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    records = audit.get("records", [])
    fieldnames = list(LockedControlAuditRecord.__dataclass_fields__)
    with (directory / "runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _validate_locked_row(row: dict[str, str], expected_config: object) -> list[str]:
    errors = []
    try:
        actual_config = json.loads(_required(row, "communication_config_json"))
        if actual_config != expected_config:
            errors.append("communication config differs from validation lock")
        if _required(row, "git_dirty").lower() != "false":
            errors.append("run is dirty")
        _required(row, "git_commit")
        _required(row, "data_sha256")
        _required(row, "meta_sha256")
    except (ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
    return errors


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def _row_seed(row: dict[str, str]) -> int | None:
    try:
        return _integer(row.get("seed"), "seed")
    except ValueError:
        return None


def _required(row: dict[str, str], key: str) -> str:
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
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    return integer


def _number(value: object, name: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
