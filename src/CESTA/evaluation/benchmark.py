"""Deterministic validation and normalization of benchmark run artifacts."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from CESTA.schema import TrainConfig
from CESTA.schema.config import load_config_file

_REQUIRED_CLASSES = ("NORMAL", "SPIKE", "DRIFT", "STUCK")
_REL_TOLERANCE = 1e-8
_ABS_TOLERANCE = 1e-8


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    path: Path


@dataclass(frozen=True)
class BenchmarkVariant:
    name: str
    config_path: Path
    communication_mode: str
    model: str
    datasets: tuple[str, ...] = ()
    expected_train_config: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class BenchmarkSpec:
    datasets: tuple[BenchmarkDataset, ...]
    seeds: tuple[int, ...]
    variants: tuple[BenchmarkVariant, ...]
    split: str = "test"


@dataclass(frozen=True)
class BenchmarkCell:
    variant: str
    dataset: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.variant}/{self.dataset}/seed{self.seed}"


@dataclass(frozen=True)
class BenchmarkRun:
    variant: str
    dataset: str
    seed: int
    split: str
    run_id: str
    artifact_path: str
    model: str
    communication_mode: str
    git_commit: str
    git_dirty: bool
    data_sha256: str
    meta_sha256: str
    macro_f1: float
    accuracy: float
    normal_f1: float
    spike_f1: float
    drift_f1: float
    stuck_f1: float
    request_ratio: float
    requested_messages: float
    possible_messages: float
    transmitted_bits: float
    bits_per_message: float
    tx_energy_j: float
    rx_energy_j: float
    total_energy_j: float
    dense_tx_energy_j: float
    dense_rx_energy_j: float
    dense_total_energy_j: float
    energy_reduction_j: float
    energy_reduction_ratio: float
    num_parameters: int | None
    total_parameters: int | None
    communication_config_json: str
    comparison_signature_json: str


@dataclass
class BenchmarkAudit:
    expected: list[str]
    valid: list[str]
    missing: list[str]
    duplicate: dict[str, list[str]]
    invalid: dict[str, list[str]]
    unmatched_runs: list[str]
    runs: list[BenchmarkRun]

    @property
    def complete(self) -> bool:
        return not self.missing and not self.duplicate and not self.invalid

    def payload(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "counts": {
                "expected": len(self.expected),
                "valid": len(self.valid),
                "missing": len(self.missing),
                "duplicate": len(self.duplicate),
                "invalid": len(self.invalid),
                "unmatched_runs": len(self.unmatched_runs),
            },
            "expected": self.expected,
            "valid": self.valid,
            "missing": self.missing,
            "duplicate": self.duplicate,
            "invalid": self.invalid,
            "unmatched_runs": self.unmatched_runs,
        }


def load_benchmark_spec(path: str | Path, *, project_root: str | Path | None = None) -> BenchmarkSpec:
    spec_path = Path(path)
    root = Path(project_root) if project_root is not None else Path.cwd()
    raw = load_config_file(spec_path)
    datasets_raw = _require_list(raw, "datasets")
    seeds_raw = _require_list(raw, "seeds")
    variants_raw = _require_list(raw, "variants")
    split = raw.get("split", "test")
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    datasets: list[BenchmarkDataset] = []
    for item in datasets_raw:
        mapping = _require_mapping(item, "dataset")
        name = _require_text(mapping, "name")
        dataset_path = _resolve_path(root, _require_text(mapping, "path"))
        datasets.append(BenchmarkDataset(name=name, path=dataset_path))

    seeds = tuple(_require_integer(seed, "seed") for seed in seeds_raw)
    variants: list[BenchmarkVariant] = []
    for item in variants_raw:
        mapping = _require_mapping(item, "variant")
        name = _require_text(mapping, "name")
        config_path = _resolve_path(root, _require_text(mapping, "config"))
        communication_mode = _require_text(mapping, "communication_mode")
        variant_datasets_raw = mapping.get("datasets", [])
        if not isinstance(variant_datasets_raw, list) or any(not isinstance(value, str) or not value for value in variant_datasets_raw):
            raise ValueError(f"Variant {name} datasets must be a list of names")
        variant_datasets = tuple(variant_datasets_raw)
        config = TrainConfig.model_validate(load_config_file(config_path))
        if config.model_kwargs.get("communication_mode") != communication_mode:
            raise ValueError(f"Variant {name} communication_mode does not match {config_path}")
        variants.append(
            BenchmarkVariant(
                name=name,
                config_path=config_path,
                communication_mode=communication_mode,
                model=config.model,
                datasets=variant_datasets,
                expected_train_config=_config_without_seed(config.model_dump(mode="json")),
            )
        )

    _require_unique([dataset.name for dataset in datasets], "dataset names")
    _require_unique(list(seeds), "seeds")
    _require_unique([variant.name for variant in variants], "variant names")
    dataset_names = {dataset.name for dataset in datasets}
    unknown_datasets = sorted({name for variant in variants for name in variant.datasets if name not in dataset_names})
    if unknown_datasets:
        raise ValueError(f"Variants reference unknown datasets: {unknown_datasets}")
    return BenchmarkSpec(datasets=tuple(datasets), seeds=seeds, variants=tuple(variants), split=split)


def audit_benchmark(spec: BenchmarkSpec, runs_root: str | Path) -> BenchmarkAudit:
    root = Path(runs_root)
    expected_cells = [
        BenchmarkCell(variant=variant.name, dataset=dataset.name, seed=seed)
        for variant in spec.variants
        for dataset in spec.datasets
        if not variant.datasets or dataset.name in variant.datasets
        for seed in spec.seeds
    ]
    candidates: dict[BenchmarkCell, list[Path]] = {cell: [] for cell in expected_cells}
    unmatched_runs: list[str] = []

    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        try:
            manifest = _load_json_mapping(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            unmatched_runs.append(str(manifest_path.parent))
            continue
        match = _match_manifest(manifest, spec)
        if match is None:
            unmatched_runs.append(str(manifest_path.parent))
            continue
        candidates[match].append(manifest_path.parent)

    missing: list[str] = []
    duplicate: dict[str, list[str]] = {}
    invalid: dict[str, list[str]] = {}
    valid_records: dict[BenchmarkCell, BenchmarkRun] = {}
    comparison_signatures: dict[str, dict[str, Any]] = {}

    for cell in expected_cells:
        paths = candidates[cell]
        if not paths:
            missing.append(cell.key)
            continue
        if len(paths) > 1:
            duplicate[cell.key] = [str(path) for path in paths]
            continue
        variant = _variant_by_name(spec, cell.variant)
        try:
            record, signature = _validate_run(paths[0], cell, variant, split=spec.split)
            expected_signature = comparison_signatures.setdefault(cell.dataset, signature)
            if signature != expected_signature:
                raise ValueError("radio, graph, payload, or dataset metadata differs from other compared runs")
            valid_records[cell] = record
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
            invalid[cell.key] = [str(error)]

    runs = [valid_records[cell] for cell in expected_cells if cell in valid_records]
    valid = [cell.key for cell in expected_cells if cell in valid_records]
    return BenchmarkAudit(
        expected=[cell.key for cell in expected_cells],
        valid=valid,
        missing=missing,
        duplicate=duplicate,
        invalid=invalid,
        unmatched_runs=unmatched_runs,
        runs=runs,
    )


def write_benchmark_audit(audit: BenchmarkAudit, output: str | Path) -> None:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "audit.json").write_text(json.dumps(audit.payload(), indent=2, sort_keys=True) + "\n")
    rows = [asdict(run) for run in audit.runs]
    fieldnames = list(asdict(audit.runs[0])) if audit.runs else [field.name for field in BenchmarkRun.__dataclass_fields__.values()]
    with (directory / "runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _match_manifest(manifest: dict[str, Any], spec: BenchmarkSpec) -> BenchmarkCell | None:
    train_config = manifest.get("train_config")
    if not isinstance(train_config, dict):
        return None
    actual_config = _config_without_seed(train_config)
    eval_config = manifest.get("eval_config")
    if spec.split == "val" and (not isinstance(eval_config, dict) or eval_config.get("split") != "val"):
        return None
    if spec.split == "test" and isinstance(eval_config, dict) and eval_config.get("split", "test") != "test":
        return None
    variants = [variant for variant in spec.variants if variant.model == manifest.get("model") and variant.expected_train_config == actual_config]
    if len(variants) != 1:
        return None
    dataset_payload = manifest.get("dataset")
    if not isinstance(dataset_payload, dict):
        return None
    dataset_path = dataset_payload.get("path")
    if not isinstance(dataset_path, str):
        return None
    datasets = [dataset for dataset in spec.datasets if Path(dataset_path).name == dataset.path.name and dataset.name == dataset.path.name]
    if len(datasets) == 1:
        variants = [variant for variant in variants if not variant.datasets or datasets[0].name in variant.datasets]
    seed = manifest.get("seed")
    if len(datasets) != 1 or not isinstance(seed, int) or isinstance(seed, bool) or seed not in spec.seeds:
        return None
    return BenchmarkCell(variant=variants[0].name, dataset=datasets[0].name, seed=seed)


def _validate_run(path: Path, cell: BenchmarkCell, variant: BenchmarkVariant, *, split: str) -> tuple[BenchmarkRun, dict[str, Any]]:
    manifest = _load_json_mapping(path / "manifest.json")
    metrics = _load_json_mapping(path / "eval_metrics.json")
    communication = _load_json_mapping(path / "communication_metrics.json")

    run_id = _require_text(manifest, "run_id")
    if run_id != path.name:
        raise ValueError("manifest run_id does not match artifact directory")
    git = _require_mapping(manifest.get("git"), "git")
    git_commit = _require_text(git, "commit")
    git_dirty = git.get("dirty")
    if not isinstance(git_dirty, bool):
        raise ValueError("git.dirty must be boolean")
    dataset_payload = _require_mapping(manifest.get("dataset"), "dataset")
    data_sha256 = _require_text(dataset_payload, "data_sha256")
    meta_sha256 = _require_text(dataset_payload, "meta_sha256")

    macro_f1 = _finite_nonnegative(metrics.get("macro_f1"), "macro_f1")
    accuracy = _finite_nonnegative(metrics.get("accuracy"), "accuracy")
    per_class = _require_mapping(metrics.get("per_class"), "per_class")
    class_f1 = {
        name: _finite_nonnegative(
            _require_mapping(per_class.get(name), f"per_class.{name}").get("f1"),
            f"per_class.{name}.f1",
        )
        for name in _REQUIRED_CLASSES
    }

    communication_config = _require_mapping(communication.get("config"), "communication.config")
    mode = _require_text(communication_config, "communication_mode")
    if mode != variant.communication_mode:
        raise ValueError(f"communication mode {mode!r} does not match expected {variant.communication_mode!r}")
    split_metrics = _require_mapping(
        _require_mapping(communication.get("splits"), "communication.splits").get(split),
        f"communication.splits.{split}",
    )
    graph = _require_mapping(communication.get("graph"), "communication.graph")
    _require_graph_metadata(graph)

    requested = _finite_nonnegative(split_metrics.get("requested_edge_count"), "requested_edge_count")
    possible = _finite_nonnegative(split_metrics.get("possible_edge_count"), "possible_edge_count")
    transmitted_bits = _finite_nonnegative(split_metrics.get("transmitted_bits_estimate"), "transmitted_bits_estimate")
    bits_per_message = _finite_nonnegative(split_metrics.get("bits_per_message"), "bits_per_message")
    request_ratio = _finite_nonnegative(split_metrics.get("active_request_ratio"), "active_request_ratio")
    requested_by_edge = _number_list(split_metrics.get("requested_edge_counts"), "requested_edge_counts")
    possible_by_edge = _number_list(split_metrics.get("possible_edge_counts"), "possible_edge_counts")
    if len(requested_by_edge) != len(possible_by_edge):
        raise ValueError("requested and possible per-edge counts have different lengths")
    paired_edge_counts = zip(requested_by_edge, possible_by_edge, strict=True)
    if any(requested_count > possible_count + _ABS_TOLERANCE for requested_count, possible_count in paired_edge_counts):
        raise ValueError("requested per-edge count exceeds possible count")
    if requested > possible + _ABS_TOLERANCE:
        raise ValueError("requested message count exceeds possible count")
    _require_close(sum(requested_by_edge), requested, "requested per-edge sum")
    _require_close(sum(possible_by_edge), possible, "possible per-edge sum")
    expected_ratio = requested / possible if possible > 0.0 else 0.0
    _require_close(request_ratio, expected_ratio, "active request ratio")
    _require_close(transmitted_bits, requested * bits_per_message, "transmitted bits")

    energy = _require_mapping(split_metrics.get("energy"), "energy")
    constants = _require_mapping(energy.get("constants"), "energy.constants")
    units = _require_mapping(energy.get("units"), "energy.units")
    distance = _require_mapping(energy.get("distance"), "energy.distance")
    selected_energy = _energy_totals(_require_mapping(energy.get("selective"), "energy.selective"), "energy.selective")
    dense_energy = _energy_totals(_require_mapping(energy.get("dense_reference"), "energy.dense_reference"), "energy.dense_reference")
    reduction = _require_mapping(energy.get("dense_vs_selective"), "energy.dense_vs_selective")

    _require_close(selected_energy["message_count"], requested, "selected energy message count")
    _require_close(selected_energy["bit_count"], transmitted_bits, "selected energy bit count")
    _require_close(selected_energy["bits_per_message"], bits_per_message, "selected energy bits per message")
    _require_close(dense_energy["message_count"], possible, "dense energy message count")
    _require_close(dense_energy["bit_count"], possible * bits_per_message, "dense energy bit count")
    _require_close(dense_energy["bits_per_message"], bits_per_message, "dense energy bits per message")
    energy_reduction_j = _finite_number(reduction.get("total_energy_reduction_j"), "total_energy_reduction_j")
    energy_reduction_ratio = _finite_number(reduction.get("total_energy_reduction_ratio"), "total_energy_reduction_ratio")
    expected_reduction = dense_energy["total_energy_j"] - selected_energy["total_energy_j"]
    expected_reduction_ratio = expected_reduction / dense_energy["total_energy_j"] if dense_energy["total_energy_j"] > 0.0 else 0.0
    _require_close(energy_reduction_j, expected_reduction, "energy reduction")
    _require_close(energy_reduction_ratio, expected_reduction_ratio, "energy reduction ratio")
    if reduction.get("selective_subset_of_dense") is not True:
        raise ValueError("energy block does not mark selective messages as a subset of dense")

    if mode == "dense":
        _require_close(request_ratio, 1.0 if possible > 0.0 else 0.0, "dense request ratio")
        _require_close(selected_energy["total_energy_j"], dense_energy["total_energy_j"], "dense selected energy")
        _require_close(energy_reduction_j, 0.0, "dense energy reduction")
        _require_close(energy_reduction_ratio, 0.0, "dense energy reduction ratio")

    signature = {
        "data_sha256": data_sha256,
        "meta_sha256": meta_sha256,
        "radio_constants": constants,
        "energy_units": units,
        "distance": distance,
        "bits_per_message": bits_per_message,
        "graph": graph,
    }
    record = BenchmarkRun(
        variant=cell.variant,
        dataset=cell.dataset,
        seed=cell.seed,
        split=split,
        run_id=run_id,
        artifact_path=str(path),
        model=variant.model,
        communication_mode=mode,
        git_commit=git_commit,
        git_dirty=git_dirty,
        data_sha256=data_sha256,
        meta_sha256=meta_sha256,
        macro_f1=macro_f1,
        accuracy=accuracy,
        normal_f1=class_f1["NORMAL"],
        spike_f1=class_f1["SPIKE"],
        drift_f1=class_f1["DRIFT"],
        stuck_f1=class_f1["STUCK"],
        request_ratio=request_ratio,
        requested_messages=requested,
        possible_messages=possible,
        transmitted_bits=transmitted_bits,
        bits_per_message=bits_per_message,
        tx_energy_j=selected_energy["tx_energy_j"],
        rx_energy_j=selected_energy["rx_energy_j"],
        total_energy_j=selected_energy["total_energy_j"],
        dense_tx_energy_j=dense_energy["tx_energy_j"],
        dense_rx_energy_j=dense_energy["rx_energy_j"],
        dense_total_energy_j=dense_energy["total_energy_j"],
        energy_reduction_j=energy_reduction_j,
        energy_reduction_ratio=energy_reduction_ratio,
        num_parameters=_optional_nonnegative_integer(manifest.get("num_parameters"), "num_parameters"),
        total_parameters=_optional_nonnegative_integer(manifest.get("total_parameters"), "total_parameters"),
        communication_config_json=json.dumps(communication_config, sort_keys=True, separators=(",", ":")),
        comparison_signature_json=json.dumps(signature, sort_keys=True, separators=(",", ":")),
    )
    return record, signature


def _energy_totals(payload: dict[str, Any], name: str) -> dict[str, float]:
    values = {
        key: _finite_nonnegative(payload.get(key), f"{name}.{key}")
        for key in ("message_count", "bit_count", "bits_per_message", "tx_energy_j", "rx_energy_j", "total_energy_j")
    }
    _require_close(values["tx_energy_j"] + values["rx_energy_j"], values["total_energy_j"], f"{name} TX+RX total")
    _require_close(values["message_count"] * values["bits_per_message"], values["bit_count"], f"{name} bit count")
    return values


def _require_graph_metadata(graph: dict[str, Any]) -> None:
    required = ("directed_edge_count", "dynamic_link_seed", "burst_params", "edge_convention", "link_mask_shape", "distance_metadata")
    missing = [key for key in required if key not in graph]
    if missing:
        raise ValueError(f"communication.graph missing fields: {', '.join(missing)}")


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing artifact: {path.name}")
    raw = json.loads(path.read_text())
    return _require_mapping(raw, path.name)


def _config_without_seed(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.pop("seed", None)
    return normalized


def _variant_by_name(spec: BenchmarkSpec, name: str) -> BenchmarkVariant:
    return next(variant for variant in spec.variants if variant.name == name)


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return value


def _require_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_nonnegative_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    integer = _require_integer(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _require_unique(values: list[object], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _finite_nonnegative(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _number_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return [_finite_nonnegative(item, name) for item in value]


def _require_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=_REL_TOLERANCE, abs_tol=_ABS_TOLERANCE):
        raise ValueError(f"{name} is inconsistent: {actual} != {expected}")
