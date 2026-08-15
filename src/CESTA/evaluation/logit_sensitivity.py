from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from CESTA.evaluation.result import EvalResult
from CESTA.schema.fault import FaultType


def build_communication_dependence_audit(results: Mapping[str, EvalResult], *, reference: str = "learned") -> dict[str, Any]:
    if set(results) != {"zero", "learned", "dense"}:
        raise ValueError("communication dependence requires zero, learned, and dense results")
    reference_result = results[reference]
    _validate_result(reference_result)
    interventions = {}
    for name, result in results.items():
        _validate_result(result)
        _validate_alignment(reference_result, result)
        changed = result.y_pred != reference_result.y_pred
        absolute_change = np.abs(result.y_logits.astype(np.float64) - reference_result.y_logits.astype(np.float64))
        interventions[name] = {
            "macro_f1": float(result.macro_f1),
            "per_class_f1": {class_name: float(value) for class_name, value in zip(FaultType.names(), result.class_metrics.f1, strict=False)},
            "changed_prediction_count": int(np.count_nonzero(changed)),
            "changed_prediction_fraction": float(np.mean(changed)) if changed.size else 0.0,
            "absolute_logit_change": _distribution_summary(absolute_change),
            "request_ratio": _communication_value(result, "active_request_ratio"),
            "total_energy_j": _energy_value(result, "total_energy_j"),
        }
    return {"split": "val", "reference": reference, "class_names": FaultType.names(), "interventions": interventions}


def build_gate_selectivity_audit(
    request_probabilities: np.ndarray[Any, np.dtype[np.floating[Any]]],
    possible_mask: np.ndarray[Any, np.dtype[np.bool_]],
    edge_costs: Sequence[float],
    edge_probabilities: Sequence[float],
    edge_index: np.ndarray[Any, np.dtype[np.integer[Any]]] | None = None,
) -> dict[str, Any]:
    probabilities = np.asarray(request_probabilities, dtype=np.float64)
    possible = np.asarray(possible_mask, dtype=bool)
    if probabilities.shape != possible.shape or probabilities.ndim != 4 or probabilities.shape[-1] != probabilities.shape[-2]:
        raise ValueError("request probabilities and possible mask must align as (batch, time, receiver, sender)")
    edge_cost_matrix = _edge_vector_matrix(edge_costs, possible, edge_index=edge_index)
    edge_probability_matrix = _edge_vector_matrix(edge_probabilities, possible, edge_index=edge_index)
    selected = probabilities[possible]
    receiver_variances = []
    for values, mask in zip(probabilities.reshape(-1, probabilities.shape[-1]), possible.reshape(-1, possible.shape[-1]), strict=True):
        if np.count_nonzero(mask) > 1:
            receiver_variances.append(float(np.var(values[mask])))
    binary_entropy = -(
        selected * np.log(np.clip(selected, 1e-12, 1.0))
        + (1.0 - selected) * np.log(np.clip(1.0 - selected, 1e-12, 1.0))
    )
    return {
        "request_probability": _distribution_summary(selected),
        "saturation_below_0p01": float(np.mean(selected < 0.01)) if selected.size else 0.0,
        "saturation_above_0p99": float(np.mean(selected > 0.99)) if selected.size else 0.0,
        "mean_binary_entropy": float(np.mean(binary_entropy)) if selected.size else 0.0,
        "mean_within_receiver_sender_variance": float(np.mean(receiver_variances)) if receiver_variances else 0.0,
        "spearman_probability_cost": _rank_correlation(selected, np.broadcast_to(edge_cost_matrix, probabilities.shape)[possible]),
        "spearman_probability_edge_probability": _rank_correlation(selected, np.broadcast_to(edge_probability_matrix, probabilities.shape)[possible]),
        "request_probability_by_cost_quartile": _quartile_means(selected, np.broadcast_to(edge_cost_matrix, probabilities.shape)[possible]),
    }


def build_edge_ablation_audit(
    baseline: EvalResult,
    ablations: Mapping[int, EvalResult],
    edge_costs: Sequence[float],
) -> dict[str, Any]:
    _validate_result(baseline)
    costs = np.asarray(edge_costs, dtype=np.float64)
    points = []
    baseline_loss = _true_class_loss(baseline)
    for edge_index, result in sorted(ablations.items()):
        if edge_index < 0 or edge_index >= len(costs):
            raise ValueError("edge ablation index is outside the canonical edge range")
        _validate_result(result)
        _validate_alignment(baseline, result)
        changed = result.y_pred != baseline.y_pred
        points.append(
            {
                "edge_index": edge_index,
                "edge_cost": float(costs[edge_index]),
                "true_class_loss_change": float(_true_class_loss(result) - baseline_loss),
                "macro_f1_change": float(result.macro_f1 - baseline.macro_f1),
                "changed_prediction_count": int(np.count_nonzero(changed)),
                "changed_prediction_fraction": float(np.mean(changed)) if changed.size else 0.0,
            }
        )
    return {"split": "val", "subset_size": int(baseline.y_true.size), "edges": points}


def build_logit_sensitivity_audit(
    results: Mapping[float, EvalResult],
    *,
    reference_threshold: float,
) -> dict[str, Any]:
    if reference_threshold not in results:
        raise ValueError("reference threshold must be present in results")
    if not results:
        raise ValueError("results must not be empty")

    reference = results[reference_threshold]
    _validate_result(reference)
    points = []
    for threshold in sorted(results):
        result = results[threshold]
        _validate_result(result)
        _validate_alignment(reference, result)
        changed = result.y_pred != reference.y_pred
        absolute_logit_change = np.abs(result.y_logits.astype(np.float64) - reference.y_logits.astype(np.float64))
        per_class_changed = {
            name: int(np.count_nonzero(changed & (reference.y_true == class_index)))
            for class_index, name in enumerate(FaultType.names())
        }
        points.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(result.macro_f1),
                "accuracy": float(result.accuracy),
                "changed_prediction_count": int(np.count_nonzero(changed)),
                "changed_prediction_fraction": float(np.mean(changed)) if changed.size else 0.0,
                "mean_absolute_logit_change": float(np.mean(absolute_logit_change)) if absolute_logit_change.size else 0.0,
                "maximum_absolute_logit_change": float(np.max(absolute_logit_change)) if absolute_logit_change.size else 0.0,
                "per_class_changed_prediction_count": per_class_changed,
                "request_ratio": _communication_value(result, "active_request_ratio"),
                "tx_energy_j": _energy_value(result, "tx_energy_j"),
                "rx_energy_j": _energy_value(result, "rx_energy_j"),
                "total_energy_j": _energy_value(result, "total_energy_j"),
            }
        )

    return {
        "split": "val",
        "reference_threshold": float(reference_threshold),
        "class_names": FaultType.names(),
        "thresholds": points,
    }


def stable_subset_indices(window_ids: Sequence[int], maximum_windows: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    if maximum_windows < 1:
        raise ValueError("maximum_windows must be positive")
    ids = np.asarray(window_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != ids.size:
        raise ValueError("window IDs must be unique and one-dimensional")
    hashes = np.mod(ids * 48_271 + 12_345, 2_147_483_647)
    return np.argsort(hashes, kind="stable")[:maximum_windows].astype(np.int64)


def validate_thresholds(thresholds: Sequence[float], reference_threshold: float) -> list[float]:
    normalized = [float(threshold) for threshold in thresholds]
    if not normalized:
        raise ValueError("at least one threshold is required")
    if any(not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0 for threshold in normalized):
        raise ValueError("thresholds must be finite values in [0, 1]")
    if len(set(normalized)) != len(normalized):
        raise ValueError("thresholds must be unique")
    if float(reference_threshold) not in normalized:
        raise ValueError("reference threshold must be included in thresholds")
    return sorted(normalized)


def _distribution_summary(values: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> dict[str, float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if not flattened.size:
        return {key: 0.0 for key in ("mean", "maximum", "p50", "p90", "p95", "p99")}
    return {
        "mean": float(np.mean(flattened)),
        "maximum": float(np.max(flattened)),
        "p50": float(np.quantile(flattened, 0.5)),
        "p90": float(np.quantile(flattened, 0.9)),
        "p95": float(np.quantile(flattened, 0.95)),
        "p99": float(np.quantile(flattened, 0.99)),
    }


def _edge_vector_matrix(
    values: Sequence[float],
    possible: np.ndarray[Any, np.dtype[np.bool_]],
    edge_index: np.ndarray[Any, np.dtype[np.integer[Any]]] | None = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    array = np.asarray(values, dtype=np.float64)
    matrix = np.zeros(possible.shape[-2:], dtype=np.float64)
    if edge_index is not None:
        canonical_edges = np.asarray(edge_index, dtype=np.int64)
        if canonical_edges.shape != (2, len(array)):
            raise ValueError("edge values and canonical edge index must align")
        matrix[canonical_edges[1], canonical_edges[0]] = array
        return matrix
    static_possible = possible.any(axis=(0, 1))
    receiver, sender = np.nonzero(static_possible)
    if array.shape != (len(receiver),):
        raise ValueError("edge values must align with statically possible directed edges")
    matrix[receiver, sender] = array
    return matrix


def _rank_correlation(left: np.ndarray[Any, np.dtype[np.float64]], right: np.ndarray[Any, np.dtype[np.float64]]) -> float | None:
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _quartile_means(probabilities: np.ndarray[Any, np.dtype[np.float64]], costs: np.ndarray[Any, np.dtype[np.float64]]) -> list[float]:
    if not probabilities.size:
        return [0.0] * 4
    boundaries = np.quantile(costs, [0.25, 0.5, 0.75])
    bins = np.digitize(costs, boundaries, right=True)
    return [float(np.mean(probabilities[bins == index])) if np.any(bins == index) else 0.0 for index in range(4)]


def _true_class_loss(result: EvalResult) -> float:
    logits = result.y_logits.astype(np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return float(-np.mean(log_probabilities[np.arange(result.y_true.size), result.y_true])) if result.y_true.size else 0.0


def _validate_result(result: EvalResult) -> None:
    if result.y_logits.ndim != 2 or result.y_logits.shape[0] != result.y_pred.shape[0]:
        raise ValueError("each result must contain aligned two-dimensional logits")
    if result.y_true.shape != result.y_pred.shape:
        raise ValueError("each result must contain aligned targets and predictions")
    if not np.isfinite(result.y_logits).all():
        raise ValueError("logits must be finite")


def _validate_alignment(reference: EvalResult, result: EvalResult) -> None:
    if result.y_logits.shape != reference.y_logits.shape:
        raise ValueError("logit shapes must match across thresholds")
    if not np.array_equal(result.y_true, reference.y_true):
        raise ValueError("target order must match across thresholds")


def _validation_split(result: EvalResult) -> dict[str, Any]:
    metrics = result.communication_metrics
    if not isinstance(metrics, dict):
        raise ValueError("communication metrics are required")
    splits = metrics.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"val"}:
        raise ValueError("logit sensitivity requires validation-only communication metrics")
    split = splits["val"]
    if not isinstance(split, dict):
        raise ValueError("validation communication metrics must be a mapping")
    return split


def _communication_value(result: EvalResult, name: str) -> float:
    value = _validation_split(result).get(name)
    if not isinstance(value, int | float) or not np.isfinite(value):
        raise ValueError(f"validation communication metric {name} must be finite")
    return float(value)


def _energy_value(result: EvalResult, name: str) -> float:
    energy = _validation_split(result).get("energy")
    selective = energy.get("selective") if isinstance(energy, dict) else None
    value = selective.get(name) if isinstance(selective, dict) else None
    if not isinstance(value, int | float) or not np.isfinite(value):
        raise ValueError(f"validation energy metric {name} must be finite")
    return float(value)
