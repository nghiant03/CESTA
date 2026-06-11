"""Communication metric aggregation for communication-aware models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from CESTA.datasets.artifact import GraphMetadata
from CESTA.evaluation.energy import compute_radio_energy_metrics
from CESTA.models.base import BaseModel


def collect_model_communication_config(model: BaseModel) -> dict[str, Any] | None:
    config = model.get_config()
    if "communication_mode" not in config:
        return None
    message_size = _model_message_size(model)
    precision_bits = config.get("precision_bits")
    return {
        "communication_mode": config.get("communication_mode"),
        "hidden_size": config.get("hidden_size"),
        "gate_hidden_size": config.get("gate_hidden_size"),
        "gumbel_temperature": config.get("gumbel_temperature"),
        "precision_bits": precision_bits,
        "message_size": message_size,
        "bits_per_message": message_size * int(precision_bits) if isinstance(precision_bits, int) and message_size is not None else None,
        "graph_edge_count": config.get("graph_edge_count", _graph_edge_count(config)),
    }


def normalize_communication_stats(stats: dict[str, object]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, int | float):
            payload[key] = float(value)
        elif isinstance(value, list | tuple):
            payload[key] = [float(item) for item in value]
    return payload


def aggregate_communication_stats(
    split_stats: dict[str, list[dict[str, Any]]],
    model: BaseModel,
    metadata: dict[str, object] | None = None,
) -> dict[str, Any] | None:
    config = collect_model_communication_config(model)
    if config is None:
        return None

    payload: dict[str, Any] = {
        "model": model.name,
        "config": config,
        "splits": {},
    }
    graph_meta = _graph_metadata(metadata)
    graph_metadata = _collect_graph_metadata(graph_meta)
    if graph_metadata is not None:
        payload["graph"] = graph_metadata
    for split_name, stats in split_stats.items():
        if not stats:
            continue
        payload["splits"][split_name] = _aggregate_split(stats, graph_meta)

    if not payload["splits"]:
        return None
    return payload


def save_communication_metrics(
    path: str | Path,
    metrics: dict[str, Any] | None,
) -> None:
    if metrics is None:
        return
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize(metrics)
    (directory / "communication_metrics.json").write_text(json.dumps(sanitized, indent=2))


def _aggregate_split(stats: list[dict[str, Any]], graph_meta: GraphMetadata | None) -> dict[str, Any]:
    totals = {
        "requested_edge_count": 0.0,
        "possible_edge_count": 0.0,
        "transmitted_bits_estimate": 0.0,
        "full_embedding_message_count": 0.0,
        "compressed_message_count": 0.0,
    }
    for item in stats:
        for key in totals:
            totals[key] += float(item.get(key, 0.0))

    active_ratio = totals["requested_edge_count"] / max(totals["possible_edge_count"], 1.0)
    total_messages = totals["full_embedding_message_count"] + totals["compressed_message_count"]
    average_compression_ratio = totals["full_embedding_message_count"] / total_messages if total_messages > 0.0 else 0.0
    requested_edge_counts = _sum_edge_counts(stats, "requested_edge_counts")
    possible_edge_counts = _sum_edge_counts(stats, "possible_edge_counts")
    bits_per_message = _bits_per_message(stats, totals)
    payload: dict[str, Any] = {
        "active_request_ratio": active_ratio,
        "requested_edge_count": totals["requested_edge_count"],
        "possible_edge_count": totals["possible_edge_count"],
        "transmitted_bits_estimate": totals["transmitted_bits_estimate"],
        "full_embedding_message_count": totals["full_embedding_message_count"],
        "compressed_message_count": totals["compressed_message_count"],
        "average_compression_ratio": average_compression_ratio,
        "bits_per_message": bits_per_message,
        "batch_count": float(len(stats)),
    }
    if requested_edge_counts is not None and possible_edge_counts is not None:
        payload["requested_edge_counts"] = requested_edge_counts.tolist()
        payload["possible_edge_counts"] = possible_edge_counts.tolist()
        if graph_meta is not None:
            payload["energy"] = compute_radio_energy_metrics(
                requested_edge_counts=requested_edge_counts.tolist(),
                possible_edge_counts=possible_edge_counts.tolist(),
                edge_distance_m=graph_meta.edge_distance_m.astype(float).tolist(),
                bits_per_message=bits_per_message,
                distance_metadata=graph_meta.distance_metadata,
            )
    return payload


def _graph_metadata(metadata: dict[str, object] | None) -> GraphMetadata | None:
    graph_meta = (metadata or {}).get("graph")
    return graph_meta if isinstance(graph_meta, GraphMetadata) else None


def _collect_graph_metadata(graph_meta: GraphMetadata | None) -> dict[str, Any] | None:
    if graph_meta is None:
        return None
    attrs = {
        "directed_edge_count": "num_edges",
        "dynamic_link_seed": "dynamic_link_seed",
        "burst_params": "burst_params",
        "edge_convention": "edge_convention",
        "link_mask_shape": "link_mask_shape",
        "distance_metadata": "distance_metadata",
    }
    payload: dict[str, Any] = {}
    for key, attr in attrs.items():
        value = getattr(graph_meta, attr, None)
        if value is None:
            continue
        if isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload or None


def _sum_edge_counts(stats: list[dict[str, Any]], key: str) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    arrays: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for item in stats:
        value = item.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"{key} must contain one-dimensional per-edge counts")
        arrays.append(array)
    if not arrays:
        return None
    edge_count = arrays[0].shape[0]
    if any(array.shape != (edge_count,) for array in arrays):
        raise ValueError(f"{key} shapes must match across batches")
    return np.sum(arrays, axis=0)


def _bits_per_message(stats: list[dict[str, Any]], totals: dict[str, float]) -> float:
    for item in stats:
        value = item.get("bits_per_message")
        if value is not None:
            return float(value)
    if totals["requested_edge_count"] > 0.0:
        return totals["transmitted_bits_estimate"] / totals["requested_edge_count"]
    return 0.0


def _model_message_size(model: BaseModel) -> int | None:
    message_size = getattr(model, "_message_size", None)
    if not callable(message_size):
        return None
    value = message_size()
    return int(value) if isinstance(value, int) else None


def _graph_edge_count(config: dict[str, Any]) -> float | None:
    edge_index = config.get("edge_index")
    if not isinstance(edge_index, list):
        return None
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    if edge_index_tensor.ndim != 2 or edge_index_tensor.shape[0] != 2:
        return None
    return float(edge_index_tensor.shape[1])


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
