"""Communication metric aggregation for communication-aware models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from CESTA.models.base import BaseModel


def collect_model_communication_config(model: BaseModel) -> dict[str, Any] | None:
    config = model.get_config()
    if "communication_mode" not in config:
        return None
    return {
        "communication_mode": config.get("communication_mode"),
        "hidden_size": config.get("hidden_size"),
        "gate_hidden_size": config.get("gate_hidden_size"),
        "gumbel_temperature": config.get("gumbel_temperature"),
        "precision_bits": config.get("precision_bits"),
        "graph_edge_count": config.get("graph_edge_count", _graph_edge_count(config)),
    }


def aggregate_communication_stats(
    split_stats: dict[str, list[dict[str, float]]],
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
    graph_metadata = _collect_graph_metadata(metadata)
    if graph_metadata is not None:
        payload["graph"] = graph_metadata
    for split_name, stats in split_stats.items():
        if not stats:
            continue
        payload["splits"][split_name] = _aggregate_split(stats)

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
    (directory / "communication_metrics.json").write_text(
        json.dumps(sanitized, indent=2)
    )


def _aggregate_split(stats: list[dict[str, float]]) -> dict[str, float]:
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

    active_ratio = totals["requested_edge_count"] / max(
        totals["possible_edge_count"], 1.0
    )
    total_messages = (
        totals["full_embedding_message_count"] + totals["compressed_message_count"]
    )
    average_compression_ratio = (
        totals["full_embedding_message_count"] / total_messages
        if total_messages > 0.0
        else 0.0
    )
    return {
        "active_request_ratio": active_ratio,
        "requested_edge_count": totals["requested_edge_count"],
        "possible_edge_count": totals["possible_edge_count"],
        "transmitted_bits_estimate": totals["transmitted_bits_estimate"],
        "full_embedding_message_count": totals["full_embedding_message_count"],
        "compressed_message_count": totals["compressed_message_count"],
        "average_compression_ratio": average_compression_ratio,
        "batch_count": float(len(stats)),
    }


def _collect_graph_metadata(metadata: dict[str, object] | None) -> dict[str, Any] | None:
    graph_meta = (metadata or {}).get("graph")
    if graph_meta is None:
        return None
    attrs = {
        "directed_edge_count": "num_edges",
        "dynamic_link_seed": "dynamic_link_seed",
        "burst_params": "burst_params",
        "edge_convention": "edge_convention",
        "link_mask_shape": "link_mask_shape",
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
