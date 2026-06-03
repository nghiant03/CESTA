"""Raw-to-canonical dataset transformation workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from CESTA.datasets.artifact import CESTADataset
from CESTA.datasets.raw import get_dataset
from CESTA.injection import FaultInjector
from CESTA.logging import logger
from CESTA.schema import TransformConfig
from CESTA.seed import seed_everything


def load_directed_edges(
    connectivity_path: str | Path,
    node_ids: list[int],
    threshold: float = 0.5,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    connectivity_path = Path(connectivity_path)
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    edges: list[tuple[int, int]] = []
    probs: list[float] = []
    with connectivity_path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            src, dst = int(parts[0]), int(parts[1])
            prob = float(parts[2])
            if src == dst or prob < threshold:
                continue
            if src in id_to_idx and dst in id_to_idx:
                edges.append((id_to_idx[src], id_to_idx[dst]))
                probs.append(prob)
    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_prob = np.asarray(probs, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_prob = np.empty((0,), dtype=np.float32)
    logger.info("Graph: {} nodes, {} directed edges (threshold={:.2f})", len(node_ids), edge_index.shape[1], threshold)
    return edge_index, edge_prob


def simulate_bursty_link_mask(
    num_timestamps: int,
    edge_index: NDArray[np.int64],
    edge_prob: NDArray[np.float32],
    *,
    seed: int,
    rho: float,
    q_bad_base: float,
    q_recover_base: float,
    bad_success_floor: float,
) -> NDArray[np.bool_]:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if not 0.0 <= bad_success_floor <= 1.0:
        raise ValueError("bad_success_floor must be in [0, 1]")
    num_edges = edge_index.shape[1]
    rng = np.random.default_rng(seed)
    link_mask = np.zeros((num_timestamps, num_edges), dtype=np.bool_)
    if num_timestamps == 0 or num_edges == 0:
        return link_mask
    q_bad_env = q_bad_base * rho * (1.0 - edge_prob)
    q_recover_env = q_recover_base * rho * edge_prob
    q_bad_dir = q_bad_base * (1.0 - rho) * (1.0 - edge_prob)
    q_recover_dir = q_recover_base * (1.0 - rho) * edge_prob
    pair_keys: list[tuple[int, int]] = [(min(int(s), int(r)), max(int(s), int(r))) for s, r in edge_index.T]
    unique_pairs = sorted(set(pair_keys))
    pair_to_idx = {pair: idx for idx, pair in enumerate(unique_pairs)}
    pair_edge_indices: dict[tuple[int, int], list[int]] = {pair: [] for pair in unique_pairs}
    for edge_idx, pair in enumerate(pair_keys):
        pair_edge_indices[pair].append(edge_idx)
    env_state = np.zeros((len(unique_pairs),), dtype=np.bool_)
    for pair, pair_idx in pair_to_idx.items():
        edge_indices = pair_edge_indices[pair]
        qb = float(np.mean(q_bad_env[edge_indices]))
        qr = float(np.mean(q_recover_env[edge_indices]))
        bad_prob = qb / max(qb + qr, 1e-12)
        env_state[pair_idx] = rng.random() < bad_prob
    dir_bad_prob = q_bad_dir / np.maximum(q_bad_dir + q_recover_dir, 1e-12)
    dir_state = rng.random(num_edges) < dir_bad_prob
    for t in range(num_timestamps):
        if t > 0:
            for pair, pair_idx in pair_to_idx.items():
                edge_indices = pair_edge_indices[pair]
                qb = float(np.mean(q_bad_env[edge_indices]))
                qr = float(np.mean(q_recover_env[edge_indices]))
                if env_state[pair_idx]:
                    env_state[pair_idx] = not (rng.random() < qr)
                else:
                    env_state[pair_idx] = rng.random() < qb
            recover = rng.random(num_edges) < q_recover_dir
            fail = rng.random(num_edges) < q_bad_dir
            dir_state = np.where(dir_state, ~recover, fail)
        env_bad = np.asarray([env_state[pair_to_idx[pair]] for pair in pair_keys], dtype=np.bool_)
        effective_bad = env_bad | dir_state
        success_prob = np.where(effective_bad, bad_success_floor, edge_prob)
        link_mask[t] = rng.random(num_edges) < success_prob
    return link_mask


def load_node_positions(mote_locs_path: str | Path | None, node_ids: list[int]) -> dict[str, Any]:
    if mote_locs_path is None:
        return {"metadata": {"source": "absent", "units": "unknown"}, "nodes": {}}
    path = Path(mote_locs_path)
    if not path.exists():
        raise FileNotFoundError(f"Node position file not found: {path}")
    node_set = set(node_ids)
    positions: dict[str, dict[str, float]] = {}
    with path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            node_id = int(float(parts[0]))
            if node_id not in node_set:
                continue
            positions[str(node_id)] = {"x": float(parts[1]), "y": float(parts[2])}
    missing = [node_id for node_id in node_ids if str(node_id) not in positions]
    return {
        "metadata": {
            "source": str(path),
            "units": "m",
            "missing_node_ids": missing,
        },
        "nodes": positions,
    }


def compute_edge_distances(
    edge_index: NDArray[np.int64],
    node_ids: list[int],
    positions: dict[str, Any],
    fallback_distance_m: float | None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    if edge_index.shape[1] == 0:
        return np.empty((0,), dtype=np.float32), {"source": "empty_graph", "units": "m"}
    nodes = positions.get("nodes", {})
    distances: list[float] = []
    fallback_edges = 0
    for sender_idx, receiver_idx in edge_index.T:
        sender_id = node_ids[int(sender_idx)]
        receiver_id = node_ids[int(receiver_idx)]
        sender = nodes.get(str(sender_id))
        receiver = nodes.get(str(receiver_id))
        if sender is None or receiver is None:
            if fallback_distance_m is None:
                raise ValueError(f"Missing positions for edge {sender_id}->{receiver_id} and no fallback_distance_m is configured")
            distances.append(float(fallback_distance_m))
            fallback_edges += 1
            continue
        dx = float(sender["x"]) - float(receiver["x"])
        dy = float(sender["y"]) - float(receiver["y"])
        distances.append(float(np.hypot(dx, dy)))
    source = "positions" if fallback_edges == 0 else "positions_with_fallback"
    if len(nodes) == 0:
        source = "fallback"
    return np.asarray(distances, dtype=np.float32), {
        "source": source,
        "units": "m",
        "fallback_distance_m": fallback_distance_m,
        "fallback_edge_count": fallback_edges,
    }


def run_transform(dataset_name: str, raw_path: Path, output: Path, config: TransformConfig) -> CESTADataset:
    if config.injection.seed is not None:
        seed_everything(config.injection.seed)
    resolved_raw_path = raw_path.parent if raw_path.is_file() else raw_path
    raw_dataset = get_dataset(dataset_name, resolved_raw_path)
    injector = FaultInjector(config.injection)
    df, feature_names = injector.run_to_frame(raw_dataset)
    group_col = config.injection.group_column
    node_ids = sorted(int(g) for g in df[group_col].unique()) if group_col in df.columns else [0]
    timestamps = sorted(df[raw_dataset.timestamp_column].unique()) if raw_dataset.timestamp_column in df.columns else list(range(len(df)))
    if config.graph.connectivity_path is not None:
        edge_index, edge_prob = load_directed_edges(config.graph.connectivity_path, node_ids, threshold=config.graph.threshold)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_prob = np.empty((0,), dtype=np.float32)
    link_mask = simulate_bursty_link_mask(
        len(timestamps),
        edge_index,
        edge_prob,
        seed=config.graph.seed,
        rho=config.graph.rho,
        q_bad_base=config.graph.q_bad_base,
        q_recover_base=config.graph.q_recover_base,
        bad_success_floor=config.graph.bad_success_floor,
    )
    positions = load_node_positions(config.graph.mote_locs_path, node_ids)
    edge_distance_m, distance_meta = compute_edge_distances(edge_index, node_ids, positions, config.graph.fallback_distance_m)
    graph_meta = {
        "dynamic_link_seed": config.graph.seed,
        "timestamps": [str(ts) for ts in timestamps],
        "burst_params": {
            "rho": config.graph.rho,
            "q_bad_base": config.graph.q_bad_base,
            "q_recover_base": config.graph.q_recover_base,
            "bad_success_floor": config.graph.bad_success_floor,
        },
        "connectivity_path": str(config.graph.connectivity_path) if config.graph.connectivity_path is not None else None,
        "masks_applied": True,
    }
    dataset = CESTADataset(
        df=df,
        config=config.injection,
        feature_names=feature_names,
        dataset_name=dataset_name,
        timestamp_column=raw_dataset.timestamp_column,
        edge_index=edge_index,
        edge_prob=edge_prob,
        node_ids=node_ids,
        threshold=config.graph.threshold,
        link_mask=link_mask,
        edge_distance_m=edge_distance_m,
        node_positions=positions,
        metadata={"graph": graph_meta, "distances": distance_meta},
    )
    dataset.save(output)
    logger.info("Saved canonical dataset to {}", output)
    return dataset
