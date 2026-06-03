"""Canonical CESTA dataset artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

from CESTA.datasets.windowed import (
    WindowedSplits,
    collect_splits,
    create_windows_with_starts,
    split_and_window,
    split_boundaries,
    validate_features,
)
from CESTA.schema import InjectionConfig
from CESTA.schema.fault import FaultType
from CESTA.schema.manifest import DatasetInfo
from CESTA.schema.window import DataSplitConfig, WindowConfig
from CESTA.utils import sha256_file


@dataclass
class GraphMetadata:
    edge_index: NDArray[np.int64]
    edge_prob: NDArray[np.float32]
    node_ids: list[int]
    num_nodes: int
    threshold: float
    edge_distance_m: NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    edge_convention: str = "sender_to_receiver"
    dynamic_link_seed: int | None = None
    burst_params: dict[str, float] = field(default_factory=dict)
    timestamps: list[Any] = field(default_factory=list)
    link_mask_shape: tuple[int, int] | None = None
    position_metadata: dict[str, Any] = field(default_factory=dict)
    distance_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass
class CESTADataset:
    df: pd.DataFrame
    config: InjectionConfig
    feature_names: list[str]
    dataset_name: str
    timestamp_column: str = "timestamp"
    edge_index: NDArray[np.int64] = field(default_factory=lambda: np.empty((2, 0), dtype=np.int64))
    edge_prob: NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    node_ids: list[int] = field(default_factory=list)
    threshold: float = 0.5
    link_mask: NDArray[np.bool_] = field(default_factory=lambda: np.empty((0, 0), dtype=np.bool_))
    edge_distance_m: NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    node_positions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def group_column(self) -> str:
        return self.config.group_column

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_groups(self) -> int:
        if self.group_column in self.df.columns:
            return self.df[self.group_column].nunique()
        return 1

    @property
    def total_timesteps(self) -> int:
        return len(self.df)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def save(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(directory / "dataset.csv", index=False)
        np.savez_compressed(
            directory / "graph_edges.npz",
            edge_index=self.edge_index.astype(np.int64),
            edge_prob=self.edge_prob.astype(np.float32),
            node_ids=np.asarray(self.node_ids, dtype=np.int64),
        )
        packed, shape = pack_link_mask(self.link_mask)
        np.savez_compressed(directory / "dynamic_link_mask.npz", link_mask=packed, shape=shape)
        (directory / "node_positions.json").write_text(json.dumps(self.node_positions, indent=2, default=str))
        np.savez_compressed(directory / "edge_distances.npz", edge_distance_m=self.edge_distance_m.astype(np.float32))
        meta = {
            "schema_version": 1,
            "dataset_name": self.dataset_name,
            "injection_config": self.config.model_dump(mode="json"),
            "feature_names": self.feature_names,
            "group_column": self.group_column,
            "timestamp_column": self.timestamp_column,
            "graph": {
                "node_ids": self.node_ids,
                "threshold": self.threshold,
                "edge_convention": "sender_to_receiver",
                "edge_count": int(self.edge_index.shape[1]),
                "num_nodes": self.num_nodes,
                "link_mask_shape": list(self.link_mask.shape),
                **dict(self.metadata.get("graph", {})),
            },
            "positions": self.node_positions.get("metadata", {}),
            "distances": dict(self.metadata.get("distances", {})),
        }
        (directory / "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> CESTADataset:
        directory = Path(path)
        required = [
            "dataset.csv",
            "dataset_meta.json",
            "graph_edges.npz",
            "dynamic_link_mask.npz",
            "node_positions.json",
            "edge_distances.npz",
        ]
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            msg = f"Canonical dataset is missing required files in {directory}: {missing}"
            raise FileNotFoundError(msg)

        meta = json.loads((directory / "dataset_meta.json").read_text())
        config = InjectionConfig.model_validate(meta["injection_config"])
        feature_names = list(meta["feature_names"])
        df = pd.read_csv(directory / "dataset.csv")
        for col in feature_names:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        if "fault_state" in df.columns:
            df["fault_state"] = df["fault_state"].astype(np.int32)

        edges = np.load(directory / "graph_edges.npz")
        link_payload = np.load(directory / "dynamic_link_mask.npz")
        distances = np.load(directory / "edge_distances.npz")
        graph_meta = dict(meta.get("graph", {}))
        return cls(
            df=df,
            config=config,
            feature_names=feature_names,
            dataset_name=str(meta["dataset_name"]),
            timestamp_column=str(meta.get("timestamp_column", "timestamp")),
            edge_index=edges["edge_index"].astype(np.int64),
            edge_prob=edges["edge_prob"].astype(np.float32),
            node_ids=[int(x) for x in edges["node_ids"].tolist()],
            threshold=float(graph_meta.get("threshold", 0.5)),
            link_mask=unpack_link_mask(link_payload["link_mask"], link_payload["shape"]),
            edge_distance_m=distances["edge_distance_m"].astype(np.float32),
            node_positions=json.loads((directory / "node_positions.json").read_text()),
            metadata={"graph": graph_meta, "distances": dict(meta.get("distances", {}))},
        )

    def describe(self, path: str | Path) -> DatasetInfo:
        directory = Path(path)
        return DatasetInfo(
            path=str(directory.resolve()),
            data_sha256=sha256_file(directory / "dataset.csv"),
            meta_sha256=sha256_file(directory / "dataset_meta.json"),
            num_features=self.num_features,
            feature_names=list(self.feature_names),
            num_groups=self.num_groups,
            total_timesteps=self.total_timesteps,
        )

    def print_summary(self) -> None:
        console = Console()
        info_table = Table(title="CESTA Dataset Summary", show_header=True)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        info_table.add_row("Dataset", self.dataset_name)
        info_table.add_row("Groups", str(self.num_groups))
        info_table.add_row("Total timesteps", f"{self.total_timesteps:,}")
        info_table.add_row("Features", str(self.num_features))
        info_table.add_row("Feature names", str(self.feature_names))
        info_table.add_row("Graph edges", str(self.edge_index.shape[1]))
        console.print(info_table)
        if "fault_state" in self.df.columns:
            console.print("\n[bold]Class Distribution:[/bold]")
            console.print(self._build_class_dist_table(self.df["fault_state"].to_numpy(dtype=np.int32)))

    def _build_class_dist_table(self, y: np.ndarray) -> Table:
        table = Table(show_header=True)
        table.add_column("Fault Type", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")
        flat = y.flatten()
        total = len(flat)
        for ft in FaultType:
            count = int(np.sum(flat == ft.value))
            pct = 100.0 * count / total if total > 0 else 0.0
            table.add_row(ft.name, f"{count:,}", f"{pct:.2f}%")
        return table

    def prepare(
        self,
        window_config: WindowConfig | None = None,
        split_config: DataSplitConfig | None = None,
        features: list[str] | None = None,
        required_metadata: set[str] | None = None,
        split_bounds: tuple[int, int, int, int] | None = None,
    ) -> WindowedSplits:
        if required_metadata is not None and "graph" in required_metadata:
            return self._prepare_graph(window_config, split_config, features)
        if split_config is not None and split_config.strategy == "connectivity-chronological":
            return self._prepare_aligned_tabular(window_config, split_config, features, required_metadata)
        return self._prepare_tabular(window_config, split_config, features, split_bounds)

    def _prepare_tabular(
        self,
        window_config: WindowConfig | None,
        split_config: DataSplitConfig | None,
        features: list[str] | None,
        split_bounds: tuple[int, int, int, int] | None = None,
    ) -> WindowedSplits:
        wc = window_config if window_config is not None else WindowConfig()
        split = split_config if split_config is not None else DataSplitConfig()
        if split.strategy != "chronological":
            msg = f"Tabular preparation supports only chronological split strategy, got {split.strategy!r}"
            raise ValueError(msg)
        if split_bounds is not None and self.group_column in self.df.columns:
            msg = "Global split bounds are only supported for a single time-aligned tabular block"
            raise ValueError(msg)
        selected_features = validate_features(features, self.feature_names)
        train_X_parts: list[NDArray[np.float32]] = []
        train_y_parts: list[NDArray[np.int32]] = []
        val_X_parts: list[NDArray[np.float32]] = []
        val_y_parts: list[NDArray[np.int32]] = []
        test_X_parts: list[NDArray[np.float32]] = []
        test_y_parts: list[NDArray[np.int32]] = []
        groups = self.df.groupby(self.group_column) if self.group_column in self.df.columns else [(None, self.df)]
        for _, group_df in groups:
            group_features = group_df[selected_features].to_numpy(dtype=np.float32)
            group_labels = group_df["fault_state"].to_numpy(dtype=np.int32)
            X_tr, y_tr, X_va, y_va, X_te, y_te = split_and_window(group_features, group_labels, wc, split, split_bounds=split_bounds)
            if len(X_tr) > 0:
                train_X_parts.append(X_tr)
                train_y_parts.append(y_tr)
            if len(X_va) > 0:
                val_X_parts.append(X_va)
                val_y_parts.append(y_va)
            if len(X_te) > 0:
                test_X_parts.append(X_te)
                test_y_parts.append(y_te)
        X_train, y_train, X_val, y_val, X_test, y_test = collect_splits(
            wc, len(selected_features), train_X_parts, train_y_parts, val_X_parts, val_y_parts, test_X_parts, test_y_parts
        )
        windowed = WindowedSplits(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
        if split_bounds is not None:
            train_start, train_end, val_end, test_end = split_bounds
            windowed.split_bounds = {"train": (train_start, train_end), "val": (train_end, val_end), "test": (val_end, test_end)}
        return windowed

    def _prepare_graph(
        self,
        window_config: WindowConfig | None,
        split_config: DataSplitConfig | None,
        features: list[str] | None,
    ) -> WindowedSplits:
        wc = window_config if window_config is not None else WindowConfig()
        split = split_config if split_config is not None else DataSplitConfig(strategy="connectivity-chronological")
        selected_features = validate_features(features, self.feature_names)
        if selected_features != ["temp"]:
            raise ValueError('Dynamic graph preparation currently supports only features=["temp"]')
        df = self.df
        timestamps = sorted(df[self.timestamp_column].unique())
        ts_index = {ts: i for i, ts in enumerate(timestamps)}
        node_index = {nid: i for i, nid in enumerate(self.node_ids)}
        T = len(timestamps)
        N = len(self.node_ids)
        X = np.zeros((T, N, 1), dtype=np.float32)
        y = np.full((T, N), -1, dtype=np.int32)
        node_mask = np.zeros((T, N), dtype=np.bool_)
        for row in df[[self.timestamp_column, self.group_column, "temp", "fault_state"]].itertuples(index=False):
            t = ts_index[getattr(row, self.timestamp_column)]
            n = node_index[int(getattr(row, self.group_column))]
            X[t, n, 0] = np.float32(getattr(row, "temp"))
            y[t, n] = np.int32(getattr(row, "fault_state"))
            node_mask[t, n] = True
        edge_mask_all = self._available_edge_mask(node_mask)
        train_start, train_end, val_end, test_end = self._split_boundaries(T, edge_mask_all, wc, split)
        X_train, y_train, train_starts = create_windows_with_starts(
            X[train_start:train_end], y[train_start:train_end], wc.window_size, wc.train_stride
        )
        X_val, y_val, val_starts = create_windows_with_starts(
            X[train_end:val_end], y[train_end:val_end], wc.window_size, wc.test_stride
        )
        X_test, y_test, test_starts = create_windows_with_starts(
            X[val_end:test_end], y[val_end:test_end], wc.window_size, wc.test_stride
        )
        train_starts = train_starts + train_start
        val_starts = val_starts + train_end
        test_starts = test_starts + val_end
        graph_meta = dict(self.metadata.get("graph", {}))
        metadata = GraphMetadata(
            edge_index=self.edge_index,
            edge_prob=self.edge_prob,
            node_ids=self.node_ids,
            num_nodes=self.num_nodes,
            threshold=self.threshold,
            edge_distance_m=self.edge_distance_m,
            dynamic_link_seed=graph_meta.get("seed"),
            burst_params=dict(graph_meta.get("burst_params", {})),
            timestamps=[str(ts) for ts in timestamps],
            link_mask_shape=tuple(self.link_mask.shape),
            position_metadata=dict(self.node_positions.get("metadata", {})),
            distance_metadata=dict(self.metadata.get("distances", {})),
        )
        return WindowedSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            metadata={"graph": metadata},
            split_bounds={"train": (train_start, train_end), "val": (train_end, val_end), "test": (val_end, test_end)},
            node_mask_train=self._window_by_starts(node_mask, train_starts, wc.window_size),
            node_mask_val=self._window_by_starts(node_mask, val_starts, wc.window_size),
            node_mask_test=self._window_by_starts(node_mask, test_starts, wc.window_size),
            edge_mask_train=self._window_by_starts(edge_mask_all, train_starts, wc.window_size),
            edge_mask_val=self._window_by_starts(edge_mask_all, val_starts, wc.window_size),
            edge_mask_test=self._window_by_starts(edge_mask_all, test_starts, wc.window_size),
        )

    def _prepare_aligned_tabular(
        self,
        window_config: WindowConfig | None,
        split_config: DataSplitConfig | None,
        features: list[str] | None,
        required_metadata: set[str] | None,
    ) -> WindowedSplits:
        wc = window_config if window_config is not None else WindowConfig()
        split = split_config if split_config is not None else DataSplitConfig(strategy="connectivity-chronological")
        selected_features = validate_features(features, self.feature_names)
        df = self.df
        timestamps = sorted(df[self.timestamp_column].unique())
        ts_index = {ts: i for i, ts in enumerate(timestamps)}
        T = len(timestamps)
        N = len(self.node_ids)
        F = len(selected_features)
        feature_blocks = np.zeros((N, T, F), dtype=np.float32)
        label_blocks = np.full((N, T), -1, dtype=np.int32)
        node_mask = np.zeros((T, N), dtype=np.bool_)
        node_index = {nid: i for i, nid in enumerate(self.node_ids)}
        for row in df[[self.timestamp_column, self.group_column, "fault_state", *selected_features]].itertuples(index=False):
            t = ts_index[getattr(row, self.timestamp_column)]
            n = node_index[int(getattr(row, self.group_column))]
            for feature_idx, feature_name in enumerate(selected_features):
                feature_blocks[n, t, feature_idx] = np.float32(getattr(row, feature_name))
            label_blocks[n, t] = np.int32(getattr(row, "fault_state"))
            node_mask[t, n] = True
        edge_mask_all = self._available_edge_mask(node_mask)
        train_start, train_end, val_end, test_end = self._split_boundaries(T, edge_mask_all, wc, split)
        train_X_parts: list[NDArray[np.float32]] = []
        train_y_parts: list[NDArray[np.int32]] = []
        val_X_parts: list[NDArray[np.float32]] = []
        val_y_parts: list[NDArray[np.int32]] = []
        test_X_parts: list[NDArray[np.float32]] = []
        test_y_parts: list[NDArray[np.int32]] = []
        train_node_parts: list[NDArray[np.int64]] = []
        val_node_parts: list[NDArray[np.int64]] = []
        test_node_parts: list[NDArray[np.int64]] = []
        for node_pos in range(N):
            valid = node_mask[:, node_pos]
            X_tr, y_tr, tr_starts = create_windows_with_starts(
                feature_blocks[node_pos, train_start:train_end],
                label_blocks[node_pos, train_start:train_end],
                wc.window_size,
                wc.train_stride,
            )
            tr_starts = tr_starts + train_start
            tr_keep = self._valid_window_starts(valid, tr_starts, wc.window_size)
            X_tr = X_tr[tr_keep]
            y_tr = y_tr[tr_keep]
            X_va, y_va, va_starts = create_windows_with_starts(
                feature_blocks[node_pos, train_end:val_end],
                label_blocks[node_pos, train_end:val_end],
                wc.window_size,
                wc.test_stride,
            )
            va_starts = va_starts + train_end
            va_keep = self._valid_window_starts(valid, va_starts, wc.window_size)
            X_va = X_va[va_keep]
            y_va = y_va[va_keep]
            X_te, y_te, te_starts = create_windows_with_starts(
                feature_blocks[node_pos, val_end:test_end],
                label_blocks[node_pos, val_end:test_end],
                wc.window_size,
                wc.test_stride,
            )
            te_starts = te_starts + val_end
            te_keep = self._valid_window_starts(valid, te_starts, wc.window_size)
            X_te = X_te[te_keep]
            y_te = y_te[te_keep]
            if len(X_tr) > 0:
                train_X_parts.append(X_tr)
                train_y_parts.append(y_tr)
                train_node_parts.append(np.full((len(X_tr),), node_pos, dtype=np.int64))
            if len(X_va) > 0:
                val_X_parts.append(X_va)
                val_y_parts.append(y_va)
                val_node_parts.append(np.full((len(X_va),), node_pos, dtype=np.int64))
            if len(X_te) > 0:
                test_X_parts.append(X_te)
                test_y_parts.append(y_te)
                test_node_parts.append(np.full((len(X_te),), node_pos, dtype=np.int64))
        X_train, y_train, X_val, y_val, X_test, y_test = collect_splits(
            wc, F, train_X_parts, train_y_parts, val_X_parts, val_y_parts, test_X_parts, test_y_parts
        )
        metadata: dict[str, Any] = {}
        if required_metadata is not None and "node_identity" in required_metadata:
            metadata["node_identity"] = {
                "num_nodes": N,
                "train_node_ids": np.concatenate(train_node_parts) if train_node_parts else np.empty((0,), dtype=np.int64),
                "val_node_ids": np.concatenate(val_node_parts) if val_node_parts else np.empty((0,), dtype=np.int64),
                "test_node_ids": np.concatenate(test_node_parts) if test_node_parts else np.empty((0,), dtype=np.int64),
                "node_ids": list(self.node_ids),
            }
        return WindowedSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata,
            split_bounds={"train": (train_start, train_end), "val": (train_end, val_end), "test": (val_end, test_end)},
        )

    def _available_edge_mask(self, node_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if self.link_mask.shape != (node_mask.shape[0], self.edge_index.shape[1]):
            raise ValueError(
                f"link_mask shape {self.link_mask.shape} does not match "
                f"(T,E)=({node_mask.shape[0]},{self.edge_index.shape[1]})"
            )
        if self.edge_index.shape[1] == 0:
            return self.link_mask.copy()
        sender = self.edge_index[0]
        receiver = self.edge_index[1]
        return self.link_mask & node_mask[:, sender] & node_mask[:, receiver]

    def _split_boundaries(
        self, num_timestamps: int, edge_mask: NDArray[np.bool_], wc: WindowConfig, split: DataSplitConfig
    ) -> tuple[int, int, int, int]:
        target_train_end, target_val_end = split_boundaries(num_timestamps, split)
        if split.strategy == "chronological":
            return 0, target_train_end, target_val_end, num_timestamps
        active_timesteps = np.flatnonzero(edge_mask.any(axis=1))
        if len(active_timesteps) == 0:
            msg = "Unable to create connectivity-chronological graph split: no active graph edges were found."
            raise ValueError(msg)
        active_start = int(active_timesteps[0])
        active_end = int(active_timesteps[-1]) + 1
        active_len = active_end - active_start
        train_end_rel, val_end_rel = split_boundaries(active_len, split)
        train_end = active_start + train_end_rel
        val_end = active_start + val_end_rel
        test_end = active_end
        train_ok = self._split_has_available_edges(edge_mask, active_start, train_end, wc.train_stride, wc.window_size)
        val_ok = self._split_has_available_edges(edge_mask, train_end, val_end, wc.test_stride, wc.window_size)
        test_ok = self._split_has_available_edges(edge_mask, val_end, test_end, wc.test_stride, wc.window_size)
        if not (train_ok and val_ok and test_ok):
            msg = "Unable to create connectivity-chronological graph split: train, validation, and test splits must each contain active-edge windows."
            raise ValueError(msg)
        return active_start, train_end, val_end, test_end

    @staticmethod
    def _split_has_available_edges(edge_mask: NDArray[np.bool_], start: int, end: int, stride: int, window_size: int) -> bool:
        if end - start < window_size:
            return False
        starts = range(start, end - window_size + 1, stride)
        return any(bool(edge_mask[i : i + window_size].any()) for i in starts)

    @staticmethod
    def _valid_window_starts(valid: NDArray[np.bool_], starts: NDArray[np.int64], window_size: int) -> NDArray[np.bool_]:
        if len(starts) == 0:
            return np.empty((0,), dtype=np.bool_)
        return np.asarray([bool(valid[i : i + window_size].all()) for i in starts], dtype=np.bool_)

    @staticmethod
    def _window_by_starts(values: NDArray[np.bool_], starts: NDArray[np.int64], window_size: int) -> NDArray[np.bool_]:
        if len(starts) == 0:
            return np.empty((0, window_size) + values.shape[1:], dtype=np.bool_)
        return np.stack([values[i : i + window_size] for i in starts]).astype(np.bool_)


def pack_link_mask(mask: NDArray[np.bool_]) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    return np.packbits(mask.reshape(-1)), np.asarray(mask.shape, dtype=np.int64)


def unpack_link_mask(packed: NDArray[np.uint8], shape: tuple[int, int] | NDArray[np.integer[Any]]) -> NDArray[np.bool_]:
    shape_tuple = tuple(int(x) for x in shape)
    total = int(np.prod(shape_tuple))
    return np.unpackbits(packed)[:total].reshape(shape_tuple).astype(np.bool_)


def load_dataset(path: str | Path) -> CESTADataset:
    return CESTADataset.load(path)
