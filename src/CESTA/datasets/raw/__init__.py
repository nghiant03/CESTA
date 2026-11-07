"""Raw dataset loaders."""

from __future__ import annotations

from pathlib import Path

from CESTA.datasets.raw.base import BaseDataset
from CESTA.datasets.raw.cesta_dht import CestaDHTDataset
from CESTA.datasets.raw.intel_lab import IntelLabDataset

_DATASET_LOADERS: dict[str, type[BaseDataset]] = {
    "intel_lab": IntelLabDataset,
    "cesta_dht": CestaDHTDataset,
}


def get_dataset(name: str, data_path: str | Path) -> BaseDataset:
    """Create a raw dataset loader by name."""
    dataset_cls = _DATASET_LOADERS.get(name.lower())
    if dataset_cls is None:
        available = ", ".join(list_datasets())
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")
    return dataset_cls(data_path)


def list_datasets() -> list[str]:
    """Return available raw dataset names."""
    return list(_DATASET_LOADERS)


__all__ = [
    "BaseDataset",
    "CestaDHTDataset",
    "IntelLabDataset",
    "get_dataset",
    "list_datasets",
]
