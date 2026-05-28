"""Dataset loaders for fault diagnosis.

Provides standardized access to sensor datasets for fault injection.

Sub-packages
------------
- ``raw`` — Raw dataset loaders (pre-injection stage).
- ``injected`` — Post-injection containers, graph topology, and windowing.
"""

from CESTA.datasets.injected.graph import GraphDataset, GraphMetadata
from CESTA.datasets.injected.loading import load_dataset
from CESTA.datasets.injected.tabular import InjectedDataset
from CESTA.datasets.injected.windowed import WindowedSplits
from CESTA.datasets.raw import get_dataset, list_datasets
from CESTA.datasets.raw.base import BaseDataset
from CESTA.datasets.raw.intel_lab import IntelLabDataset

__all__ = [
    "BaseDataset",
    "GraphDataset",
    "GraphMetadata",
    "InjectedDataset",
    "IntelLabDataset",
    "WindowedSplits",
    "load_dataset",
    "get_dataset",
    "list_datasets",
]
