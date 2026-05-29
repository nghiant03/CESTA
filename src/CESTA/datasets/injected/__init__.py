"""Injected dataset containers, graph topology, and windowing utilities."""

from CESTA.datasets.injected.loading import load_dataset
from CESTA.datasets.injected.tabular import InjectedDataset

__all__ = [
    "InjectedDataset",
    "load_dataset",
]
