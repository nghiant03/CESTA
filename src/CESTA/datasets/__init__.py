"""Dataset loaders for fault diagnosis.

Provides standardized access to sensor datasets for fault injection.

Sub-packages
------------
- ``raw`` — Raw dataset loaders (pre-injection stage).
- ``injected`` — Post-injection containers, graph topology, and windowing.
"""

from CESTA.datasets.injected.loading import load_dataset
from CESTA.datasets.injected.tabular import InjectedDataset
from CESTA.datasets.raw import get_dataset, list_datasets

__all__ = [
    "InjectedDataset",
    "load_dataset",
    "get_dataset",
    "list_datasets",
]
