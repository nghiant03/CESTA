"""Spatial models for fault diagnosis."""

from CESTA.models.spatial.cesta import CESTAClassifier
from CESTA.models.spatial.hifinet import HiFiNetClassifier
from CESTA.models.spatial.stgcn import STGCNClassifier

__all__ = [
    "CESTAClassifier",
    "HiFiNetClassifier",
    "STGCNClassifier",
]
