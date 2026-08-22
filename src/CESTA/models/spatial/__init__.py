"""Spatial models for fault diagnosis."""

from CESTA.models.spatial.cesta import CESTAClassifier
from CESTA.models.spatial.dcrnn import DCRNNClassifier
from CESTA.models.spatial.hifinet import HiFiNetClassifier
from CESTA.models.spatial.hmct import HMCTClassifier
from CESTA.models.spatial.stgcn import STGCNClassifier

__all__ = [
    "CESTAClassifier",
    "DCRNNClassifier",
    "HiFiNetClassifier",
    "HMCTClassifier",
    "STGCNClassifier",
]
