"""Deep learning models for fault diagnosis.

This module provides model architectures and a registry system for
managing different model implementations.

Subpackages:
    temporal/  - Temporal sequence classifiers
    spatial/   - CESTA, DCRNN, HiFiNet, ST-GCN
"""

from CESTA.models.base import BaseModel
from CESTA.models.registry import (
    create_model,
    get_model_class,
    is_registered,
    list_models,
    register_model,
)
from CESTA.models.spatial import CESTAClassifier, DCRNNClassifier, HiFiNetClassifier, STGCNClassifier
from CESTA.models.temporal import (
    AutoformerClassifier,
    CNN1DClassifier,
    HydraClassifier,
    InformerClassifier,
    ModernTCNClassifier,
    PatchTSTClassifier,
    TransformerClassifier,
)

__all__ = [
    "AutoformerClassifier",
    "BaseModel",
    "CESTAClassifier",
    "CNN1DClassifier",
    "DCRNNClassifier",
    "HiFiNetClassifier",
    "HydraClassifier",
    "InformerClassifier",
    "ModernTCNClassifier",
    "PatchTSTClassifier",
    "STGCNClassifier",
    "TransformerClassifier",
    "create_model",
    "get_model_class",
    "is_registered",
    "list_models",
    "register_model",
]
