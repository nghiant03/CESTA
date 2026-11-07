"""Training-only initialization of the portable PatchTST graph encoder."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from CESTA.models.base import BaseModel
from CESTA.schema import TrainConfig


def initialize_temporal_transfer(model: BaseModel, config: TrainConfig, data: Path) -> None:
    from CESTA.artifacts import load_checkpoint
    from CESTA.models.spatial.cesta.model import CESTAClassifier
    from CESTA.models.spatial.cesta.temporal import PatchTSTTemporalEncoder
    from CESTA.models.temporal.patchtst import PatchTSTClassifier
    from CESTA.utils import sha256_file

    path = Path(config.temporal_transfer_checkpoint or "")
    manifest = json.loads((path / "manifest.json").read_text())
    tc = TrainConfig.model_validate(manifest["train_config"])
    if tc.features != config.features or tc.data != config.data:
        raise ValueError("Transfer requires matching teacher features and split/window recipe")
    for filename, key in (("dataset.csv", "data_sha256"), ("dataset_meta.json", "meta_sha256")):
        if sha256_file(data / filename) != manifest["dataset"][key]:
            raise ValueError("Transfer dataset hash mismatch")
    if not isinstance(model, CESTAClassifier) or model.standardize_window_features:
        raise ValueError("Transfer requires CESTA with raw inputs and internal PatchTST scaling")
    encoder = model.temporal_encoder
    if not isinstance(encoder, PatchTSTTemporalEncoder) or not encoder.standalone_compatible:
        raise ValueError("Transfer requires standalone-compatible PatchTST backend")
    teacher = load_checkpoint(path, map_location="cpu")
    if not isinstance(teacher, PatchTSTClassifier):
        raise ValueError("Transfer teacher must be PatchTST")
    encoder.encoder.load_state_dict(teacher.encoder.state_dict(), strict=True)
    encoder.layer_norm.load_state_dict(teacher.layer_norm.state_dict(), strict=True)
    width = teacher.fc.in_features
    if encoder.projection.out_features < width:
        raise ValueError("Payload too small for identity embedding of teacher hidden states")
    with torch.no_grad():
        encoder.projection.weight.zero_()
        encoder.projection.bias.zero_()
        encoder.projection.weight[:width].copy_(torch.eye(width))
        model.classifier.weight.zero_()
        model.classifier.weight[:, :width].copy_(teacher.fc.weight)
        model.classifier.bias.copy_(teacher.fc.bias)
    if model.count_parameters() > 719327:
        raise ValueError("Transfer model exceeds original DHT parameter budget")
