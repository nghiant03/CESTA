"""Run artifact and checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch

from CESTA.schema import RunManifest
from CESTA.schema.manifest import GitInfo
from CESTA.utils import generate_run_id


class CheckpointModel(Protocol):
    @property
    def name(self) -> str: ...

    def get_config(self) -> dict[str, object]: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> object: ...

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> object: ...


ModelT = TypeVar("ModelT", bound=CheckpointModel)


def create_run_dir(output_root: Path, *, model: str, seed: int, git: GitInfo) -> Path:
    run_dir = output_root / generate_run_id(model, seed, git)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(path: str | Path, manifest: RunManifest) -> Path:
    manifest_path = Path(path) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2))
    return manifest_path


def save_checkpoint(model: CheckpointModel, path: str | Path, config_dict: dict[str, object] | None = None) -> None:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_weight_path(directory))
    checkpoint_metadata_path(directory).write_text(json.dumps(build_checkpoint_metadata(model, config_dict), indent=2))


def build_checkpoint_metadata(model: CheckpointModel, config_dict: dict[str, object] | None = None) -> dict[str, object]:
    meta: dict[str, object] = {
        "model_name": model.name,
        "model_config": model.get_config(),
    }
    if config_dict is not None:
        meta["train_config"] = config_dict
    return meta


def checkpoint_metadata_path(path: str | Path) -> Path:
    return Path(path) / "config.json"


def checkpoint_weight_path(path: str | Path) -> Path:
    return Path(path) / "weight.pt"


def load_checkpoint_metadata(path: str | Path) -> dict[str, object]:
    return json.loads(checkpoint_metadata_path(path).read_text())  # type: ignore[no-any-return]


def load_checkpoint_train_config(path: str | Path) -> dict[str, object] | None:
    meta = load_checkpoint_metadata(path)
    train_config = meta.get("train_config")
    return train_config if isinstance(train_config, dict) else None


def instantiate_model_from_config(config: dict[str, object], model_cls: type[ModelT]) -> ModelT:
    return model_cls(**config)


def load_checkpoint_weights(model: ModelT, path: str | Path, *, map_location: object | None = None) -> ModelT:
    load_kwargs: dict[str, Any] = {"weights_only": True}
    if map_location is not None:
        load_kwargs["map_location"] = map_location
    state_dict: dict[str, Any] = torch.load(checkpoint_weight_path(path), **load_kwargs)
    model_state = model.state_dict()
    if "edge_distance_m" in state_dict and isinstance(model_state, dict) and "edge_distance_m" not in model_state:
        state_dict.pop("edge_distance_m")
    model.load_state_dict(state_dict)
    return model


def load_checkpoint(path: str | Path, *, map_location: object | None = None) -> CheckpointModel:
    from CESTA.models.registry import get_model_class

    meta = load_checkpoint_metadata(path)
    model_name = meta.get("model_name")
    model_config = meta.get("model_config")
    if not isinstance(model_name, str):
        raise ValueError("Checkpoint metadata is missing model_name")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint metadata is missing model_config")
    model = instantiate_model_from_config(model_config, get_model_class(model_name))
    return load_checkpoint_weights(model, path, map_location=map_location)
