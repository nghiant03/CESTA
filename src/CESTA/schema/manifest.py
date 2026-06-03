"""Run manifest schema.

A ``RunManifest`` captures everything needed to reproduce and fairly
compare a single training/evaluation invocation: code version,
environment, dataset identity, timing, and the full resolved configs.
It is written as ``manifest.json`` inside each run directory.

Runtime collectors that populate these models live in
:mod:`CESTA.utils`; ``DatasetInfo`` is produced by
``CESTADataset.describe()``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EnvInfo(BaseModel):
    """Execution environment snapshot."""

    model_config = ConfigDict(frozen=True)

    python_version: str
    platform: str
    hostname: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None = None
    device: str
    device_name: str | None = None
    cesta_version: str


class GitInfo(BaseModel):
    """Git state at run time."""

    model_config = ConfigDict(frozen=True)

    commit: str | None = None
    short_sha: str | None = None
    branch: str | None = None
    dirty: bool = False


class DatasetInfo(BaseModel):
    """Identity of the canonical dataset used for the run."""

    model_config = ConfigDict(frozen=True)

    path: str
    data_sha256: str | None = None
    meta_sha256: str | None = None
    num_features: int = 0
    feature_names: list[str] = Field(default_factory=list)
    num_groups: int = 0
    total_timesteps: int = 0


class Timing(BaseModel):
    """Wall-clock timing information for a run."""

    model_config = ConfigDict(frozen=True)

    started_at: str
    ended_at: str | None = None
    duration_seconds: float | None = None
    epochs_run: int | None = None


class RunManifest(BaseModel):
    """Full manifest persisted as ``manifest.json`` per run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    kind: str = "train"
    seed: int
    model: str
    num_parameters: int | None = None

    git: GitInfo = Field(default_factory=GitInfo)
    env: EnvInfo
    dataset: DatasetInfo
    timing: Timing

    train_config: dict[str, Any] | None = None
    eval_config: dict[str, Any] | None = None
    injection_config: dict[str, Any] | None = None
