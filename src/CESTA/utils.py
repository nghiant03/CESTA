"""Shared runtime helpers used across CLI subcommands.

Collectors snapshot git, environment, and produce run identifiers
that populate :class:`CESTA.schema.manifest.RunManifest`. Dataset
identity comes from ``CESTADataset.describe()``.
"""

from __future__ import annotations

import hashlib
import platform
import socket
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from CESTA.schema.manifest import EnvInfo, GitInfo


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Stream-hash a file with SHA-256."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_git_info(cwd: Path | None = None) -> GitInfo:
    """Return git commit/branch/dirty state of ``cwd`` via dulwich."""
    from dulwich.errors import NotGitRepository
    from dulwich.porcelain import status
    from dulwich.repo import Repo

    wd = cwd if cwd is not None else Path.cwd()
    try:
        repo = Repo.discover(str(wd))
    except NotGitRepository:
        return GitInfo()

    try:
        head_sha_bytes = repo.head()
    except KeyError:
        return GitInfo()
    commit = head_sha_bytes.decode("ascii")

    branch: str | None = None
    try:
        from dulwich.refs import Ref

        ref = repo.refs.read_ref(Ref(b"HEAD"))
        if ref is not None and ref.startswith(b"ref: refs/heads/"):
            branch = ref[len(b"ref: refs/heads/") :].decode("utf-8")
    except (KeyError, OSError):
        branch = None

    try:
        st = status(repo)
        dirty = bool(
            st.staged.get("add")
            or st.staged.get("delete")
            or st.staged.get("modify")
            or st.unstaged
            or st.untracked
        )
    except Exception:
        dirty = False

    repo.close()
    return GitInfo(
        commit=commit,
        short_sha=commit[:7],
        branch=branch,
        dirty=dirty,
    )


def _cesta_version() -> str:
    try:
        return metadata.version("CESTA")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def collect_env_info(device: Any | None = None) -> EnvInfo:
    """Snapshot the current execution environment.

    ``device`` may be a ``torch.device``, a device string (``"cuda"``,
    ``"cpu"``), or ``None`` to auto-select.
    """
    import torch

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device) if isinstance(device, str) else device

    device_name: str | None = None
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(idx)

    torch_version_module = getattr(torch, "version", None)
    cuda_version = getattr(torch_version_module, "cuda", None)

    return EnvInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        hostname=socket.gethostname(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=cuda_version,
        device=str(dev),
        device_name=device_name,
        cesta_version=_cesta_version(),
    )


def utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_run_id(model: str, seed: int, git: GitInfo) -> str:
    """Generate a ``<timestamp>_<model>_seed<seed>_<shortsha>`` run ID.

    Non-pure: samples the wall clock at call time.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git.short_sha or "nogit"
    return f"{ts}_{model}_seed{seed}_{sha}"
