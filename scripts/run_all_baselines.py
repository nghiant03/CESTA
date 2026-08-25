from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

import yaml

from CESTA.schema import TrainConfig
from CESTA.schema.config import load_config_file
from CESTA.utils import sha256_file

DEFAULT_CONFIGS = (
    Path("config/training/cnn-1d.yaml"),
    Path("config/training/transformer.yaml"),
    Path("config/training/autoformer.yaml"),
    Path("config/training/informer.yaml"),
    Path("config/training/patch-tst.yaml"),
    Path("config/training/modern-tcn.yaml"),
    Path("config/training/hydra.yaml"),
    Path("config/training/st-gcn.yaml"),
    Path("config/training/hifinet.yaml"),
    Path("config/training/hmct.yaml"),
    Path("config/training/dcrnn.yaml"),
    Path("config/training/cesta.yaml"),
)
DEFAULT_DATASETS = (
    Path("data/datasets/Intel_fault05"),
    Path("data/datasets/Intel_fault10"),
    Path("data/datasets/Intel_fault15"),
    Path("data/datasets/Intel_fault20"),
)
DEFAULT_SEEDS = (12, 42, 1242)


@dataclass(frozen=True)
class BaselineTask:
    config_path: Path
    dataset_path: Path
    seed: int
    model: str

    @property
    def key(self) -> str:
        payload = {
            "config": str(self.config_path.resolve()),
            "dataset": str(self.dataset_path.resolve()),
            "seed": self.seed,
        }
        return json.dumps(payload, sort_keys=True)

    def as_record(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "config": str(self.config_path),
            "dataset": str(self.dataset_path),
            "seed": self.seed,
        }


@dataclass
class RunningTask:
    task: BaselineTask
    process: subprocess.Popen[bytes]
    gpu: str
    started_at: str
    started: float
    position: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def raise_keyboard_interrupt(_signum: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def resolve_gpu_ids(num_gpus: int) -> list[str]:
    if num_gpus < 1:
        msg = "--num-gpus must be at least 1"
        raise ValueError(msg)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return [str(index) for index in range(num_gpus)]
    gpu_ids = [gpu_id.strip() for gpu_id in visible_devices.split(",") if gpu_id.strip()]
    if len(gpu_ids) < num_gpus:
        msg = f"--num-gpus={num_gpus} exceeds the {len(gpu_ids)} devices in CUDA_VISIBLE_DEVICES"
        raise ValueError(msg)
    return gpu_ids[:num_gpus]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every baseline config on the four canonical datasets with seeds 12, 42, and 1242.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Root directory for run outputs and progress logs.")
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_CONFIGS), help="Training config files to run.")
    parser.add_argument("--datasets", nargs="+", type=Path, default=list(DEFAULT_DATASETS), help="Canonical CESTA dataset directories to train on.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Seeds to run for each config/dataset pair.")
    parser.add_argument("--runner", nargs="+", default=["uv", "run", "cesta"], help="Command prefix used to invoke the CESTA CLI.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of visible GPUs to use, with one independent training run per GPU.")
    parser.add_argument("--early-stopping", action="store_true", help="Forward --early-stopping to cesta train.")
    parser.add_argument("--no-test-evaluation", action="store_true", help="Train and checkpoint using validation data without evaluating test data.")
    parser.add_argument("--restart", action="store_true", help="Ignore existing progress state and start from scratch.")
    parser.add_argument("--keep-progress-log", action="store_true", help="Keep progress logs after every run finishes.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned and remaining run count without launching training.")
    return parser.parse_args()


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        msg = f"Config file must contain a mapping: {path}"
        raise ValueError(msg)
    return raw


def model_name_from_config(path: Path) -> str:
    raw = load_yaml_mapping(path)
    train_section = raw.get("train", raw)
    if not isinstance(train_section, dict):
        msg = f"Config train section must contain a mapping: {path}"
        raise ValueError(msg)
    model = train_section.get("model")
    if not isinstance(model, str) or not model:
        msg = f"Config does not define train.model: {path}"
        raise ValueError(msg)
    return model


def validate_paths(configs: list[Path], datasets: list[Path]) -> None:
    missing_configs = [str(path) for path in configs if not path.is_file()]
    missing_datasets = [str(path) for path in datasets if not path.is_dir()]
    if missing_configs or missing_datasets:
        details = []
        if missing_configs:
            details.append("missing configs: " + ", ".join(missing_configs))
        if missing_datasets:
            details.append("missing datasets: " + ", ".join(missing_datasets))
        raise FileNotFoundError("; ".join(details))


def build_tasks(configs: list[Path], datasets: list[Path], seeds: list[int]) -> list[BaselineTask]:
    validate_paths(configs, datasets)
    model_names = {config: model_name_from_config(config) for config in configs}
    return [
        BaselineTask(config_path=config, dataset_path=dataset, seed=seed, model=model_names[config])
        for dataset in datasets
        for config in configs
        for seed in seeds
    ]


def discover_completed_tasks(tasks: list[BaselineTask], runs_dir: Path, *, require_test_artifacts: bool = True) -> dict[str, dict[str, Any]]:
    expected = {_task_signature(task): task for task in tasks}
    completed: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(runs_dir.glob("*/*/manifest.json")):
        run_dir = manifest_path.parent
        if not _has_complete_run_artifacts(run_dir, require_test_artifacts=require_test_artifacts):
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        signature = _manifest_signature(manifest)
        if signature is None:
            continue
        task = expected.get(signature)
        if task is None or task.key in completed:
            continue
        completed[task.key] = task.as_record() | {
            "completed_at": manifest.get("timing", {}).get("ended_at"),
            "run_id": manifest.get("run_id"),
            "run_path": str(run_dir),
            "source": "manifest",
        }
    return completed


def _task_signature(task: BaselineTask) -> str:
    raw_config = copy.deepcopy(load_config_file(task.config_path))
    train_section = raw_config.get("train")
    if isinstance(train_section, dict):
        train_section["seed"] = task.seed
    else:
        raw_config["seed"] = task.seed
    train_config = TrainConfig.model_validate(raw_config).model_dump(mode="json")
    dataset = {
        "data_sha256": sha256_file(task.dataset_path / "dataset.csv"),
        "meta_sha256": sha256_file(task.dataset_path / "dataset_meta.json"),
    }
    return json.dumps(
        {"model": task.model, "seed": task.seed, "train_config": train_config, "dataset": dataset},
        sort_keys=True,
    )


def _manifest_signature(manifest: object) -> str | None:
    if not isinstance(manifest, dict):
        return None
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        return None
    model = manifest.get("model")
    seed = manifest.get("seed")
    train_config = manifest.get("train_config")
    data_sha256 = dataset.get("data_sha256")
    meta_sha256 = dataset.get("meta_sha256")
    if not isinstance(model, str) or not isinstance(seed, int) or not isinstance(train_config, dict):
        return None
    if not isinstance(data_sha256, str) or not isinstance(meta_sha256, str):
        return None
    return json.dumps(
        {
            "model": model,
            "seed": seed,
            "train_config": train_config,
            "dataset": {"data_sha256": data_sha256, "meta_sha256": meta_sha256},
        },
        sort_keys=True,
    )


def _has_complete_run_artifacts(run_dir: Path, *, require_test_artifacts: bool = True) -> bool:
    required = ["manifest.json", "config.json", "weight.pt", "history.jsonl"]
    if require_test_artifacts:
        required.extend(["eval_metrics.json", "predictions.npz"])
    return all((run_dir / name).is_file() for name in required)


def initial_progress(total: int) -> dict[str, Any]:
    now = utc_now()
    return {
        "started_at": now,
        "updated_at": now,
        "total": total,
        "completed": {},
        "failures": [],
        "current": None,
    }


def load_progress(path: Path, total: int, restart: bool) -> dict[str, Any]:
    if restart or not path.exists():
        return initial_progress(total)
    progress = json.loads(path.read_text())
    if not isinstance(progress, dict):
        msg = f"Progress state must be a JSON object: {path}"
        raise ValueError(msg)
    progress.setdefault("completed", {})
    progress.setdefault("failures", [])
    progress["total"] = total
    progress["current"] = None
    progress["updated_at"] = utc_now()
    return progress


def write_progress(path: Path, progress: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(event, sort_keys=True) + "\n")


def prepare_temp_config(config_path: Path, seed: int, temp_dir: Path) -> Path:
    raw = copy.deepcopy(load_yaml_mapping(config_path))
    train_section = raw.get("train")
    if isinstance(train_section, dict):
        train_section["seed"] = seed
    else:
        raw["seed"] = seed
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{config_path.stem}_seed{seed}.yaml"
    temp_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return temp_path


def start_task(
    task: BaselineTask,
    temp_config: Path,
    runs_dir: Path,
    runner: list[str],
    early_stopping: bool,
    gpu: str,
    test_evaluation: bool = True,
) -> subprocess.Popen[bytes]:
    output_root = runs_dir / task.model
    command = [*runner, "train", str(temp_config), str(task.dataset_path), "--output", str(output_root)]
    if early_stopping:
        command.append("--early-stopping")
    if not test_evaluation:
        command.append("--no-test-evaluation")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"GPU {gpu} running:", " ".join(command), flush=True)
    return subprocess.Popen(command, env=environment, start_new_session=True)


def terminate_processes(running: dict[str, RunningTask]) -> None:
    for active in running.values():
        if active.process.poll() is None:
            os.killpg(active.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while any(active.process.poll() is None for active in running.values()) and time.monotonic() < deadline:
        time.sleep(0.2)
    for active in running.values():
        if active.process.poll() is None:
            os.killpg(active.process.pid, signal.SIGKILL)
        active.process.wait()


def cleanup_progress(paths: list[Path], temp_dir: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
    shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    args = parse_args()
    gpu_ids = resolve_gpu_ids(args.num_gpus)
    runs_dir = args.runs_dir
    state_path = runs_dir / "baseline_sweep_state.json"
    events_path = runs_dir / "baseline_sweep_events.jsonl"
    temp_dir = runs_dir / "baseline_sweep_configs"
    tasks = build_tasks(args.configs, args.datasets, args.seeds)
    progress = load_progress(state_path, len(tasks), args.restart)
    progress["completed"] = (
        {} if args.restart else discover_completed_tasks(tasks, runs_dir, require_test_artifacts=not args.no_test_evaluation)
    )
    completed = progress["completed"]
    remaining = [task for task in tasks if task.key not in completed]

    if args.dry_run:
        print(f"Planned runs: {len(tasks)} | already completed: {len(tasks) - len(remaining)} | remaining: {len(remaining)}")
        print(f"GPU workers: {args.num_gpus}")
        print(f"Progress state: {state_path}")
        if remaining:
            next_task = remaining[0]
            print(f"Next run: {next_task.model} {next_task.dataset_path.name} seed={next_task.seed}")
        return 0

    progress["current"] = []
    write_progress(state_path, progress)
    print(f"Progress state: {state_path}", flush=True)
    print(f"Progress events: {events_path}", flush=True)

    if not remaining:
        if not args.keep_progress_log:
            cleanup_progress([state_path, events_path], temp_dir)
            print("All runs already completed; progress logs cleaned up.", flush=True)
        else:
            print("All runs already completed.", flush=True)
        return 0

    pending = iter(remaining)
    running: dict[str, RunningTask] = {}
    next_position = len(progress["completed"]) + 1
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)

    def write_active_progress() -> None:
        progress["current"] = [
            active.task.as_record()
            | {"gpu": active.gpu, "position": active.position, "started_at": active.started_at}
            for active in sorted(running.values(), key=lambda item: item.gpu)
        ]
        progress["updated_at"] = utc_now()
        write_progress(state_path, progress)

    def launch(gpu: str, task: BaselineTask, position: int) -> None:
        record = task.as_record()
        started_at = utc_now()
        temp_config = prepare_temp_config(task.config_path, task.seed, temp_dir / f"gpu-{gpu}")
        process = start_task(
            task,
            temp_config,
            runs_dir,
            args.runner,
            args.early_stopping,
            gpu,
            test_evaluation=not args.no_test_evaluation,
        )
        running[gpu] = RunningTask(
            task=task,
            process=process,
            gpu=gpu,
            started_at=started_at,
            started=time.perf_counter(),
            position=position,
        )
        append_event(
            events_path,
            {"event": "start", "at": started_at, "completed": len(progress["completed"]), "total": len(tasks), "gpu": gpu, **record},
        )
        print(f"[{position}/{len(tasks)}] GPU {gpu}: {task.model} {task.dataset_path.name} seed={task.seed}", flush=True)

    try:
        for gpu in gpu_ids:
            task = next(pending, None)
            if task is None:
                break
            launch(gpu, task, next_position)
            next_position += 1
        write_active_progress()

        while running:
            finished_gpu = next((gpu for gpu, active in running.items() if active.process.poll() is not None), None)
            if finished_gpu is None:
                time.sleep(0.5)
                continue
            active = running.pop(finished_gpu)
            returncode = active.process.returncode
            duration = time.perf_counter() - active.started
            finished_at = utc_now()
            record = active.task.as_record()

            if returncode != 0:
                failure = record | {
                    "gpu": active.gpu,
                    "returncode": returncode,
                    "failed_at": finished_at,
                    "duration_seconds": duration,
                }
                progress["failures"].append(failure)
                append_event(events_path, {"event": "failure", "at": finished_at, **failure})
                terminate_processes(running)
                for interrupted in running.values():
                    append_event(events_path, {"event": "interrupted", "at": utc_now(), "gpu": interrupted.gpu, **interrupted.task.as_record()})
                running.clear()
                write_active_progress()
                print(f"Failed with return code {returncode}. Resume with the same command after fixing the issue.", file=sys.stderr, flush=True)
                return returncode

            success = record | {"gpu": active.gpu, "completed_at": finished_at, "duration_seconds": duration}
            progress["completed"][active.task.key] = success
            append_event(
                events_path,
                {"event": "success", "at": finished_at, "completed": len(progress["completed"]), "total": len(tasks), **success},
            )
            task = next(pending, None)
            if task is not None:
                launch(finished_gpu, task, next_position)
                next_position += 1
            write_active_progress()
    except KeyboardInterrupt:
        terminate_processes(running)
        for interrupted in running.values():
            append_event(events_path, {"event": "interrupted", "at": utc_now(), "gpu": interrupted.gpu, **interrupted.task.as_record()})
        running.clear()
        write_active_progress()
        print("Sweep interrupted. Resume with the same command.", file=sys.stderr, flush=True)
        return 130

    if not args.keep_progress_log:
        cleanup_progress([state_path, events_path], temp_dir)
        print("All runs completed; progress logs cleaned up.", flush=True)
    else:
        progress["finished_at"] = utc_now()
        write_progress(state_path, progress)
        print("All runs completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
