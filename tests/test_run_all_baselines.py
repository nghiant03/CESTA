from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


def _load_sweep_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "run_all_baselines.py"
    spec = importlib.util.spec_from_file_location("run_all_baselines", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = _load_sweep_module()


class BaselineCompletionDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "config.yaml"
        self.dataset_path = self.root / "dataset"
        self.runs_dir = self.root / "runs"
        self.dataset_path.mkdir()
        self.runs_dir.mkdir()
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "train": {"model": "cesta", "epochs": 1, "seed": 42},
                    "model_kwargs": {"communication_mode": "dense"},
                },
                sort_keys=False,
            )
        )
        (self.dataset_path / "dataset.csv").write_text("value\n1\n")
        (self.dataset_path / "dataset_meta.json").write_text("{}\n")
        self.task = SWEEP.build_tasks([self.config_path], [self.dataset_path], [12])[0]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_discovers_complete_matching_manifest_without_progress_state(self) -> None:
        run_dir = self._write_run("run-a")

        completed = SWEEP.discover_completed_tasks([self.task], self.runs_dir)

        self.assertEqual(list(completed), [self.task.key])
        self.assertEqual(completed[self.task.key]["run_path"], str(run_dir))
        self.assertEqual(completed[self.task.key]["source"], "manifest")

    def test_ignores_incomplete_run(self) -> None:
        run_dir = self._write_run("run-a")
        (run_dir / "predictions.npz").unlink()

        completed = SWEEP.discover_completed_tasks([self.task], self.runs_dir)

        self.assertEqual(completed, {})

    def test_ignores_different_resolved_config(self) -> None:
        self._write_run("run-a", train_updates={"learning_rate": 0.5})

        completed = SWEEP.discover_completed_tasks([self.task], self.runs_dir)

        self.assertEqual(completed, {})

    def test_ignores_different_dataset_hash(self) -> None:
        self._write_run("run-a", data_sha256="different")

        completed = SWEEP.discover_completed_tasks([self.task], self.runs_dir)

        self.assertEqual(completed, {})

    def test_duplicate_matching_runs_count_once(self) -> None:
        self._write_run("run-a")
        self._write_run("run-b")

        completed = SWEEP.discover_completed_tasks([self.task], self.runs_dir)

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[self.task.key]["run_id"], "run-a")

    def _write_run(
        self,
        run_id: str,
        *,
        train_updates: dict[str, Any] | None = None,
        data_sha256: str | None = None,
    ) -> Path:
        from CESTA.schema import TrainConfig
        from CESTA.schema.config import load_config_file
        from CESTA.utils import sha256_file

        raw = load_config_file(self.config_path)
        raw["train"]["seed"] = 12
        train_config = TrainConfig.model_validate(raw).model_dump(mode="json")
        train_config.update(train_updates or {})
        run_dir = self.runs_dir / "cesta" / run_id
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": run_id,
            "model": "cesta",
            "seed": 12,
            "train_config": train_config,
            "dataset": {
                "data_sha256": data_sha256 or sha256_file(self.dataset_path / "dataset.csv"),
                "meta_sha256": sha256_file(self.dataset_path / "dataset_meta.json"),
            },
            "timing": {"ended_at": "2026-07-23T00:00:00Z"},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        for name in ("config.json", "weight.pt", "eval_metrics.json", "predictions.npz"):
            (run_dir / name).write_text("")
        return run_dir


if __name__ == "__main__":
    unittest.main()
