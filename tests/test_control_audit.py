from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from CESTA.evaluation.control_audit import audit_locked_control_runs
from CESTA.evaluation.control_budgets import write_yaml_payload


class LockedControlAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.lock_path = self.root / "lock.yaml"
        self.runs_csv = self.root / "runs.csv"
        self.config = {"communication_mode": "entropy", "control_entropy_threshold": 0.5}
        lock = {
            "version": 1,
            "selection_split": "val",
            "budget_content_sha256": "budget-hash",
            "expected_seeds": [12, 42],
            "matching": {"minimum_target_ratio": 0.98, "maximum_target_ratio": 1.0},
            "locks": [
                {
                    "budget_id": "learned__dataset",
                    "source_variant": "learned",
                    "dataset": "dataset",
                    "controller": "entropy",
                    "selection_status": "matched",
                    "selected_variant": "entropy-0p5",
                    "selected_communication_config": self.config,
                    "target_total_energy_j": 10.0,
                    "validation_energy_ratio": 0.99,
                }
            ],
        }
        lock["content_sha256"] = self._hash(lock)
        write_yaml_payload(lock, self.lock_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_complete_audit_reports_test_budget_drift_without_reselection(self) -> None:
        self._write_rows([self._row(12, 9.9), self._row(42, 10.2)])

        audit = audit_locked_control_runs(self.lock_path, self.runs_csv)

        self.assertTrue(audit["complete"])
        statuses = [record["test_budget_status"] for record in audit["records"]]
        self.assertEqual(statuses, ["matched", "over_budget_drift"])

    def test_rejects_non_test_input(self) -> None:
        row = self._row(12, 9.9)
        row["split"] = "val"
        self._write_rows([row])

        with self.assertRaisesRegex(ValueError, "test records only"):
            audit_locked_control_runs(self.lock_path, self.runs_csv)

    def test_reports_missing_duplicate_and_config_mismatch(self) -> None:
        mismatch = self._row(12, 9.9)
        mismatch["communication_config_json"] = json.dumps({"communication_mode": "entropy", "control_entropy_threshold": 0.4})
        self._write_rows([mismatch])

        audit = audit_locked_control_runs(self.lock_path, self.runs_csv)

        self.assertFalse(audit["complete"])
        self.assertIn("communication config differs", audit["invalid"]["entropy-0p5/dataset/seed12"][0])
        self.assertEqual(audit["missing"], ["entropy-0p5/dataset/seed42"])

    def _row(self, seed: int, energy: float) -> dict[str, Any]:
        return {
            "variant": "entropy-0p5",
            "dataset": "dataset",
            "seed": seed,
            "split": "test",
            "artifact_path": f"runs/entropy-0p5/seed{seed}",
            "git_commit": "abc123",
            "git_dirty": "False",
            "data_sha256": "data-hash",
            "meta_sha256": "meta-hash",
            "macro_f1": 0.8,
            "request_ratio": 0.5,
            "total_energy_j": energy,
            "communication_config_json": json.dumps(self.config, sort_keys=True, separators=(",", ":")),
        }

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        with self.runs_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _hash(payload: dict[str, Any]) -> str:
        import hashlib

        without_hash = dict(payload)
        without_hash.pop("content_sha256", None)
        canonical = json.dumps(without_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


if __name__ == "__main__":
    unittest.main()
