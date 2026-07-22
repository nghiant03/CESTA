from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from CESTA.evaluation.benchmark import audit_benchmark, load_benchmark_spec, write_benchmark_audit


class BenchmarkAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir()
        self.spec_path = self.root / "benchmark.yaml"
        self.config_path = self.root / "config.yaml"
        self.dataset_path = self.root / "Intel_fault15"
        self.dataset_path.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_selective_artifact_exports_required_metrics(self) -> None:
        self._write_config("gumbel_request")
        self._write_spec("gumbel_request")
        self._write_run("run-a", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0))

        audit = self._audit()

        self.assertTrue(audit.complete)
        self.assertEqual(audit.valid, ["variant/Intel_fault15/seed12"])
        self.assertAlmostEqual(audit.runs[0].total_energy_j, 0.6)
        output = self.root / "output"
        write_benchmark_audit(audit, output)
        with (output / "runs.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["spike_f1"], "0.8")
        self.assertEqual(rows[0]["dense_total_energy_j"], "0.8")

    def test_valid_dense_artifact_requires_zero_reduction(self) -> None:
        self._write_config("dense")
        self._write_spec("dense")
        self._write_run("run-a", mode="dense", requested=(2.0, 2.0), possible=(2.0, 2.0))

        audit = self._audit()

        self.assertTrue(audit.complete)
        self.assertEqual(audit.runs[0].request_ratio, 1.0)
        self.assertEqual(audit.runs[0].energy_reduction_ratio, 0.0)

    def test_missing_duplicate_and_invalid_cells_are_reported(self) -> None:
        self._write_config("gumbel_request")
        self._write_spec("gumbel_request", seeds=(12, 42, 1242))
        self._write_run("run-a", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=12)
        self._write_run("run-b", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=12)
        invalid_path = self._write_run("run-c", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=42)
        (invalid_path / "communication_metrics.json").unlink()

        audit = self._audit()

        self.assertFalse(audit.complete)
        self.assertEqual(audit.missing, ["variant/Intel_fault15/seed1242"])
        self.assertEqual(len(audit.duplicate["variant/Intel_fault15/seed12"]), 2)
        self.assertIn("missing artifact", audit.invalid["variant/Intel_fault15/seed42"][0])

    def test_nonfinite_and_requested_above_possible_fail(self) -> None:
        self._write_config("gumbel_request")
        self._write_spec("gumbel_request", seeds=(12, 42))
        nan_path = self._write_run("run-a", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=12)
        metrics = self._read_json(nan_path / "eval_metrics.json")
        metrics["macro_f1"] = math.nan
        self._write_json(nan_path / "eval_metrics.json", metrics)
        self._write_run("run-b", mode="gumbel_request", requested=(3.0, 1.0), possible=(2.0, 2.0), seed=42)

        audit = self._audit()

        self.assertIn("must be finite", audit.invalid["variant/Intel_fault15/seed12"][0])
        self.assertIn("exceeds possible", audit.invalid["variant/Intel_fault15/seed42"][0])

    def test_inconsistent_bits_and_energy_totals_fail(self) -> None:
        self._write_config("gumbel_request")
        self._write_spec("gumbel_request", seeds=(12, 42))
        bits_path = self._write_run("run-a", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=12)
        communication = self._read_json(bits_path / "communication_metrics.json")
        communication["splits"]["test"]["transmitted_bits_estimate"] = 99.0
        self._write_json(bits_path / "communication_metrics.json", communication)
        energy_path = self._write_run("run-b", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=42)
        communication = self._read_json(energy_path / "communication_metrics.json")
        communication["splits"]["test"]["energy"]["selective"]["total_energy_j"] = 8.0
        self._write_json(energy_path / "communication_metrics.json", communication)

        audit = self._audit()

        self.assertIn("transmitted bits", audit.invalid["variant/Intel_fault15/seed12"][0])
        self.assertIn("TX+RX total", audit.invalid["variant/Intel_fault15/seed42"][0])

    def test_missing_per_class_metric_and_provenance_fail(self) -> None:
        self._write_config("gumbel_request")
        self._write_spec("gumbel_request", seeds=(12, 42))
        class_path = self._write_run("run-a", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=12)
        metrics = self._read_json(class_path / "eval_metrics.json")
        del metrics["per_class"]["SPIKE"]
        self._write_json(class_path / "eval_metrics.json", metrics)
        provenance_path = self._write_run("run-b", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=42)
        manifest = self._read_json(provenance_path / "manifest.json")
        manifest["git"]["commit"] = None
        self._write_json(provenance_path / "manifest.json", manifest)

        audit = self._audit()

        self.assertIn("per_class.SPIKE", audit.invalid["variant/Intel_fault15/seed12"][0])
        self.assertIn("commit", audit.invalid["variant/Intel_fault15/seed42"][0])

    def test_comparison_metadata_mismatch_fails_later_run(self) -> None:
        self._write_config("gumbel_request")
        self._write_spec("gumbel_request", seeds=(12, 42))
        self._write_run("run-a", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=12)
        mismatch_path = self._write_run("run-b", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=42)
        communication = self._read_json(mismatch_path / "communication_metrics.json")
        communication["splits"]["test"]["energy"]["constants"]["electronics_j_per_bit"] = 99.0
        self._write_json(mismatch_path / "communication_metrics.json", communication)

        audit = self._audit()

        self.assertEqual(audit.valid, ["variant/Intel_fault15/seed12"])
        self.assertIn("metadata differs", audit.invalid["variant/Intel_fault15/seed42"][0])

    def test_output_order_uses_spec_not_metric_values(self) -> None:
        self._write_config("gumbel_request")
        self._write_spec("gumbel_request", seeds=(42, 12))
        run_12 = self._write_run("run-z", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=12)
        run_42 = self._write_run("run-a", mode="gumbel_request", requested=(2.0, 1.0), possible=(2.0, 2.0), seed=42)
        metrics = self._read_json(run_12 / "eval_metrics.json")
        metrics["macro_f1"] = 0.99
        self._write_json(run_12 / "eval_metrics.json", metrics)
        metrics = self._read_json(run_42 / "eval_metrics.json")
        metrics["macro_f1"] = 0.01
        self._write_json(run_42 / "eval_metrics.json", metrics)

        audit = self._audit()

        self.assertEqual([run.seed for run in audit.runs], [42, 12])

    def _audit(self):
        spec = load_benchmark_spec(self.spec_path, project_root=self.root)
        return audit_benchmark(spec, self.runs_root)

    def _write_config(self, mode: str) -> None:
        config = {
            "train": {"model": "cesta", "epochs": 1, "seed": 42},
            "model_kwargs": {"communication_mode": mode, "precision_bits": 8},
        }
        self.config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    def _write_spec(self, mode: str, seeds: tuple[int, ...] = (12,)) -> None:
        spec = {
            "datasets": [{"name": "Intel_fault15", "path": "Intel_fault15"}],
            "seeds": list(seeds),
            "variants": [{"name": "variant", "config": "config.yaml", "communication_mode": mode}],
        }
        self.spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))

    def _write_run(
        self,
        run_id: str,
        *,
        mode: str,
        requested: tuple[float, float],
        possible: tuple[float, float],
        seed: int = 12,
    ) -> Path:
        path = self.runs_root / "cesta" / run_id
        path.mkdir(parents=True)
        train_config = self._resolved_config(seed)
        requested_total = sum(requested)
        possible_total = sum(possible)
        bits_per_message = 8.0
        selected = self._energy_totals(requested_total, bits_per_message)
        dense = self._energy_totals(possible_total, bits_per_message)
        reduction = dense["total_energy_j"] - selected["total_energy_j"]
        reduction_ratio = reduction / dense["total_energy_j"] if dense["total_energy_j"] else 0.0
        manifest = {
            "run_id": run_id,
            "seed": seed,
            "model": "cesta",
            "git": {"commit": "abc123", "dirty": False},
            "dataset": {
                "path": str(self.dataset_path),
                "data_sha256": "data-hash",
                "meta_sha256": "meta-hash",
            },
            "train_config": train_config,
        }
        metrics = {
            "macro_f1": 0.75,
            "accuracy": 0.9,
            "per_class": {name: {"f1": value} for name, value in zip(("NORMAL", "SPIKE", "DRIFT", "STUCK"), (0.9, 0.8, 0.7, 0.6), strict=True)},
        }
        graph = {
            "directed_edge_count": 2,
            "dynamic_link_seed": 42,
            "burst_params": {"rho": 0.5},
            "edge_convention": "sender_to_receiver",
            "link_mask_shape": [10, 2],
            "distance_metadata": {"source": "positions", "units": "m"},
        }
        communication = {
            "config": {"communication_mode": mode},
            "splits": {
                "test": {
                    "active_request_ratio": requested_total / possible_total if possible_total else 0.0,
                    "requested_edge_count": requested_total,
                    "possible_edge_count": possible_total,
                    "transmitted_bits_estimate": requested_total * bits_per_message,
                    "bits_per_message": bits_per_message,
                    "requested_edge_counts": list(requested),
                    "possible_edge_counts": list(possible),
                    "energy": {
                        "constants": {"radio_model": "test", "electronics_j_per_bit": 0.1},
                        "units": {"energy": "J", "distance": "m", "payload": "bit"},
                        "distance": {"source": "positions", "units": "m", "edge_count": 2},
                        "selective": selected,
                        "dense_reference": dense,
                        "dense_vs_selective": {
                            "selective_subset_of_dense": all(left <= right for left, right in zip(requested, possible, strict=True)),
                            "total_energy_reduction_j": reduction,
                            "total_energy_reduction_ratio": reduction_ratio,
                        },
                    },
                }
            },
            "graph": graph,
        }
        self._write_json(path / "manifest.json", manifest)
        self._write_json(path / "eval_metrics.json", metrics)
        self._write_json(path / "communication_metrics.json", communication)
        return path

    def _resolved_config(self, seed: int) -> dict[str, object]:
        raw = yaml.safe_load(self.config_path.read_text())
        raw["train"]["seed"] = seed
        from CESTA.schema import TrainConfig

        return TrainConfig.model_validate(raw).model_dump(mode="json")

    @staticmethod
    def _energy_totals(messages: float, bits_per_message: float) -> dict[str, float]:
        bit_count = messages * bits_per_message
        tx_energy = messages * 0.1
        rx_energy = messages * 0.1
        return {
            "message_count": messages,
            "bit_count": bit_count,
            "bits_per_message": bits_per_message,
            "tx_energy_j": tx_energy,
            "rx_energy_j": rx_energy,
            "total_energy_j": tx_energy + rx_energy,
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    unittest.main()
