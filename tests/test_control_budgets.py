from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any

from CESTA.evaluation.control_budgets import (
    derive_control_budgets,
    load_yaml_payload,
    select_control_policies,
    write_yaml_payload,
)


class ControlBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source_csv = self.root / "source.csv"
        self.control_csv = self.root / "controls.csv"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_derives_dataset_budget_from_complete_validation_cells(self) -> None:
        self._write_rows(
            self.source_csv,
            [self._row("learned", "dataset", seed, energy, 0.8) for seed, energy in ((12, 9.0), (42, 10.0), (1242, 11.0))],
        )

        payload = derive_control_budgets(
            self.source_csv,
            source_variants=["learned"],
            expected_seeds=[12, 42, 1242],
        )

        budget = payload["budgets"][0]
        self.assertEqual(payload["selection_split"], "val")
        self.assertEqual(budget["target_total_energy_j"], 10.0)
        self.assertEqual(budget["minimum_matched_total_energy_j"], 9.8)
        self.assertEqual([record["seed"] for record in budget["source_records"]], [12, 42, 1242])

    def test_rejects_test_records_and_incomplete_seed_coverage(self) -> None:
        rows = [self._row("learned", "dataset", 12, 10.0, 0.8)]
        rows[0]["split"] = "test"
        self._write_rows(self.source_csv, rows)

        with self.assertRaisesRegex(ValueError, "validation records only"):
            derive_control_budgets(self.source_csv, source_variants=["learned"], expected_seeds=[12])

        rows[0]["split"] = "val"
        self._write_rows(self.source_csv, rows)
        with self.assertRaisesRegex(ValueError, "coverage"):
            derive_control_budgets(self.source_csv, source_variants=["learned"], expected_seeds=[12, 42])

    def test_selects_highest_validation_f1_inside_energy_band(self) -> None:
        self._write_rows(
            self.source_csv,
            [self._row("learned", "dataset", seed, 10.0, 0.8) for seed in (12, 42, 1242)],
        )
        budgets = derive_control_budgets(
            self.source_csv,
            source_variants=["learned"],
            expected_seeds=[12, 42, 1242],
        )
        rows = []
        for seed in (12, 42, 1242):
            rows.extend(
                [
                    self._row("entropy-low", "dataset", seed, 9.9, 0.81, mode="entropy", parameter=0.4),
                    self._row("entropy-high", "dataset", seed, 9.85, 0.84, mode="entropy", parameter=0.5),
                    self._row("entropy-over", "dataset", seed, 10.1, 0.99, mode="entropy", parameter=0.6),
                ]
            )
        self._write_rows(self.control_csv, rows)

        lock = select_control_policies(
            budgets,
            self.control_csv,
            {"entropy": ["entropy-low", "entropy-high", "entropy-over"]},
        )

        selected = lock["locks"][0]
        self.assertEqual(selected["selected_variant"], "entropy-high")
        self.assertEqual(selected["selection_status"], "matched")
        self.assertEqual(selected["selected_communication_config"]["control_entropy_threshold"], 0.5)

    def test_falls_back_to_nearest_candidate_below_target(self) -> None:
        self._write_rows(
            self.source_csv,
            [self._row("learned", "dataset", seed, 10.0, 0.8) for seed in (12, 42, 1242)],
        )
        budgets = derive_control_budgets(
            self.source_csv,
            source_variants=["learned"],
            expected_seeds=[12, 42, 1242],
        )
        rows = []
        for seed in (12, 42, 1242):
            rows.extend(
                [
                    self._row("static-one", "dataset", seed, 7.0, 0.85, mode="static_topk", parameter=1),
                    self._row("static-two", "dataset", seed, 9.5, 0.80, mode="static_topk", parameter=2),
                ]
            )
        self._write_rows(self.control_csv, rows)

        lock = select_control_policies(
            budgets,
            self.control_csv,
            {"static_topk": ["static-one", "static-two"]},
        )

        selected = lock["locks"][0]
        self.assertEqual(selected["selected_variant"], "static-two")
        self.assertEqual(selected["selection_status"], "under_budget_unmatched")

    def test_hashed_yaml_detects_manual_changes(self) -> None:
        self._write_rows(
            self.source_csv,
            [self._row("learned", "dataset", seed, 10.0, 0.8) for seed in (12, 42, 1242)],
        )
        payload = derive_control_budgets(
            self.source_csv,
            source_variants=["learned"],
            expected_seeds=[12, 42, 1242],
        )
        path = self.root / "budgets.yaml"
        write_yaml_payload(payload, path)

        self.assertEqual(load_yaml_payload(path), payload)
        path.write_text(path.read_text().replace("target_total_energy_j: 10.0", "target_total_energy_j: 9.0"))
        with self.assertRaisesRegex(ValueError, "Content hash"):
            load_yaml_payload(path)

    @staticmethod
    def _row(
        variant: str,
        dataset: str,
        seed: int,
        energy: float,
        macro_f1: float,
        *,
        mode: str = "gumbel_request",
        parameter: float | int = 0.0,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {"communication_mode": mode}
        if mode == "entropy":
            config["control_entropy_threshold"] = parameter
        if mode == "static_topk":
            config["control_static_topk"] = parameter
        return {
            "variant": variant,
            "dataset": dataset,
            "seed": seed,
            "split": "val",
            "artifact_path": f"runs/{variant}/{dataset}/seed{seed}",
            "git_commit": "abc123",
            "data_sha256": "data-hash",
            "meta_sha256": "meta-hash",
            "macro_f1": macro_f1,
            "request_ratio": energy / 20.0,
            "requested_messages": energy * 10.0,
            "possible_messages": 200.0,
            "transmitted_bits": energy * 80.0,
            "tx_energy_j": energy * 0.6,
            "rx_energy_j": energy * 0.4,
            "total_energy_j": energy,
            "dense_total_energy_j": 20.0,
            "num_parameters": 100,
            "total_parameters": 120,
            "communication_config_json": __import__("json").dumps(config, sort_keys=True, separators=(",", ":")),
            "comparison_signature_json": '{"signature":"same"}',
        }

    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
