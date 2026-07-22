from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from CESTA.evaluation.benchmark_summary import (
    ClassificationRecord,
    ValidationRecord,
    aggregate_classification,
    paired_summary,
    rank_validation_comparators,
    write_benchmark_summary,
)


class BenchmarkSummaryTest(unittest.TestCase):
    def test_validation_ranking_uses_only_validation_values_and_locked_tiebreakers(self) -> None:
        records = [
            ValidationRecord("stable", "dataset", 12, 0.8, 200),
            ValidationRecord("stable", "dataset", 42, 0.8, 200),
            ValidationRecord("variable", "dataset", 12, 0.9, 100),
            ValidationRecord("variable", "dataset", 42, 0.7, 100),
        ]

        ranking = rank_validation_comparators(records, expected_cells=2)

        self.assertEqual([record.variant for record in ranking], ["stable", "variable"])
        self.assertEqual([record.rank for record in ranking], [1, 2])

    def test_validation_ranking_rejects_incomplete_coverage(self) -> None:
        records = [ValidationRecord("candidate", "dataset", 12, 0.8, 100)]

        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            rank_validation_comparators(records, expected_cells=2)

    def test_aggregation_is_deterministic_and_uses_sample_standard_deviation(self) -> None:
        records = [self._record("candidate", "dataset-b", 42, 0.8), self._record("candidate", "dataset-a", 12, 0.6)]

        first = aggregate_classification(records)
        second = aggregate_classification(reversed(records))

        self.assertEqual(first, second)
        self.assertEqual(
            [(summary.variant, summary.dataset) for summary in first],
            [("candidate", "ALL"), ("candidate", "dataset-a"), ("candidate", "dataset-b")],
        )
        self.assertAlmostEqual(first[0].macro_f1_mean, 0.7)
        self.assertAlmostEqual(first[0].macro_f1_std, 0.02**0.5)

    def test_paired_summary_matches_dataset_and_seed(self) -> None:
        records = [
            self._record("candidate", "dataset-a", 12, 0.8),
            self._record("reference", "dataset-a", 12, 0.7),
            self._record("candidate", "dataset-b", 42, 0.6),
            self._record("reference", "dataset-b", 42, 0.65),
        ]

        summary = paired_summary(records, variant="candidate", reference="reference", bootstrap_resamples=100)

        self.assertAlmostEqual(summary.mean_difference, 0.025)
        self.assertEqual([(difference.dataset, difference.seed) for difference in summary.differences], [("dataset-a", 12), ("dataset-b", 42)])

    def test_paired_summary_rejects_missing_pair(self) -> None:
        records = [self._record("candidate", "dataset-a", 12, 0.8), self._record("reference", "dataset-b", 12, 0.7)]

        with self.assertRaisesRegex(ValueError, "Paired cells differ"):
            paired_summary(records, variant="candidate", reference="reference", bootstrap_resamples=100)

    def test_bootstrap_interval_is_reproducible(self) -> None:
        records = []
        for seed, candidate, reference in ((12, 0.8, 0.7), (42, 0.75, 0.7), (1242, 0.7, 0.72)):
            records.extend([self._record("candidate", "dataset-a", seed, candidate), self._record("reference", "dataset-a", seed, reference)])

        first = paired_summary(records, variant="candidate", reference="reference", bootstrap_resamples=500, bootstrap_seed=9)
        second = paired_summary(records, variant="candidate", reference="reference", bootstrap_resamples=500, bootstrap_seed=9)

        self.assertEqual(first, second)
        self.assertLessEqual(first.confidence_low, first.mean_difference)
        self.assertGreaterEqual(first.confidence_high, first.mean_difference)

    def test_comparison_requires_explicit_variant_and_reference(self) -> None:
        records = [
            self._record("highest-test-score", "dataset-a", 12, 0.99),
            self._record("locked-reference", "dataset-a", 12, 0.7),
            self._record("candidate", "dataset-a", 12, 0.8),
        ]

        summary = paired_summary(records, variant="candidate", reference="locked-reference", bootstrap_resamples=100)

        self.assertEqual(summary.reference, "locked-reference")
        self.assertAlmostEqual(summary.mean_difference, 0.1)

    def test_writes_machine_readable_outputs(self) -> None:
        records = [self._record("candidate", "dataset-a", 12, 0.8), self._record("reference", "dataset-a", 12, 0.7)]
        aggregates = aggregate_classification(records)
        comparison = paired_summary(records, variant="candidate", reference="reference", bootstrap_resamples=100)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)

            write_benchmark_summary(aggregates, [comparison], output)

            self.assertTrue((output / "aggregates.csv").is_file())
            payload = json.loads((output / "paired_comparisons.json").read_text())
            self.assertEqual(payload[0]["reference"], "reference")
            self.assertEqual(payload[0]["differences"][0]["dataset"], "dataset-a")

    @staticmethod
    def _record(variant: str, dataset: str, seed: int, macro_f1: float) -> ClassificationRecord:
        return ClassificationRecord(
            variant=variant,
            dataset=dataset,
            seed=seed,
            macro_f1=macro_f1,
            accuracy=macro_f1,
            normal_f1=macro_f1,
            spike_f1=macro_f1,
            drift_f1=macro_f1,
            stuck_f1=macro_f1,
        )


if __name__ == "__main__":
    unittest.main()
