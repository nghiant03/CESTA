from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import yaml

from CESTA.evaluation.benchmark import load_benchmark_spec


def _load_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "generate_control_tuning.py"
    spec = importlib.util.spec_from_file_location("generate_control_tuning", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_module()


class GenerateControlTuningTest(unittest.TestCase):
    def test_generates_predeclared_validation_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            template = root / "template.yaml"
            dataset = root / "dataset"
            dataset.mkdir()
            template.write_text(
                yaml.safe_dump(
                    {
                        "train": {"model": "cesta", "epochs": 1},
                        "model_kwargs": {"communication_mode": "entropy", "control_entropy_threshold": 0.5},
                    },
                    sort_keys=False,
                )
            )
            source = root / "grid.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "datasets": [{"name": "dataset", "path": str(dataset)}],
                        "seeds": [12, 42],
                        "controllers": {
                            "entropy": {
                                "template": str(template),
                                "parameter": "control_entropy_threshold",
                                "values": [0.2, 0.8],
                            }
                        },
                    },
                    sort_keys=False,
                )
            )
            output = root / "generated"

            manifest = GENERATOR.generate_control_tuning(source, output)
            benchmark = load_benchmark_spec(output / "validation-benchmark.yaml", project_root=root)

            self.assertEqual(manifest["variant_count"], 2)
            self.assertEqual(benchmark.split, "val")
            self.assertEqual([variant.name for variant in benchmark.variants], ["cesta_entropy_0p2", "cesta_entropy_0p8"])
            thresholds = [variant.expected_train_config["model_kwargs"]["control_entropy_threshold"] for variant in benchmark.variants]
            self.assertEqual(thresholds, [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
