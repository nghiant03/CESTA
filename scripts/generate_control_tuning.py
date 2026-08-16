from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from CESTA.schema import TrainConfig
from CESTA.schema.config import load_config_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predefined validation-only control tuning configs and benchmark spec.")
    parser.add_argument("--spec", type=Path, default=Path("config/benchmarks/control-tuning.yaml"), help="Control tuning grid YAML.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for generated configs and validation benchmark spec.")
    return parser.parse_args()


def generate_control_tuning(spec_path: str | Path, output: str | Path) -> dict[str, Any]:
    raw = load_config_file(spec_path)
    datasets = _list(raw, "datasets")
    seeds = _list(raw, "seeds")
    controllers = _mapping(raw.get("controllers"), "controllers")
    output_path = Path(output)
    configs_path = output_path / "configs"
    configs_path.mkdir(parents=True, exist_ok=True)
    variants = []
    for controller, controller_raw in sorted(controllers.items()):
        controller_spec = _mapping(controller_raw, f"controllers.{controller}")
        template_path = Path(_text(controller_spec.get("template"), f"controllers.{controller}.template"))
        parameter = _text(controller_spec.get("parameter"), f"controllers.{controller}.parameter")
        values = controller_spec.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"controllers.{controller}.values must be a non-empty list")
        template = load_config_file(template_path)
        for value in values:
            config = copy.deepcopy(template)
            model_kwargs = _mapping(config.get("model_kwargs"), f"{template_path}.model_kwargs")
            model_kwargs[parameter] = value
            TrainConfig.model_validate(config)
            value_slug = _value_slug(value)
            name = f"cesta_{controller}_{value_slug}"
            config_path = configs_path / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            variants.append(
                {
                    "name": name,
                    "config": str(config_path),
                    "communication_mode": controller,
                }
            )
    benchmark = {"split": "val", "datasets": datasets, "seeds": seeds, "variants": variants}
    benchmark_path = output_path / "validation-benchmark.yaml"
    benchmark_path.write_text(yaml.safe_dump(benchmark, sort_keys=False))
    manifest = {
        "source_spec": str(spec_path),
        "variant_count": len(variants),
        "benchmark_spec": str(benchmark_path),
        "configs": [variant["config"] for variant in variants],
    }
    (output_path / "generation.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = generate_control_tuning(args.spec, args.output)
    print(f"Control tuning: {manifest['variant_count']} candidates written to {args.output}")
    return 0


def _value_slug(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Control grid values must be numeric")
    return str(value).replace("-", "m").replace(".", "p")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
