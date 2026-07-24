from __future__ import annotations

import argparse
from pathlib import Path

from CESTA.evaluation.control_budgets import load_yaml_payload, select_control_policies, write_yaml_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock validation-selected communication controls without reading test results.")
    parser.add_argument("--budgets", type=Path, required=True, help="Hashed control-budget YAML.")
    parser.add_argument("--validation-runs-csv", type=Path, required=True, help="Validation-only control audit runs.csv.")
    parser.add_argument(
        "--controller",
        nargs="+",
        action="append",
        required=True,
        metavar=("CONTROLLER", "VARIANT"),
        help="Controller followed by one or more candidate variant names. Repeat per controller.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output policy-lock YAML.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controller_variants: dict[str, list[str]] = {}
    for values in args.controller:
        controller, *variants = values
        if not variants:
            raise ValueError(f"Controller {controller} requires at least one variant")
        if controller in controller_variants:
            raise ValueError(f"Duplicate controller: {controller}")
        controller_variants[controller] = variants
    payload = select_control_policies(
        load_yaml_payload(args.budgets),
        args.validation_runs_csv,
        controller_variants,
    )
    write_yaml_payload(payload, args.output)
    print(f"Control lock: {len(payload['locks'])} policies written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
