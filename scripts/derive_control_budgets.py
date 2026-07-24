from __future__ import annotations

import argparse
from pathlib import Path

from CESTA.evaluation.control_budgets import derive_control_budgets, write_yaml_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive validation-only communication budgets from audited CESTA runs.")
    parser.add_argument("--runs-csv", type=Path, required=True, help="Validation audit runs.csv.")
    parser.add_argument("--source-variant", action="append", required=True, help="Locked CESTA source variant. Repeat as needed.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[12, 42, 1242], help="Expected source seeds.")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Matched-budget relative tolerance below the energy target.")
    parser.add_argument("--output", type=Path, required=True, help="Output budget YAML.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = derive_control_budgets(
        args.runs_csv,
        source_variants=args.source_variant,
        expected_seeds=args.seeds,
        tolerance_ratio=args.tolerance,
    )
    write_yaml_payload(payload, args.output)
    print(f"Control budgets: {len(payload['budgets'])} targets written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
