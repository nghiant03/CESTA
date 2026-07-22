from __future__ import annotations

import argparse
from pathlib import Path

from CESTA.evaluation.benchmark import audit_benchmark, load_benchmark_spec, write_benchmark_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit decisive-comparison run artifacts and export normalized metrics.")
    parser.add_argument("--spec", type=Path, required=True, help="Benchmark specification YAML or JSON file.")
    parser.add_argument("--runs-root", type=Path, required=True, help="Root containing model run directories.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for audit.json and runs.csv.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Return success when expected runs are missing or invalid.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_benchmark_spec(args.spec)
    audit = audit_benchmark(spec, args.runs_root)
    write_benchmark_audit(audit, args.output)
    counts = audit.payload()["counts"]
    print(
        "Benchmark audit: "
        f"valid={counts['valid']}/{counts['expected']} "
        f"missing={counts['missing']} duplicate={counts['duplicate']} invalid={counts['invalid']} "
        f"unmatched={counts['unmatched_runs']}"
    )
    print(f"Audit report: {args.output / 'audit.json'}")
    return 0 if audit.complete or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
