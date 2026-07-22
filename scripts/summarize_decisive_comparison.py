from __future__ import annotations

import argparse
from pathlib import Path

from CESTA.evaluation.benchmark_summary import aggregate_classification, load_classification_records, paired_summary, write_benchmark_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate audited benchmark records and compute predefined paired comparisons.")
    parser.add_argument("--runs-csv", type=Path, required=True, help="Normalized run-level CSV produced by a benchmark audit.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for aggregate and paired-comparison outputs.")
    parser.add_argument(
        "--comparison",
        action="append",
        nargs=2,
        metavar=("VARIANT", "REFERENCE"),
        default=[],
        help="Explicit paired comparison. Repeat for multiple locked comparisons.",
    )
    parser.add_argument("--metric", default="macro_f1", help="Classification metric used for paired differences.")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000, help="Number of paired bootstrap resamples.")
    parser.add_argument("--bootstrap-seed", type=int, default=20260723, help="Fixed bootstrap random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_classification_records(args.runs_csv)
    aggregates = aggregate_classification(records)
    comparisons = [
        paired_summary(
            records,
            variant=variant,
            reference=reference,
            metric=args.metric,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for variant, reference in args.comparison
    ]
    write_benchmark_summary(aggregates, comparisons, args.output)
    print(f"Benchmark summary: runs={len(records)} aggregates={len(aggregates)} paired_comparisons={len(comparisons)}")
    print(f"Summary directory: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
