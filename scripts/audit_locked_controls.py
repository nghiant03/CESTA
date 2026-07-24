from __future__ import annotations

import argparse
from pathlib import Path

from CESTA.evaluation.control_audit import audit_locked_control_runs, write_locked_control_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit test runs against validation-locked communication controls.")
    parser.add_argument("--lock", type=Path, required=True, help="Hashed validation policy-lock YAML.")
    parser.add_argument("--test-runs-csv", type=Path, required=True, help="Audited test records for locked controls.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for audit records.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Return success for missing or invalid cells.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_locked_control_runs(args.lock, args.test_runs_csv)
    write_locked_control_audit(audit, args.output)
    print(
        f"Locked-control audit: valid={len(audit['valid'])}/{len(audit['expected'])} "
        f"missing={len(audit['missing'])} duplicate={len(audit['duplicate'])} invalid={len(audit['invalid'])}"
    )
    return 0 if audit["complete"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
