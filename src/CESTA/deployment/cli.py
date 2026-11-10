from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from CESTA.artifacts import load_checkpoint_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export parity-validated CESTA, HiFiNet, or STGCN firmware artifacts.")
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint directory containing config.json, weight.pt and manifest.json.")
    parser.add_argument("--architecture", choices=("auto", "cesta", "hifinet", "stgcn"), default="auto")
    parser.add_argument("--output", type=Path, required=True, help="Empty output bundle directory.")
    parser.add_argument("--validate-only", action="store_true", help="HiFiNet/STGCN: validate PyTorch stages without TFLite conversion.")
    parser.add_argument("--target", choices=("node", "local", "graph"), help="CESTA only; currently only node is supported.")
    parser.add_argument("--receiver-index", type=int, help="CESTA: receiver node index.")
    parser.add_argument("--validation-artifact", action="store_true", help="CESTA: label untrained conversion-validation artifacts.")
    args = parser.parse_args(argv)
    detected = load_checkpoint_metadata(args.model).get("model_name")
    if detected not in {"cesta", "hifinet", "stgcn"}:
        parser.error(f"unsupported checkpoint architecture: {detected!r}")
    if args.architecture != "auto" and args.architecture != detected:
        parser.error(f"--architecture {args.architecture} disagrees with checkpoint architecture {detected}")
    if detected == "cesta":
        if args.validate_only:
            parser.error("CESTA requires converted parity validation; --validate-only is supported only for HiFiNet/STGCN")
        if args.receiver_index is None:
            parser.error("CESTA node export requires --receiver-index")
        args.target = args.target or "node"
    else:
        if args.target is not None or args.receiver_index is not None or args.validation_artifact:
            parser.error("--target, --receiver-index, --validation-artifact are CESTA-only options")
        if args.output is None:
            parser.error("staged export requires --output")
    return import_module(f"CESTA.deployment.{detected}").export(args)
