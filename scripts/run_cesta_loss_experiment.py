from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

from CESTA.schema.config import TrainConfig, load_config_file
from CESTA.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["CESTA-DHT_1min_fault15_stuck0_drift125", "Intel_fault05", "Intel_fault10", "Intel_fault15", "Intel_fault20"]
VARIANTS = {
    "control": {},
    "crf001": {"crf_loss_weight": 0.01},
    "gamma1-crf001": {"focal_gamma": 1.0, "crf_loss_weight": 0.01},
    "gamma0-crf001": {"focal_gamma": 0.0, "crf_loss_weight": 0.01},
    "local01": {"local_ce_loss_weight": 0.1},
    "local03": {"local_ce_loss_weight": 0.3},
    "gamma1-crf001-local01": {"focal_gamma": 1.0, "crf_loss_weight": 0.01, "local_ce_loss_weight": 0.1},
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_cell(output: Path, variant: str, config: TrainConfig, dataset: str, seed: int, *, early_stopping: bool = True) -> dict[str, Any]:
    cell = output / dataset / variant / f"seed{seed}"
    cell.mkdir(parents=True, exist_ok=False)
    config = TrainConfig.model_validate({**config.model_dump(mode="json"), "seed": seed})
    config_path = cell / "training.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    data = ROOT / "data/datasets" / dataset
    print(f"Training {dataset} / {variant} / seed {seed}", flush=True)
    with (cell / "training.log").open("w") as log:
        subprocess.run(
            ["uv", "run", "cesta", "train", str(config_path), str(data), "--output", str(cell / "runs"),
             *(["--early-stopping"] if early_stopping else []), "--no-test-evaluation"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    manifests = list((cell / "runs").glob("*/manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one run in {cell}")
    run = manifests[0].parent
    if (run / "eval_metrics.json").exists():
        raise RuntimeError("Unexpected test evaluation")
    with (cell / "validation.log").open("w") as log:
        subprocess.run(
            ["uv", "run", "cesta", "evaluate", "--model", str(run), "--data", str(data),
             "--split", "val", "--batch-size", str(config.batch_size)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    metrics = json.loads((run / "validation/eval_metrics.json").read_text())
    communication = json.loads((run / "validation/communication_metrics.json").read_text())["splits"]["val"]
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
    best = max(history, key=lambda row: row["val_macro_f1"])
    row = {
        "dataset": dataset, "variant": variant, "seed": seed, "run": str(run), "best_epoch": best["epoch"],
        "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "per_class": metrics["per_class"],
        "request_ratio": communication["active_request_ratio"], "bits": communication["transmitted_bits_estimate"],
        "energy_j": communication["energy"]["selective"]["total_energy_j"],
    }
    if not all(math.isfinite(row[key]) for key in ("macro_f1", "accuracy", "request_ratio", "bits", "energy_j")):
        raise RuntimeError(f"Nonfinite validation metrics in {run}")
    write_json(cell / "result.json", row)
    return row


def assess(rows: list[dict[str, Any]], variant: str, seeds: list[int]) -> dict[str, Any]:
    deltas = {}
    for dataset in DATASETS:
        paired = []
        for seed in seeds:
            control = next(row for row in rows if (row["dataset"], row["variant"], row["seed"]) == (dataset, "control", seed))
            candidate = next(row for row in rows if (row["dataset"], row["variant"], row["seed"]) == (dataset, variant, seed))
            paired.append({key: candidate[key] - control[key] for key in ("macro_f1", "request_ratio", "bits", "energy_j")})
        deltas[dataset] = {key: sum(pair[key] for pair in paired) / len(paired) for key in paired[0]}
    eligible = deltas[DATASETS[0]]["macro_f1"] > 0 and all(
        delta["macro_f1"] >= 0 and delta["request_ratio"] <= 0 and delta["bits"] <= 0 and delta["energy_j"] <= 0
        for delta in deltas.values()
    )
    return {"variant": variant, "eligible": eligible, "deltas": deltas}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("Epochs must be positive")
    base = TrainConfig.model_validate(load_config_file(ROOT / "config/experiments/ablations/cesta-window-standardization.yaml"))
    base = TrainConfig.model_validate({**base.model_dump(mode="json"), "epochs": args.epochs, "local_ce_loss_weight": 0.0})
    configs = {name: TrainConfig.model_validate({**base.model_dump(mode="json"), **changes}) for name, changes in VARIANTS.items()}
    provenance = {
        "datasets": {name: {filename: sha256_file(ROOT / "data/datasets" / name / filename)
                            for filename in ("dataset.csv", "dataset_meta.json", "graph_edges.npz", "dynamic_link_mask.npz",
                                             "node_positions.json", "edge_distances.npz")} for name in DATASETS},
        "configs": {name: config.model_dump(mode="json") for name, config in configs.items()},
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path)
                          for path in [Path(__file__).resolve(), *sorted((ROOT / "src/CESTA").rglob("*.py"))]},
        "selection": "Seed 12 validation only; DHT macro-F1 strictly improves; no F1 regression on any Intel dataset; "
                     "no increase in request ratio, bits or estimated energy on any dataset. Rank eligible by DHT gain, then name.",
        "confirmation": "Lock before seeds 42 and 1242; compare candidate and control on all datasets; "
                        "apply the same constraints to three-seed means. No fallback candidate after confirmation.",
        "test_evaluation": False,
        "limitations": "Exploratory DHT recipe; validation reuse; no statistical noninferiority claim; energy is modeled, not measured.",
    }
    if args.dry_run:
        print(json.dumps(provenance, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment_spec.json", provenance)
    rows = []
    for dataset in DATASETS:
        for name, config in configs.items():
            rows.append(run_cell(output / "screening", name, config, dataset, 12))
            write_json(output / "results.json", rows)
    assessments = [assess(rows, name, [12]) for name in configs if name != "control"]
    eligible = [item for item in assessments if item["eligible"]]
    eligible.sort(key=lambda item: (-item["deltas"][DATASETS[0]]["macro_f1"], item["variant"]))
    selected = eligible[0]["variant"] if eligible else None
    selection = {"selected": selected, "assessments": assessments, "status": "locked" if selected else "no_eligible_candidate"}
    if selected is not None:
        locked = output / "locked.yaml"
        locked.write_text(yaml.safe_dump(configs[selected].model_dump(mode="json"), sort_keys=False))
        selection["locked_config_sha256"] = sha256_file(locked)
    write_json(output / "selection.json", selection)
    if selected is None:
        print("No candidate met all validation constraints; no confirmation launched.", flush=True)
        return
    for dataset in DATASETS:
        for name in ("control", selected):
            for seed in (42, 1242):
                rows.append(run_cell(output / "confirmation", name, configs[name], dataset, seed))
                write_json(output / "results.json", rows)
    write_json(output / "confirmation.json", assess(rows, selected, [12, 42, 1242]))
    print("Validation-only experiment completed; test data was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
