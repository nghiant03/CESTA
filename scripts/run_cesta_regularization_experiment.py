from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from run_cesta_loss_experiment import ROOT, run_cell, write_json

from CESTA.metrics import compute_class_metrics, macro_f1
from CESTA.schema.config import TrainConfig, load_config_file
from CESTA.utils import sha256_file

DATASET = "CESTA-DHT_1min_fault15_stuck0_drift125"
VARIANTS = {
    "control": {},
    "affine-mild": {"augmentation_scale": 0.01, "augmentation_offset": 0.1},
    "affine-moderate": {"augmentation_scale": 0.02, "augmentation_offset": 0.2},
    "adamw-zero": {"optimizer": "adamw", "weight_decay": 0.0},
    "adamw-0001": {"optimizer": "adamw", "weight_decay": 0.001},
    "adamw-001": {"optimizer": "adamw", "weight_decay": 0.01},
}


def diagnose_decoder(run: Path) -> dict[str, Any]:
    metrics_path = run / "validation/eval_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if not metrics["train_config"]["model_kwargs"].get("use_crf"):
        raise ValueError("Decoder diagnosis requires CRF-enabled validation predictions")
    prediction_path = run / "validation/predictions.npz"
    with np.load(prediction_path) as arrays:
        targets = arrays["y_true"]
        decoded = arrays["y_pred"]
        logits = arrays["y_logits"]
    if not np.isfinite(logits).all() or targets.shape != decoded.shape or logits.shape[:-1] != targets.shape:
        raise ValueError("Invalid validation prediction arrays")
    raw = logits.argmax(-1)
    report: dict[str, Any] = {"run": str(run), "predictions_sha256": sha256_file(prediction_path)}
    for name, predictions in (("argmax", raw), ("crf", decoded)):
        classes = compute_class_metrics([torch.from_numpy(predictions)], [torch.from_numpy(targets)], logits.shape[-1])
        report[name] = {"macro_f1": macro_f1(classes), "per_class": asdict(classes)}
    report.update({
        "helped": int(((raw != targets) & (decoded == targets)).sum()),
        "harmed": int(((raw == targets) & (decoded != targets)).sum()),
        "disagreements": int((raw != decoded).sum()),
        "decision": "Retain existing CRF; diagnosis does not change decoding or tune on test data",
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-run", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    diagnosis = diagnose_decoder(args.diagnostic_run.resolve())
    base = TrainConfig.model_validate(load_config_file(ROOT / "config/experiments/ablations/cesta-window-standardization.yaml"))
    base = TrainConfig.model_validate({**base.model_dump(mode="json"), "epochs": args.epochs,
                                      "focal_gamma": 1.0, "crf_loss_weight": 0.01, "local_ce_loss_weight": 0.0})
    configs = {name: TrainConfig.model_validate({**base.model_dump(mode="json"), **changes}) for name, changes in VARIANTS.items()}
    spec = {
        "dataset": DATASET,
        "dataset_hashes": {name: sha256_file(ROOT / "data/datasets" / DATASET / name)
                           for name in ("dataset.csv", "dataset_meta.json", "graph_edges.npz", "dynamic_link_mask.npz",
                                        "node_positions.json", "edge_distances.npz")},
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in
                          [Path(__file__).resolve(), ROOT / "scripts/run_cesta_loss_experiment.py",
                           *sorted((ROOT / "src/CESTA").rglob("*.py"))]},
        "configs": {name: config.model_dump(mode="json") for name, config in configs.items()},
        "order": list(configs), "decoder_diagnosis": diagnosis,
        "augmentation": "Uniform independent scale and raw-unit offset per window/node/feature, constant across time. "
                        "Missing samples unchanged. Standardization largely cancels this; primarily tests raw correction features.",
        "optimizer": "AdamW decays all parameters; zero-decay AdamW isolates optimizer implementation from weight decay.",
        "selection": "Seed12 validation macro-F1 improvement over fresh control, no increase in request ratio, bits or energy; "
                     "rank by F1 then name. Independent changes only; no combinations in this experiment.",
        "confirmation": "Lock before paired candidate/control seeds42/1242; report three-seed validation means. No fallback selection.",
        "test_evaluation": False,
        "limitations": "Exploratory validation reuse, not proof of test superiority over PatchTST; Intel is not retrained.",
    }
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment_spec.json", spec)
    write_json(output / "decoder_diagnosis.json", diagnosis)
    rows = []
    for name, config in configs.items():
        rows.append(run_cell(output / "screening", name, config, DATASET, 12))
        write_json(output / "results.json", rows)
    control = rows[0]
    eligible = [row for row in rows[1:] if row["macro_f1"] > control["macro_f1"]
                and all(row[key] <= control[key] for key in ("request_ratio", "bits", "energy_j"))]
    eligible.sort(key=lambda row: (-row["macro_f1"], row["variant"]))
    selected = eligible[0]["variant"] if eligible else None
    selection = {"selected": selected, "status": "locked" if selected else "no_eligible_candidate"}
    if selected is not None:
        locked = output / "locked.yaml"
        locked.write_text(yaml.safe_dump(configs[selected].model_dump(mode="json"), sort_keys=False))
        selection["locked_config_sha256"] = sha256_file(locked)
    write_json(output / "selection.json", selection)
    if selected is None:
        print("No eligible candidate; screening complete.", flush=True)
        return
    for seed in (42, 1242):
        for name in ("control", selected):
            rows.append(run_cell(output / "confirmation", name, configs[name], DATASET, seed))
            write_json(output / "results.json", rows)
    means = {name: {key: sum(row[key] for row in rows if row["variant"] == name) / 3
                    for key in ("macro_f1", "request_ratio", "bits", "energy_j")} for name in ("control", selected)}
    confirmed = means[selected]["macro_f1"] > means["control"]["macro_f1"] and all(
        means[selected][key] <= means["control"][key] for key in ("request_ratio", "bits", "energy_j")
    )
    write_json(output / "confirmation.json", {"means": means, "confirmed": confirmed, "test_evaluation": False})
    print("Ordered validation-only experiments completed.", flush=True)


if __name__ == "__main__":
    main()
