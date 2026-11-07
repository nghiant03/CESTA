"""Bounded validation-only frozen-teacher screen; never evaluates test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_cesta_loss_experiment import ROOT, run_cell, write_json

from CESTA.schema.config import TrainConfig, load_config_file
from CESTA.utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(ensemble=False)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = "CESTA-DHT_1min_fault15_stuck0_drift0625"
    teachers = {}
    for path in (ROOT / "runs/patchtst-dht-new-config-restart-20261106/patchtst").glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest["seed"] in teachers:
            raise ValueError("Duplicate teacher seed")
        if manifest["timing"]["epochs_run"] != 100 or not (path.parent / "weight.pt").is_file():
            raise ValueError("Incomplete teacher")
        teachers[manifest["seed"]] = path.parent
    if set(teachers) != {12, 42, 1242}:
        raise ValueError("Required teacher seeds missing or unexpected")
    base = TrainConfig.model_validate(load_config_file(ROOT / "config/experiments/ablations/cesta-dht-gru-control.yaml"))
    variants = {
        "control": {},
        "kd200": {"distillation_weight": .3, "distillation_local_weight": 1.0, "epochs": 200},
        "warm100-joint100": {"distillation_weight": .3, "distillation_local_weight": 1.0, "local_warmup_epochs": 100},
        "warm100-joint100-low": {"distillation_weight": .3, "distillation_local_weight": 1.0, "local_warmup_epochs": 100, "joint_learning_rate_scale": .3},
    }
    for path in teachers.values():
        tc = TrainConfig.model_validate(json.loads((path / "config.json").read_text())["train_config"])
        if tc.model != "patchtst" or tc.features != base.features or tc.data != base.data:
            raise ValueError("Incompatible teacher recipe")
    teacher_f1 = max(json.loads(line)["val_macro_f1"] for line in (teachers[12] / "history.jsonl").read_text().splitlines())
    write_json(output / "experiment_spec.json", {
        "dataset": dataset, "base": base.model_dump(mode="json"), "variants": variants,
        "temperature": 2, "seeds": [12, 42, 1242], "test_evaluation": False,
        "selection": ("Exploratory ensemble: best seed12 above original GRU; lock then confirm seeds42/1242; mean must beat teacher mean and all paired GRU gains positive." if args.ensemble else "New exploratory original-BiGRU staged screen. Control100, KD200, localwarm100+joint100 at LR .001 or .0003. Best seed12 above control and teacher; lock before seeds42/1242. No test or communication veto."),
        "teacher_hashes": {str(seed): {name: sha256_file(path / name) for name in ("weight.pt", "config.json", "manifest.json")} for seed, path in teachers.items()},
        "data_hashes": {p.name: sha256_file(p) for p in (ROOT / "data/datasets" / dataset).iterdir() if p.is_file()},
        "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__).resolve(), ROOT / "scripts/run_cesta_loss_experiment.py", *sorted((ROOT / "src/CESTA").rglob("*.py"))]},
    })
    rows = []
    if args.ensemble:
        write_json(output / "exploratory_phase.json", {
            "status": "New exploratory screen after prior confirmation, not an extension of its locked candidate",
            "selection": "Best seed12 candidate above original control; confirm seeds42/1242. Acceptance: mean above .8986931144798077 and all paired original GRU gains positive.",
            "rationale": "Average all three frozen teacher probabilities at T2 to reduce teacher seed sensitivity; second variant doubles training-label DRIFT timestep KL weights and normalizes by weight sum.",
            "historical_graph_identity_verified": False,
        })

    def cell(name: str, seed: int) -> dict:
        config = TrainConfig.model_validate({**base.model_dump(mode="json"), "seed": seed,
            **variants[name], "distillation_teacher": str(teachers[seed]) if variants[name] else None})
        if args.ensemble:
            config = TrainConfig.model_validate({**config.model_dump(mode="json"), "distillation_teacher": None,
                "distillation_teachers": [str(teachers[s]) for s in (12, 42, 1242)]})
        row = run_cell(output, name, config, dataset, seed, early_stopping=False)
        rows.append(row)
        write_json(output / "results.json", rows)
        return row

    control = {"macro_f1": 0.8657012357450247} if args.ensemble else cell("control", 12)
    for name in variants:
        if name != "control":
            cell(name, 12)
    eligible = [r for r in (rows if args.ensemble else rows[1:]) if r["macro_f1"] > (control["macro_f1"] if args.ensemble else max(control["macro_f1"], teacher_f1))]
    selected = max(eligible, key=lambda r: r["macro_f1"])["variant"] if eligible else None
    write_json(output / "selection.json", {"selected": selected, "locked": True})
    if selected:
        for seed in (42, 1242):
            if not args.ensemble:
                cell("control", seed)
            cell(selected, seed)
    write_json(output / "completion.json", {"selected": selected, "complete": True, "test_evaluated": False})


if __name__ == "__main__":
    main()
