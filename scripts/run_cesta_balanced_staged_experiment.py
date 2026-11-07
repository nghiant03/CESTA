"""Exploratory exact focal20+joint80 screen; original paper BiGRU only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_cesta_loss_experiment import ROOT, run_cell, write_json

from CESTA.datasets import load_dataset
from CESTA.schema.config import TrainConfig, load_config_file
from CESTA.utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    dataset = "CESTA-DHT_1min_fault15_stuck0_drift0625"
    base = TrainConfig.model_validate(load_config_file(ROOT / "config/experiments/ablations/cesta-dht-gru-control.yaml"))
    data = load_dataset(ROOT/"data/datasets"/dataset).prepare(window_config=base.data.window, split_config=base.data.split,
                                                            features=base.features, required_metadata={"graph"})
    valid = (data.y_train >= 0) & data.node_mask_train
    counts = np.bincount(data.y_train[valid], minlength=4)
    if (counts == 0).any():
        raise ValueError("Training classes must be nonempty")
    frequencies = counts / counts.sum()
    weights = np.minimum(np.sqrt(frequencies.max()/frequencies), 3.0)
    weights /= (weights * frequencies).sum()
    teachers = {}
    for p in (ROOT/"runs/patchtst-dht-new-config-restart-20261106/patchtst").glob("*/manifest.json"):
        m = json.loads(p.read_text())
        if m["seed"] in teachers or m["timing"]["epochs_run"] != 100:
            raise ValueError("Duplicate/incomplete teacher")
        tc = TrainConfig.model_validate(m["train_config"])
        if tc.features != base.features or tc.data != base.data:
            raise ValueError("Teacher data recipe mismatch")
        teachers[m["seed"]] = p.parent
    if set(teachers) != {12,42,1242}:
        raise ValueError("Missing teachers")
    variants = {
        "joint-balanced": {"focal_alpha": weights.tolist()},
        "staged-balanced": {"focal_alpha": weights.tolist(), "epochs": 80, "local_warmup_epochs": 20,
                            "local_warmup_preserve_optimizer": True, "local_warmup_use_task_loss": True},
        "staged-unweighted": {"epochs": 80, "local_warmup_epochs": 20,
                              "local_warmup_preserve_optimizer": True, "local_warmup_use_task_loss": True},
    }
    spec = {
        "phase": "New exploratory exact reviewer recipe; validation reused; no test",
        "counts_train_window_timesteps": counts.tolist(), "focal_weights": weights.tolist(), "variants": variants,
        "weight_rule": "inverse sqrt train frequency, cap max:min3:1, normalize train-frequency-weighted mean1; supervised focal only",
        "schedule": "Total100 each: joint100 or local20+joint80; gamma1 focal + localKD1 warmup; original joint focal+CRF.01+localKD1+finalKD.3; T2; LR.001; preserved local Adam moments; joint Gumbel starts epoch1",
        "selection": "Best seed12 above originalGRU and same-seed PatchTST .891308571329575; lock before42/1242; no control retraining. Mean target .8986931144798077",
        "reference_results": str(ROOT/"runs/cesta-distillation-full100-20261106/results.json"),
        "historical_graph_identity_verified": False,
        "teacher_hashes": {str(s):{n:sha256_file(p/n) for n in ("weight.pt","config.json","manifest.json")} for s,p in teachers.items()},
        "data_hashes": {p.name:sha256_file(p) for p in (ROOT/"data/datasets"/dataset).iterdir() if p.is_file()},
        "source_hashes": {str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__), ROOT/"scripts/run_cesta_loss_experiment.py", *sorted((ROOT/"src/CESTA").rglob("*.py"))]},
    }
    if args.preflight_only:
        print(json.dumps(spec,indent=2)); return
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False); write_json(output/"experiment_spec.json",spec)
    rows = []
    def cell(name: str, seed: int) -> None:
        c = TrainConfig.model_validate({**base.model_dump(mode="json"), **variants[name], "seed":seed,
            "distillation_teacher":str(teachers[seed]), "distillation_local_weight":1.0, "distillation_weight":.3})
        rows.append(run_cell(output,name,c,dataset,seed,early_stopping=False)); write_json(output/"results.json",rows)
    for name in variants: cell(name,12)
    best = max(rows,key=lambda r:r["macro_f1"])
    selected = best["variant"] if best["macro_f1"] > .891308571329575 else None
    write_json(output/"selection.json",{"selected":selected,"locked":True})
    if selected:
        for seed in (42,1242): cell(selected,seed)
    write_json(output/"completion.json",{"complete":True,"test_evaluated":False})


if __name__ == "__main__":
    main()
