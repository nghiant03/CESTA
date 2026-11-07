"""Exploratory conflict-aware KD screen; original paper BiGRU only."""
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
        "h128-d05": {"model_kwargs": {**base.model_kwargs, "hidden_size": 128, "dropout": .5}},
        "h128-d06": {"model_kwargs": {**base.model_kwargs, "hidden_size": 128, "dropout": .6}},
        "h96-d05": {"model_kwargs": {**base.model_kwargs, "hidden_size": 96, "dropout": .5}},
        "h128-d04-ensemble": {"model_kwargs": {**base.model_kwargs, "hidden_size": 128, "dropout": .4},
                             "distillation_teachers": [str(teachers[s]) for s in (12,42,1242)]},
    }
    from CESTA.models import create_model
    import torch

    parameter_counts = {}
    for name, changes in variants.items():
        model = create_model("cesta", input_size=data.input_size, num_classes=4, metadata=data.metadata, **changes["model_kwargs"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(data.X_train[:1]))
        if not torch.isfinite(logits).all() or model.count_parameters() > 719327:
            raise ValueError("Capacity/runtime check failed")
        parameter_counts[name] = {"active": model.count_parameters(), "total": model.count_all_parameters(), "payload_width": model.encoder_output_size}
    spec = {
        "phase": "New exploratory dropout followup after repeated prior completed improvement phases (see prior_phase_inventory.json); validation reused; no test",
        "variants": variants, "parameter_counts": parameter_counts,
        "prior_audit": "Previous capacity screen dropout.4 confirmed mean.8942348591547078; ensemble dropout.2 mean.8887138277071185. This is a new exploratory phase, not independent validation.",
        "schedule": "100 joint no warmup. Original unweighted gamma1 focal+CRF.01+communication objectives; localKD1/finalKD.3 T2 LR.001. Only hidden width/dropout vary; two bidirectional GRU layers retained.",
        "selection": "Best seed12 strictly above previous dropout.4 best .9037873225276213; all four screened before lock; confirm42/1242 no fallback. Mean target .8986931144798077",
        "reference_results": str(ROOT/"runs/cesta-distillation-full100-20261106/results.json"),
        "historical_graph_identity_verified": False,
        "teacher_hashes": {str(s):{n:sha256_file(p/n) for n in ("weight.pt","config.json","manifest.json")} for s,p in teachers.items()},
        "data_hashes": {p.name:sha256_file(p) for p in (ROOT/"data/datasets"/dataset).iterdir() if p.is_file()},
        "source_hashes": {str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__), ROOT/"scripts/run_cesta_loss_experiment.py", *sorted((ROOT/"src/CESTA").rglob("*.py"))]},
    }
    if args.preflight_only:
        print(json.dumps(spec,indent=2)); return
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False); write_json(output/"experiment_spec.json",spec)
    prior = ["cesta-distillation-final-20261106", "cesta-distillation-full100-20261106", "cesta-ensemble-exploratory-20261106",
             "cesta-patch-transfer-20261106", "cesta-gru-staged-20261106", "cesta-gru-refinement-20261106",
             "cesta-balanced-staged-20261106", "cesta-conflict-20261106", "cesta-gru-capacity-20261106"]
    write_json(output / "prior_phase_inventory.json", {
        "completed_directories": prior, "count": len(prior),
        "note": "Includes early-stopped superseded screen and architecture-ineligible PatchTST transfer; excludes earlier pre-distillation studies. Exploratory validation reused."})
    rows = []
    def cell(name: str, seed: int) -> None:
        c = TrainConfig.model_validate({**base.model_dump(mode="json"), **variants[name], "seed":seed,
            "distillation_teacher":None if name.endswith("ensemble") else str(teachers[seed]), "distillation_local_weight":1.0, "distillation_weight":.3})
        rows.append(run_cell(output,name,c,dataset,seed,early_stopping=False)); write_json(output/"results.json",rows)
    for name in variants: cell(name,12)
    best = max(rows,key=lambda r:r["macro_f1"])
    selected = best["variant"] if best["macro_f1"] > .9037873225276213 else None
    write_json(output/"selection.json",{"selected":selected,"locked":True})
    if selected:
        for seed in (42,1242): cell(selected,seed)
    write_json(output/"completion.json",{"complete":True,"test_evaluated":False})


if __name__ == "__main__":
    main()
