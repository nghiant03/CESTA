"""Distinct architecture screen: portable standalone-compatible local PatchTST."""
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
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = "CESTA-DHT_1min_fault15_stuck0_drift0625"
    teachers = {}
    for p in (ROOT / "runs/patchtst-dht-new-config-restart-20261106/patchtst").glob("*/manifest.json"):
        m = json.loads(p.read_text())
        if m["seed"] in teachers or m["timing"]["epochs_run"] != 100:
            raise ValueError("Duplicate or incomplete teacher")
        teachers[m["seed"]] = p.parent
    if set(teachers) != {12, 42, 1242}:
        raise ValueError("Missing teachers")
    base = TrainConfig.model_validate(load_config_file(ROOT / "config/experiments/ablations/cesta-dht-gru-control.yaml"))
    kwargs = {**base.model_kwargs, "temporal_backend": "patchtst", "standardize_window_features": False,
              "patchtst_config": {"d_model": 32, "standalone_compatible": True}}
    base = TrainConfig.model_validate({**base.model_dump(mode="json"), "model_kwargs": kwargs})
    configs = {name: TrainConfig.model_validate({**base.model_dump(mode="json"),
               "temporal_transfer_checkpoint": str(teachers[12]) if name == "pretrained" else None}) for name in ("fresh", "pretrained")}
    write_json(output / "experiment_spec.json", {
        "configs": {k: v.model_dump(mode="json") for k, v in configs.items()},
        "phase": "Exploratory architecture replacement after KD phases; full100 no early stop, no test",
        "selection": "Best seed12 above original GRU .8657012357450247; lock then paired seeds42/1242; mean must exceed PatchTST .8986931144798077",
        "historical_graph_identity_verified": False,
        "teacher_hashes": {str(s): {n: sha256_file(p/n) for n in ("weight.pt", "config.json", "manifest.json")} for s,p in teachers.items()},
        "data_hashes": {p.name: sha256_file(p) for p in (ROOT/"data/datasets"/dataset).iterdir() if p.is_file()},
        "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__), ROOT/"scripts/run_cesta_loss_experiment.py", *sorted((ROOT/"src/CESTA").rglob("*.py"))]},
    })
    rows = []
    def cell(name: str, seed: int) -> None:
        config = TrainConfig.model_validate({**configs[name].model_dump(mode="json"), "seed": seed,
                  "temporal_transfer_checkpoint": str(teachers[seed]) if name == "pretrained" else None})
        rows.append(run_cell(output, name, config, dataset, seed, early_stopping=False))
        write_json(output/"results.json", rows)
    for name in configs:
        cell(name, 12)
    best = max(rows, key=lambda r: r["macro_f1"])
    selected = best["variant"] if best["macro_f1"] > .8657012357450247 else None
    write_json(output/"selection.json", {"selected": selected, "locked": True})
    if selected:
        for seed in (42, 1242):
            cell(selected, seed)
    write_json(output/"completion.json", {"complete": True, "test_evaluated": False})


if __name__ == "__main__":
    main()
