from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from run_cesta_loss_experiment import ROOT, run_cell, write_json

from CESTA.schema.config import TrainConfig
from CESTA.training.schedule import epoch_learning_rate
from CESTA.utils import sha256_file

SEEDS = (12, 42, 2026)
DATASET = "CESTA-DHT_1min_fault15_stuck0_drift0625"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def intel_inventory() -> dict[str, str]:
    return {str(p.resolve()): sha256_file(p) for p in sorted((ROOT / "config/training/Intel").rglob("*")) if p.is_file()}


def prepare(output: Path) -> None:
    from CESTA.artifacts import load_checkpoint
    from CESTA.models.base import BaseModel

    previous = ROOT / "runs/cesta-gru-communication-followup-20261106/protocol.json"
    refs = read(previous)
    raw = refs["baseline_config"]
    base = TrainConfig.model_validate(raw).model_dump(mode="json")
    added = {"learning_rate_schedule": "none", "learning_rate_schedule_start_epoch": None, "learning_rate_final": None}
    if base != {**raw, **added}:
        raise ValueError("Unexpected baseline drift beyond newly added disabled schedule fields")
    if any(base[k] for k in ("local_warmup_epochs", "training_initialization_checkpoint", "temporal_transfer_checkpoint")):
        raise ValueError("Unexpected initialization/warmup")
    variants = {"constant": {}, "late-cosine": {"learning_rate_schedule": "late_cosine",
                "learning_rate_schedule_start_epoch": 51, "learning_rate_final": 0.0001}}
    configs = {name: {str(seed): TrainConfig.model_validate({**base, **change, "seed": seed,
               "distillation_teacher": refs["teachers"][str(seed)]["run"]}).model_dump(mode="json") for seed in SEEDS}
               for name, change in variants.items()}
    loaded = []
    for group in ("incumbents", "dense_incumbents", "teachers"):
        for seed in SEEDS:
            run = Path(refs[group][str(seed)]["run"])
            model = load_checkpoint(run, map_location="cpu")
            if not isinstance(model, BaseModel):
                raise ValueError("Unexpected checkpoint type")
            if group != "teachers" and (model.count_parameters() != 429279 or model.count_all_parameters() != 466528):
                raise ValueError("Old checkpoint parameter mismatch")
            loaded.append(str(run))
            del model
    hashes = {p: sha256_file(Path(p)) for p in refs["hashes"]}
    paths = [previous, Path(__file__).resolve(), ROOT / "scripts/run_cesta_loss_experiment.py",
             *sorted((ROOT / "src/CESTA").rglob("*.py")), *sorted((ROOT / "data/datasets" / DATASET).iterdir())]
    hashes.update({str(p): sha256_file(p) for p in paths if p.is_file()})
    intel = intel_inventory()
    hashes.update(intel)
    order = [{"variant": name, "seed": seed} for seed in SEEDS for name in variants]
    protocol = {"dataset": DATASET, "configs": configs, "order": order, "baseline_config": raw,
                "schedule": "epochs1..50=.001;51..100=.0001+(.001-.0001)/2*(1+cos(pi*(e-51)/49)); no optimizer reset",
                "primary": "Mean paired late-cosine minus fresh constant validation-selected macro-F1 over12/42/2026",
                "secondary": "Final epoch validation F1; epochs81..100 validation F1 mean and population SD; selected per-class and communication",
                "selection": "None: run all six in fixed order irrespective of scores; no triage/fallback/secondary winner switching",
                "references": {k: refs[k] for k in ("incumbents", "dense_incumbents", "teachers", "historical_summary")},
                "parameter_counts": {"active": 429279, "total": 466528, "payload_width": 192},
                "epochs": 100, "early_stopping": False, "test_evaluated": False,
                "limitations": "Exploratory reused seeds/validation; no independent confirmation; energy modeled; original1242 preserved",
                "old_checkpoints_loaded": loaded, "intel_inventory": intel, "hashes": hashes}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol_lock.json", {"sha256": sha256_file(output / "protocol.json")})
    print(f"Preflight ready: {output / 'protocol.json'}")


def verify(output: Path, spec: dict[str, Any]) -> None:
    if sha256_file(output / "protocol.json") != read(output / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol changed")
    for path, digest in spec["hashes"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Locked file changed: {path}")
    if intel_inventory() != spec["intel_inventory"]:
        raise ValueError("Intel inventory/content changed")


def execute(output: Path) -> None:
    spec = read(output / "protocol.json")
    verify(output, spec)
    with (output / "execution_started.json").open("x") as handle:
        json.dump({"approved": True}, handle)
    rows = []
    for cell in spec["order"]:
        name, seed = cell["variant"], cell["seed"]
        config = TrainConfig.model_validate(spec["configs"][name][str(seed)])
        row = run_cell(output, name, config, DATASET, seed, early_stopping=False)
        run = Path(row["run"])
        manifest = read(run / "manifest.json")
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        if len(history) != 100 or manifest["timing"]["epochs_run"] != 100 or manifest["train_config"] != config.model_dump(mode="json"):
            raise ValueError("Schedule/config mismatch")
        if manifest["num_parameters"] != 429279 or manifest["total_parameters"] != 466528:
            raise ValueError("Parameter drift")
        for epoch, item in enumerate(history, 1):
            if item["epoch"] != epoch or item["learning_rates"] != [epoch_learning_rate(config, epoch)]:
                raise ValueError("Logged learning rate differs from protocol")
        row["first_max_f1_epoch"] = row.pop("best_epoch")
        row["final_epoch_f1"] = history[-1]["val_macro_f1"]
        row["late20_mean_f1"] = statistics.mean(h["val_macro_f1"] for h in history[80:])
        row["late20_population_sd_f1"] = statistics.pstdev(h["val_macro_f1"] for h in history[80:])
        row["communication"] = read(run / "validation/communication_metrics.json")["splits"]["val"]
        rows.append(row)
        write_json(output / "results.json", rows)
    paired = []
    for seed in SEEDS:
        c = next(r for r in rows if r["seed"] == seed and r["variant"] == "constant")
        s = next(r for r in rows if r["seed"] == seed and r["variant"] == "late-cosine")
        keys = ("macro_f1", "final_epoch_f1", "late20_mean_f1", "late20_population_sd_f1", "request_ratio", "bits", "energy_j")
        paired.append({"seed": seed, "cosine_minus_constant": {k: s[k] - c[k] for k in keys}})
    series = {name: [r["macro_f1"] for r in rows if r["variant"] == name] for name in ("constant", "late-cosine")}
    series.update({name: [spec["references"][key][str(seed)]["macro_f1"] for seed in SEEDS]
                   for name, key in (("historical_sparse", "incumbents"), ("historical_dense", "dense_incumbents"), ("teacher", "teachers"))})
    stats = {name: {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)} for name, values in series.items()}
    summary = {"paired": paired, "statistics": stats,
               "primary_mean_delta": statistics.mean(p["cosine_minus_constant"]["macro_f1"] for p in paired),
               "primary_delta_sample_sd": statistics.stdev(p["cosine_minus_constant"]["macro_f1"] for p in paired),
               "historical_original": spec["references"]["historical_summary"]["original"], "independent_superiority_claim": False}
    write_json(output / "comparison.json", summary)
    verify(output, spec)
    write_json(output / "completion.json", {"complete": True, "six_runs": True, "all_hashes_verified": True,
               "full_intel_inventory_unchanged": True, "test_evaluated": False, "selection": None, "fallback": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if args.execute_approved:
        execute(args.output.resolve())
    else:
        prepare(args.output.resolve())


if __name__ == "__main__":
    main()
