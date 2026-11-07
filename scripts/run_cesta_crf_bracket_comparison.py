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
    previous = ROOT / "runs/cesta-gru-schedule-comparison-20261106"
    refs = read(previous / "protocol.json")
    controls = {str(r["seed"]): r for r in read(previous / "results.json") if r["variant"] == "constant"}
    if set(controls) != {"12", "42", "2026"}:
        raise ValueError("Missing fixed controls")
    configs = {}
    for seed in SEEDS:
        raw = read(Path(controls[str(seed)]["run"]) / "config.json")["train_config"]
        if raw != refs["configs"]["constant"][str(seed)] or raw["learning_rate_schedule"] != "none":
            raise ValueError("Control configuration drift")
        config = TrainConfig.model_validate({**raw, "crf_loss_weight": 0.02}).model_dump(mode="json")
        if {k: config[k] for k in config if config[k] != raw[k]} != {"crf_loss_weight": 0.02}:
            raise ValueError("Unexpected candidate changes")
        configs[str(seed)] = config
    historical = {}
    for path in sorted((ROOT / "runs").rglob("config.json")):
        item = read(path)
        if isinstance(item, dict) and isinstance(item.get("train_config"), dict):
            historical[str(path)] = item["train_config"]
    if any(config == old for config in configs.values() for old in historical.values()):
        raise ValueError("Exact candidate configuration already exists")
    hashes = {}
    for path, digest in refs["hashes"].items():
        current = sha256_file(Path(path))
        if current != digest:
            raise ValueError(f"Prior study input changed: {path}")
        hashes[path] = current
    paths = [Path(__file__).resolve(), previous / "protocol.json", previous / "protocol_lock.json",
             previous / "results.json", previous / "comparison.json", previous / "completion_audit.json",
             *sorted((ROOT / "runs/cesta-train-stage-diagnostic-20261106").rglob("*")),
             *sorted((ROOT / "src/CESTA").rglob("*.py")), *map(Path, historical)]
    for row in controls.values():
        run = Path(row["run"])
        manifest = read(run / "manifest.json")
        if manifest["timing"]["epochs_run"] != 100 or read(run / "validation/eval_metrics.json")["macro_f1"] != row["macro_f1"]:
            raise ValueError("Invalid reused control")
        paths.extend(run / name for name in ("weight.pt", "config.json", "manifest.json", "history.jsonl",
                                            "validation/eval_metrics.json", "validation/communication_metrics.json"))
    negative = ROOT / "runs/cesta-gru-crf-weight-comparison-20261106"
    paths.extend(p for p in negative.rglob("*") if p.is_file())
    hashes.update({str(p): sha256_file(p) for p in paths if p.is_file()})
    intel = intel_inventory()
    hashes.update(intel)
    protocol = {"dataset": DATASET, "configs": configs, "seeds": list(SEEDS), "controls": controls,
                "references": refs["references"], "parameter_counts": refs["parameter_counts"],
                "primary": "Mean paired candidate minus reused fresh constant control validation-selected macro-F1 over12/42/2026",
                "secondary": "STUCK precision/recall/F1 all seeds; all-class metrics; communication; final/late20 metrics. No rescue or switching",
                "control_reuse": "Prior schedule study fresh constant runs, not newly concurrent training",
                "prior_negative": read(negative / "comparison.json"),
                "directional_bracket": "Single .02 hypothesis after failed .005; keep negative visible, no retrospective best-of-weight claim",
                "mechanism": "Training CRF loss weight .01 to .02 only; decoder retained; not inference scaling or guaranteed boundary FP fix",
                "selection": "None: all three100epoch candidates; no triage, fallback, test or extra validation boundary diagnostics",
                "duplicate_check": {"historical_configs": len(historical), "exact_repeats": 0},
                "limitations": "Hypothesis chosen after reused validation and training diagnostics; descriptive, no statistical superiority claim",
                "epochs": 100, "early_stopping": False, "test_evaluated": False, "intel_inventory": intel, "hashes": hashes}
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
    for seed in spec["seeds"]:
        name = "crf-002"
        config = TrainConfig.model_validate(spec["configs"][str(seed)])
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
        c = spec["controls"][str(seed)]
        s = next(r for r in rows if r["seed"] == seed)
        keys = ("macro_f1", "final_epoch_f1", "late20_mean_f1", "late20_population_sd_f1", "request_ratio", "bits", "energy_j")
        paired.append({"seed": seed, "candidate_minus_control": {k: s[k] - c[k] for k in keys}})
    series = {"crf-002": [r["macro_f1"] for r in rows],
              "reused_constant": [spec["controls"][str(seed)]["macro_f1"] for seed in SEEDS]}
    series.update({name: [spec["references"][key][str(seed)]["macro_f1"] for seed in SEEDS]
                   for name, key in (("historical_sparse", "incumbents"), ("historical_dense", "dense_incumbents"), ("teacher", "teachers"))})
    stats = {name: {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)} for name, values in series.items()}
    summary = {"paired": paired, "statistics": stats,
               "primary_mean_delta": statistics.mean(p["candidate_minus_control"]["macro_f1"] for p in paired),
               "primary_delta_sample_sd": statistics.stdev(p["candidate_minus_control"]["macro_f1"] for p in paired),
               "historical_original": spec["references"]["historical_summary"]["original"],
               "prior_negative": spec["prior_negative"],
               "stuck_paired": [{"seed": r["seed"], "candidate": r["per_class"]["STUCK"],
                                 "control": spec["controls"][str(r["seed"])]["per_class"]["STUCK"]} for r in rows],
               "independent_superiority_claim": False}
    write_json(output / "comparison.json", summary)
    verify(output, spec)
    write_json(output / "completion.json", {"complete": True, "three_new_runs": True, "three_reused_controls": True, "all_hashes_verified": True,
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
