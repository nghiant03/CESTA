from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from run_cesta_loss_experiment import ROOT, run_cell, write_json

from CESTA.schema.config import TrainConfig
from CESTA.utils import sha256_file

DATASET = "CESTA-DHT_1min_fault15_stuck0_drift0625"
SEEDS = (12, 42, 2026)
VARIANTS = {
    "comm-0003": {"communication_penalty_weight": 0.003},
    "comm-003": {"communication_penalty_weight": 0.03},
    "comm-001-halfkd": {"distillation_local_weight": 0.5, "distillation_weight": 0.15},
    "comm-001-entropy0": {"gate_entropy_weight": 0.0},
}


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def intel_inventory() -> dict[str, str]:
    return {str(p.resolve()): sha256_file(p) for p in sorted((ROOT / "config/training/Intel").rglob("*")) if p.is_file()}


def prepare(output: Path) -> None:
    import torch

    from CESTA.datasets import load_dataset
    from CESTA.models import create_model

    prior = ROOT / "runs/cesta-gru-dropout-followup-20261106/results.json"
    replacement = ROOT / "runs/cesta-seed2026-paired-20261106"
    summary = read(replacement / "paired_summary.json")
    incumbents = {str(r["seed"]): r for r in read(prior) if r["variant"] == "h96-d05"}
    if set(incumbents) != {"12", "42", "1242"}:
        raise ValueError("Missing original incumbent seeds")
    incumbents["2026"] = read(replacement / "new_candidate_result.json")
    base_path = Path(incumbents["12"]["run"]) / "config.json"
    raw = read(base_path)["train_config"]
    base = TrainConfig.model_validate(raw)
    if base.model_dump(mode="json") != raw:
        raise ValueError("Resolved baseline schema/default drift")
    if (base.epochs, base.gate_entropy_weight, base.features) != (100, 0.001, ["temp"]):
        raise ValueError("Unexpected baseline")
    teachers = {}
    hashes = {}
    for seed, row in incumbents.items():
        student = Path(row["run"])
        config = read(student / "config.json")["train_config"]
        expected = {**raw, "seed": int(seed), "distillation_teacher": config["distillation_teacher"]}
        if config != expected:
            raise ValueError("Incumbent configuration differs beyond seed/teacher")
        teacher = Path(config["distillation_teacher"])
        tc = TrainConfig.model_validate(read(teacher / "config.json")["train_config"])
        if tc.seed != int(seed) or tc.features != base.features or tc.data != base.data:
            raise ValueError("Teacher seed/data mismatch")
        for run in (student, teacher):
            if read(run / "manifest.json")["timing"]["epochs_run"] != 100:
                raise ValueError("Incomplete reference")
            for filename in ("weight.pt", "config.json", "manifest.json", "history.jsonl"):
                path = run / filename
                hashes[str(path)] = sha256_file(path)
        metric_path = teacher / "validation/eval_metrics.json"
        if metric_path.exists():
            hashes[str(metric_path)] = sha256_file(metric_path)
            teacher_f1 = read(metric_path)["macro_f1"]
        else:
            teacher_f1 = max(json.loads(line)["val_macro_f1"] for line in (teacher / "history.jsonl").read_text().splitlines())
        student_metrics = student / "validation/eval_metrics.json"
        hashes[str(student_metrics)] = sha256_file(student_metrics)
        if read(student_metrics)["macro_f1"] != row["macro_f1"]:
            raise ValueError("Student evaluation disagrees with reference row")
        pair = next(p for p in summary["original" if seed == "1242" else "replacement"]["paired"] if p["seed"] == int(seed))
        if teacher_f1 != pair["teacher_f1"] or row["macro_f1"] != pair["candidate_f1"]:
            raise ValueError("Reference metric mismatch")
        teachers[seed] = {"run": str(teacher), "macro_f1": teacher_f1,
                          "metric_source": str(metric_path if metric_path.exists() else teacher / "history.jsonl")}
    dense_incumbents = incumbents
    sparse_directory = ROOT / "runs/cesta-gru-optimization-followup-20261106"
    sparse_results = sparse_directory / "results.json"
    incumbents = {str(r["seed"]): r for r in read(sparse_results) if r["variant"] == "comm-001"}
    if set(incumbents) != {"12", "42", "2026"}:
        raise ValueError("Missing sparse reference seeds")
    for seed, row in incumbents.items():
        run = Path(row["run"])
        config = read(run / "config.json")["train_config"]
        expected = {**raw, "communication_penalty_weight": 0.01, "seed": int(seed),
                    "distillation_teacher": teachers[seed]["run"]}
        if config != expected or read(run / "validation/eval_metrics.json")["macro_f1"] != row["macro_f1"]:
            raise ValueError("Sparse reference recipe/metric mismatch")
        if read(run / "manifest.json")["timing"]["epochs_run"] != 100:
            raise ValueError("Incomplete sparse reference")
        for filename in ("weight.pt", "config.json", "manifest.json", "history.jsonl",
                         "validation/eval_metrics.json", "validation/communication_metrics.json"):
            path = run / filename
            hashes[str(path)] = sha256_file(path)
    base_path = Path(incumbents["12"]["run"]) / "config.json"
    raw = read(base_path)["train_config"]
    base = TrainConfig.model_validate(raw)
    if base.model_dump(mode="json") != raw or incumbents["12"]["macro_f1"] != 0.9145803133390049:
        raise ValueError("Sparse baseline drift")
    configs = {
        name: {str(seed): TrainConfig.model_validate({**raw, **changes, "seed": seed,
                "distillation_teacher": teachers[str(seed)]["run"]}).model_dump(mode="json") for seed in SEEDS}
        for name, changes in VARIANTS.items()
    }
    historical_configs = {}
    for path in sorted((ROOT / "runs").rglob("config.json")):
        saved = read(path)
        if isinstance(saved, dict) and isinstance(saved.get("train_config"), dict):
            historical_configs[str(path.resolve())] = saved["train_config"]
    duplicates = [{"variant": name, "seed": seed, "historical_path": path}
                  for name, by_seed in configs.items() for seed, config in by_seed.items()
                  for path, historical in historical_configs.items() if config == historical]
    if duplicates:
        raise ValueError(f"Exact historical configuration repeats: {duplicates}")
    historical_hashes = {path: sha256_file(Path(path)) for path in historical_configs}
    intel_files = intel_inventory()
    if not intel_files:
        raise ValueError("Empty Intel inventory")
    hashes.update(intel_files)
    hashes.update(historical_hashes)
    for filename in ("protocol.json", "protocol_lock.json", "results.json", "selection.json", "confirmation.json", "completion_audit.json"):
        path = sparse_directory / filename
        hashes[str(path)] = sha256_file(path)
    data = load_dataset(ROOT / "data/datasets" / DATASET).prepare(
        window_config=base.data.window, split_config=base.data.split, features=base.features, required_metadata={"graph"})
    model = create_model("cesta", input_size=data.input_size, num_classes=4, metadata=data.metadata, **base.model_kwargs)
    counts = {"active": model.count_parameters(), "total": model.count_all_parameters(), "payload_width": model.encoder_output_size}
    baseline_manifest = read(base_path.parent / "manifest.json")
    if counts != {"active": baseline_manifest["num_parameters"], "total": baseline_manifest["total_parameters"], "payload_width": 192}:
        raise ValueError("Parameter counts differ from incumbent")
    if counts["active"] != 429279:
        raise ValueError(f"Architecture count mismatch: {counts}")
    model.eval()
    with torch.no_grad():
        if not torch.isfinite(model(torch.from_numpy(data.X_train[:1]))).all():
            raise ValueError("Nonfinite CPU preflight")
    paths = [Path(__file__).resolve(), ROOT / "scripts/run_cesta_loss_experiment.py", prior,
             replacement / "paired_summary.json", replacement / "new_candidate_result.json",
             *sorted((ROOT / "src/CESTA").rglob("*.py")), *sorted((ROOT / "data/datasets" / DATASET).iterdir())]
    hashes.update({str(p): sha256_file(p) for p in paths if p.is_file()})
    spec = {
        "dataset": DATASET, "baseline_config": raw, "variants": VARIANTS, "configs": configs,
        "screen_order": list(VARIANTS), "screen_seed": 12, "confirmation_seeds": [42, 2026],
        "epochs": 100, "early_stopping": False, "test_evaluation": False, "parameter_counts": counts,
        "incumbents": incumbents, "dense_incumbents": dense_incumbents, "teachers": teachers, "historical_summary": summary,
        "intel_inventory": intel_files, "historical_config_inventory": historical_hashes,
        "exact_repeat_audit": {"historical_config_count": len(historical_configs), "prospective_configs": 12, "duplicates": duplicates},
        "selection": "All four seed12 cells; highest macro-F1 strictly above incumbent; exact ties lexicographic; lock before42/2026; no fallback",
        "acceptance": "Mean over12/42/2026 strictly above dense incumbent, sparse incumbent and PatchTST means; communication reported, not vetoed",
        "limitations": ("Exploratory repeated validation reuse;2026 was post-hoc replacement; no independent superiority claim; "
                        "historical graph identity unverified; modeled energy; "
                        "seed12 record threshold biases seed12 specialists, not generalization; no rescue on42/2026"),
        "hashes": hashes,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", spec)
    write_json(output / "protocol_lock.json", {"sha256": sha256_file(output / "protocol.json"), "training_launched": False})
    print(f"Preflight ready: {output / 'protocol.json'}", flush=True)


def execute(output: Path) -> None:
    protocol = output / "protocol.json"
    if sha256_file(protocol) != read(output / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol changed after lock")
    spec = read(protocol)
    for filename, digest in spec["hashes"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError(f"Locked input changed: {filename}")
    if intel_inventory() != spec["intel_inventory"]:
        raise ValueError("Intel file inventory/content changed")
    with (output / "execution_started.json").open("x") as handle:
        json.dump({"approved_execution": True, "protocol_sha256": sha256_file(protocol)}, handle)
    rows = []

    def cell(name: str, seed: int) -> None:
        config = TrainConfig.model_validate(spec["configs"][name][str(seed)])
        row = run_cell(output, name, config, spec["dataset"], seed, early_stopping=False)
        run = Path(row["run"])
        manifest = read(run / "manifest.json")
        if manifest["timing"]["epochs_run"] != 100 or manifest["num_parameters"] != spec["parameter_counts"]["active"]:
            raise ValueError("Run schedule or parameter count mismatch")
        if manifest["train_config"] != spec["configs"][name][str(seed)]:
            raise ValueError("Run config drift")
        if manifest["total_parameters"] != spec["parameter_counts"]["total"]:
            raise ValueError("Total parameter count drift")
        row["first_max_f1_epoch"] = row.pop("best_epoch")
        row["parameter_counts"] = spec["parameter_counts"]
        row["communication"] = read(run / "validation/communication_metrics.json")["splits"]["val"]
        rows.append(row)
        write_json(output / "results.json", rows)

    for name in spec["screen_order"]:
        cell(name, 12)
    best = sorted(rows, key=lambda row: (-row["macro_f1"], row["variant"]))[0]
    selected = best["variant"] if best["macro_f1"] > spec["incumbents"]["12"]["macro_f1"] else None
    write_json(output / "selection.json", {"selected": selected, "locked": True, "screen_results": rows.copy(),
               "selected_configs": spec["configs"][selected] if selected else None, "protocol_sha256": sha256_file(protocol)})
    if selected:
        for seed in spec["confirmation_seeds"]:
            cell(selected, seed)
        paired = []
        for seed in SEEDS:
            candidate = next(r for r in rows if r["variant"] == selected and r["seed"] == seed)
            incumbent = spec["incumbents"][str(seed)]
            teacher = spec["teachers"][str(seed)]
            dense = spec["dense_incumbents"][str(seed)]
            paired.append({"seed": seed, "candidate_f1": candidate["macro_f1"], "incumbent_f1": incumbent["macro_f1"],
                           "teacher_f1": teacher["macro_f1"], "dense_f1": dense["macro_f1"],
                           "delta_dense": candidate["macro_f1"] - dense["macro_f1"],
                           "dense_communication_deltas": {k: candidate[k] - dense[k] for k in ("request_ratio", "bits", "energy_j")},
                           "delta_incumbent": candidate["macro_f1"] - incumbent["macro_f1"],
                           "delta_teacher": candidate["macro_f1"] - teacher["macro_f1"],
                           "communication_deltas": {k: candidate[k] - incumbent[k] for k in ("request_ratio", "bits", "energy_j")}})
        stats = {name: {"mean": statistics.mean(p[name] for p in paired), "sample_sd": statistics.stdev(p[name] for p in paired)}
                 for name in ("candidate_f1", "incumbent_f1", "teacher_f1", "dense_f1")}
        write_json(output / "confirmation.json", {"paired": paired, "statistics": stats,
                   "accepted_descriptively": stats["candidate_f1"]["mean"] > max(
                       stats[n]["mean"] for n in ("incumbent_f1", "dense_f1", "teacher_f1")),
                   "historical_original": spec["historical_summary"]["original"], "independent_superiority_claim": False})
    if intel_inventory() != spec["intel_inventory"]:
        raise ValueError("Intel inventory/content changed during execution")
    write_json(output / "completion.json", {"complete": True, "selected": selected, "test_evaluated": False, "fallback": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if args.execute_approved:
        execute(args.output.resolve())
    else:
        prepare(args.output.resolve())


if __name__ == "__main__":
    main()
