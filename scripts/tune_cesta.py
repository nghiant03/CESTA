from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import optuna
import yaml

from CESTA.schema.config import TrainConfig, load_config_file
from CESTA.utils import sha256_file


def suggest_config(base: TrainConfig, trial: optuna.Trial) -> TrainConfig:
    values = base.model_dump(mode="json")
    values["learning_rate"] = trial.suggest_float("learning_rate", 0.0002, 0.002, log=True)
    values["crf_loss_weight"] = trial.suggest_categorical("crf_loss_weight", [0.0, 0.01, 0.05, 0.1])
    values["model_kwargs"]["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64, 128])
    values["model_kwargs"]["num_layers"] = trial.suggest_int("num_layers", 1, 2)
    values["model_kwargs"]["dropout"] = trial.suggest_float("dropout", 0.05, 0.3)
    values["model_kwargs"]["use_crf"] = values["crf_loss_weight"] > 0
    return TrainConfig.model_validate(values)


def validation_score(run_dir: Path) -> tuple[float, dict]:
    if (run_dir / "eval_metrics.json").exists():
        raise RuntimeError("Tuning trial unexpectedly evaluated test data")
    records = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line.strip()]
    valid = [row for row in records if row.get("val_macro_f1") is not None and math.isfinite(row["val_macro_f1"])]
    if not valid:
        raise RuntimeError("Trial has no finite validation macro-F1")
    best = max(valid, key=lambda row: row["val_macro_f1"])
    return float(best["val_macro_f1"]), best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = TrainConfig.model_validate(load_config_file(args.base_config))
    if base.model != "cesta" or base.model_kwargs.get("temporal_backend", "gru") != "gru":
        raise ValueError("This study tunes the CESTA GRU backend")
    if not base.model_kwargs.get("standardize_window_features"):
        raise ValueError("This study requires standardized inputs")
    if args.trials < 1 or args.epochs < 1:
        raise ValueError("Trials and epochs must be positive")
    base = base.model_copy(update={"seed": 12, "epochs": args.epochs})
    data = args.data.resolve()
    provenance = {
        "dataset": str(data),
        "data_sha256": sha256_file(data / "dataset.csv"),
        "meta_sha256": sha256_file(data / "dataset_meta.json"),
        "base_config": base.model_dump(mode="json"),
        "trials": args.trials,
        "selection": "maximum validation macro-F1; early-stopping patience 10; no test evaluation during search",
        "communication": "fixed base penalties, report achieved communication; no hard budget constraint",
    }
    if args.dry_run:
        print(json.dumps(provenance, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "study_spec.json").write_text(json.dumps(provenance, indent=2) + "\n")
    study = optuna.create_study(
        study_name="cesta-dht-validation",
        storage=f"sqlite:///{output / 'study.db'}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=12, n_startup_trials=5),
    )
    study.enqueue_trial({"learning_rate": base.learning_rate, "crf_loss_weight": base.crf_loss_weight,
                         "hidden_size": base.model_kwargs["hidden_size"], "num_layers": base.model_kwargs["num_layers"],
                         "dropout": base.model_kwargs["dropout"]})

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(base, trial)
        trial_root = output / f"trial-{trial.number:02d}"
        trial_root.mkdir()
        config_path = trial_root / "training.yaml"
        config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
        trial.set_user_attr("config_path", str(config_path))
        with (trial_root / "training.log").open("w") as log:
            subprocess.run(
                ["uv", "run", "cesta", "train", str(config_path), str(data), "--output", str(trial_root / "runs"),
                 "--early-stopping", "--no-test-evaluation"], stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        manifests = list((trial_root / "runs").glob("*/manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("Expected exactly one completed trial run")
        score, best = validation_score(manifests[0].parent)
        trial.set_user_attr("run_path", str(manifests[0].parent))
        trial.set_user_attr("best_epoch", best["epoch"])
        trial.set_user_attr("val_request_ratio", best.get("val_request_ratio"))
        return score

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    best = study.best_trial
    locked = output / "locked.yaml"
    locked.write_text(Path(best.user_attrs["config_path"]).read_text())
    (output / "selection.json").write_text(json.dumps({
        "trial": best.number, "validation_macro_f1": best.value, "parameters": best.params,
        "attributes": best.user_attrs, "locked_config_sha256": sha256_file(locked),
    }, indent=2) + "\n")
    subprocess.run([
        "uv", "run", "python", "scripts/run_all_baselines.py", "--configs", str(locked), "--datasets", str(data),
        "--seeds", "12", "42", "1242", "--num-gpus", "1", "--runs-dir", str(output / "confirmation"),
        "--early-stopping", "--keep-progress-log",
    ], check=True)


if __name__ == "__main__":
    main()
