"""Post-hoc user-requested single seed replacement; no seed search or test evaluation."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import yaml
from run_cesta_loss_experiment import ROOT, run_cell, write_json

from CESTA.schema.config import TrainConfig
from CESTA.utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replacement-seed", type=int, default=2026)
    args = parser.parse_args()
    seed = args.replacement_seed
    if seed in (12, 42, 1242):
        raise ValueError("Replacement must differ from existing seeds")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    prior = ROOT / "runs/cesta-gru-dropout-followup-20261106"
    candidates = {r["seed"]: r for r in json.loads((prior/"results.json").read_text()) if r["variant"] == "h96-d05"}
    teachers = {}
    for path in (ROOT/"runs/patchtst-dht-new-config-restart-20261106/patchtst").glob("*/manifest.json"):
        m = json.loads(path.read_text())
        if m["seed"] in teachers: raise ValueError("Duplicate teacher seed")
        teachers[m["seed"]] = path.parent
    if set(teachers) != {12,42,1242} or set(candidates) != {12,42,1242}: raise ValueError("Missing prior cells")
    teacher_config = TrainConfig.model_validate(json.loads((teachers[12]/"config.json").read_text())["train_config"])
    teacher_config = TrainConfig.model_validate({**teacher_config.model_dump(mode="json"), "seed":seed})
    student_config = TrainConfig.model_validate(json.loads((Path(candidates[12]["run"])/"config.json").read_text())["train_config"])
    student_template = {**student_config.model_dump(mode="json"), "seed":seed, "distillation_teacher":"RESOLVED_NEW_SAME_SEED_TEACHER"}
    if teacher_config.features != student_config.features or teacher_config.data != student_config.data: raise ValueError("Recipe mismatch")
    dataset = ROOT/"data/datasets/CESTA-DHT_1min_fault15_stuck0_drift0625"
    spec = {"reason":"User requested replacing seed1242; parent explicitly chose2026, not a seed search",
        "replacement_seed":seed,"new_seeds":[12,42,seed],"old_seeds":[12,42,1242],
        "limitation":"Post-hoc replacement after observing1242; not independent superiority evidence. No further replacement based on outcome.",
        "teacher_config":teacher_config.model_dump(mode="json"),"student_config_template":student_template,
        "prior_candidate_rows":candidates,"prior_teacher_runs":{str(s):str(p) for s,p in teachers.items()},
        "old_artifact_hashes":{str(p):{n:sha256_file(p/n) for n in ("weight.pt","config.json","manifest.json")} for p in [*teachers.values(),*[Path(r["run"]) for r in candidates.values()]]},
        "dataset_hashes":{p.name:sha256_file(p) for p in dataset.iterdir() if p.is_file()},
        "source_hashes":{str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),ROOT/"scripts/run_cesta_loss_experiment.py",*sorted((ROOT/"src/CESTA").rglob("*.py"))]},
        "test_evaluation":False,"epochs_each":100,"early_stopping":False}
    write_json(output/"protocol.json",spec)
    config_path=output/"teacher.yaml"; config_path.write_text(yaml.safe_dump(teacher_config.model_dump(mode="json"),sort_keys=False))
    with (output/"teacher_training.log").open("w") as log:
        subprocess.run(["uv","run","cesta","train",str(config_path),str(dataset),"--output",str(output/"teacher"),"--no-test-evaluation"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    paths=list((output/"teacher").glob("*/manifest.json"))
    if len(paths)!=1: raise ValueError("Expected one new teacher")
    teacher=paths[0].parent
    with (output/"teacher_validation.log").open("w") as log:
        subprocess.run(["uv","run","cesta","evaluate","--model",str(teacher),"--data",str(dataset),"--split","val","--batch-size","32"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    student=TrainConfig.model_validate({**student_template,"distillation_teacher":str(teacher)})
    row=run_cell(output,"h96-d05",student,dataset.name,seed,early_stopping=False)
    write_json(output/"new_candidate_result.json",row)
    tf1={s:max(json.loads(line)["val_macro_f1"] for line in (p/"history.jsonl").read_text().splitlines()) for s,p in teachers.items()}
    tf1[seed]=json.loads((teacher/"validation/eval_metrics.json").read_text())["macro_f1"]
    cf1={s:r["macro_f1"] for s,r in candidates.items()}; cf1[seed]=row["macro_f1"]
    summary={"post_hoc_seed_replacement":True,"independent_superiority_claim":False,"new_teacher_run":str(teacher),"new_student_run":row["run"]}
    for name,seeds in (("original",[12,42,1242]),("replacement",[12,42,seed])):
        pairs=[{"seed":s,"teacher_f1":tf1[s],"candidate_f1":cf1[s],"delta":cf1[s]-tf1[s]} for s in seeds]
        summary[name]={"paired":pairs,"teacher_mean":statistics.mean(tf1[s] for s in seeds),"teacher_sample_sd":statistics.stdev(tf1[s] for s in seeds),
                       "candidate_mean":statistics.mean(cf1[s] for s in seeds),"candidate_sample_sd":statistics.stdev(cf1[s] for s in seeds)}
    write_json(output/"paired_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
