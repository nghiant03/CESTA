from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_cesta_loss_experiment import ROOT, write_json

from CESTA.artifacts import load_checkpoint
from CESTA.batch import GraphWindowBatch
from CESTA.datasets import load_dataset
from CESTA.schema.config import TrainConfig
from CESTA.training.batch_utils import make_window_loader, prepare_batch
from CESTA.utils import sha256_file


def metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    cm = np.bincount(4 * y[mask] + pred[mask], minlength=16).reshape(4, 4)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(4, dtype=float), where=denominator > 0)
    return {"count": int(mask.sum()), "confusion_true_rows": cm.tolist(), "f1": f1.tolist(), "macro_f1": float(f1.mean())}


def summarize(y: np.ndarray, predictions: dict[str, np.ndarray], valid: np.ndarray, near: np.ndarray,
              complete: np.ndarray, activity: np.ndarray | None, probability: np.ndarray | None) -> dict[str, Any]:
    cohorts = {"all_valid": valid, "complete_nodes": valid & complete[:, None, :],
               "near_transition": valid & near, "interior": valid & ~near}
    cohorts.update({f"node_index{n}": valid & (np.arange(y.shape[2])[None, None, :] == n) for n in range(y.shape[2])})
    result: dict[str, Any] = {"stages": {stage: {name: metrics(y, pred, mask) for name, mask in cohorts.items()}
                                            for stage, pred in predictions.items()}}
    names = list(predictions)
    changes = {}
    for a, b in zip(names, names[1:]):
        changes[f"{a}_to_{b}"] = {
            cohort: [{"class": k, "support": int((mask & (y == k)).sum()),
                      "fixed": int((mask & (y == k) & (predictions[a] != y) & (predictions[b] == y)).sum()),
                      "introduced": int((mask & (y == k) & (predictions[a] == y) & (predictions[b] != y)).sum())}
                     for k in range(4)] for cohort, mask in cohorts.items()}
    result["stage_changes"] = changes
    if activity is not None and probability is not None:
        final = predictions[names[-1]]
        result["requests_conditioned"] = [{"class": k, "final_error": error,
            "count": int((valid & (y == k) & ((final != y) == error)).sum()),
            "mean_receiver_request_fraction": float(activity[valid & (y == k) & ((final != y) == error)].mean())
                if (valid & (y == k) & ((final != y) == error)).any() else None,
            "mean_receiver_soft_probability": float(probability[valid & (y == k) & ((final != y) == error)].mean())
                if (valid & (y == k) & ((final != y) == error)).any() else None}
            for k in range(4) for error in (False, True)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    previous = ROOT / "runs/cesta-gru-schedule-comparison-20261106"
    rows = json.loads((previous / "results.json").read_text())
    spec = json.loads((previous / "protocol.json").read_text())
    config = TrainConfig.model_validate(spec["configs"]["constant"]["12"])
    dataset = ROOT / "data/datasets" / spec["dataset"]
    data = load_dataset(dataset).prepare(window_config=config.data.window, split_config=config.data.split,
                                        features=config.features, required_metadata={"graph"})
    width = config.data.window.window_size
    stride = config.data.window.train_stride
    if width % stride:
        raise ValueError("Nonintegral nonoverlapping subsampling")
    indices = np.arange(0, len(data.X_train), width // stride)
    split = data.train.select(indices)
    if split.node_mask is None or split.edge_mask is None:
        raise ValueError("Missing masks")
    y = split.y
    valid = split.node_mask & (y >= 0)
    complete = np.asarray(valid.all(axis=1))
    timeline_y = y.transpose(0, 1, 2).reshape(-1, y.shape[2])
    timeline_valid = valid.reshape(-1, y.shape[2])
    changes = (timeline_y[1:] != timeline_y[:-1]) & timeline_valid[1:] & timeline_valid[:-1]
    near_flat = np.zeros_like(timeline_valid)
    for t, n in np.argwhere(changes):
        near_flat[max(0, t - 2):min(len(near_flat), t + 4), n] = True
    near = near_flat.reshape(y.shape)
    runs = [(r["variant"], r["seed"], Path(r["run"]), False) for r in rows]
    runs += [("teacher", seed, Path(spec["references"]["teachers"][str(seed)]["run"]), True) for seed in (12, 42, 2026)]
    paths = [Path(__file__).resolve(), previous / "results.json", previous / "protocol.json",
             *sorted((ROOT / "src/CESTA").rglob("*.py")), *[p for p in dataset.iterdir() if p.is_file()]]
    paths += [run / name for _, _, run, _ in runs for name in ("weight.pt", "config.json", "manifest.json")]
    hashes = {str(p): sha256_file(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "cohort.npz", indices=indices, y=y, valid=valid, complete=complete, near_transition=near,
                        edge_mask=split.edge_mask)
    write_json(output / "protocol.json", {"split": "train", "window_indices": indices.tolist(), "hashes": hashes,
        "definitions": {"class_order": ["NORMAL", "SPIKE", "DRIFT", "STUCK"],
            "stages": "local argmax; fused argmax before correction; corrected argmax; CRF decode of corrected emissions",
            "near_transition": "Within two timesteps of either side of observed valid label change; cross-window boundaries included",
            "teacher_cohort": "Only node windows with all60 valid labels/node masks; identical graph-derived raw values and indices",
            "requests": ("Receiver mean fraction/probability over possible senders, then mean across eligible timesteps; "
                         "not global edge-weighted ratio"),
            "limitations": ("Frozen validation-selected checkpoints on training data; no new validation/test inference; "
                             "descriptive associations, not causal; nonoverlap is not independence")},
        "cohort": {"graph_windows": len(indices), "valid_timesteps": int(valid.sum()), "complete_node_windows": int(complete.sum()),
                   "near_timesteps": int((near & valid).sum()), "training_bounds": data.split_bounds.get("train")}})
    results = []
    for variant, seed, run, is_teacher in runs:
        model: Any = load_checkpoint(run, map_location="cpu")
        model.eval()
        predictions: dict[str, list[np.ndarray]] = {"teacher": []} if is_teacher else {k: [] for k in ("local", "fused", "corrected", "crf")}
        activities, probabilities = [], []
        loader = make_window_loader(split.X, y, 16, shuffle=False, metadata=data.metadata,
                                    node_mask=split.node_mask, edge_mask=split.edge_mask, seed=0, node_identity_split="train")
        with torch.inference_mode():
            for batch in loader:
                inp, labels, mask, _ = prepare_batch(batch, torch.device("cpu"))
                if not isinstance(inp, GraphWindowBatch) or mask is None:
                    raise ValueError("Expected masked graph batch")
                if is_teacher:
                    b, t, n = labels.shape
                    x = inp.x.reshape(b, t, n, -1).permute(0, 2, 1, 3).reshape(b * n, t, -1)
                    eligible = (mask & (labels >= 0)).all(dim=1).reshape(-1)
                    pred = torch.zeros((b * n, t), dtype=torch.long)
                    if eligible.any():
                        pred[eligible] = model(x[eligible]).argmax(-1)
                    predictions["teacher"].append(pred.reshape(b, n, t).permute(0, 2, 1).numpy())
                else:
                    logits = model(inp)
                    stages = {"local": model.local_logits.argmax(-1), "fused": model.communication_logits.argmax(-1),
                              "corrected": logits.argmax(-1), "crf": model.crf_decode(logits, mask)}
                    for name, pred in stages.items():
                        predictions[name].append(pred.numpy())
                    activities.append(model.communication_activity.numpy())
                    probabilities.append(model.receiver_request_probability.numpy())
        joined = {name: np.concatenate(values) for name, values in predictions.items()}
        activity = np.concatenate(activities) if activities else None
        probability = np.concatenate(probabilities) if probabilities else None
        selected = valid & complete[:, None, :] if is_teacher else valid
        result = {"variant": variant, "seed": seed, "run": str(run),
                  **summarize(y, joined, selected, near, complete, activity, probability)}
        results.append(result)
        np.savez_compressed(output / f"{variant}-seed{seed}.npz", **joined,
                            **({"activity": activity, "probability": probability} if activity is not None else {}))
        write_json(output / "results.json", results)
        print(variant, seed, {stage: item["complete_nodes"]["f1"] for stage, item in result["stages"].items()}, flush=True)
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("Input changed during diagnostic")
    write_json(output / "completion.json", {"complete": True, "all_hashes_verified": True, "training": False,
                                           "validation_inference": False, "test_inference": False})


if __name__ == "__main__":
    main()
