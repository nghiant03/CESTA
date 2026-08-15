from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from CESTA.artifacts import load_checkpoint_metadata, load_checkpoint_weights
from CESTA.datasets import load_dataset
from CESTA.evaluation import Evaluator
from CESTA.evaluation.logit_sensitivity import (
    build_communication_dependence_audit,
    build_gate_selectivity_audit,
    build_logit_sensitivity_audit,
    validate_thresholds,
)
from CESTA.models import create_model, get_model_class
from CESTA.models.base import BaseModel
from CESTA.schema import EvaluateConfig
from CESTA.schema.fault import FaultType
from CESTA.schema.window import DataConfig

DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit validation prediction and logit sensitivity across request thresholds.")
    parser.add_argument("--model", type=Path, required=True, help="Trained CESTA run directory.")
    parser.add_argument("--data", type=Path, required=True, help="Canonical dataset directory.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS, help="Predefined deterministic threshold grid.")
    parser.add_argument("--reference-threshold", type=float, default=0.50, help="Threshold used as the logit and prediction reference.")
    parser.add_argument("--batch-size", type=int, default=64, help="Validation evaluation batch size.")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device; defaults to automatic selection.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON; defaults to the model run directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = validate_thresholds(args.thresholds, args.reference_threshold)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    dataset = load_dataset(args.data)
    metadata = load_checkpoint_metadata(args.model)
    model_name = metadata.get("model_name")
    train_config = metadata.get("train_config")
    if not isinstance(model_name, str) or not isinstance(train_config, dict):
        raise ValueError("checkpoint metadata must contain model_name and train_config")

    raw_data_config = train_config.get("data")
    data_config = DataConfig.model_validate(raw_data_config) if isinstance(raw_data_config, dict) else DataConfig()
    saved_features = train_config.get("features")
    features = saved_features if isinstance(saved_features, list) and all(isinstance(value, str) for value in saved_features) else None
    model_kwargs = train_config.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}

    requested_metadata = set(get_model_class(model_name).required_metadata)
    if model_kwargs.get("node_embedding_dim", 0):
        requested_metadata.add("node_identity")
    prepared = dataset.prepare(
        window_config=data_config.window,
        split_config=data_config.split,
        features=features,
        required_metadata=requested_metadata,
    )
    if not prepared.has_val:
        raise ValueError("dataset has no validation split")

    model = create_model(
        model_name,
        input_size=prepared.input_size,
        num_classes=FaultType.count(),
        metadata=prepared.metadata,
        **model_kwargs,
    )
    if not isinstance(model, BaseModel):
        raise TypeError("created model does not implement BaseModel")
    load_checkpoint_weights(model, args.model, map_location="cpu")
    if not hasattr(model, "request_threshold"):
        raise ValueError("model does not expose request_threshold")

    evaluator = Evaluator(EvaluateConfig(batch_size=args.batch_size, split="val"), device=args.device)
    results = {}
    for threshold in thresholds:
        setattr(model, "request_threshold", threshold)
        results[threshold] = evaluator.evaluate(
            model,
            prepared.val.X,
            prepared.val.y,
            metadata=prepared.metadata,
            node_mask=prepared.val.node_mask,
            edge_mask=prepared.val.edge_mask,
            split="val",
        )

    audit = build_logit_sensitivity_audit(results, reference_threshold=args.reference_threshold)
    original_mode = getattr(model, "communication_mode", None)
    original_threshold = float(getattr(model, "request_threshold"))
    if original_mode != "gumbel_request":
        raise ValueError("communication-dependence audit requires a learned Gumbel checkpoint")
    intervention_results = {"learned": results[args.reference_threshold]}
    try:
        for name, mode in (("zero", "none"), ("dense", "dense")):
            setattr(model, "communication_mode", mode)
            intervention_results[name] = evaluator.evaluate(
                model,
                prepared.val.X,
                prepared.val.y,
                metadata=prepared.metadata,
                node_mask=prepared.val.node_mask,
                edge_mask=prepared.val.edge_mask,
                split="val",
            )
    finally:
        setattr(model, "communication_mode", original_mode)
        setattr(model, "request_threshold", original_threshold)
    audit["communication_dependence"] = build_communication_dependence_audit(intervention_results)
    probabilities, possible_masks = _collect_gate_probabilities(model, evaluator, prepared)
    if probabilities:
        graph = prepared.metadata.get("graph")
        edge_distances = getattr(graph, "edge_distance_m", None)
        edge_probabilities = getattr(graph, "edge_prob", None)
        if edge_distances is None or edge_probabilities is None:
            raise ValueError("gate selectivity audit requires graph edge distances and probabilities")
        audit["gate_selectivity"] = build_gate_selectivity_audit(
            np.concatenate(probabilities),
            np.concatenate(possible_masks),
            edge_distances,
            edge_probabilities,
            edge_index=getattr(graph, "edge_index", None),
        )
    audit["model"] = str(args.model)
    audit["data"] = str(args.data)
    output = args.output or args.model / "validation_logit_sensitivity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2))
    print(f"Validation logit sensitivity: {len(thresholds)} thresholds written to {output}")
    return 0


def _collect_gate_probabilities(model: BaseModel, evaluator: Evaluator, prepared: object) -> tuple[list[np.ndarray], list[np.ndarray]]:
    from CESTA.training.batch_utils import make_window_loader, prepare_batch

    if not hasattr(model, "_edge_gate_features"):
        return [], []
    loader = make_window_loader(
        prepared.val.X,
        prepared.val.y,
        evaluator.config.batch_size,
        shuffle=False,
        metadata=prepared.metadata,
        node_mask=prepared.val.node_mask,
        edge_mask=prepared.val.edge_mask,
        seed=0,
        node_identity_split="val",
    )
    model = model.to(evaluator.device)
    model.eval()
    probabilities = []
    possible_masks = []
    with torch.no_grad():
        for batch in loader:
            model_input, _, _, _ = prepare_batch(batch, evaluator.device)
            model(model_input)
            local_logits = getattr(model, "local_logits", None)
            if not isinstance(local_logits, torch.Tensor):
                continue
            local_hidden = model.temporal_encoder(
                model_input.x.permute(0, 2, 1, 3).reshape(-1, model_input.x.shape[1], model_input.x.shape[-1])
            )[0]
            local_hidden = local_hidden.view(model_input.x.shape[0], model.num_nodes, model_input.x.shape[1], -1).permute(0, 2, 1, 3)
            possible = model._possible_message_mask(local_hidden, edge_index=model_input.edge_index, edge_mask=model_input.edge_mask)
            features = model._edge_gate_features(local_hidden, possible, edge_index=model_input.edge_index)
            probability = torch.softmax(model.request_gate(features), dim=-1)[..., 1] * possible
            probabilities.append(probability.cpu().numpy())
            possible_masks.append(possible.bool().cpu().numpy())
    return probabilities, possible_masks


if __name__ == "__main__":
    raise SystemExit(main())
