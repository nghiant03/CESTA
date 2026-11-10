from __future__ import annotations

import argparse
import hashlib
from typing import Any

import torch

from CESTA.batch import GraphWindowBatch
from CESTA.deployment.common import (
    WINDOW_SIZE,
    convert_stages,
    invoke_stage,
    load_deployment_checkpoint,
    validate_checkpoint_recipe,
    write_stage_bundle,
)
from CESTA.models.spatial.stgcn import STGCNClassifier
from CESTA.models.spatial.stgcn.deployment import STGCNClassifierStage, STGCNFinishStage, STGCNGraphStage, STGCNPrepareStage


def export(args: argparse.Namespace) -> int:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty to avoid mixing stage bundles")
    model = load_deployment_checkpoint(args.model, "stgcn", STGCNClassifier)
    interval = validate_checkpoint_recipe(args.model)
    torch.set_num_threads(1)
    classifier = STGCNClassifierStage(model).eval()
    graph = [[STGCNGraphStage(model, receiver, layer).eval() for receiver in range(model.num_nodes)] for layer in range(model.num_st_blocks)]
    stages: dict[str, Any] = {"classifier": classifier}
    for layer, receivers in enumerate(graph):
        stages[f"prepare-{layer}"] = STGCNPrepareStage(model, layer).eval()
        stages[f"finish-{layer}"] = STGCNFinishStage(model, layer).eval()
        for receiver, stage in enumerate(receivers):
            stages[f"graph-{layer}-node-{receiver}"] = stage
    contents, interpreters, stage_metadata = convert_stages(stages, args.validate_only)
    errors = {}

    def invoke(name: str, packed: torch.Tensor, converted: bool) -> torch.Tensor:
        return invoke_stage(stages[name], interpreters[name] if converted else None, packed)

    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for case in range(8):
            features = 22.0 + torch.randn(1, WINDOW_SIZE, model.num_nodes, model.features_per_node, generator=generator)
            if case == 0:
                features.zero_()
            if case == 4:
                features[:, 20:30] += 12.0
                features[:, 40:] = features[:, 39:40]
            mask = torch.ones(1, WINDOW_SIZE, model.edge_index.shape[1], dtype=torch.bool)
            if case in (1, 5):
                mask.zero_()
            if case in (2, 6):
                mask = torch.rand(mask.shape, generator=generator) > 0.5
            batch = GraphWindowBatch(
                x=features,
                y=torch.zeros(features.shape[:3], dtype=torch.long),
                node_mask=torch.ones(features.shape[:3], dtype=torch.bool),
                edge_index=model.edge_index,
                edge_mask=mask,
            )
            reference = model(batch)
            if case >= 5:
                degree = torch.ones(model.num_nodes)
                for source, target in model.edge_index.T.tolist():
                    if source != target:
                        degree[target] += 1
                native = features.permute(0, 3, 2, 1)
                for block in model.blocks:
                    residual = block.residual(native)
                    temporal = block.temporal1(native).relu().permute(0, 3, 2, 1)
                    projected = block.gcn.lin(temporal)
                    spatial = projected / degree.reshape(1, 1, -1, 1)
                    for edge, (source, target) in enumerate(model.edge_index.T.tolist()):
                        if source != target:
                            spatial[:, :, target] += projected[:, :, source] * mask[..., edge : edge + 1] / (degree[source] * degree[target]).sqrt()
                    spatial = (spatial + block.gcn.bias).relu().permute(0, 3, 2, 1)
                    native = block.temporal2(spatial).relu() + residual
                reference = model.fc(native.permute(0, 3, 2, 1))
            for converted in [False] if args.validate_only else [False, True]:
                degrees = torch.ones(1, WINDOW_SIZE, model.num_nodes)
                for edge, (source, target) in enumerate(model.edge_index.T.tolist()):
                    if source != target:
                        degrees[..., target] += 1 if case >= 5 else mask[..., edge]
                inverse = degrees.rsqrt()
                hidden = [features[:, :, node] for node in range(model.num_nodes)]
                for layer, receivers in enumerate(graph):
                    prepared = [invoke(f"prepare-{layer}", value, converted) for value in hidden]
                    following = []
                    for receiver, stage in enumerate(receivers):
                        indices = [receiver, *stage.sender_indices]
                        payload = torch.cat(
                            [*[prepared[node] for node in indices], inverse[..., indices], mask[..., stage.edge_indices].float()], dim=-1
                        )
                        spatial = invoke(f"graph-{layer}-node-{receiver}", payload, converted)
                        following.append(invoke(f"finish-{layer}", torch.cat([hidden[receiver], spatial], -1), converted))
                    hidden = following
                logits = torch.stack([invoke("classifier", value, converted) for value in hidden], dim=2)
                torch.testing.assert_close(logits, reference, rtol=1e-4, atol=3e-4)
                torch.testing.assert_close(logits.argmax(-1), reference.argmax(-1), rtol=0, atol=0)
                key = f"{'tflite' if converted else 'torch'}-case-{case}"
                errors[key] = (logits - reference).abs().max().item()
                print(f"{key}: max logit error {errors[key]}")
    bundle = {
        "normalization_policy": "fixed_full_topology",
        "missing_payload_policy": "zero_mask_fixed_normalization_approximation",
        "format": "stgcn-staged-host-v1",
        "firmware_supported": True,
        "converted": not args.validate_only,
        "source_checkpoint": str(args.model.resolve()),
        "source_checkpoint_sha256": hashlib.sha256((args.model / "weight.pt").read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256((args.model / "manifest.json").read_bytes()).hexdigest(),
        "model_config": model.get_config(),
        "window_size": WINDOW_SIZE,
        "sample_interval_secs": interval,
        "preprocessing": "raw features; no window standardization; complete consecutive windows only",
        "decoder": "per-timestep argmax of raw class logits; first-index tie breaking; no CRF",
        "exchange_rounds": model.num_st_blocks,
        "round_payload_widths": [model.st_hidden_size] * model.num_st_blocks,
        "graph_input_layout": ["local_temporal", "sender_temporal_in_edge_order", "local_and_sender_inverse_sqrt_indegree", "received_mask"],
        "classifier_input_layout": ["final_block_embedding"],
        "exchange_contract": "synchronous rounds; same checkpoint/window/timestep/round; no in-place updates; self always available",
        "missing_payload_contract": (
            "masks define graph; inverse degrees include self and all active incoming edges at each source; "
            "unknown degrees cannot claim native parity"
        ),
        "receivers": [
            {"receiver_index": receiver, "sender_indices": graph[0][receiver].sender_indices, "edge_indices": graph[0][receiver].edge_indices}
            for receiver in range(model.num_nodes)
        ],
        "stages": stage_metadata,
        "validation_max_logit_errors": errors,
    }
    write_stage_bundle(args.output, contents, bundle)
    return 0


