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
from CESTA.models.spatial.hifinet import HiFiNetClassifier
from CESTA.models.spatial.hifinet.deployment import HiFiNetClassifierStage, HiFiNetGraphStage, HiFiNetTemporalStage


def export(args: argparse.Namespace) -> int:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty to avoid mixing stage bundles")
    model = load_deployment_checkpoint(args.model, "hifinet", HiFiNetClassifier)
    interval = validate_checkpoint_recipe(args.model)
    torch.set_num_threads(1)
    temporal = HiFiNetTemporalStage(model).eval()
    classifier = HiFiNetClassifierStage(model).eval()
    graph = [[HiFiNetGraphStage(model, receiver, layer).eval() for receiver in range(model.num_nodes)]
             for layer in range(model.num_gat_layers)]
    stages: dict[str, Any] = {"temporal": temporal, "classifier": classifier}
    for layer, receivers in enumerate(graph):
        for receiver, stage in enumerate(receivers):
            stages[f"graph-{layer}-node-{receiver}"] = stage
    contents, interpreters, stage_metadata = convert_stages(stages, args.validate_only)
    errors = {}

    def invoke(name: str, packed: torch.Tensor, converted: bool) -> torch.Tensor:
        return invoke_stage(stages[name], interpreters[name] if converted else None, packed)

    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for case in range(5):
            features = 22.0 + torch.randn(1, WINDOW_SIZE, model.num_nodes, model.features_per_node, generator=generator)
            if case == 0:
                features.zero_()
            if case == 4:
                features[:, 20:30] += 12.0
                features[:, 40:] = features[:, 39:40]
            mask = torch.ones(1, WINDOW_SIZE, model.edge_index.shape[1], dtype=torch.bool)
            if case == 1:
                mask.zero_()
            if case == 2:
                mask = torch.rand(mask.shape, generator=generator) > 0.5
            batch = GraphWindowBatch(x=features, y=torch.zeros(features.shape[:3], dtype=torch.long),
                                     node_mask=torch.ones(features.shape[:3], dtype=torch.bool),
                                     edge_index=model.edge_index, edge_mask=mask)
            reference = model(batch)
            for converted in ([False] if args.validate_only else [False, True]):
                local = [invoke("temporal", features[:, :, node], converted) for node in range(model.num_nodes)]
                original_local, _ = model.temporal_encoder_1(features.permute(0, 2, 1, 3).reshape(
                    model.num_nodes, WINDOW_SIZE, model.features_per_node))
                original_local, _ = model.temporal_encoder_2(original_local)
                original_hidden = original_local.reshape(1, model.num_nodes, WINDOW_SIZE, -1).permute(0, 2, 1, 3)
                torch.testing.assert_close(torch.stack(local, dim=2), original_hidden, rtol=1e-4, atol=3e-4)
                batched_edges = model._batch_edge_index(model.edge_index, mask, batch_size=1,
                                                       sequence_length=WINDOW_SIZE, device=features.device)
                hidden = local
                for layer, receivers in enumerate(graph):
                    following = []
                    for receiver, stage in enumerate(receivers):
                        payload = torch.cat([hidden[receiver], *[hidden[sender] for sender in stage.sender_indices],
                                             mask[..., stage.edge_indices].float()], dim=-1)
                        following.append(invoke(f"graph-{layer}-node-{receiver}", payload, converted))
                    original_hidden = torch.relu(model.graph_norms[layer](model.graph_layers[layer](
                        original_hidden.reshape(WINDOW_SIZE * model.num_nodes, -1), batched_edges)))
                    original_hidden = original_hidden.reshape(1, WINDOW_SIZE, model.num_nodes, -1)
                    torch.testing.assert_close(torch.stack(following, dim=2), original_hidden, rtol=1e-4, atol=3e-4)
                    hidden = following
                logits = torch.stack([invoke("classifier", torch.cat([local[node], hidden[node]], dim=-1), converted)
                                      for node in range(model.num_nodes)], dim=2)
                torch.testing.assert_close(logits, reference, rtol=1e-4, atol=3e-4 if converted else 1e-5)
                torch.testing.assert_close(logits.argmax(-1), reference.argmax(-1), rtol=0, atol=0)
                key = f"{'tflite' if converted else 'torch'}-case-{case}"
                errors[key] = (logits - reference).abs().max().item()
                print(f"{key}: max logit error {errors[key]}")
    bundle = {
        "format": "hifinet-staged-host-v1", "firmware_supported": False,
        "converted": not args.validate_only, "source_checkpoint": str(args.model.resolve()),
        "source_checkpoint_sha256": hashlib.sha256((args.model / "weight.pt").read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256((args.model / "manifest.json").read_bytes()).hexdigest(),
        "model_config": model.get_config(), "window_size": WINDOW_SIZE, "sample_interval_secs": interval,
        "preprocessing": "raw features; no window standardization; complete consecutive windows only",
        "decoder": "per-timestep argmax of raw class logits; first-index tie breaking; no CRF",
        "exchange_rounds": model.num_gat_layers,
        "round_payload_widths": [model.embedding_size] + [model.gat_hidden_size] * (model.num_gat_layers - 1),
        "graph_input_layout": ["local_previous_layer", "sender_previous_layer_in_edge_order", "received_mask"],
        "classifier_input_layout": ["retained_local_temporal_embedding", "final_graph_embedding"],
        "exchange_contract": "synchronous rounds; same checkpoint/window/timestep/round; no in-place updates; self always available",
        "missing_payload_contract": "zero-fill absent payloads; finite inputs; received masks are binary and graph-edge aligned",
        "receivers": [{"receiver_index": receiver, "sender_indices": graph[0][receiver].sender_indices,
                       "edge_indices": graph[0][receiver].edge_indices} for receiver in range(model.num_nodes)],
        "stages": stage_metadata, "validation_max_logit_errors": errors,
    }
    write_stage_bundle(args.output, contents, bundle)
    return 0


