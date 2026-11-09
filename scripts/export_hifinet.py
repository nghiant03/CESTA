from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from export_cesta_firmware import SUPPORTED_OPERATORS, WINDOW_SIZE, inspect_model, split_large_concatenations, validate_checkpoint_recipe

from CESTA.artifacts import load_checkpoint_metadata, load_checkpoint_weights
from CESTA.batch import GraphWindowBatch
from CESTA.models.spatial.hifinet import HiFiNetClassifier
from CESTA.models.spatial.hifinet.deployment import HiFiNetClassifierStage, HiFiNetGraphStage, HiFiNetTemporalStage


def main() -> int:
    parser = argparse.ArgumentParser(description="Export parity-validated HiFiNet stages for host inference and firmware bundle installation.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="Validate PyTorch stages without converting to TFLite")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty to avoid mixing stage bundles")
    metadata = load_checkpoint_metadata(args.model)
    if metadata.get("model_name") != "hifinet" or not isinstance(metadata.get("model_config"), dict):
        raise ValueError("expected a HiFiNet checkpoint with model_config")
    model = load_checkpoint_weights(HiFiNetClassifier(**metadata["model_config"]), args.model, map_location="cpu").eval()
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("checkpoint contains nonfinite weights")
    interval = validate_checkpoint_recipe(args.model)
    torch.set_num_threads(1)
    temporal = HiFiNetTemporalStage(model).eval()
    classifier = HiFiNetClassifierStage(model).eval()
    graph = [[HiFiNetGraphStage(model, receiver, layer).eval() for receiver in range(model.num_nodes)]
             for layer in range(model.num_gat_layers)]
    stages = {"temporal": temporal, "classifier": classifier}
    for layer, receivers in enumerate(graph):
        for receiver, stage in enumerate(receivers):
            stages[f"graph-{layer}-node-{receiver}"] = stage
    contents = {}
    interpreters = {}
    stage_metadata = {}
    if not args.validate_only:
        import litert_torch
        from ai_edge_litert.interpreter import Interpreter

        for name, stage in stages.items():
            example = torch.zeros(1, WINDOW_SIZE, stage.input_width)
            original = litert_torch.convert(stage, (example,)).model_content()
            content, splits = split_large_concatenations(original)
            operators, input_shape, output_shape = inspect_model(content)
            if operators - SUPPORTED_OPERATORS:
                raise ValueError(f"{name}: operators absent from the current Micro resolver: {sorted(operators - SUPPORTED_OPERATORS)}")
            if input_shape != list(example.shape) or output_shape != [1, WINDOW_SIZE, stage.output_width]:
                raise ValueError(f"{name}: unexpected stage shapes {input_shape}, {output_shape}")
            interpreter = Interpreter(model_content=content)
            interpreter.allocate_tensors()
            interpreters[name] = interpreter
            contents[name] = content
            stage_metadata[name] = {
                "file": f"{name}.tflite", "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                "input_shape": input_shape, "output_shape": output_shape, "operators": sorted(operators),
                "split_concatenations": splits, "missing_micro_operators": sorted(operators - SUPPORTED_OPERATORS),
            }
    errors = {}

    def invoke(name: str, packed: torch.Tensor, converted: bool) -> torch.Tensor:
        expected = stages[name](packed)
        if not converted:
            return expected
        interpreter = interpreters[name]
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], packed.contiguous().numpy())
        interpreter.invoke()
        actual = torch.from_numpy(interpreter.get_tensor(interpreter.get_output_details()[0]["index"]))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-4)
        return actual

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
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (args.output / f"{name}.tflite").write_bytes(content)
    (args.output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
