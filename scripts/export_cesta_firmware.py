from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import cast

import torch

from CESTA.artifacts import load_checkpoint_metadata, load_checkpoint_weights
from CESTA.models.spatial.cesta import CESTAClassifier
from CESTA.models.spatial.cesta.deployment import CESTADeploymentModel, CESTALocalDeploymentModel, CESTANodeDeploymentModel

WINDOW_SIZE = 60
SUPPORTED_OPERATORS = {
    "ABS",
    "ADD",
    "BATCH_MATMUL",
    "BROADCAST_TO",
    "CAST",
    "CONCATENATION",
    "DIV",
    "EXP",
    "FULLY_CONNECTED",
    "GATHER",
    "GREATER_EQUAL",
    "LOG",
    "LOGISTIC",
    "MAXIMUM",
    "MINIMUM",
    "MUL",
    "NOT_EQUAL",
    "PACK",
    "REDUCE_MAX",
    "RELU",
    "RESHAPE",
    "REVERSE_V2",
    "SELECT_V2",
    "SLICE",
    "SOFTMAX",
    "SPLIT",
    "SQRT",
    "SUB",
    "SUM",
    "TANH",
    "TRANSPOSE",
    "UNPACK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained CESTA checkpoint for ESP32-S3 TensorFlow Lite Micro inference.")
    parser.add_argument("--model", type=Path, required=True, help="Trained CESTA run directory containing weight.pt and config.json.")
    parser.add_argument("--output", type=Path, default=Path("firmware/model"), help="Directory for model.tflite and model.json.")
    parser.add_argument(
        "--target",
        choices=("node", "local", "graph"),
        default="node",
        help="Export distributed per-node, local-only, or centralized full-graph inference.",
    )
    parser.add_argument("--receiver-index", type=int, help="Graph node index embedded in a distributed node export.")
    parser.add_argument("--validation-artifact", action="store_true", help="Mark an untrained export as conversion validation only.")
    parser.add_argument("--install", action="store_true", help="Install the export into firmware/model after validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = load_cesta_checkpoint(args.model)
    validate_model(model, args.target)

    model.eval()
    node_metadata: dict[str, object] = {}
    if args.target == "node":
        if args.receiver_index is None:
            raise ValueError("node export requires --receiver-index")
        deployment_model = CESTANodeDeploymentModel(model, args.receiver_index).eval()
        example = torch.zeros(1, WINDOW_SIZE, deployment_model.input_width, dtype=torch.float32)
        expected_input = [1, WINDOW_SIZE, deployment_model.input_width]
        expected_output = [1, WINDOW_SIZE, deployment_model.output_width]
        node_metadata = {
            "receiver_index": args.receiver_index,
            "sender_indices": cast(torch.Tensor, deployment_model.sender_indices).tolist(),
            "neighbor_count": deployment_model.neighbor_count,
            "hidden_size": deployment_model.hidden_size,
            "input_layout": ["local_features", "neighbor_hidden_and_features", "possible_mask", "received_mask"],
            "output_layout": ["class_probabilities", "local_hidden", "request_mask"],
        }
    elif args.target == "local":
        deployment_model = CESTALocalDeploymentModel(model).eval()
        example = torch.zeros(1, WINDOW_SIZE, model.features_per_node, dtype=torch.float32)
        expected_input = [1, WINDOW_SIZE, model.features_per_node]
        expected_output = [1, WINDOW_SIZE, model.num_classes]
    else:
        deployment_model = CESTADeploymentModel(model).eval()
        example = torch.zeros(1, WINDOW_SIZE, model.num_nodes, model.features_per_node, dtype=torch.float32)
        expected_input = [1, WINDOW_SIZE, model.num_nodes, model.features_per_node]
        expected_output = [1, WINDOW_SIZE, model.num_nodes, model.num_classes]
    with torch.no_grad():
        reference = deployment_model(example).cpu()

    import litert_torch

    converted = litert_torch.convert(deployment_model, (example,))
    model_content = converted.model_content()
    operators, input_shape, output_shape = inspect_model(model_content)
    unsupported = operators - SUPPORTED_OPERATORS
    if unsupported:
        raise ValueError(f"firmware does not register exported operators: {', '.join(sorted(unsupported))}")
    if input_shape != expected_input or output_shape != expected_output:
        raise ValueError(f"unexpected exported shapes: input={input_shape}, output={output_shape}")
    verify_numerical_parity(model_content, example, reference)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.tflite"
    metadata_path = output / "model.json"
    model_path.write_bytes(model_content)
    metadata = {
        "format": "cesta-tflite-micro-v2",
        "source_checkpoint": None if args.validation_artifact else str(args.model),
        "trained_checkpoint": not args.validation_artifact,
        "target": args.target,
        "window_size": WINDOW_SIZE,
        "num_nodes": model.num_nodes,
        "features_per_node": model.features_per_node,
        "num_classes": model.num_classes,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "communication_mode": model.communication_mode,
        "request_threshold": model.request_threshold,
        "deployment_contract": (
            "receiver-local request, neighbor response, per-timestep classification"
            if args.target == "node"
            else "centralized tensor inference"
        ),
        "operators": sorted(operators),
        **node_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if args.install:
        firmware_model = Path("firmware/model")
        firmware_model.mkdir(parents=True, exist_ok=True)
        if model_path.resolve() != (firmware_model / "model.tflite").resolve():
            shutil.copy2(model_path, firmware_model / "model.tflite")
        if metadata_path.resolve() != (firmware_model / "model.json").resolve():
            shutil.copy2(metadata_path, firmware_model / "model.json")

    print(f"Exported {len(model_content)} bytes to {model_path}")
    return 0


def load_cesta_checkpoint(path: Path) -> CESTAClassifier:
    metadata = load_checkpoint_metadata(path)
    if metadata.get("model_name") != "cesta":
        raise ValueError("checkpoint must contain a CESTA model")
    raw_config = metadata.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint metadata is missing model_config")
    constructor_keys = set(inspect.signature(CESTAClassifier.__init__).parameters)
    model = CESTAClassifier(**{key: value for key, value in raw_config.items() if key in constructor_keys})
    return load_checkpoint_weights(model, path, map_location="cpu")


def validate_model(model: CESTAClassifier, target: str) -> None:
    if model.features_per_node != 1:
        raise ValueError("firmware export requires one feature per node")
    if model.num_classes != 4:
        raise ValueError("firmware export requires NORMAL, SPIKE, DRIFT, and STUCK outputs")
    if model.communication_mode not in {"none", "dense", "gumbel_request"}:
        raise ValueError("firmware export supports none, dense, and gumbel_request communication")
    if target == "node" and model.communication_mode not in {"dense", "gumbel_request"}:
        raise ValueError("distributed node export requires dense or gumbel_request communication")
    if model.structured_request_topk > 0:
        raise ValueError("firmware export does not support structured top-k requests")


def inspect_model(model_content: bytes) -> tuple[set[str], list[int], list[int]]:
    import tflite

    if len(model_content) < 8 or model_content[4:8] != b"TFL3":
        raise ValueError("converter did not produce a TensorFlow Lite FlatBuffer")
    flatbuffer = tflite.Model.GetRootAsModel(model_content, 0)
    operators = {
        tflite.opcode2name(flatbuffer.OperatorCodes(index).BuiltinCode())
        for index in range(flatbuffer.OperatorCodesLength())
    }
    subgraph = flatbuffer.Subgraphs(0)
    if subgraph.InputsLength() != 1 or subgraph.OutputsLength() != 1:
        raise ValueError("firmware model must have exactly one input and one output")
    input_tensor = subgraph.Tensors(subgraph.Inputs(0))
    output_tensor = subgraph.Tensors(subgraph.Outputs(0))
    return operators, input_tensor.ShapeAsNumpy().tolist(), output_tensor.ShapeAsNumpy().tolist()


def verify_numerical_parity(model_content: bytes, example: torch.Tensor, reference: torch.Tensor) -> None:
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], example.numpy())
    interpreter.invoke()
    actual = interpreter.get_tensor(output_details["index"])
    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(main())
