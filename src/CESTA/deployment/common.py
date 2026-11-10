from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from CESTA.artifacts import load_checkpoint_metadata

WINDOW_SIZE = 60
# esp-tflite-micro 1.3.5 kernels/concatenation.cc uses fixed arrays of this size.
MAX_CONCATENATION_INPUTS = 10
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
    "MEAN",
    "MINIMUM",
    "MUL",
    "NOT_EQUAL",
    "PACK",
    "REDUCE_MAX",
    "RELU",
    "RESHAPE",
    "REVERSE_V2",
    "RSQRT",
    "SELECT_V2",
    "SLICE",
    "SOFTMAX",
    "SPLIT",
    "SQRT",
    "SQUARED_DIFFERENCE",
    "SUB",
    "SUM",
    "TANH",
    "TRANSPOSE",
    "UNPACK",
}


def validate_checkpoint_recipe(path: Path) -> int:
    metadata: Any = load_checkpoint_metadata(path)
    manifest = json.loads((path / "manifest.json").read_text())
    for source in (metadata, manifest):
        window = source.get("train_config", {}).get("data", {}).get("window", {}).get("window_size")
        if window != WINDOW_SIZE:
            raise ValueError(f"checkpoint configuration and manifest must explicitly declare window_size={WINDOW_SIZE}")
    frequency = manifest.get("injection_config", {}).get("resample_freq")
    if frequency not in {"1min", "60s", "60sec"}:
        raise ValueError(f"node firmware requires a verified 60-second resample recipe, got {frequency!r}")
    return 60


def split_large_concatenations(model_content: bytes) -> tuple[bytes, int]:
    """Lower wide float concatenations to an ordered tree after converter fusion.

    Rewriting the final FlatBuffer prevents the converter from fusing the tree
    back into the 60-input concatenations produced by the unrolled RNN.
    Existing output tensor IDs (including signature references) stay intact.
    """
    flatbuffers = import_module("flatbuffers")
    schema = import_module("ai_edge_litert.schema_py_generated")

    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model_content, 0))
    rewritten = 0
    for graph in model.subgraphs:
        operators = []
        for operator in graph.operators:
            code = model.operatorCodes[operator.opcodeIndex]
            if code.builtinCode != schema.BuiltinOperator.CONCATENATION or len(operator.inputs) <= MAX_CONCATENATION_INPUTS:
                operators.append(operator)
                continue
            output = graph.tensors[operator.outputs[0]]
            if output.type != schema.TensorType.FLOAT32 or any(graph.tensors[index].type != output.type for index in operator.inputs):
                raise ValueError("splitting concatenations currently requires float32 tensors")
            axis = operator.builtinOptions.axis % len(output.shape)
            inputs = list(operator.inputs)
            while len(inputs) > MAX_CONCATENATION_INPUTS:
                grouped = []
                for start in range(0, len(inputs), MAX_CONCATENATION_INPUTS):
                    chunk = inputs[start : start + MAX_CONCATENATION_INPUTS]
                    if len(chunk) == 1:
                        grouped.extend(chunk)
                        continue
                    tensor = deepcopy(output)
                    tensor.name = f"cesta_concat_{len(graph.tensors)}".encode()
                    tensor.shape = list(output.shape)
                    tensor.shape[axis] = sum(int(graph.tensors[index].shape[axis]) for index in chunk)
                    tensor.shapeSignature = None
                    tensor.buffer = 0
                    tensor_index = len(graph.tensors)
                    graph.tensors.append(tensor)
                    intermediate = deepcopy(operator)
                    intermediate.inputs = chunk
                    intermediate.outputs = [tensor_index]
                    # Apply a fused activation only once, at the original output.
                    intermediate.builtinOptions.fusedActivationFunction = schema.ActivationFunctionType.NONE
                    operators.append(intermediate)
                    grouped.append(tensor_index)
                inputs = grouped
            operator.inputs = inputs
            operators.append(operator)
            rewritten += 1
        graph.operators = operators
    if not rewritten:
        return model_content, 0
    builder = flatbuffers.Builder(len(model_content))
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output()), rewritten


def inspect_model(model_content: bytes) -> tuple[set[str], list[int], list[int]]:
    tflite = import_module("tflite")

    if len(model_content) < 8 or model_content[4:8] != b"TFL3":
        raise ValueError("converter did not produce a TensorFlow Lite FlatBuffer")
    flatbuffer = tflite.Model.GetRootAsModel(model_content, 0)
    operators = {
        tflite.opcode2name(flatbuffer.OperatorCodes(index).BuiltinCode())
        for index in range(flatbuffer.OperatorCodesLength())
    }
    subgraph = flatbuffer.Subgraphs(0)
    for graph_index in range(flatbuffer.SubgraphsLength()):
        graph = flatbuffer.Subgraphs(graph_index)
        for index in range(graph.OperatorsLength()):
            operator = graph.Operators(index)
            code = flatbuffer.OperatorCodes(operator.OpcodeIndex()).BuiltinCode()
            if code == tflite.BuiltinOperator.CONCATENATION and operator.InputsLength() > MAX_CONCATENATION_INPUTS:
                raise ValueError(f"subgraph {graph_index} concatenation {index} exceeds the TFLite Micro input limit")
    if subgraph.InputsLength() != 1 or subgraph.OutputsLength() != 1:
        raise ValueError("firmware model must have exactly one input and one output")
    input_tensor = subgraph.Tensors(subgraph.Inputs(0))
    output_tensor = subgraph.Tensors(subgraph.Outputs(0))
    return operators, input_tensor.ShapeAsNumpy().tolist(), output_tensor.ShapeAsNumpy().tolist()


def convert_stage(stage: torch.nn.Module, example: torch.Tensor, output_shape: list[int]) -> tuple[bytes, bytes, dict[str, Any]]:
    original = import_module("litert_torch").convert(stage, (example,)).model_content()
    content, splits = split_large_concatenations(original)
    operators, actual_input, actual_output = inspect_model(content)
    if operators - SUPPORTED_OPERATORS:
        raise ValueError(f"operators absent from the Micro resolver: {sorted(operators - SUPPORTED_OPERATORS)}")
    if actual_input != list(example.shape) or actual_output != output_shape:
        raise ValueError(f"unexpected stage shapes: {actual_input}, {actual_output}")
    return original, content, {
        "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
        "input_shape": actual_input, "output_shape": actual_output, "operators": sorted(operators),
        "split_concatenations": splits, "missing_micro_operators": [],
    }


def convert_stages(stages: dict[str, Any], validate_only: bool) -> tuple[dict, dict, dict]:
    contents, interpreters, metadata = {}, {}, {}
    if not validate_only:
        for name, stage in stages.items():
            example = torch.zeros(1, WINDOW_SIZE, stage.input_width)
            _, content, details = convert_stage(stage, example, [1, WINDOW_SIZE, stage.output_width])
            interpreter = import_module("ai_edge_litert.interpreter").Interpreter(model_content=content)
            interpreter.allocate_tensors()
            contents[name], interpreters[name] = content, interpreter
            metadata[name] = {"file": f"{name}.tflite", **details}
    return contents, interpreters, metadata


def invoke_stage(stage: torch.nn.Module, interpreter: Any, packed: torch.Tensor) -> torch.Tensor:
    expected = stage(packed)
    if interpreter is None:
        return expected
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], packed.contiguous().numpy())
    interpreter.invoke()
    actual = torch.from_numpy(interpreter.get_tensor(interpreter.get_output_details()[0]["index"]))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-4)
    return actual


def load_deployment_checkpoint(path: Path, architecture: str, constructor: Any) -> Any:
    from CESTA.artifacts import load_checkpoint_weights

    metadata = load_checkpoint_metadata(path)
    config = metadata.get("model_config")
    if metadata.get("model_name") != architecture or not isinstance(config, dict):
        raise ValueError(f"expected a {architecture} checkpoint with model_config")
    descriptive = getattr(constructor, "checkpoint_metadata_fields", frozenset())
    model = constructor(**{key: value for key, value in config.items() if key not in descriptive})
    model = load_checkpoint_weights(model, path, map_location="cpu").eval()
    reconstructed = model.get_config()
    for key in descriptive:
        if key in config and config[key] != reconstructed.get(key):
            raise ValueError(f"checkpoint descriptive metadata mismatch: {key}")
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("checkpoint contains nonfinite weights")
    if not torch.equal(model.edge_index, torch.tensor(config["edge_index"], dtype=torch.long)):
        raise ValueError("checkpoint topology differs from architecture config")
    return model


def write_stage_bundle(output: Path, contents: dict[str, bytes], metadata: dict) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty to avoid mixing stage bundles")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (output / f"{name}.tflite").write_bytes(content)
    (output / "bundle.json").write_text(json.dumps(metadata, indent=2) + "\n")
