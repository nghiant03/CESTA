from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from importlib import import_module
from pathlib import Path


def inspect_flatbuffer(data: bytes) -> tuple[list[int], list[int], set[str]]:
    tflite = import_module("tflite")

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    try:
        require(data[4:8] == b"TFL3", "invalid TFLite identifier")
        model = tflite.Model.GetRootAsModel(data, 0)
        require(model.Version() == 3 and model.SubgraphsLength() == 1, "unsupported TFLite schema/subgraphs")
        graph = model.Subgraphs(0)
        require(graph.InputsLength() == graph.OutputsLength() == 1, "one input/output required")
        shapes = []
        for index in (graph.Inputs(0), graph.Outputs(0)):
            require(0 <= index < graph.TensorsLength(), "invalid tensor index")
            tensor = graph.Tensors(index)
            require(tensor.Type() == tflite.TensorType.FLOAT32 and not tensor.IsVariable(), "float32 nonvariable I/O required")
            shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
            require(len(shape) == 3 and all(dim > 0 for dim in shape), "static rank-three I/O required")
            signature = [tensor.ShapeSignature(i) for i in range(tensor.ShapeSignatureLength())]
            require(not signature or signature == shape, "dynamic I/O shape signature unsupported")
            shapes.append(shape)
        names = {value: name for name, value in vars(tflite.BuiltinOperator).items() if isinstance(value, int)}
        operators = set()
        for index in range(graph.OperatorsLength()):
            operator = graph.Operators(index)
            require(operator.OpcodeIndex() < model.OperatorCodesLength(), "invalid opcode index")
            code = model.OperatorCodes(operator.OpcodeIndex())
            builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
            require(builtin in names and builtin != tflite.BuiltinOperator.CUSTOM, "unknown/custom operator")
            operators.add(names[builtin])
        require(bool(operators), "empty stage graph")
        return shapes[0], shapes[1], operators
    except (IndexError, TypeError, OverflowError) as error:
        raise ValueError("malformed TFLite structure") from error


def validate_staged(bundle: Path) -> dict:
    metadata = read_metadata(bundle)

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    stgcn = metadata["format"] == "stgcn-staged-host-v1"
    require(metadata["format"] in {"stgcn-staged-host-v1", "hifinet-staged-host-v1"} and metadata["converted"] is True,
            "converted staged bundle required")
    if stgcn:
        require(metadata["normalization_policy"] == "fixed_full_topology", "unsupported normalization")
        require(metadata["missing_payload_policy"] == "zero_mask_fixed_normalization_approximation", "unsupported missing payload policy")
    config = metadata["model_config"]
    require(metadata["window_size"] == 60 and metadata["sample_interval_secs"] == 60, "unsupported cadence/window")
    require(config["num_st_blocks" if stgcn else "num_gat_layers"] == metadata["exchange_rounds"] == (2 if stgcn else 3), "invalid round count")
    require(config["input_size"] == config["num_nodes"] and config["num_classes"] == 4, "one temperature feature and four classes required")
    require(metadata["preprocessing"] == "raw features; no window standardization; complete consecutive windows only", "unsupported preprocessing")
    require(metadata["decoder"] == "per-timestep argmax of raw class logits; first-index tie breaking; no CRF", "unsupported decoder")
    require(
        metadata["graph_input_layout"]
        == (["local_temporal", "sender_temporal_in_edge_order", "local_and_sender_inverse_sqrt_indegree", "received_mask"] if stgcn
            else ["local_previous_layer", "sender_previous_layer_in_edge_order", "received_mask"]),
        "graph layout mismatch",
    )
    require(metadata["classifier_input_layout"] == (["final_block_embedding"] if stgcn
            else ["retained_local_temporal_embedding", "final_graph_embedding"]), "classifier layout mismatch")
    widths = [config["st_hidden"]] * 2 if stgcn else [config["embedding_size"], config["gat_hidden_size"], config["gat_hidden_size"]]
    require(metadata["round_payload_widths"] == widths and all(type(w) is int and 0 < w <= 256 for w in widths), "invalid trained widths")
    for key in ("source_checkpoint_sha256", "source_config_sha256", "source_manifest_sha256"):
        require(len(metadata[key]) == 64 and len(bytes.fromhex(metadata[key])) == 32, "invalid provenance hash")
    errors = metadata["validation_max_logit_errors"]
    require(all(f"tflite-case-{i}" in errors for i in range(8 if stgcn else 5)), "missing TFLite parity validation")
    require(all(math.isfinite(v) and 0 <= v <= 0.001 for v in errors.values()), "invalid parity results")
    source, target = config["edge_index"]
    require(len(source) == len(target) and len(set(zip(source, target))) == len(source), "invalid graph edges")
    require(all(type(n) is int and 0 <= n < config["num_nodes"] for n in source + target), "invalid graph node")
    if stgcn:
        expected = {"classifier": (widths[0], 4)}
        require(config["st_hidden"] == 32 and config["temporal_kernel"] == 3, "unsupported architecture")
        for layer in range(2):
            original = 1 if layer == 0 else widths[0]
            expected[f"prepare-{layer}"] = (original, widths[0])
            expected[f"finish-{layer}"] = (original + widths[0], widths[0])
    else:
        expected = {"temporal": (1, widths[0]), "classifier": (widths[0] + widths[1], 4)}
    require(len(metadata["receivers"]) == config["num_nodes"], "receiver count mismatch")
    for receiver, entry in enumerate(metadata["receivers"]):
        indices = [i for i, (s, t) in enumerate(zip(source, target)) if t == receiver and s != receiver]
        require(
            entry == {"receiver_index": receiver, "sender_indices": [source[i] for i in indices], "edge_indices": indices},
            "receiver topology/order mismatch",
        )
        require(all((receiver, s) in set(zip(source, target)) for s in entry["sender_indices"]), "radio requires symmetric peer topology")
        for layer, width in enumerate(widths):
            expected[f"graph-{layer}-node-{receiver}"] = ((1 + len(indices)) * (width + int(stgcn)) + len(indices), width if stgcn else widths[1])
    require(set(metadata["stages"]) == set(expected), "missing or extra stages")
    supported = {
        "GREATER_EQUAL",
        "SQRT",
        "MINIMUM",
        "MUL",
        "MEAN",
        "BROADCAST_TO",
        "CAST",
        "RSQRT",
        "DIV",
        "GATHER",
        "REDUCE_MAX",
        "UNPACK",
        "TRANSPOSE",
        "SPLIT",
        "ADD",
        "EXP",
        "CONCATENATION",
        "REVERSE_V2",
        "RESHAPE",
        "SUM",
        "SLICE",
        "BATCH_MATMUL",
        "PACK",
        "RELU",
        "LOGISTIC",
        "SQUARED_DIFFERENCE",
        "TANH",
        "LOG",
        "FULLY_CONNECTED",
        "NOT_EQUAL",
        "MAXIMUM",
        "ABS",
        "SOFTMAX",
        "SELECT_V2",
        "SUB",
    }
    for name, (incoming, outgoing) in expected.items():
        stage = metadata["stages"][name]
        require(stage["file"] == f"{name}.tflite", "unsafe stage filename")
        data = (bundle / stage["file"]).read_bytes()
        require(
            data[4:8] == b"TFL3" and len(data) == stage["bytes"] and hashlib.sha256(data).hexdigest() == stage["sha256"],
            f"stage integrity failure: {name}",
        )
        require(stage["input_shape"] == [1, 60, incoming] and stage["output_shape"] == [1, 60, outgoing], "stage shape mismatch")
        require((not stage["missing_micro_operators"] if stgcn else set(stage["missing_micro_operators"]) <= supported)
                and set(stage["operators"]) <= supported, "unsupported operators")
        actual_input, actual_output, actual_operators = inspect_flatbuffer(data)
        require(actual_input == stage["input_shape"] and actual_output == stage["output_shape"], "actual tensor shape mismatch")
        require(actual_operators == set(stage["operators"]) and actual_operators <= supported, "actual operator mismatch")
    return metadata


def stgcn_fingerprint(metadata: dict) -> str:
    keys = (
        "normalization_policy",
        "missing_payload_policy",
        "format",
        "model_config",
        "window_size",
        "sample_interval_secs",
        "preprocessing",
        "decoder",
        "exchange_rounds",
        "round_payload_widths",
        "graph_input_layout",
        "classifier_input_layout",
        "exchange_contract",
        "missing_payload_contract",
        "receivers",
        "source_checkpoint_sha256",
        "source_config_sha256",
        "source_manifest_sha256",
    )
    contract = {key: metadata[key] for key in keys}
    source, target = metadata["model_config"]["edge_index"]
    degrees = [1 + sum(s != t and t == node for s, t in zip(source, target)) for node in range(metadata["model_config"]["num_nodes"])]
    contract["inverse_degrees"] = [
        [degrees[n] ** -0.5 for n in [entry["receiver_index"], *entry["sender_indices"]]] for entry in metadata["receivers"]
    ]
    contract["wire_version"] = 5
    contract["stages"] = {
        name: {key: stage[key] for key in ("sha256", "input_shape", "output_shape")} for name, stage in sorted(metadata["stages"].items())
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hifinet_fingerprint(metadata: dict) -> str:
    keys = (
        "format",
        "model_config",
        "window_size",
        "sample_interval_secs",
        "preprocessing",
        "decoder",
        "exchange_rounds",
        "round_payload_widths",
        "graph_input_layout",
        "classifier_input_layout",
        "exchange_contract",
        "missing_payload_contract",
        "receivers",
        "source_checkpoint_sha256",
        "source_config_sha256",
        "source_manifest_sha256",
    )
    contract = {key: metadata[key] for key in keys}
    contract["wire_version"] = 4
    contract["stages"] = {
        name: {key: stage[key] for key in ("sha256", "input_shape", "output_shape")} for name, stage in sorted(metadata["stages"].items())
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_metadata(bundle: Path) -> dict:
    path = bundle / "bundle.json"
    if not path.exists():
        path = bundle / "model.json"
    return json.loads(path.read_text())


def validate_cesta(bundle: Path) -> dict:
    metadata = read_metadata(bundle)
    if metadata["format"] != "cesta-tflite-micro-v3" or metadata["target"] != "node" or not metadata.get("converted", True):
        raise ValueError("converted CESTA node bundle required")
    if metadata["window_size"] != 60 or metadata["sample_interval_secs"] != 60 or metadata["num_classes"] != 4:
        raise ValueError("unsupported CESTA cadence/classes")
    if metadata["trained_checkpoint"] is not True or metadata["temporal_backend"] != "gru":
        raise ValueError("trained GRU checkpoint required")
    if metadata["gru_bidirectional"] is not True or metadata["gru_num_layers"] != 2 or metadata["features_per_node"] != 1:
        raise ValueError("unsupported CESTA encoder")
    if metadata["output_layout"] != ["class_logits", "local_hidden", "request_mask"]:
        raise ValueError("unsupported output layout")
    if metadata["communication_mode"] not in {"dense", "gumbel_request"} or not 0 <= metadata["request_threshold"] <= 1:
        raise ValueError("unsupported communication contract")
    senders, receiver = metadata["sender_indices"], metadata["receiver_index"]
    neighbors, width = metadata["neighbor_count"], metadata["hidden_size"]
    if neighbors != len(senders) or neighbors == 0 or width <= 0 or width % 2:
        raise ValueError("invalid hidden width or neighbors")
    valid_nodes = all(type(n) is int and 0 <= n < metadata["num_nodes"] for n in [receiver, *senders])
    if len(set(senders)) != neighbors or receiver in senders or not valid_nodes:
        raise ValueError("invalid receiver topology")
    if metadata["input_shape"] != [1, 60, 1 + neighbors * (width + 3)] or metadata["output_shape"] != [1, 60, 4 + width + neighbors]:
        raise ValueError("invalid CESTA shapes")
    for key in ("source_checkpoint_sha256", "source_config_sha256"):
        if len(metadata[key]) != 64 or len(bytes.fromhex(metadata[key])) != 32:
            raise ValueError("invalid provenance hash")
    if metadata["use_crf"]:
        transitions = metadata["crf_transitions"]
        if len(transitions) != 4 or any(len(row) != 4 or any(not math.isfinite(v) for v in row) for row in transitions):
            raise ValueError("invalid CRF transitions")
    if "stages" not in metadata:
        if (bundle / "bundle.json").exists():
            raise ValueError("canonical bundle requires stage integrity metadata")
        data = (bundle / "model.tflite").read_bytes()
        metadata["stages"] = {"node": {
            "file": "model.tflite", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "input_shape": metadata["input_shape"], "output_shape": metadata["output_shape"], "operators": metadata["operators"],
        }}
    if set(metadata["stages"]) != {"node"}:
        raise ValueError("CESTA node stage required")
    stage = metadata["stages"]["node"]
    if stage["file"] != ("node.tflite" if (bundle / "bundle.json").exists() else "model.tflite"):
        raise ValueError("unsafe stage filename")
    data = (bundle / stage["file"]).read_bytes()
    if len(data) != stage["bytes"] or hashlib.sha256(data).hexdigest() != stage["sha256"]:
        raise ValueError("stage integrity failure")
    incoming, outgoing, operators = inspect_flatbuffer(data)
    if incoming != metadata["input_shape"] or outgoing != metadata["output_shape"] or operators != set(metadata["operators"]):
        raise ValueError("actual stage contract mismatch")
    if incoming != stage["input_shape"] or outgoing != stage["output_shape"] or operators != set(stage["operators"]):
        raise ValueError("stage metadata mismatch")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate/install firmware bundles or generate receiver embeddings")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--architecture", choices=("cesta", "hifinet", "stgcn"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rust-output", type=Path)
    parser.add_argument("--receiver", type=int)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    metadata = read_metadata(bundle)
    formats = {"cesta-tflite-micro-v3": "cesta", "hifinet-staged-host-v1": "hifinet", "stgcn-staged-host-v1": "stgcn"}
    architecture = formats.get(metadata.get("format", ""))
    if architecture is None or args.architecture not in (None, architecture):
        raise ValueError("unsupported or mismatched bundle architecture")
    metadata = {"cesta": validate_cesta, "hifinet": validate_staged, "stgcn": validate_staged}[architecture](bundle)
    receiver = args.receiver
    if architecture == "cesta":
        if args.rust_output or receiver not in (None, metadata["receiver_index"]):
            raise ValueError("CESTA requires matching receiver and file installation, not Rust generation")
    elif args.rust_output or receiver is not None:
        if receiver is None or not 0 <= receiver < len(metadata["receivers"]):
            raise ValueError("valid receiver required")
    if args.output:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("installation destination must be empty")
        args.output.mkdir(parents=True, exist_ok=True)
        if architecture == "cesta":
            shutil.copy2(bundle / metadata["stages"]["node"]["file"], args.output / "model.tflite")
            metadata["stages"]["node"]["file"] = "model.tflite"
            (args.output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
        else:
            for name in ["bundle.json", *[stage["file"] for stage in metadata["stages"].values()]]:
                shutil.copy2(bundle / name, args.output / name)
            bundle = args.output.resolve()
    if args.rust_output:
        if architecture == "stgcn":
            names = [name for r in range(2) for name in (f"prepare-{r}", f"graph-{r}-node-{receiver}", f"finish-{r}")] + ["classifier"]
            identity = stgcn_fingerprint(metadata)
        else:
            names = ["temporal", *[f"graph-{r}-node-{receiver}" for r in range(3)], "classifier"]
            identity = hifinet_fingerprint(metadata)
        lines = [
            f"pub const RECEIVER: usize = {receiver};",
            f'pub const FINGERPRINT: &str = "{identity}";',
            f'pub const CHECKPOINT: &str = "{metadata["source_checkpoint_sha256"]}";',
            f"pub const WIDTHS: [usize; {len(metadata['round_payload_widths'])}] = {metadata['round_payload_widths']!r};",
            f"pub const SENDERS: &[usize] = &{metadata['receivers'][receiver]['sender_indices']!r};",
        ]
        if architecture == "stgcn":
            source, target = metadata["model_config"]["edge_index"]
            degrees = [1 + sum(s != t and t == node for s, t in zip(source, target)) for node in range(metadata["model_config"]["num_nodes"])]
            inverse = [degrees[node] ** -0.5 for node in [receiver, *metadata["receivers"][receiver]["sender_indices"]]]
            lines.append(f"pub const INVERSE_DEGREES: &[f32] = &{inverse!r};")
        lines.append(f"pub static STAGES: [(&[u8], usize, usize); {len(names)}] = [")
        for name in names:
            stage = metadata["stages"][name]
            lines.append(f"(include_bytes!({json.dumps(str(bundle / stage['file']))}), {stage['input_shape'][2]}, {stage['output_shape'][2]}),")
        lines.append("];")
        args.rust_output.write_text("\n".join(lines) + "\n")
    print(f"Validated {architecture} bundle: {bundle}")


if __name__ == "__main__":
    main()
