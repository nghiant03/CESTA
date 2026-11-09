from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def validate(bundle: Path) -> dict:
    metadata = json.loads((bundle / "bundle.json").read_text())

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(metadata["format"] == "hifinet-staged-host-v1" and metadata["converted"] is True, "converted HiFiNet bundle required")
    config = metadata["model_config"]
    require(metadata["window_size"] == 60 and metadata["sample_interval_secs"] == 60, "unsupported cadence/window")
    require(config["num_gat_layers"] == metadata["exchange_rounds"] == 3, "three graph rounds required")
    require(config["input_size"] == config["num_nodes"] and config["num_classes"] == 4, "one temperature feature and four classes required")
    require(metadata["preprocessing"] == "raw features; no window standardization; complete consecutive windows only", "unsupported preprocessing")
    require(metadata["decoder"] == "per-timestep argmax of raw class logits; first-index tie breaking; no CRF", "unsupported decoder")
    require(
        metadata["graph_input_layout"] == ["local_previous_layer", "sender_previous_layer_in_edge_order", "received_mask"], "graph layout mismatch"
    )
    require(metadata["classifier_input_layout"] == ["retained_local_temporal_embedding", "final_graph_embedding"], "classifier layout mismatch")
    widths = [config["embedding_size"], config["gat_hidden_size"], config["gat_hidden_size"]]
    require(metadata["round_payload_widths"] == widths and all(type(w) is int and 0 < w <= 256 for w in widths), "invalid trained widths")
    for key in ("source_checkpoint_sha256", "source_config_sha256", "source_manifest_sha256"):
        require(len(metadata[key]) == 64 and len(bytes.fromhex(metadata[key])) == 32, "invalid provenance hash")
    errors = metadata["validation_max_logit_errors"]
    require(all(f"tflite-case-{i}" in errors for i in range(5)), "missing TFLite parity validation")
    require(all(math.isfinite(v) and 0 <= v <= 0.001 for v in errors.values()), "invalid parity results")
    source, target = config["edge_index"]
    require(len(source) == len(target) and len(set(zip(source, target))) == len(source), "invalid graph edges")
    require(all(type(n) is int and 0 <= n < config["num_nodes"] for n in source + target), "invalid graph node")
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
            expected[f"graph-{layer}-node-{receiver}"] = ((1 + len(indices)) * width + len(indices), widths[1])
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
        require(set(stage["missing_micro_operators"]) <= supported and set(stage["operators"]) <= supported, "unsupported operators")
    return metadata


def compatibility_fingerprint(metadata: dict) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate/install a complete HiFiNet bundle or generate firmware embedding")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rust-output", type=Path)
    parser.add_argument("--receiver", type=int)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    metadata = validate(bundle)
    if args.output:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("installation destination must be empty")
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ["bundle.json", *[s["file"] for s in metadata["stages"].values()]]:
            shutil.copy2(bundle / name, args.output / name)
    if args.rust_output:
        receiver = args.receiver
        if receiver is None or not 0 <= receiver < len(metadata["receivers"]):
            raise ValueError("valid receiver required")
        names = ["temporal", *[f"graph-{r}-node-{receiver}" for r in range(3)], "classifier"]
        identity = compatibility_fingerprint(metadata)
        lines = [
            f"pub const RECEIVER: usize = {receiver};",
            f'pub const FINGERPRINT: &str = "{identity}";',
            f'pub const CHECKPOINT: &str = "{metadata["source_checkpoint_sha256"]}";',
            f"pub const WIDTHS: [usize; 3] = {metadata['round_payload_widths']!r};",
            f"pub const SENDERS: &[usize] = &{metadata['receivers'][receiver]['sender_indices']!r};",
        ]
        lines.append("pub static STAGES: [(&[u8], usize, usize); 5] = [")
        for name in names:
            stage = metadata["stages"][name]
            lines.append(f"(include_bytes!({json.dumps(str(bundle / stage['file']))}), {stage['input_shape'][2]}, {stage['output_shape'][2]}),")
        lines.append("];")
        args.rust_output.write_text("\n".join(lines) + "\n")
    print(f"Validated HiFiNet bundle: {bundle}")


if __name__ == "__main__":
    main()
