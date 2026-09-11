"""Post-hoc radio-energy accounting for spatial baselines on the decisive test split.

Estimates the TX+RX energy that each message-passing spatial baseline (ST-GCN, DCRNN,
HiFiNet, HMCT) would spend on the decisive 70/15/15 test split, using:

- the serialized per-edge possible (active) message counts from the canonical dense
  CESTA runs (archive, commit 4f5b9aa), cross-checked against a fresh reconstruction
  of the test-split dynamic link mask from the canonical datasets,
- each baseline's per-round exchanged vector size, derived from its actual decisive-run
  model configuration (one distinct config per model, asserted),
- the first-order radio model shared with CESTA (CESTA.evaluation.energy).

Every graph baseline respects the dynamic link mask, so the set of active
(window, timestep, edge) slots is identical to dense CESTA's possible slots; only the
number of exchange rounds per slot and the per-round payload differ. Two accounting
schemes are reported:

- as-implemented: each round carries the model's actual exchanged vector at 32-bit floats;
- payload-matched: each round carries CESTA's serialized 8192-bit payload.

This is a model-based estimate, not a measurement; caveats are serialized in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from CESTA.datasets.artifact import CESTADataset
from CESTA.evaluation.energy import RadioEnergyConfig
from CESTA.schema.window import DataSplitConfig, WindowConfig

CANONICAL_COMMIT = "4f5b9aa"
DATASETS = ("Intel_fault05", "Intel_fault10", "Intel_fault15", "Intel_fault20")
CESTA_VARIANTS = {
    "cesta_dense_crf0p05": "Dense (CRF 0.05)",
    "cesta_selective_p1e-3": "Selective p=1e-3",
    "cesta_selective_p1e-3_commcond": "Selective p=1e-3 + comm-cond",
    "cesta_lowbit_p1e-2_commcond_voi": "Lowbit p=1e-2 + comm-cond + VOI",
}
PRECISION_BITS = 32
PAYLOAD_MATCHED_BITS = 8192.0
GRAPH_MODELS = ("stgcn", "dcrnn", "hifinet", "hmct")
TEMPORAL_MODELS = ("cnn1d", "transformer", "autoformer", "informer", "patchtst", "modern_tcn", "hydra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc spatial-baseline radio-energy accounting on the decisive test split.")
    parser.add_argument("--archive-root", type=Path, default=Path("imported/runs/archive/pre-final-20260816"),
                        help="Archive root containing the canonical CESTA matrix (commit 4f5b9aa).")
    parser.add_argument("--baseline-roots", type=Path, nargs="*", default=[Path("imported/runs"), Path("runs")],
                        help="Roots containing decisive baseline run directories.")
    parser.add_argument("--datasets-root", type=Path, default=Path("data/datasets"), help="Canonical dataset root.")
    parser.add_argument("--output", type=Path, default=Path("runs/posthoc-spatial-energy"), help="Output directory.")
    return parser.parse_args()


def sha256_prefix(path: Path, length: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def classify_cesta_variant(run_config: dict[str, Any]) -> str:
    train = run_config.get("train_config", {})
    model = run_config.get("model_config", {})
    penalty = float(train.get("communication_penalty_weight", 0.0))
    voi = float(train.get("voi_gate_loss_weight", 0.0))
    comm_cond = bool(train.get("communication_conditioned_correction", model.get("use_communication_conditioned_correction", False)))
    if penalty == 0.0:
        return "cesta_dense_crf0p05"
    if comm_cond and voi == 0.0:
        return "cesta_selective_p1e-3_commcond"
    if comm_cond and voi > 0.0:
        return "cesta_lowbit_p1e-2_commcond_voi"
    return "cesta_selective_p1e-3"


def collect_cesta_runs(archive_root: Path) -> dict[str, dict[str, dict[int, Path]]]:
    runs: dict[str, dict[str, dict[int, Path]]] = {variant: {} for variant in CESTA_VARIANTS}
    for run_dir in sorted((archive_root / "cesta").iterdir()):
        if not run_dir.is_dir() or not (run_dir / "communication_metrics.json").exists():
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text())
        commit = str(manifest.get("git", {}).get("commit", ""))
        if not commit.startswith(CANONICAL_COMMIT):
            continue
        config = json.loads((run_dir / "config.json").read_text())
        variant = classify_cesta_variant(config)
        if variant not in CESTA_VARIANTS:
            continue
        dataset = Path(manifest["dataset"]["path"]).name
        seed = int(manifest["seed"])
        runs[variant].setdefault(dataset, {})[seed] = run_dir
    for variant, by_dataset in runs.items():
        for dataset in by_dataset:
            missing = set(seed_range()) - set(by_dataset[dataset])
            if missing:
                raise ValueError(f"Canonical matrix incomplete: {variant} {dataset} missing seeds {sorted(missing)}")
    return runs


def seed_range() -> tuple[int, int, int]:
    return 12, 42, 1242


def collect_baseline_configs(baseline_roots: list[Path]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for model in (*GRAPH_MODELS, *TEMPORAL_MODELS):
        seen: dict[str, Path] = {}
        for root in baseline_roots:
            model_root = root / model
            if not model_root.is_dir():
                continue
            for run_dir in sorted(model_root.iterdir()):
                config_path = run_dir / "config.json"
                if not run_dir.is_dir() or not config_path.exists():
                    continue
                config = json.loads(config_path.read_text())
                if config.get("model_name") != model:
                    continue
                model_config = {key: value for key, value in config["model_config"].items() if key != "edge_index"}
                fingerprint = json.dumps(model_config, sort_keys=True)
                seen.setdefault(fingerprint, run_dir)
        if not seen:
            raise ValueError(f"No decisive runs found for baseline {model}")
        if len(seen) > 1:
            raise ValueError(f"Baseline {model} has {len(seen)} distinct decisive configurations; expected one")
        canonical_dir = next(iter(seen.values()))
        configs[model] = json.loads((canonical_dir / "config.json").read_text())["model_config"]
    return configs


def communication_pattern(model: str, model_config: dict[str, Any]) -> dict[str, Any]:
    """Derive per-slot exchange rounds and per-round payload (floats) from the actual model configuration.

    One slot is one active (window, timestep, directed edge). Payload is the node vector that the
    sender transmits in that round under the model's implementation.
    """
    if model == "stgcn":
        rounds = int(model_config["num_st_blocks"])
        payload = [int(model_config["st_hidden"])] * rounds
    elif model == "dcrnn":
        supports = 2 if model_config["bidirectional_diffusion"] else 1
        hops_per_conv = int(model_config["diffusion_steps"]) * supports
        convs_per_cell = 2
        layers = int(model_config["num_layers"])
        features_per_node = int(model_config["input_size"]) // int(model_config["num_nodes"])
        hidden = int(model_config["hidden_size"])
        payload: list[int] = []
        layer_input = features_per_node
        for _ in range(layers):
            payload.extend([layer_input + hidden] * (convs_per_cell * hops_per_conv))
            layer_input = hidden
        rounds = len(payload)
    elif model == "hifinet":
        rounds = int(model_config["num_gat_layers"])
        payload = [int(model_config["embedding_size"])] + [int(model_config["gat_hidden_size"])] * (rounds - 1)
    elif model == "cesta":
        rounds = 1
        payload = [int(model_config["message_size"])]
    else:
        rounds = 0
        payload = []
    return {"rounds_per_active_slot": rounds, "payload_floats_per_round": payload}


def pattern_bits(pattern: dict[str, Any], scheme: str) -> list[float]:
    if scheme == "payload-matched":
        return [PAYLOAD_MATCHED_BITS] * pattern["rounds_per_active_slot"]
    return [float(values) * PRECISION_BITS for values in pattern["payload_floats_per_round"]]


def pattern_energy_j(
    possible_edge_counts: np.ndarray,
    edge_distance_m: np.ndarray,
    bits_per_round: list[float],
    constants: RadioEnergyConfig,
) -> dict[str, float]:
    tx_per_edge = np.zeros_like(possible_edge_counts, dtype=np.float64)
    rx_per_edge = np.zeros_like(possible_edge_counts, dtype=np.float64)
    distances = np.asarray(edge_distance_m, dtype=np.float64)
    free_space = distances < constants.crossover_distance_m
    amplifier_j_per_bit = np.where(
        free_space,
        constants.free_space_j_per_bit_m2 * np.square(distances),
        constants.multipath_j_per_bit_m4 * np.power(distances, 4),
    )
    for bits in bits_per_round:
        tx_per_edge += bits * (constants.electronics_j_per_bit + amplifier_j_per_bit)
        rx_per_edge += bits * constants.electronics_j_per_bit
    tx_energy = float((possible_edge_counts * tx_per_edge).sum())
    rx_energy = float((possible_edge_counts * rx_per_edge).sum())
    return {
        "tx_energy_j": tx_energy,
        "rx_energy_j": rx_energy,
        "total_energy_j": tx_energy + rx_energy,
        "message_count_per_slot": float(len(bits_per_round)),
        "bits_per_active_slot": float(sum(bits_per_round)),
    }


def load_dataset_context(dataset_root: Path) -> tuple[np.ndarray, int, np.ndarray, dict[str, Any]]:
    dataset = CESTADataset.load(dataset_root)
    splits = dataset.prepare(WindowConfig(), DataSplitConfig(), ["temp"], required_metadata={"graph"})
    edge_mask_test = splits.edge_mask_test
    if edge_mask_test is None:
        raise ValueError("Graph preparation did not produce a test edge mask")
    window_count = int(edge_mask_test.shape[0])
    window_size = int(edge_mask_test.shape[1])
    counts = np.asarray(edge_mask_test, dtype=np.float64).sum(axis=(0, 1))
    graph = splits.metadata["graph"]
    return counts, window_count, np.asarray(graph.edge_distance_m, dtype=np.float64), {
        "dataset_sha256_prefix": sha256_prefix(dataset_root / "dataset.csv"),
        "meta_sha256_prefix": sha256_prefix(dataset_root / "dataset_meta.json"),
        "test_window_count": window_count,
        "window_size": window_size,
        "active_edge_count": float(counts.sum()),
        "split_bounds": {name: list(bounds) for name, bounds in splits.split_bounds.items()},
    }


def serialize_cesta_energy(cesta_runs: dict[str, dict[str, dict[int, Path]]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, by_dataset in cesta_runs.items():
        out[variant] = {}
        for dataset, by_seed in by_dataset.items():
            energies = []
            ratios = []
            bits = []
            for seed, run_dir in sorted(by_seed.items()):
                metrics = json.loads((run_dir / "communication_metrics.json").read_text())
                test = metrics["splits"]["test"]
                energies.append(float(test["energy"]["selective"]["total_energy_j"]))
                ratios.append(float(test["active_request_ratio"]))
                bits.append(float(test["energy"]["selective"]["bits_per_message"]))
            out[variant][dataset] = {
                "test_total_energy_j_mean": float(np.mean(energies)),
                "test_total_energy_j_values": energies,
                "request_ratio_mean": float(np.mean(ratios)),
                "bits_per_message": bits[0],
            }
    return out


def main() -> int:
    args = parse_args()
    constants = RadioEnergyConfig()
    cesta_runs = collect_cesta_runs(args.archive_root)
    baseline_configs = collect_baseline_configs(args.baseline_roots)

    cesta_energy = serialize_cesta_energy(cesta_runs)
    dense_energy_by_dataset = {dataset: values["test_total_energy_j_mean"] for dataset, values in cesta_energy["cesta_dense_crf0p05"].items()}

    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        reconstructed, window_count, distances, context = load_dataset_context(args.datasets_root / dataset)
        seed12_dir = cesta_runs["cesta_dense_crf0p05"][dataset][12]
        metrics = json.loads((seed12_dir / "communication_metrics.json").read_text())
        serialized = np.asarray(metrics["splits"]["test"]["possible_edge_counts"], dtype=np.float64)
        if not np.array_equal(serialized, reconstructed):
            raise ValueError(f"{dataset}: serialized possible counts differ from dataset reconstruction")
        manifest = json.loads((seed12_dir / "manifest.json").read_text())
        if manifest["dataset"]["data_sha256"][:16] != context["dataset_sha256_prefix"]:
            raise ValueError(f"{dataset}: local dataset hash differs from the canonical run manifest")
        cesta_pattern = communication_pattern("cesta", metrics["config"])
        cesta_check = pattern_energy_j(serialized, distances, pattern_bits(cesta_pattern, "as-implemented"), constants)
        reference = float(metrics["splits"]["test"]["energy"]["dense_reference"]["total_energy_j"])
        if abs(cesta_check["total_energy_j"] - reference) > 1e-6 * reference:
            raise ValueError(f"{dataset}: dense CESTA energy reconstruction mismatch: {cesta_check['total_energy_j']} vs {reference}")
        per_dataset[dataset] = {
            "context": context,
            "possible_edge_counts": serialized.tolist(),
            "edge_distance_m": distances.tolist(),
        }

    window_count = per_dataset[DATASETS[0]]["context"]["test_window_count"]
    models: dict[str, Any] = {}
    for model in (*GRAPH_MODELS, *TEMPORAL_MODELS):
        pattern = communication_pattern(model, baseline_configs[model])
        if pattern["rounds_per_active_slot"] == 0:
            zero_scheme = {"tx_energy_j": 0.0, "rx_energy_j": 0.0, "total_energy_j": 0.0, "message_count_per_slot": 0.0, "bits_per_active_slot": 0.0}
            models[model] = {
                "pattern": pattern,
                "schemes": {"as-implemented": zero_scheme, "payload-matched": zero_scheme},
                "per_dataset_total_energy_j": {dataset: 0.0 for dataset in DATASETS},
            }
            continue
        entry: dict[str, Any] = {"pattern": pattern, "schemes": {}, "per_dataset_total_energy_j": {}}
        for scheme in ("as-implemented", "payload-matched"):
            per_dataset_energy: dict[str, dict[str, Any]] = {}
            for dataset in DATASETS:
                counts = np.asarray(per_dataset[dataset]["possible_edge_counts"], dtype=np.float64)
                distances = np.asarray(per_dataset[dataset]["edge_distance_m"], dtype=np.float64)
                per_dataset_energy[dataset] = pattern_energy_j(counts, distances, pattern_bits(pattern, scheme), constants)
            entry["schemes"][scheme] = per_dataset_energy[DATASETS[0]]
            entry["per_dataset_total_energy_j"][scheme] = {key: value["total_energy_j"] for key, value in per_dataset_energy.items()}
        models[model] = entry

    payload = {
        "method": "posthoc_radio_energy_model",
        "provenance": {
            "canonical_commit": CANONICAL_COMMIT,
            "archive_root": str(args.archive_root),
            "datasets": {dataset: per_dataset[dataset]["context"] for dataset in DATASETS},
            "radio_constants": constants.to_payload(),
            "precision_bits": PRECISION_BITS,
            "payload_matched_bits": PAYLOAD_MATCHED_BITS,
            "cesta_variant_signature": "dense / penalty 1e-3 / penalty 1e-3 + comm-cond / penalty 1e-2 + comm-cond + VOI (train_config fields)",
        },
        "cesta_variants": cesta_energy,
        "models": models,
        "caveats": [
            "Model-based accounting, not a measurement: each aggregated directed edge slot carries one radio message per exchange round.",
            "All graph baselines respect the dynamic link mask (verified in model code); slots are identical to dense CESTA possible slots "
            "and cross-checked against serialized canonical artifacts.",
            "Per-round payload is the sender's transmitted node vector from the decisive-run model configuration; "
            "32-bit floats assumed for all models, matching CESTA's serialized accounting.",
            "DCRNN multi-hop diffusion counts one transmission per hop and per conv (gates and candidate), including forwarded hop-2 states; "
            "bidirectional diffusion would double the forward count via reversed-edge messages.",
            "GAT attention coefficients are computed receiver-side from the transmitted feature vector; no extra bits are charged for attention.",
            "Request/response handshake, MAC overhead, and idle listening are excluded, identically for all models.",
            "CESTA selective energies are the serialized test-split totals of the canonical matrix (commit 4f5b9aa), "
            "not the Sep-4 batch-8 re-run.",
            "HMCT and the temporal baselines perform no message passing and therefore spend zero radio energy under this accounting.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "posthoc_spatial_energy.json").write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append("# Post-hoc spatial-baseline radio energy (decisive test split, commit 4f5b9aa)\n")
    first_context = per_dataset[DATASETS[0]]["context"]
    lines.append(
        f"Test windows per dataset: {window_count}; "
        f"active directed-edge slots per dataset: {first_context['active_edge_count']:.0f}.\n"
    )
    lines.append("\n## Communication pattern per active (window, timestep, edge) slot\n")
    lines.append("| Model | Rounds | Payload floats/round | Bits/slot (as-implemented) | Bits/slot (payload-matched) |")
    lines.append("|---|---:|---|---:|---:|")
    for model in (*GRAPH_MODELS, *TEMPORAL_MODELS):
        pattern = models[model]["pattern"]
        payloads = ", ".join(str(value) for value in pattern["payload_floats_per_round"]) or "—"
        bits_impl = models[model]["schemes"]["as-implemented"]["bits_per_active_slot"]
        bits_matched = models[model]["schemes"]["payload-matched"]["bits_per_active_slot"]
        lines.append(f"| {model} | {pattern['rounds_per_active_slot']} | {payloads} | {bits_impl:.0f} | {bits_matched:.0f} |")

    dense_total = dense_energy_by_dataset[DATASETS[0]]
    single_round_dense = models["stgcn"]["schemes"]["as-implemented"]["total_energy_j"]
    dcrnn_total = models["dcrnn"]["schemes"]["as-implemented"]["total_energy_j"]

    lines.append("\n## Test-split TX+RX energy (J, as-implemented payloads)\n")
    lines.append("Identical across the four datasets: message-passing baselines use every active slot,")
    lines.append("and the test windows and link masks are the same on all datasets.\n")
    lines.append("| Model | TX (J) | RX (J) | Total (J) | Per window (J) | vs single-round dense (45.20 J) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    ordered = ("dcrnn", "hifinet", "stgcn", "hmct", *TEMPORAL_MODELS)
    for model in ordered:
        energy = models[model]["schemes"]["as-implemented"]
        total = energy["total_energy_j"]
        ratio = total / single_round_dense if single_round_dense > 0 else 0.0
        lines.append(
            f"| {model} | {energy['tx_energy_j']:.2f} | {energy['rx_energy_j']:.2f} | {total:.2f} | "
            f"{(total / window_count):.2f} | {ratio:.2f}x |"
        )
    lines.append(f"| cesta dense (serialized) | — | — | {dense_total:.2f} | {(dense_total / window_count):.2f} | 1.00x |")

    lines.append("\n## CESTA energy savings vs message-passing baselines (per dataset, as-implemented)\n")
    header_left = "vs single-round dense (ST-GCN / HiFiNet / dense CESTA, 45.20 J): f05 / f10 / f15 / f20 / agg"
    header_right = "vs DCRNN (271.93 J): f05 / f10 / f15 / f20 / agg"
    lines.append(f"| CESTA variant | {header_left} | {header_right} |")
    lines.append("|---|---|---|")

    def saving_row(variant: str) -> str:
        per_dataset_energy = [cesta_energy[variant][dataset]["test_total_energy_j_mean"] for dataset in DATASETS]
        aggregate = float(np.mean(per_dataset_energy))
        cells = []
        for reference in (single_round_dense, dcrnn_total):
            savings = [(1.0 - value / reference) * 100 for value in per_dataset_energy]
            savings.append((1.0 - aggregate / reference) * 100)
            cells.append(" / ".join(f"{value:.1f}%" for value in savings))
        return f"| {CESTA_VARIANTS[variant]} | {cells[0]} | {cells[1]} |"

    lines.append(saving_row("cesta_selective_p1e-3"))
    lines.append(saving_row("cesta_selective_p1e-3_commcond"))
    lines.append(saving_row("cesta_lowbit_p1e-2_commcond_voi"))

    lines.append("\n## Payload-matched scheme (all rounds at 8192 bits)\n")
    lines.append("| Model | Total (J) | vs single-round dense (45.20 J) |")
    lines.append("|---|---:|---:|")
    for model in ordered:
        total = models[model]["schemes"]["payload-matched"]["total_energy_j"]
        ratio = total / dense_total if dense_total > 0 else 0.0
        lines.append(f"| {model} | {total:.2f} | {ratio:.2f}x |")
    lines.append(f"| cesta dense (serialized) | {dense_total:.2f} | 1.00x |")
    (args.output / "posthoc_spatial_energy.md").write_text("\n".join(lines) + "\n")
    print(f"Post-hoc spatial energy written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
