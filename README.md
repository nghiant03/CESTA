# CESTA

Communication-Efficient Spatial-Temporal Aggregation for sensor-network fault diagnosis.

CESTA provides reproducible raw-data transformation, synthetic fault injection, temporal and graph model training, communication-energy evaluation, and run artifact persistence. The research asks whether receiver-side selective communication can outperform strong temporal models while using less TX+RX energy than dense spatial message passing. See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the consolidated aim, protocol, evidence, and work plan.

## Setup

Requirements: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and a PyTorch-compatible environment.

```bash
git clone https://github.com/Sinner/CESTA.git
cd CESTA
uv sync
uv run cesta --help
```

## Workflow

```bash
# Transform raw Intel data into a canonical dataset
uv run cesta transform intel_lab data/raw/Intel/data.txt data/canon/intel_lab \
  --config config/datasets/intel-lab/fault-15.yaml

# Train the communication-aware CESTA model
uv run cesta train config/training/cesta.yaml data/canon/intel_lab

# Evaluate a run, or create validation-only tuning artifacts
uv run cesta evaluate --model runs/cesta/<run_id> --data data/canon/intel_lab
uv run cesta evaluate --model runs/cesta/<run_id> --data data/canon/intel_lab --split val --output runs/control-validation
```

Training is config-file-first. Default training configurations under `config/training/` use the connectivity-chronological `70/15/15` split; diagnosis studies live under `config/experiments/`, and comparison specifications live under `config/benchmarks/`. Use `uv run cesta <command> --help` for all command options. Hydra uses a portable PyTorch implementation of the published bidirectional quasiseparable mixer, so it runs through the standard environment without requiring the official CUDA-only kernel package.

Run independent baseline tasks concurrently across visible GPUs with:

```bash
uv run python scripts/run_all_baselines.py --num-gpus 4 --keep-progress-log
```

Each GPU receives one training process. Interrupt with `Ctrl-C` and rerun the same command to resume from completed manifests; interrupted tasks restart from the beginning. Set `CUDA_VISIBLE_DEVICES` before the command to select specific GPUs. Use `--restart` only to ignore completed runs intentionally.

## Capabilities

- Markov injection of `SPIKE`, `DRIFT`, and `STUCK` sensor faults.
- Temporal models: CNN1D, Transformer, Autoformer, Informer, PatchTST, ModernTCN, and Hydra.
- Spatial models: dynamic ST-GCN, HiFiNet, HMCT, DCRNN, and CESTA.
- CESTA modes: no communication, dense communication, receiver-side Gumbel request gating, and random, static top-k, or local-change rule-based controls.
- Canonical graph datasets with node masks, dynamic edge masks, node positions, and edge distances.
- Classification, communication, and theoretical TX+RX radio-energy metrics.
- Reproducible manifests, configs, checkpoints, histories, predictions, and Optuna searches.
- Optional ESP32-S3 DHT11/MQTT firmware under `firmware/`.

## Artifacts

Every training invocation creates a new directory:

```text
runs/<model>/<run_id>/
├── weight.pt
├── config.json
├── history.jsonl
├── manifest.json
├── eval_metrics.json
├── predictions.npz
└── communication_metrics.json  # communication-aware models
```

Audit and summarize the decisive comparison with:

```bash
uv run python scripts/audit_decisive_comparison.py \
  --spec config/benchmarks/decisive-comparison.yaml \
  --runs-root runs \
  --output runs/decisive-comparison-audit

uv run python scripts/summarize_decisive_comparison.py \
  --runs-csv runs/decisive-comparison-audit/runs.csv \
  --output runs/decisive-comparison-summary \
  --comparison <variant> <locked-reference>
```

Budget-matched controls use validation artifacts only until policies are locked:

```bash
uv run python scripts/generate_control_tuning.py \
  --spec config/benchmarks/control-tuning.yaml \
  --output runs/control-tuning/generated

uv run python scripts/derive_control_budgets.py \
  --runs-csv runs/cesta-validation-audit/runs.csv \
  --source-variant <learned-variant> \
  --output runs/control-tuning/control-budgets.yaml

uv run python scripts/lock_control_policies.py \
  --budgets runs/control-tuning/control-budgets.yaml \
  --validation-runs-csv runs/control-validation-audit/runs.csv \
  --controller local_change <candidate-variants> \
  --output runs/control-tuning/control-lock.yaml

uv run python scripts/audit_locked_controls.py \
  --lock runs/control-tuning/control-lock.yaml \
  --test-runs-csv runs/control-test-audit/runs.csv \
  --output runs/control-locked-audit
```

Use `cesta train --no-test-evaluation` while generating tuning checkpoints. Validation evaluation defaults to a checkpoint-local `validation/` directory when no output is supplied, preventing test-artifact replacement. Budget and lock files are content-hashed; the scripts reject test records during derivation and selection.

## Layout

```text
src/CESTA/
├── schema/          # Config and manifest schemas
├── cli/             # Typer commands
├── workflows/       # Train/evaluate orchestration
├── injection/       # Fault generation and injection
├── datasets/        # Raw loaders and canonical artifacts
├── models/          # Temporal and spatial models
├── training/        # Trainer, losses, objectives, callbacks
├── evaluation/      # Metrics, energy, audit, and summaries
└── optimization/    # Optuna search

config/
├── datasets/        # Dataset transformation and fault-injection configs
├── training/        # Canonical model training configs
├── experiments/     # Diagnosis ablations, controls, and sweeps
└── benchmarks/      # Comparison and tuning-grid specifications
docs/RESEARCH.md     # Research aim, protocol, evidence, and work plan
firmware/             # ESP32-S3 firmware and deployment notes
notebooks/            # Analysis notebooks
runs/                 # Generated experiment artifacts
```

## Firmware

The optional ESP32-S3 firmware reads DHT11 sensors, synchronizes time through the configured SNTP server, and publishes readings to `cesta/readings/<device_id>`. Each node runs the receiver-side CESTA pipeline: encode its local 60-sample window, decide requests from receiver-local state, exchange selected hidden-state payloads with neighbors over MQTT, aggregate replies, and produce per-timestep `NORMAL`, `SPIKE`, `DRIFT`, and `STUCK` probabilities. Install `espup`, `espflash`, `ldproxy`, and ESP-IDF host dependencies, then configure WiFi, MQTT, `DEVICE_ID`, pins, `NODE_INDEX`, `NEIGHBORS`, NTP timing, `INFERENCE_ENABLED`, the tensor-arena size, and `FAULT_CONFIG` in `firmware/src/config.rs`.

Distributed inference uses a request/response exchange beside telemetry. Each device subscribes to its own mailboxes `cesta/exchange/<device_id>/request` and `cesta/exchange/<device_id>/response`. After each reading the node runs a receiver-local request pass, thresholds the exported request probabilities at the model's `request_threshold`, and publishes one binary request per neighbor listing the timesteps it needs. Neighbors answer from their most recent cached window with the requested hidden-state rows only (a zero-count response means no cached window yet), and the receiver reruns the model with the received payloads and publishes the aggregated diagnosis. `EXCHANGE_WAIT_MS` bounds the wait, `EXCHANGE_BUFFER_BYTES` must cover a full-window response, and `EXCHANGE_TOPIC_PREFIX` locates the mailboxes.

Export a trained checkpoint before flashing; one firmware image is required per deployed node. The checked-in artifact only validates conversion and is rejected by the firmware until replaced with a trained export. The default `node` target embeds one graph receiver together with its sender list, so pass the device's graph index as `--receiver-index` and list the same senders in `NEIGHBORS` in `sender_indices` order. The `local` target deploys the shared per-sensor CESTA temporal encoder and classifier without communication, and the `graph` target exports centralized fixed-topology CESTA inference for a runtime that supplies every node window. Export validates shapes, registered TensorFlow Lite Micro operators, and numerical parity with PyTorch.

```bash
uv run --isolated \
  --with 'litert-torch==0.9.4' --with tflite --with 'pydantic<2.12' \
  --with pyyaml --with loguru --with dulwich --with numpy \
  python scripts/export_cesta_firmware.py \
  --model runs/cesta/<run_id> --output firmware/model \
  --target node --receiver-index <node_index>

cd firmware
cargo check
cargo build --release
espflash flash target/xtensa-esp32s3-espidf/release/cesta-firmware --monitor
```

Inference uses Espressif's TensorFlow Lite Micro component and requires an ESP32-S3 board with octal PSRAM and at least 8 MB flash; large MQTT and window buffers are allocated in PSRAM. Only normal and SPIKE hardware profiles are implemented. SPIKE reads `SPIKE_DHT_PIN` and expects a MOSFET or open-drain transistor to disturb the sensor DATA line. `firmware/src/main.rs` must construct SNTP from `NTP_SERVER`, not `EspSntp::new_default()`.

Payloads:

```json
{"device_id": "esp32_01", "timestamp": 1718000000, "temperature": 25.3, "humidity": 60.1}
{"device_id": "esp32_01", "timestamp": 1718000003, "type": "inference", "window_id": 12, "communication_mode": "gumbel_request", "label": "NORMAL", "class": 0, "confidence": 0.98, "probabilities": [0.98, 0.01, 0.0, 0.01], "requested": [["esp32_02", 7]], "received": [["esp32_02", 7]], "request_elapsed_ms": 210, "aggregate_elapsed_ms": 195}
```

A lab deployment can use Mosquitto, Telegraf, InfluxDB, Grafana, and a Python subscriber that exports CSV data to `data/raw/esp32_dht11/`.

## Development

```bash
uv run ruff check src/CESTA
uv run ruff format src/CESTA
uv run pyright src/CESTA
```

Run targeted tests for changed behavior before broad lint and type checks. The repository is active research software; checked-in configs are the executable experiment definitions, and evidence remains provisional until the protocol in `docs/RESEARCH.md` is complete.
