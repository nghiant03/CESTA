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
  --config config/data/intel_fault15.yaml

# Train temporal and communication-aware models
uv run cesta train config/model/gru.yaml data/canon/intel_lab
uv run cesta train config/model/cesta.yaml data/canon/intel_lab

# Evaluate a run
uv run cesta evaluate --model runs/cesta/<run_id> --data data/canon/intel_lab
```

Training is config-file-first. Diagnosis and benchmark configurations live under `config/model/diagnosis/`; use `uv run cesta <command> --help` for all command options.

## Capabilities

- Markov injection of `SPIKE`, `DRIFT`, and `STUCK` sensor faults.
- Temporal models: CNN1D, LSTM, GRU, Transformer, Autoformer, Informer, PatchTST, and ModernTCN.
- Spatial models: dynamic ST-GCN and CESTA.
- CESTA modes: no communication, dense communication, receiver-side Gumbel request gating, and random, static top-k, entropy, margin, local-change, or combined rule-based controls.
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
  --spec config/benchmark/decisive-comparison.yaml \
  --runs-root runs \
  --output runs/decisive-comparison-audit

uv run python scripts/summarize_decisive_comparison.py \
  --runs-csv runs/decisive-comparison-audit/runs.csv \
  --output runs/decisive-comparison-summary \
  --comparison <variant> <locked-reference>
```

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

config/              # Data, model, diagnosis, and benchmark configs
config/model/diagnosis/controls/  # Rule-control templates for validation tuning
docs/RESEARCH.md     # Research aim, protocol, evidence, and work plan
firmware/             # ESP32-S3 firmware and deployment notes
notebooks/            # Analysis notebooks
runs/                 # Generated experiment artifacts
```

## Firmware

The optional ESP32-S3 firmware reads DHT11 sensors, synchronizes time through the configured SNTP server, and publishes readings every 30 seconds to `cesta/readings/<device_id>`. Install `espup`, `espflash`, `ldproxy`, and ESP-IDF host dependencies, then configure WiFi, MQTT, device, pins, NTP timing, build tag, and `FAULT_CONFIG` in `firmware/src/config.rs`.

Only normal and SPIKE profiles are implemented. SPIKE reads `SPIKE_DHT_PIN` and expects a MOSFET or open-drain transistor to disturb the sensor DATA line. `firmware/src/main.rs` must construct SNTP from `NTP_SERVER`, not `EspSntp::new_default()`.

```bash
cd firmware
cargo check
cargo build --release
espflash flash target/xtensa-esp32s3-espidf/release/cesta-firmware --monitor
```

Payload:

```json
{"device_id": "esp32_01", "timestamp": 1718000000, "temperature": 25.3, "humidity": 60.1}
```

A lab deployment can use Mosquitto, Telegraf, InfluxDB, Grafana, and a Python subscriber that exports CSV data to `data/raw/esp32_dht11/`.

## Development

```bash
uv run ruff check src/CESTA
uv run ruff format src/CESTA
uv run pyright src/CESTA
```

Run targeted tests for changed behavior before broad lint and type checks. The repository is active research software; checked-in configs are the executable experiment definitions, and evidence remains provisional until the protocol in `docs/RESEARCH.md` is complete.
