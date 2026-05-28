# CESTA

**Communication-Efficient Spatial-Temporal Aggregation for sensor fault diagnosis.**

CESTA is a research codebase for studying communication-aware fault diagnosis in sensor networks. It provides a reproducible pipeline for fault injection, graph-aware dataset preparation, temporal and spatial-temporal model training, communication-cost evaluation, and run artifact persistence.

The current research focus is a receiver-side learned request mechanism: each sensor node first reasons from local temporal evidence, then selectively requests neighbor information over existing graph edges when communication is expected to improve diagnosis.

## Research aim

Dense spatial-temporal models can improve diagnosis by sharing neighbor context, but they often assume all graph communication is always available and free. CESTA targets the harder edge-oriented setting where radio communication is costly, dynamic, and should be justified by diagnostic value.

The main experimental question is:

> Can selective receiver-side communication exceed strong temporal-only macro-F1 while reducing communication energy relative to dense spatial message passing?

See [`docs/PROPOSAL.md`](docs/PROPOSAL.md) and [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for the research motivation, hypotheses, baselines, current diagnosis results, and planned ablations.

## Highlights

- Markov-chain injection of realistic sensor faults: `SPIKE`, `DRIFT`, and `STUCK`.
- Temporal baselines: CNN1D, LSTM, GRU, Transformer, Autoformer, Informer, PatchTST, and ModernTCN.
- Spatial-temporal baselines: ST-GCN and dense CESTA over graph-prepared datasets.
- Communication-aware CESTA with receiver-side Gumbel request gating, optional neighbor belief messages, communication-conditioned correction, VOI-style objectives, and CRF sequence decoding.
- Dynamic graph preparation from Intel connectivity data, including graph-aligned windows, node masks, edge masks, and per-node labels.
- Reproducible run artifacts: manifests, configs, checkpoints, histories, evaluation metrics, predictions, and communication metrics.
- Optuna hyperparameter optimization for reproducible model selection.
- Optional ESP32-S3 Rust firmware for lab sensor collection over MQTT.

## Installation

### Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) for environment management
- CUDA-compatible PyTorch is configured through `pyproject.toml` via the PyTorch CUDA 12.4 index

### Setup

```bash
git clone https://github.com/Sinner/CESTA.git
cd CESTA
uv sync
```

Verify the CLI:

```bash
uv run cesta --help
uv run cesta list models
uv run cesta list datasets
```

## Quick start

The standard workflow is fault injection, optional graph preparation, training, and evaluation.

```bash
# 1. Inject faults into raw Intel sensor data
uv run cesta inject intel_lab data/raw/Intel/data.txt data/injected/intel_lab

# 2. Add graph topology for spatial-temporal models
uv run cesta prepare graph data/injected/intel_lab data/raw/Intel/connectivity.txt

# 3. Train a temporal baseline
uv run cesta train config/model/gru.yaml data/injected/intel_lab

# 4. Train CESTA
uv run cesta train config/model/cesta.yaml data/injected/intel_lab

# 5. Evaluate a run
uv run cesta evaluate --model runs/cesta/<run_id> --data data/injected/intel_lab
```

Diagnosis-focused configurations live under `config/model/diagnosis/` and encode the current higher-end research settings for dense, request-gated, residual, and capacity-matched CESTA variants.

## Data pipeline

CESTA separates the data lifecycle into raw datasets and injected datasets.

1. Raw loaders normalize source data into a common sensor-time table.
2. Fault injection adds synthetic labels and corrupted readings using configurable Markov transitions.
3. Graph preparation attaches topology and dynamic communication metadata.
4. Window preparation produces chronological train/validation/test splits for temporal or graph-aligned training.

Graph models receive per-window tensors with node masks and edge masks, so missing node readings and unavailable communication links can be handled without requiring complete-case timestamps.

## Models

| Family | Models | Notes |
|---|---|---|
| Temporal | `cnn1d`, `lstm`, `gru`, `transformer`, `autoformer`, `informer`, `patchtst`, `modern_tcn` | Strong local-only baselines for fault diagnosis. |
| Spatial-temporal | `stgcn`, `cesta` | Require graph metadata from `prepare graph`. |
| Communication-aware | `cesta` | Supports dense, no-communication, and receiver-side Gumbel request modes. |

CESTA can be configured with graph residual fusion, learned request gates, communication penalties, neighbor belief features, boundary supervision, communication-conditioned correction, structured top-k requests, VOI-style gate loss, and CRF decoding.

## Configuration

Training is config-file-first. Model, optimizer, data-window, split, loss, and communication settings are stored in YAML and validated by Pydantic.

```bash
uv run cesta train config/model/lstm.yaml data/injected/intel_lab
uv run cesta train config/model/diagnosis/cesta_diag_70_15_15_dense.yaml data/injected/Intel_fault15
```

Large command surfaces use YAML or JSON config files; smaller utility commands keep direct CLI options. This keeps experiment settings reproducible and avoids hidden command-line state.

## Run artifacts

Each training invocation creates a new run directory and never overwrites previous runs.

```text
runs/<model>/<utc_ts>_<model>_seed<seed>_<shortsha>/
├── weight.pt
├── config.json
├── history.jsonl
├── manifest.json
├── eval_metrics.json
└── predictions.npz
```

Communication-aware models also write communication metrics during evaluation when available.

## CLI overview

```text
cesta
├── inject              # Inject faults into raw sensor data
├── prepare
│   └── graph           # Attach graph topology and communication metadata
├── train               # Train from a YAML/JSON config
├── evaluate            # Evaluate a trained run
├── optimize            # Run Optuna hyperparameter search
│   └── show            # Display study results
└── list                # List datasets, models, or metrics
```

Run `uv run cesta <command> --help` for command-specific options.

## Project structure

```text
src/CESTA/
├── schema/            # Pydantic configs and manifest schemas
├── batch.py           # Runtime batch contracts
├── artifacts.py       # Run artifact and checkpoint helpers
├── workflows/         # Reusable train/evaluate orchestration
├── cli/               # Typer CLI
├── injection/         # Markov generator and fault injectors
├── datasets/          # Raw and injected dataset handling
├── models/            # Temporal, spatial, and CESTA model definitions
├── training/          # Trainer, losses, callbacks, objectives
├── evaluation/        # Metrics, evaluator, communication reporting
├── optimization/      # Optuna search spaces and optimizer
├── utils.py           # Runtime helpers
└── seed.py            # Reproducibility helper

config/                # Model and diagnosis YAML configs
docs/                  # Proposal, experiment plan, and research notes
firmware/              # ESP32-S3 Rust firmware for optional data collection
notebooks/             # Analysis notebooks
runs/                  # Generated experiment artifacts
```

## Development

Use the project tools through `uv`.

```bash
uv run ruff check src/CESTA
uv run ruff format src/CESTA
uv run pyright src/CESTA
```

After code changes, run the most targeted validation first, then broaden to ruff and pyright when appropriate.

## Firmware

The optional firmware stack targets ESP32-S3 devices collecting DHT11 readings and publishing JSON payloads over MQTT. See [`firmware/README.md`](firmware/README.md) for build, flash, MQTT, and lab deployment details.

## Extension points

### Add a dataset

1. Implement a `BaseDataset` subclass in `src/CESTA/datasets/raw/`.
2. Define `name`, feature columns, grouping, timestamp handling, loading, and preprocessing.
3. Register it in `src/CESTA/datasets/raw/__init__.py`.

### Add a fault type

1. Add the enum value in `src/CESTA/schema/fault.py`.
2. Implement an injector in `src/CESTA/injection/faults.py`.
3. Register it in `src/CESTA/injection/registry.py`.
4. Add default Markov settings in `MarkovConfig._default_fault_configs()` if it should be part of the standard injection profile.

### Add a model

1. Implement a `BaseModel` subclass under `src/CESTA/models/temporal/` or `src/CESTA/models/spatial/`.
2. Declare required metadata such as `graph` or `node_identity` when needed.
3. Add metadata extraction logic in `src/CESTA/models/registry.py` if the constructor needs dataset metadata.
4. Register the model in `src/CESTA/models/registry.py`.

## Repository status

This is an active research repository. APIs and experiment settings may change as hypotheses are tested. Treat `docs/PROPOSAL.md`, `docs/EXPERIMENT.md`, and checked-in configs as the canonical references for current research intent and experimental protocol.
