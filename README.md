# CESTA

**Communication-Efficient Spatial-Temporal Aggregation** for sensor-network fault diagnosis.

CESTA is the research artifact for studying whether receiver-side selective communication can match strong temporal and spatial baselines on per-timestep sensor-fault diagnosis while using less TX+RX radio energy than dense spatial message passing. It provides reproducible raw-data transformation, Markov fault injection (`SPIKE`, `DRIFT`, `STUCK`), temporal and graph model training, communication and radio-energy evaluation, run artifact persistence, and optional ESP32-S3 firmware for distributed on-device deployment.

**Contents:** [Setup](#setup) · [Quickstart](#quickstart) · [Capabilities](#capabilities) · [Reproducing the experiments](#reproducing-the-experiments) · [Firmware](#firmware) · [Repository layout](#repository-layout) · [Development](#development) · [Citation](#citation) · [License](#license)

## Setup

Requirements: Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). `uv sync` creates the environment with all dependencies, including PyTorch.

```bash
git clone https://github.com/Sinner/CESTA.git
cd CESTA
uv sync
uv run cesta --help
```

## Quickstart

Three steps from raw data to an evaluated model:

```bash
# 1. Transform raw Intel Lab data into a canonical fault-injected dataset
uv run cesta transform intel_lab data/raw/Intel/data.txt data/datasets/Intel_fault15 \
  --config config/datasets/intel-lab/fault-15.yaml

# 2. Train the communication-aware CESTA model
uv run cesta train config/training/cesta.yaml data/datasets/Intel_fault15

# 3. Evaluate a run (test split, or validation-only for tuning)
uv run cesta evaluate --model runs/cesta/<run_id> --data data/datasets/Intel_fault15
uv run cesta evaluate --model runs/cesta/<run_id> --data data/datasets/Intel_fault15 --split val
```

Training is config-file-first. Canonical model configs live under `config/training/` (connectivity-chronological `70/15/15` split), dataset and fault-injection configs under `config/datasets/`, diagnosis studies under `config/experiments/`, and comparison specifications under `config/benchmarks/`. Run `uv run cesta <command> --help` for all options.

## Capabilities

**Models**

- Temporal baselines: CNN1D, Transformer, Autoformer, Informer, PatchTST, ModernTCN, and a portable PyTorch Hydra (no CUDA-only kernel required).
- Spatial models: dynamic ST-GCN, HiFiNet, HMCT, DCRNN, and CESTA.
- CESTA communication modes: none, dense, receiver-side learned Gumbel request gating, and random, static top-k, or local-change rule-based controls.

**Data and evaluation**

- Markov injection of `SPIKE`, `DRIFT`, and `STUCK` sensor faults with per-event randomization.
- Canonical graph datasets with node masks, dynamic edge masks, node positions, and edge distances.
- Classification, communication, and theoretical TX+RX radio-energy metrics.

**Reproducibility**

- Immutable per-run artifacts: manifests, configs, checkpoints, histories, and predictions.
- Optuna hyperparameter search and audit tooling for benchmark matrices.
- ESP32-S3 firmware for distributed on-device deployment (see [Firmware](#firmware)).

## Reproducing the experiments

Every training invocation creates a new, never-overwritten run:

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

The full experimental protocol — locked data cohort, baseline matrix, budget-matched controls, and audits — is executable from the checked-in configs and scripts:

```bash
# Baseline matrix (inspect with --dry-run first; resumable, multi-GPU)
uv run python scripts/run_all_baselines.py --num-gpus 4 --keep-progress-log

# Audit run artifacts against the decisive-comparison specification
uv run python scripts/audit_decisive_comparison.py \
  --spec config/benchmarks/decisive-comparison.yaml \
  --runs-root runs --output runs/decisive-comparison-audit

# Paired summaries against a locked reference variant
uv run python scripts/summarize_decisive_comparison.py \
  --runs-csv runs/decisive-comparison-audit/runs.csv \
  --output runs/decisive-comparison-summary \
  --comparison <variant> <locked-reference>

# Budget-matched controls: tune on validation artifacts only, lock, then audit
uv run python scripts/generate_control_tuning.py \
  --spec config/benchmarks/control-tuning.yaml \
  --output runs/control-tuning/generated
uv run python scripts/derive_control_budgets.py \
  --runs-csv <validation-runs.csv> --source-variant <variant> \
  --output runs/control-tuning/control-budgets.yaml
uv run python scripts/lock_control_policies.py \
  --budgets runs/control-tuning/control-budgets.yaml \
  --validation-runs-csv <validation-runs.csv> \
  --controller <control> <variants...> \
  --output runs/control-tuning/control-lock.yaml
uv run python scripts/audit_locked_controls.py \
  --lock runs/control-tuning/control-lock.yaml \
  --test-runs-csv <test-runs.csv> --output runs/control-locked-audit
```

## Firmware

The optional Rust firmware under [`firmware/`](firmware/README.md) turns each ESP32-S3 board into a distributed CESTA node: local window encoding with an exported TFLite Micro model, selective hidden-state exchange with neighbors directly over ESP-NOW, and per-timestep fault diagnoses published over MQTT. It supports hardware (`SPIKE`) and software-injected (`DRIFT`, `STUCK`) fault profiles covering every fault type in the training pipeline.

See [`firmware/README.md`](firmware/README.md) for hardware requirements, configuration, model export, build/flash commands, and telemetry formats.

## Repository layout

```text
src/CESTA/        # Library: schema, datasets, injection, models, training,
                  # evaluation, workflows, CLI
config/           # Datasets, training, experiments, and benchmark configs
firmware/         # ESP32-S3 Rust firmware (see firmware/README.md)
runs/             # Generated experiment artifacts
scripts/          # Baseline runner, audit/summary, control tuning, export
```

## Development

```bash
uv run ruff check src/CESTA
uv run ruff format src/CESTA
uv run pyright src/CESTA
```

The repository is active research software; checked-in configs are the executable experiment definitions, and evidence remains provisional until the experimental protocol is complete.

## Citation

The paper describing CESTA is under preparation. Until it is published, please cite this repository:

```bibtex
@software{cesta,
  title   = {CESTA: Communication-Efficient Spatial-Temporal Aggregation
             for Sensor-Network Fault Diagnosis},
  author  = {CESTA authors},
  year    = {2026},
  url     = {https://github.com/Sinner/CESTA},
  note    = {Paper under preparation}
}
```

## License

[MIT](LICENSE)
