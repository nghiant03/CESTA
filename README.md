# CESTA

**Communication-Efficient Spatial-Temporal Aggregation** for sensor-network fault diagnosis.

Official implementation of the CESTA paper.

**Contents:** [Setup](#setup) · [Quickstart](#quickstart) · [Capabilities](#capabilities) · [Usage](#usage) · [Firmware](#firmware) · [Citation](#citation) · [License](#license)

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

## Capabilities

**Models**

- Temporal baselines: CNN1D, Transformer, Autoformer, Informer, PatchTST, ModernTCN, and Hydra.
- Spatial models: dynamic ST-GCN, HiFiNet, HMCT, DCRNN, and CESTA.

**Data and evaluation**

- Markov injection of `SPIKE`, `DRIFT`, and `STUCK` sensor faults with per-event randomization.
- Canonical graph datasets with node masks, dynamic edge masks, node positions, and edge distances.
- Classification, communication, and theoretical TX+RX radio-energy metrics.

**Reproducibility**

- Immutable per-run artifacts: manifests, configs, checkpoints, histories, and predictions.
- Optuna hyperparameter search and audit tooling for benchmark matrices.
- ESP32-S3 firmware for distributed on-device deployment (see [Firmware](#firmware)).

## Usage


The full experimental protocol is executable from the checked-in configs and scripts:

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

## Firmware

See [`firmware/README.md`](firmware/README.md) for hardware requirements, configuration, model export, build/flash commands, and telemetry formats.

## Citation

## License

[MIT](LICENSE)
