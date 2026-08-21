# AGENTS.md

Keep this file current after code changes. CESTA is a research project for communication-aware sensor fault diagnosis.

## Development rules

- Use `uv` for Python environments and commands.
- Do not add or maintain automated tests; validate changes with targeted runtime checks, `uv run ruff check src/CESTA`, and `uv run pyright src/CESTA`.
- Ruff uses line length 150 and import sorting from `pyproject.toml`.
- Use `from __future__ import annotations` and lazy function imports instead of `typing.TYPE_CHECKING`.
- Reconsider names whenever their purpose changes.
- Keep every notebook import block in its top cell.

## Documentation

- `README.md`: setup, workflow, capabilities, firmware, and repository layout.
- `docs/RESEARCH.md`: the single research document containing aim, protocol, evidence, work plan, and hard boundaries.
- Do not create separate proposal, experiment, result, plan, or subsystem documents. Merge durable project guidance into `README.md` or `AGENTS.md`, and research updates into `docs/RESEARCH.md`.

## Architecture

```text
src/CESTA/
├── schema/          # Pydantic configuration and artifact schemas
├── batch.py         # Runtime batch contracts
├── metrics.py       # Shared classification metrics
├── artifacts.py     # Run directories, manifests, and checkpoints
├── workflows/       # Train/evaluate orchestration
├── cli/             # Thin Typer wrappers
├── injection/       # Markov faults and injectors
├── datasets/        # Raw loaders and canonical artifacts
├── models/          # Temporal and spatial models, including portable Hydra
├── training/        # Trainer, objectives, losses, and callbacks
├── evaluation/      # Evaluation, communication, energy, and benchmarks
├── optimization/    # Optuna search
├── utils.py         # Git, environment, ID, time, and hashing helpers
└── seed.py          # Reproducibility helper

config/
├── datasets/        # Dataset transformation and fault-injection configs
├── training/        # Canonical model training configs
├── experiments/     # Diagnosis ablations, controls, and sweeps
└── benchmarks/      # Comparison and tuning-grid specifications
docs/RESEARCH.md     # Consolidated research record
firmware/             # ESP32-S3 Rust firmware
notebooks/            # Analysis notebooks
runs/                 # Generated artifacts
```

## Core contracts

- Import configuration schemas from their domain modules under `schema/`; `schema/types.py` is a compatibility shim.
- `GraphWindowBatch` in `batch.py` is the native graph runtime contract used by loaders, trainers, evaluators, ST-GCN, HiFiNet, DCRNN, and CESTA.
- Import classification metrics from `CESTA.metrics`; `evaluation/metrics.py` is a compatibility shim.
- Keep `artifacts.py` independent of models, training, evaluation, CLI, and workflows. It uses a structural checkpoint protocol.
- Keep CLI modules thin. Cross-package train/evaluate behavior belongs in `workflows/`.
- Prefer config-file-first command surfaces for large runtime settings; validate YAML/JSON directly into Pydantic models.
- All active workflows use the connectivity-chronological `70/15/15` split; plain chronological and `80/10/10` splits are unsupported.

## Data

`CESTADataset` is the canonical post-transform artifact. A dataset requires `dataset.csv`, `dataset_meta.json`, `graph_edges.npz`, `dynamic_link_mask.npz`, `node_positions.json`, and `edge_distances.npz`; legacy names are unsupported.

`CESTADataset.prepare()` returns `WindowedSplits`. Graph metadata travels in `WindowedSplits.metadata["graph"]`. Graph models declare `required_metadata = {"graph"}`. `create_model()` validates requirements and extracts metadata-backed constructor arguments.

`WindowedSplit.select()` must apply one index selection to every aligned field. Preserve missing nodes and unavailable links through node and edge masks. For non-graph models using `connectivity-chronological`, preserve the graph cohort's active communication block and split boundaries while dropping only incomplete node windows.

To add a raw dataset, subclass `BaseDataset` under `datasets/raw/` and register it in `datasets/raw/__init__.py`. To add a fault, update `schema/fault.py`, implement it in `injection/faults.py`, register it in `injection/registry.py`, and add defaults to `MarkovConfig` when applicable.

## Training and evaluation

- Call `seed_everything(config.seed)` before model construction.
- `Trainer` receives validation data explicitly and supports focal loss, aligned oversampling, callbacks, and composed auxiliary objectives.
- Keep shared masked loss, decoding, and auxiliary objectives in `training/objectives.py`, not model-specific branches in the trainer.
- `Evaluator` handles device placement, masked predictions, metrics, split-aware communication aggregation, and validation-only checkpoint evaluation.
- `EvalResult.save()` writes classification artifacts and `communication_metrics.json` when available. Validation evaluation must not overwrite test artifacts.
- Energy accounting belongs in `evaluation/energy.py`, outside models. Count TX and RX for each active directed message using graph-aligned distances and serialize constants, units, distance source, shares, totals, and dense-reference reductions.

Each training invocation creates a new, never-overwritten run:

```text
runs/<model>/<run_id>/
├── weight.pt
├── config.json
├── history.jsonl
├── manifest.json
├── eval_metrics.json
├── predictions.npz
└── communication_metrics.json  # when applicable
```

## Temporal models

Hydra is implemented as a portable PyTorch quasiseparable bidirectional mixer under `models/temporal/hydra.py`. Keep its per-timestep head and `(batch, time, classes)` output contract; do not replace it with window-level pooling. It intentionally avoids the official CUDA-only kernel dependency so baseline training and artifact loading remain portable.

## CESTA model

`CESTAClassifier` is under `models/spatial/cesta/` and accepts graph-aligned input `(batch, window, nodes * features)`. Modes are `none`, `dense`, `gumbel_request`, `random`, `static_topk`, and `local_change`.

Request decisions must use receiver-local state, local uncertainty, and edge metadata only. They cannot inspect sender hidden states before communication. Aggregation uses receiver queries and received sender keys/values, softmax over the received set only, and zero graph context when none are received.

The model may expose communication statistics, soft receiver request probabilities, neighbor beliefs, boundary logits, CRF decoding, communication-conditioned correction, structured top-k requests, VOI objectives, and rule controls. Receiver request probabilities must exclude unavailable edges before aggregation. Rule-control decisions use receiver-local scores and edge metadata, and their validation-tuned parameters must be persisted in communication artifacts. Random controls derive decisions from stable window, timestep, receiver, sender, and controller-seed identities. Freeze inactive learned-gate parameters in rule modes and persist active and total parameter counts. Keep transmitted-bit estimates aligned with the actual payload and preserve gradient-bearing communication ratios and expected energy for training penalties. Evaluation aggregates per-edge requested/possible counts against canonical graph distances.

## Firmware

The ESP32-S3 firmware lives under `firmware/`. `firmware/src/config.rs` selects normal or SPIKE profiles; SPIKE reads `SPIKE_DHT_PIN` and expects an external DATA-line disturbance. `main.rs` must construct SNTP from `NTP_SERVER`, not `EspSntp::new_default()`. Build and deployment commands are in `README.md`.

## Commands

```bash
uv run cesta transform intel_lab data/raw/Intel/data.txt data/datasets/intel_lab --config config/datasets/intel-lab/fault-15.yaml
uv run cesta train config/training/cesta.yaml data/canon/intel_lab
uv run cesta evaluate --model runs/cesta/<run_id> --data data/canon/intel_lab
uv run cesta optimize --data data/canon/intel_lab --model cnn1d --n-trials 20 --epochs 10

uv run python scripts/run_all_baselines.py --dry-run
uv run python scripts/audit_decisive_comparison.py --spec config/benchmarks/decisive-comparison.yaml --runs-root runs --output runs/decisive-comparison-audit --allow-incomplete
uv run python scripts/summarize_decisive_comparison.py --runs-csv runs/decisive-comparison-audit/runs.csv --output runs/decisive-comparison-summary --comparison <variant> <locked-reference>
uv run python scripts/generate_control_tuning.py --spec config/benchmarks/control-tuning.yaml --output runs/control-tuning/generated
uv run python scripts/derive_control_budgets.py --runs-csv <validation-runs.csv> --source-variant <variant> --output runs/control-tuning/control-budgets.yaml
uv run python scripts/lock_control_policies.py --budgets <budgets.yaml> --validation-runs-csv <validation-runs.csv> --controller <control> <variants...> --output runs/control-tuning/control-lock.yaml
uv run python scripts/audit_locked_controls.py --lock <control-lock.yaml> --test-runs-csv <test-runs.csv> --output runs/control-locked-audit
uv run python scripts/audit_validation_logit_sensitivity.py --model <run> --data <dataset>
```

The baseline runner reconciles completed cells from manifests and resolved configs. All default model configs use the connectivity-chronological `70/15/15` split for direct CESTA accuracy comparisons; historical `80/10/10` runs are descriptive only. The benchmark auditor rejects missing, duplicate, inconsistent, or incomparable cells without selecting by test performance.
