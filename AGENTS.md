# AGENTS.md

CESTA is a research project for communication-aware sensor fault diagnosis. Keep this file current after code changes.

## Development rules

- Use `uv` for Python environments and commands.
- Do not add or maintain automated tests; validate changes with targeted runtime checks, `uv run ruff check src/CESTA`, and `uv run pyright src/CESTA`.
- Follow the Ruff configuration in `pyproject.toml`: line length 150 and import sorting.
- Use `from __future__ import annotations` and lazy function imports instead of `typing.TYPE_CHECKING`.
- Reconsider names whenever their purpose changes.

## Documentation

- `README.md`: setup, workflow, capabilities, firmware summary, and repository layout.
- `firmware/README.md`: firmware hardware requirements, configuration, export, build/flash, and telemetry.
- Do not create separate proposal, experiment, result, plan, or subsystem documents. Merge durable project guidance into `README.md` or `AGENTS.md`.

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
├── models/          # Temporal and spatial models
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
firmware/            # ESP32-S3 Rust firmware
runs/                # Generated artifacts
```

## Core contracts

- Import configuration schemas from their domain modules under `schema/`; `schema/types.py` is a compatibility shim.
- `GraphWindowBatch` in `batch.py` is the native graph runtime contract used by loaders, trainers, evaluators, ST-GCN, HiFiNet, HMCT, DCRNN, and CESTA.
- Import classification metrics from `CESTA.metrics`; `evaluation/metrics.py` is a compatibility shim.
- Keep `artifacts.py` independent of models, training, evaluation, CLI, and workflows. It uses a structural checkpoint protocol.
- Keep CLI modules thin. Cross-package train/evaluate behavior belongs in `workflows/`.
- Prefer configuration files for large runtime settings; validate YAML/JSON directly into Pydantic models.
- All active workflows and default model configs use the connectivity-chronological `70/15/15` split for comparable accuracy results. Plain chronological and `80/10/10` splits are unsupported; historical `80/10/10` runs are descriptive only.

## Data

`CESTADataset` is the canonical post-transform artifact. Required files (legacy names are unsupported):

- `dataset.csv`
- `dataset_meta.json`
- `graph_edges.npz`
- `dynamic_link_mask.npz`
- `node_positions.json`
- `edge_distances.npz`

Window and metadata contracts:

- `CESTADataset.prepare()` returns `WindowedSplits`, with graph metadata in `WindowedSplits.metadata["graph"]`.
- Graph models declare `required_metadata = {"graph"}`. `create_model()` validates requirements and extracts metadata-backed constructor arguments.
- `WindowedSplit.select()` must apply one index selection to every aligned field.
- Preserve missing nodes and unavailable links through node and edge masks.
- For non-graph models, preserve the graph cohort's active communication block and split boundaries while dropping only incomplete node windows.

Extension points:

- **Raw dataset:** subclass `BaseDataset` under `datasets/raw/` and register it in `datasets/raw/__init__.py`.
- **Fault:** update `schema/fault.py`, implement it in `injection/faults.py`, register it in `injection/registry.py`, and add defaults to `MarkovConfig` when applicable.

## Training and evaluation

- Call `seed_everything(config.seed)` before model construction.
- `Trainer` receives validation data explicitly and supports focal loss, aligned oversampling, callbacks, and composed auxiliary objectives.
- Keep shared masked loss, decoding, and auxiliary objectives in `training/objectives.py`, not model-specific branches in the trainer.
- `Evaluator` handles device placement, masked predictions, metrics, split-aware communication aggregation, and validation-only checkpoint evaluation.
- `EvalResult.save()` writes classification artifacts and `communication_metrics.json` when available. Validation evaluation must not overwrite test artifacts.
- Energy accounting belongs in `evaluation/energy.py`, outside models. Count TX and RX for each active directed message using graph-aligned distances and serialize constants, units, distance source, shares, totals, and dense-reference reductions.
- The baseline runner reconciles completed cells from manifests and resolved configs. The benchmark auditor rejects missing, duplicate, inconsistent, or incomparable cells without selecting by test performance.

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

## Models

### Hydra

- Implementation: `models/temporal/hydra.py`, a portable PyTorch quasiseparable bidirectional mixer.
- Preserve the per-timestep head and `(batch, time, classes)` output; do not replace it with window-level pooling.
- Avoid the official CUDA-only kernel dependency so baseline training and artifact loading remain portable.

### HMCT

- Implementation: `models/spatial/hmct.py`, independent per-node temporal diagnosis using first differences, residual multi-scale dilated convolutions, sinusoidal position encoding, and a Transformer encoder.
- Preserve the `(batch, time, nodes, classes)` graph-model output contract. HMCT does not perform graph message passing.

### CESTA

`CESTAClassifier` is under `models/spatial/cesta/` and accepts graph-aligned input `(batch, window, nodes * features)`. Modes are `none`, `dense`, `gumbel_request`, `random`, `static_topk`, and `local_change`.

- Request decisions use receiver-local state, local uncertainty, and edge metadata only. They must not inspect sender hidden states before communication.
- Exclude unavailable edges from receiver request probabilities before aggregation.
- Aggregate receiver queries and received sender keys/values with softmax over the received set only. Use zero graph context when none are received.
- Rule controls use receiver-local scores and edge metadata. Persist validation-tuned parameters in communication artifacts.
- Random controls derive decisions from stable window, timestep, receiver, sender, and controller-seed identities.
- Freeze inactive learned-gate parameters in rule modes and persist active and total parameter counts.
- Align transmitted-bit estimates with the actual payload. Preserve gradient-bearing communication ratios and expected energy for training penalties.
- During evaluation, aggregate per-edge requested/possible counts against canonical graph distances.

Optional outputs and features include communication statistics, soft receiver request probabilities, neighbor beliefs, boundary logits, CRF decoding, communication-conditioned correction, structured top-k requests, VOI objectives, and rule controls.

## Firmware

The ESP32-S3 Rust application lives in `firmware/`. Build and deployment commands are in `README.md`; detailed hardware, export, and troubleshooting guidance belongs in `firmware/README.md`.

### Build and memory configuration

- Pin ESP-IDF v5.3.6. Use 8 MB QIO flash and octal PSRAM, both at 40 MHz, with large allocations routed to PSRAM through `firmware/sdkconfig.defaults`.
- Select `CONFIG_ESPTOOLPY_FLASHMODE_QIO=y` and, independently, `CONFIG_SPIRAM_MODE_OCT=y`. The derived `CONFIG_ESPTOOLPY_FLASHMODE="dio"` and ROM image header are intentional; do not manually change them to `"qio"`.
- Keep `CONFIG_SPIRAM_MEMTEST=y`, including during startup diagnosis or memory-timing changes. PSRAM identification alone does not validate memory access.
- Match the flashing tool's flash frequency to the compiled configuration. Verify the resolved build configuration and both firmware modes after memory-configuration changes.
- Keep standard build-and-flash documentation concise. Document the matching project-built bootloader procedure under PSRAM-startup troubleshooting in `firmware/README.md`; its QIO initialization resolved the tested N16R8 startup failure.
- After C++ component edits, touch `firmware/sdkconfig.defaults` before rebuilding so `esp-idf-sys` reruns CMake. A Rust-only rebuild can retain a stale bridge library.

### Model export and inference

- Embed `firmware/model/model.tflite` and call Espressif TensorFlow Lite Micro through the `firmware/components/cesta_tflite` C++ bridge.
- `scripts/export_cesta_firmware.py` exports a trained CESTA checkpoint, validates numerical parity and registered operators, and writes metadata beside the model.
- Preserve receiver-local requests, neighbor responses, and per-timestep many-to-many diagnosis. Firmware performs the communication; do not replace the deployment with a local-only classifier.
- The node artifact consumes a local 60-sample window plus selected neighbor payloads and returns per-timestep probabilities and request decisions.
- Parse `model.json` and reject exports whose target, receiver, sender list, shapes, or training status do not match `firmware/src/config.rs`.
- Preserve watchdog servicing during long float32 passes: the bridge yields at operator boundaries and temporarily lowers the caller to idle priority during tensor allocation, then restores it.

### Communication and telemetry

- Use binary request/response exchanges directly over ESP-NOW. `espnow.rs` fragments frames into 250-byte packets and reassembles them per peer. Its worker serves cached hidden-state rows for requested timesteps and sends queued frames.
- Peers are the static `NEIGHBORS` station MAC addresses, unencrypted and on the current station channel. MQTT is for telemetry only.
- The main loop thresholds request probabilities at the model's `request_threshold` (evaluation semantics), waits up to `EXCHANGE_WAIT_MS`, then reruns inference with received payloads.
- Publish JSON telemetry to `cesta/readings/<device_id>`:
  - Readings: `device_id`, `timestamp`, `temperature`, `humidity`, `path`, `fault_mode`, `gpio`.
  - Inference: `type: "inference"`, `window_id`, `label`, `class`, `confidence`, `probabilities`, per-neighbor timestep counts (`requested`/`received`), and pass latencies.
- Lab deployments can collect telemetry with Mosquitto, Telegraf, InfluxDB, and Grafana.
- In `main.rs`, construct SNTP from `NTP_SERVER`, not `EspSntp::new_default()`.

### Fault profiles and diagnostics

- `firmware/src/config.rs` selects normal, SPIKE, DRIFT, or STUCK profiles, with one fault profile per type in `schema/fault.py`. Inject faults in `fault.rs` on the normal `DHT_PIN` reading, mirroring the Python injectors with per-event randomization over `event_duration_samples` events.
- Normal builds run the sensor/radio application. Synthetic diagnosis is opt-in through `INFERENCE_SYNTHETIC_DIAGNOSTIC` in `firmware/src/config.rs`. Keep it `false` by default and restore it after diagnostic testing; do not add a Cargo feature for this switch.
- Synthetic mode logs a 15-second startup countdown, runs its cases once, and repeats the final PASS/FAIL every five seconds for serial reconnection. It does not start the normal sensor or radio loop.
- Use UART0 at 115200 baud as the primary console and native USB Serial/JTAG as secondary. Prefer the USB-UART bridge for startup and panic capture; ESP32 resets can disconnect native USB and its USB/IP forwarding.

## Commands

### Data and training

```bash
uv run cesta transform intel_lab data/raw/Intel/data.txt data/datasets/Intel_fault15 --config config/datasets/intel-lab/fault-15.yaml
uv run cesta train config/training/cesta.yaml data/datasets/Intel_fault15
uv run cesta evaluate --model runs/cesta/<run_id> --data data/datasets/Intel_fault15
uv run cesta optimize --data data/datasets/Intel_fault15 --model cnn1d --n-trials 20 --epochs 10
```

### Baselines and comparison audits

```bash
uv run python scripts/run_all_baselines.py --dry-run
uv run python scripts/audit_decisive_comparison.py --spec config/benchmarks/decisive-comparison.yaml --runs-root runs --output runs/decisive-comparison-audit --allow-incomplete
uv run python scripts/summarize_decisive_comparison.py --runs-csv runs/decisive-comparison-audit/runs.csv --output runs/decisive-comparison-summary --comparison <variant> <locked-reference>
```

### Control tuning and analysis

```bash
uv run python scripts/generate_control_tuning.py --spec config/benchmarks/control-tuning.yaml --output runs/control-tuning/generated
uv run python scripts/derive_control_budgets.py --runs-csv <validation-runs.csv> --source-variant <variant> --output runs/control-tuning/control-budgets.yaml
uv run python scripts/lock_control_policies.py --budgets <budgets.yaml> --validation-runs-csv <validation-runs.csv> --controller <control> <variants...> --output runs/control-tuning/control-lock.yaml
uv run python scripts/audit_locked_controls.py --lock <control-lock.yaml> --test-runs-csv <test-runs.csv> --output runs/control-locked-audit
uv run python scripts/audit_validation_logit_sensitivity.py --model <run> --data <dataset>
uv run python scripts/posthoc_spatial_energy.py --output runs/posthoc-spatial-energy
```
