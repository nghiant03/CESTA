# Developer contracts

Keep these contracts current after code changes.

## Development rules

- Use `uv` for Python environments and commands.
- Do not add or maintain automated tests. Validate with targeted runtime checks, `uv run ruff check src/CESTA`, and `uv run pyright src/CESTA`.
- Follow `pyproject.toml`: Ruff line length 150 and import sorting.
- Use `from __future__ import annotations` and lazy function imports instead of `typing.TYPE_CHECKING`.
- Reconsider names when their purpose changes.
- Keep documentation developer-facing: workflows and artifact references in [README.md](README.md), implementation invariants here, firmware details in [firmware/README.md](firmware/README.md). Do not duplicate guidance or create separate proposal, experiment, result, plan, or subsystem documents. Store detailed experiment records in run artifacts.

## Package boundaries

Paths below are relative to `src/CESTA/`.

| Path | Responsibility |
|---|---|
| `schema/` | Pydantic configuration and artifact schemas |
| `batch.py` | Runtime batch contracts |
| `metrics.py` | Shared classification metrics |
| `artifacts.py` | Run directories, manifests, checkpoints |
| `workflows/` | Train/evaluate orchestration |
| `cli/` | Thin Typer wrappers |
| `datasets/` | Raw loaders and canonical artifacts |
| `injection/` | Markov faults and injectors |
| `models/` | Temporal and spatial models |
| `training/` | Trainer, objectives, losses, callbacks |
| `evaluation/` | Evaluation, communication, energy, benchmarks |
| `optimization/` | Optuna search |
| `utils.py`, `seed.py` | Environment/provenance helpers and reproducibility |

- Import schemas from domain modules under `schema/`; `schema/types.py` is a compatibility shim. Import classification metrics from `CESTA.metrics`; `evaluation/metrics.py` is a shim.
- `GraphWindowBatch` is the native contract for graph loaders, trainers, evaluators, ST-GCN, HiFiNet, HMCT, DCRNN, and CESTA.
- Keep `artifacts.py` independent of models, training, evaluation, CLI, and workflows; use its structural checkpoint protocol.
- Cross-package train/evaluate behavior belongs in `workflows/`. Validate large YAML/JSON settings directly into Pydantic models.
- Models may declare `checkpoint_metadata_fields` as a frozenset of descriptive legacy `get_config()` fields. Reconstruction excludes only declared fields from constructor kwargs, then validates reconstructed metadata. Unknown architecture kwargs must fail. CESTA declares `graph_edge_count` and `gate_input_schema`.

## Data contracts

`CESTADataset` requires `dataset.csv`, `dataset_meta.json`, `graph_edges.npz`, `dynamic_link_mask.npz`, `node_positions.json`, and `edge_distances.npz`. Legacy names are unsupported.

- `CESTADataset.prepare()` returns `WindowedSplits`; graph metadata belongs in `metadata["graph"]`.
- Graph models declare `required_metadata = {"graph"}`; `create_model()` validates requirements and extracts metadata-backed constructor arguments.
- `WindowedSplit.select()` applies the same indices to every aligned field. Preserve missing nodes and unavailable links through masks.
- Non-graph models retain the graph cohort's active communication block and split boundaries, dropping only incomplete node windows.
- Add raw loaders by subclassing `BaseDataset` under `datasets/raw/` and registering in `datasets/raw/__init__.py`. CESTA-DHT ingestion rules are in the root README.
- Add faults in `schema/fault.py`, `injection/faults.py`, and `injection/registry.py`; update `MarkovConfig` defaults when applicable.

## Training and evaluation

- Call `seed_everything(config.seed)` before model construction.
- `Trainer` receives validation data explicitly; preserve aligned oversampling and composed auxiliary objectives. Shared masked loss, decoding, and auxiliary logic belongs in `training/objectives.py`, not model-specific trainer branches.
- Focal loss must have finite gradients at saturated correct-class probabilities, including fractional gamma below one. Abort on nonfinite loss/gradients before optimizer updates.
- CRF likelihood/decoding processes contiguous valid segments independently, without transitions across missing timesteps. Sum segment losses per node-window, then average over nonempty node-windows; masked emissions have zero gradient.
- `standardize_window_features` standardizes per window/node/feature using valid samples, population variance plus `1e-5`, and zero masked outputs. Keep correction features in raw units; deployment and gate audits use identical preprocessing.
- `local_ce_loss_weight` defaults to zero; positive values require local logits and add training-only masked CE without inference changes.
- Default optimizer is Adam with zero decay and no augmentation. `optimizer: adamw` applies decoupled `weight_decay` to all parameters.
- `augmentation_scale`/`augmentation_offset` sample uniform training-only affine transforms per window/node/feature, constant across time, preserving masks and STUCK segments. Offsets use raw units; scale radius must be below one. Window standardization largely cancels these transforms, leaving raw correction features as the main perturbation.
- `Evaluator` owns device placement, masked predictions, metrics, and split-aware communication aggregation. `EvalResult.save()` preserves split-specific artifacts.
- Energy accounting belongs in `evaluation/energy.py`, outside models. Count TX and RX per active directed message using graph-aligned distances; serialize constants, units, distance source, shares, totals, and dense-reference reductions.
- Baseline reconciliation uses manifests and resolved configs. Audits reject missing, duplicate, inconsistent, or incomparable cells without selecting by test performance.

### Distillation

- `distillation_teacher` is a checkpoint directory; `distillation_teachers` is the mutually exclusive ensemble form. `distillation_weight` defaults to zero, `distillation_temperature` to 2, `distillation_local_weight` to zero, and `distillation_drift_weight` to 1.
- The trainer owns frozen eval-mode teachers, excluded from student checkpoints/deployment. Invoke them only on complete training node windows reshaped directly from the aligned graph minibatch; exclude incomplete windows entirely.
- Average ensemble probabilities, not logits. Compute detached targets once per minibatch and reuse for local/final KL; never persist the cache or reuse across batches. Average KL over valid node-window timesteps and multiply by temperature squared. DRIFT reweighting normalizes by summed timestep weights.
- Both KD weights zero must preserve the original path without loading teachers or consuming RNG. Validation retains the original objective/decoder; reject KD plus augmentation.
- Reject dataset hash/feature/window/split mismatches, duplicate/missing teacher seeds, and incomplete checkpoints. Persist teacher/source hashes, including the shared experiment helper. Current graph/link hashes cannot prove historical graph-derived split identity when old manifests lack those hashes.
- `distillation_correct_only` selects detached teacher-argmax agreement with training labels, normalizes selected weights, and returns differentiable zero for empty selections.
- `distillation_task_protection` computes global shared-parameter task/KD dot products and task norm; subtract negative KD projection only when task norm² > `1e-12`. Protect the entire original non-KD objective; leave nonshared gradients unchanged. This does not guarantee per-loss or Adam-step protection. Log per-class retention/projection statistics; zero-norm cosine placeholder 0 is undefined, not orthogonality.

### Initialization, warmup, and scheduling

- `training_initialization_checkpoint` strictly loads matching architecture/data-recipe weights after CSV/metadata hash checks. It creates a fresh optimizer, not an optimizer-state resume; never train on validation teacher outputs.
- `local_warmup_epochs` defaults to 0. Warmup updates only GRU/classifier using training CE plus optional local teacher KL, without graph/CRF updates or validation selection. Joint training uses terminal warmup weights and normally a fresh optimizer; `joint_learning_rate_scale` applies when warmup is enabled.
- `local_warmup_use_task_loss: true` uses the configured focal criterion; `local_warmup_preserve_optimizer: true` retains shared Adam moments across stages. Count warmup epochs separately; selected global epoch is warmup count plus recorded joint epoch. Balanced staged recipes restart Gumbel annealing at joint epoch 1 and derive capped inverse-sqrt weights from valid overlapping training labels only.
- `learning_rate_schedule: late_cosine` requires a start before the final epoch and positive `learning_rate_final` no greater than initial LR. Reject schedules with warmup or nonunit joint LR scale. Update existing optimizer groups before epoch updates without resetting Adam, and log actual `learning_rates`. Default `none` leaves groups untouched.

## Model contracts

### Hydra and HMCT

- `models/temporal/hydra.py` implements a portable PyTorch quasiseparable bidirectional mixer with per-timestep `(batch, time, classes)` output. Do not replace it with window pooling or add the CUDA-only official kernel dependency.
- `models/spatial/hmct.py` performs independent per-node diagnosis using first differences, residual multi-scale dilated convolutions, sinusoidal positions, and a Transformer encoder. Preserve `(batch, time, nodes, classes)` output; HMCT does not perform graph message passing.

### CESTA communication

`CESTAClassifier` in `models/spatial/cesta/` accepts graph-aligned `(batch, window, nodes * features)` input. Modes: `none`, `dense`, `gumbel_request`, `random`, `static_topk`, `local_change`.

- Gates inspect receiver-local state, uncertainty, and edge metadata only, never unreceived sender hidden states. Mask unavailable edges before request aggregation.
- Attention uses receiver queries and received sender keys/values, softmax over the received set only; return zero graph context for an empty set.
- Rule controls use receiver-local scores and edge metadata; persist validation-tuned policies. Random decisions use stable window/timestep/receiver/sender/controller-seed identities.
- Freeze inactive learned-gate parameters in rule modes; persist active and total counts. Payload width follows twice GRU hidden size; report bits as well as request ratios.
- Match bit estimates to transmitted payloads; retain gradient-bearing communication ratios and expected energy for penalties. Evaluation aggregates per-edge requested/possible counts against canonical distances.

### Temporal backends

All alternative backends are opt-in and rejected by firmware export. Preserve default GRU checkpoint keys and Intel recipes.

- `gru`: default shared BiGRU; the main-method architecture constraint is documented in the root README.
- `patchtst`: receiver-local patches, per-token LayerNorm, interpolation to every timestep, and projection to unchanged communication width. Configure via `patchtst_config`; external mask-aware standardization replaces internal scaling.
- `gru_dilated_conv`: replace the second BiGRU layer with LayerNorm, two depthwise kernel-3 convolutions (dilations 1/4), GELU, and pointwise projection. Requires two-layer bidirectional configuration; retains one recurrent layer and payload width. Hidden 128 saves 228,096 parameters.
- `patch_gru`: trailing four-sample stride-one patches, zero left padding, no interpolation, two BiGRU layers. Internal width is seven-eighths of hidden size rounded down to a multiple of eight; projection restores payload width. Requires bidirectionality, two layers, hidden size ≥ 16, and no more encoder parameters than original GRU. Zero masked values/outputs; patches cannot cross missing samples, but recurrence still spans gaps. Hidden 128 saves 33,024 parameters (686,303 versus 719,327 active). Desktop CUDA latency was about 6% slower; accuracy/firmware gains are unestablished.
- `patchtst_config.standalone_compatible: true`: standalone standard-deviation scaling and BatchNorm with frozen running statistics, including during training. Disable external standardization. `temporal_transfer_checkpoint` strictly initializes matched encoder/LayerNorm and embeds the head through an identity-padded payload projection; reload needs no teacher. Width-32 transfer has 356,575 active parameters versus original GRU 719,327.
- Compatible internal scaling includes zero-filled missing samples; incomplete windows are not equivalent to standalone complete-window inference. Masks still control loss/decoding/communication. Do not change trained preprocessing without a separately validated recipe.

Retain tracked experiment files only when they support the paper's Intel results, planned DHT accuracy/macro-F1 comparison, or their reproducibility dependencies. The retained `cesta-dht-gru-control.yaml` and `cesta-window-standardization.yaml` support the DHT training lineage and shared runner. Alternative-backend implementation contracts above remain relevant to checkpoint compatibility, but unrelated experiment recipes and diagnostic results are not maintained here. Do not delete untracked or ignored run artifacts during tracked-file cleanup.
