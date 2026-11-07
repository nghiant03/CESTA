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
- Models may declare `checkpoint_metadata_fields` as an explicit frozenset of descriptive fields retained in legacy `get_config()` outputs. Generic reconstruction excludes only these declared fields from constructor arguments, then validates them against reconstructed metadata; unknown architecture kwargs must still fail. CESTA declares `graph_edge_count` and `gate_input_schema`.
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

### CESTA-DHT ingestion

- `datasets/raw/cesta_dht.py` registers `cesta_dht`; use `config/datasets/cesta-dht.yaml` with the existing transform CLI. It bypasses synthetic injection and preserves supplied labels.
- Explicit node mapping: `esp32_00` through `esp32_03` map to nodes 1 through 4. `edges.csv` contains unique undirected `node_a,node_b` pairs, expanded into both directed edges.
- Select the fault stream for devices that publish one, otherwise normal. Keep paired clean readings as reference columns, excluded from feature names. Never average the two paths or fill missing faulty readings from clean references.
- Preserve three-second sampling and missing-node masks. Reject whole-network gaps rather than silently compressing time; split such recordings into contiguous sessions.
- Static topology and fallback distances are assumptions, not measured radio behavior. Each current device has a constant class, so chronological accuracy is confounded by device identity. Record rotating profiles and held-out sessions for generalization claims.

Extension points:

- **Raw dataset:** subclass `BaseDataset` under `datasets/raw/` and register it in `datasets/raw/__init__.py`.
- **Fault:** update `schema/fault.py`, implement it in `injection/faults.py`, register it in `injection/registry.py`, and add defaults to `MarkovConfig` when applicable.

## Training and evaluation

- Call `seed_everything(config.seed)` before model construction.
- Original training configurations are preserved in `config/training/Intel/`; top-level training configs are smaller-width defaults. The Intel baseline runner uses the preserved configs; pass `--configs config/training/*.yaml` for the smaller DHT sweep.
- Focal loss must keep finite gradients at saturated correct-class probabilities, including fractional gamma below one. The trainer aborts on nonfinite loss or gradients before updating the optimizer.
- `Trainer` receives validation data explicitly and supports focal loss, aligned oversampling, callbacks, and composed auxiliary objectives.
- Keep shared masked loss, decoding, and auxiliary objectives in `training/objectives.py`, not model-specific branches in the trainer.
- Training-only PatchTST distillation uses `distillation_teacher` (checkpoint directory), `distillation_weight` (default zero), and `distillation_temperature` (default 2). The frozen eval-mode teacher is owned by the trainer, never the student or its checkpoint. Only training minibatches invoke it; complete node windows are reshaped directly from the graph batch, preserving shuffled/oversampled alignment. Incomplete node windows are excluded entirely, matching the standalone cohort. KL is averaged over valid node-window timesteps and scaled by temperature squared. The workflow rejects dataset hash/recipe mismatches and augmentation; teacher hashes are persisted. Validation still uses the original objective and decoder. Intel defaults and inference are unchanged. `scripts/run_cesta_distillation_experiment.py --output runs/<new-experiment>` runs full 100 epochs without early stopping: weights 0.3/1 at temperature 2, probability targets at temperature 1, and local-plus-final supervision against a fresh original-GRU control at seed 12. `distillation_local_weight` adds a separately weighted KL term on existing local logits (local-only supervision is also supported). The best validation candidate beating both the control and same-seed teacher is locked before paired seeds 42/1242; communication increases are reported, not vetoed. No test evaluation.
- `local_ce_loss_weight` defaults to zero and adds training-only masked cross entropy on the existing local logits. Positive weights require a model exposing local logits; inference parameters, forward computation, and decoding remain unchanged.
- Training defaults remain Adam with zero weight decay and no augmentation. `optimizer: adamw` and `weight_decay` opt into decoupled decay on all parameters. `augmentation_scale` and `augmentation_offset` sample uniform affine transforms per window/node/feature, constant across time, training-only, preserving masks and constant STUCK segments. Offsets use raw feature units; scale is a fractional radius below one. Window standardization largely cancels these transforms, so CESTA augmentation primarily perturbs raw correction features.
- `scripts/run_cesta_regularization_experiment.py --output runs/<new-experiment> --diagnostic-run <prior-run>` diagnoses saved validation argmax versus CRF, then screens a fresh DHT-only gamma-1/CRF-0.01 control, two affine strengths, and AdamW decay 0/0.001/0.01 in that order. Changes are isolated, not combined. It retains CRF and never retrains Intel or evaluates test. Improved validation candidates with no communication increase are locked before paired seeds 42/1242 confirmation; artifacts and source hashes are preserved. Use `uv run python` and `--dry-run` to inspect the recipe.
- `uv run python scripts/run_cesta_loss_experiment.py --output runs/<new-experiment>` runs seven fixed loss variants on adjusted DHT (`CESTA-DHT_1min_fault15_stuck0_drift125`) and Intel fault05/10/15/20. It uses fresh seed-12 controls, fixed standardized GRU architecture and CRF decoding, up to 100 epochs, and patience 10. Selection requires strictly improved DHT validation macro-F1, zero Intel validation F1 regression, and no increase in requests, bits, or estimated energy on any dataset. It locks the eligible candidate with highest DHT gain before paired control/candidate seeds 42 and 1242, then checks three-seed means without selecting a replacement. All evaluation is validation-only; no test runs launch automatically. Output directories must be new; provenance, source hashes, incremental results, selection and confirmation are persisted. `--dry-run` validates the matrix without creating output. These exploratory validation results do not establish test superiority or statistical noninferiority.
- CESTA CRF likelihood and decoding treat each contiguous valid segment independently, without transitions across missing timesteps. Sum segment losses per node-window, then average over nonempty node-windows; masked emissions contribute no gradient.
- CESTA optionally standardizes GRU inputs per window, node, and feature with `standardize_window_features`; exclude missing samples, use population variance plus `1e-5`, and zero masked outputs. Keep correction features in raw units and use the same preprocessing in deployment and gate audits. The opt-in Intel-sized seed-12 ablation is `config/experiments/ablations/cesta-window-standardization.yaml`; preserved Intel defaults remain unchanged.
- `Evaluator` handles device placement, masked predictions, metrics, split-aware communication aggregation, and validation-only checkpoint evaluation.
- `EvalResult.save()` writes classification artifacts and `communication_metrics.json` when available. Validation evaluation must not overwrite test artifacts.
- Energy accounting belongs in `evaluation/energy.py`, outside models. Count TX and RX for each active directed message using graph-aligned distances and serialize constants, units, distance source, shares, totals, and dense-reference reductions.
- The baseline runner reconciles completed cells from manifests and resolved configs. The benchmark auditor rejects missing, duplicate, inconsistent, or incomparable cells without selecting by test performance.
- `scripts/tune_cesta.py` tunes standardized GRU-based CESTA using 15 seed-12 trials, validation macro-F1 and early stopping (patience 10). It preserves base communication penalties, searches learning rate/width/depth/dropout/CRF weight, stores an Optuna SQLite study in a new output directory, locks the selected YAML before three-seed test confirmation, and never evaluates test during search. Existing output directories are rejected to prevent accidental study mixing.

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

Ensemble distillation: `distillation_teachers` accepts multiple checkpoint directories (mutually exclusive with `distillation_teacher`). Average their probabilities at the configured temperature, not their logits. `distillation_drift_weight` defaults to 1; larger values weight DRIFT-labeled training timesteps and normalize by the sum of timestep weights. Both teacher weights zero preserve the original path without loading teachers or consuming RNG. Configuration validation rejects KD plus augmentation. Teacher provenance includes current graph/link hashes but explicitly cannot verify historical graph-derived split identity for old manifests lacking those hashes. Use `scripts/run_cesta_distillation_experiment.py --ensemble --output runs/<new-experiment>` for a separately labeled exploratory two-candidate phase after previous confirmation; all runs use 100 epochs and validation only. Teacher preflight rejects duplicate/missing seeds, incomplete checkpoints and incompatible recipes. Source provenance includes the shared experiment helper.

Paper-aligned improvement must retain the original shared BiGRU, receiver-local gate, received-set attention/correction, and CRF. Alternative PatchTST-transfer results are not acceptance for that constraint. Training-only local warmup (`local_warmup_epochs`, default 0) updates only GRU/classifier from training batches using CE plus local teacher KL; no graph module/CRF updates or validation selection occur during warmup. Joint training uses a fresh optimizer, optionally scaled by `joint_learning_rate_scale`; extra warmup epochs must be disclosed separately. `scripts/run_cesta_staged_experiment.py` is a new exploratory screen, not a continuation of previous locked confirmations.

`training_initialization_checkpoint` is an opt-in training-only strict weight initialization; workflow checks architecture, feature/window/split recipe and dataset CSV/metadata hashes. It starts a new optimizer, not an exact optimizer-state resume. `scripts/run_cesta_refinement_experiment.py` records source checkpoint hashes and tests fixed label-supervised refinement schedules; it never trains on validation teacher outputs. Staged/extended-KD/refinement experiments have not met the original-BiGRU PatchTST-superiority target. Warmup zero and initialization null leave Intel defaults unchanged.

Exact staged warmup is opt-in via `local_warmup_use_task_loss: true` and `local_warmup_preserve_optimizer: true`: configured focal criterion and shared Adam state are used across local/joint stages. Defaults retain earlier behavior and leave Intel unchanged. The reviewed three-recipe screen in `scripts/run_cesta_balanced_staged_experiment.py` uses total100 epochs (joint100 or local20+joint80), training-only capped inverse-sqrt focal weights, and reused reference controls. It completed with no eligible seed12 candidate; do not imply confirmation or teacher superiority. Report selected global epoch as warmup count plus recorded joint epoch.

Conflict-aware KD is training-only and optional: `distillation_correct_only` selects detached teacher-argmax agreement with training labels on complete node windows, normalized by selected weights, with differentiable zero for empty selection. `distillation_task_protection` computes global shared-parameter task/KD dot and task norm, subtracts negative KD projection only when norm²>1e-12, and leaves nonshared gradients unchanged. Protect the entire original non-KD objective, not just focal; do not claim per-loss or Adam-step protection. Logs include per-class retention and projection statistics; zero-norm cosine placeholder0 is undefined, not orthogonality. Audited two-candidate100epoch screen completed without eligible seed12 candidate; no confirmation or test evaluation.

The capacity screen `scripts/run_cesta_capacity_experiment.py` prospectively compares GRU hidden96/dropout.2, hidden64/.2, hidden128/.4, hidden96/.4 with unchanged two-layer bidirectionality and KD objective. Payload width follows twice hidden size; report bits rather than request ratio alone. Hidden128/.4 was locked and confirmed at89.4235% mean, still below PatchTST89.8693%; no fallback selected after confirmation. All variants remain opt-in and Intel configs unchanged.

Dropout-followup locked hidden96/.5 reached89.8583% three-seed mean versus PatchTST89.8693% (target still unmet, no rounding to a win). It has429,279 active parameters and192-wide payload but100% available-edge requests on all seeds; lower bits arise from payload reduction, not selective gates. Keep this distinction in claims. Artifacts `runs/cesta-gru-dropout-followup-20261106` preserve all four prospective screening cells and locked confirmation.

`scripts/run_cesta_seed_replacement.py` is a user-directed post-hoc paired comparison, not a tuning loop: one chosen new seed trains teacher then fixed student,100epochs each validation-only, reusing12/42 and preserving1242. It records prior artifact hashes and both original/replacement sample-SD summaries. Do not rewrite original conclusions or describe replacement as independent confirmation.

The fixed-width optimization followup `scripts/run_cesta_optimization_followup.py` clones the exact saved hidden96/dropout0.5 config and refuses schema/default drift. Its six prospective variants change LR0.0005/0.0015, KD temperature1/4, halve both KD weights, or raise communication penalty to0.01; gate entropy stays0.001. Preflight writes a hashed protocol; `--execute-approved` verifies inputs before100-epoch sequential training. The completed screen locked communication0.01 at seed12 F1 0.9145803133390049 before42/2026; confirmation mean0.898689289553445 improves incumbent0.8984286702591195 but remains below PatchTST0.8989147627978472. No fallback or new seed was selected. Mean requests39.0104%, bits32,212,992, modeled TX+RX energy3.221621J; all counts unchanged (429,279 active/466,528 total, payload192). Artifacts `runs/cesta-gru-optimization-followup-20261106` preserve original1242 history and full protocol audit. Treat this as exploratory reused validation/seeds, not independent superiority. The shared helper's `best_epoch` is first-max validation F1, not guaranteed saved checkpoint identity under energy ties. Temperature1 has prior final-only precedent; temperature4 is not just weaker KD. No tests or Intel default changes.

The four-cell communication followup `scripts/run_cesta_communication_followup.py` completed100 epochs per seed12 recipe: penalty0.003/0.03, penalty0.01+halfKD, and penalty0.01+entropy0. F1 values0.9045083990193599/0.9049702664198804/0.9004201799642552/0.9082140839698073 all missed the locked threshold0.9145803133390049. No candidate was selected; no42/2026 confirmation, rescue, fallback, or test evaluation occurred. Artifacts `runs/cesta-gru-communication-followup-20261106` include exact-repeat comparison against331 saved configs, full recursive Intel file inventory/content,489 locked hashes, and a completion audit rechecking all hashes and100-epoch resolved configs. Counts and architecture remain unchanged. Record-threshold screening favors seed12 specialists, not generalization; preserve earlier dense/sparse/teacher paired conclusions and historical1242. The main result field `first_max_f1_epoch` does not assert saved checkpoint epoch identity.

Optional training-only `learning_rate_schedule: late_cosine` requires `learning_rate_schedule_start_epoch` before the final epoch and positive `learning_rate_final` no greater than initial LR. Default `none` leaves optimizer groups untouched; schedule plus warmup or nonunit joint LR scale is rejected. The trainer changes existing optimizer group rates before epoch updates without resetting Adam state and logs actual `learning_rates` in history. `scripts/run_cesta_schedule_comparison.py` preflights a fixed six-run paired comparison: fresh sparse constant then late cosine at12,42,2026; all100 epochs, no triage or fallback. Cosine starts at epoch51 with0.001 and ends at100 with0.0001. Primary is mean paired validation-selected F1 delta; final-epoch F1 and81–100 mean/population SD are secondary, never replacement selection rules. Intel configs and inference remain unchanged.

The paired schedule comparison completed all six100-epoch runs: cosine mean0.8984795541383741 versus fresh constant0.898689289553445, primary paired delta−0.00020973541507089225 (sample SD0.0026929552180701834). No primary improvement or PatchTST superiority; final-epoch/stability improvements cannot rescue this result. Timing was chosen after observed validation deterioration, so results are exploratory, not independent. Three interrupted attempts (83/9/0 logged epochs) were preserved and restarted unchanged; disclose92 additional logged epochs. All492 lock hashes and full Intel inventory passed completion audit; paired histories agree exactly through51, and the restarted cosine12 reproduces the original83 epochs. Artifacts `runs/cesta-gru-schedule-comparison-20261106` preserve runtime checks, interruptions, selected per-class/communication results, fixed secondary summaries, and original1242 history. No fallback/test evaluation.

The config-only CRF-weight comparison `scripts/run_cesta_crf_weight_comparison.py` completed fixed12/42/2026 candidates at training CRF0.005 versus reused sparse constant controls at0.01. Decoder retained, no inference scaling. Mean0.8962342626444991 versus control0.898689289553445; paired mean delta−0.0024550269089458343, no primary improvement. Mean request31.2054%, bits25,767,936, modeled energy2.577051J do not rescue failure; selected STUCK F1 declines all three seeds. Artifacts `runs/cesta-gru-crf-weight-comparison-20261106` preserve539 hashes, full Intel inventory, diagnostic provenance, historical1242, and fixed secondaries. Seed2026 interruption after95 logged epochs was preserved and restarted unchanged, adding95 logged epochs beyond300 completed. No fallback, extra validation boundary diagnostic, test evaluation, or statistical superiority claim.

The separately locked CRF0.02 directional bracket `scripts/run_cesta_crf_bracket_comparison.py` completed100 epochs each12/42/2026 against reused sparse0.01 controls. Candidate mean0.9009683617958909; paired control mean gain0.0022790722424459453 (sample SD0.001264192836334102), all three positive. Teacher mean0.8989147627978472 is lower descriptively, but teacher wins42/2026; no independent/statistical superiority claim. Direction chosen after failed0.005, whose negative result remains preserved; do not call this retrospective best-of-weight selection. Stronger0.05/0.1 had different historical recipes. STUCK precision/F1 decline all three, contradicting proposed STUCK mechanism. Mean requests40.3051%, bits33,282,048, modeled energy3.328538J increase versus sparse control. All590 hashes/full Intel inventory passed. Artifacts `runs/cesta-gru-crf-bracket-comparison-20261106` preserve a seed42 interruption before logged epochs and unchanged restart. No test/fallback/seed switching; decoder and counts unchanged.

The user-authorized locked test evaluation completed all nine existing CRF0.02/PatchTST/sparse0.01 checkpoints at12/42/2026 in `runs/cesta-locked-test-20261106-2`, without retraining or reselection. Means89.5615/88.4438/89.7496%; primary paired teacher gain+1.1176pp (sample SD1.0871pp), secondary sparse delta−0.1882pp (SD0.3065pp). Candidate beats teacher all three but does not improve sparse mean; never switch candidates from these test scores. Exact matched6,720-target cohorts, no partial windows; fixed windowing drops12 trailing timestamps. All829 input hashes passed before narrow documentation updates. Historical same-dataset results already expose identical test target sequences (nine earlier CESTA/PatchTST/HMCT artifacts); none are these selected checkpoints. Disclose extensive validation optimization, post-hoc seed replacement and prior test exposure; do not claim pristine holdout or statistically established superiority. Original1242 preserved. Unsuffixed directory retains a pre-inference runner shape-check failure, corrected only in the fresh runner before any evaluation. Per-class/confusion and communication details are persisted; candidate modeled mean energy3.512262J exceeds sparse3.411900J, with1m fallback distances. Source/Intel configs unchanged; Ruff/Pyright passed.

The remaining standalone DHT comparison completed all30 fixed12/42/2026 cells in `runs/dht-remaining-baselines-20261106`: ten canonical configs, fresh100epochs without early stopping, explicit no-test training followed by one test evaluation per validation-selected checkpoint. No exact historical recipe was reusable. Every evaluation matches the6,720-target cohort; all locked input hashes and complete Intel inventory passed. DCRNN2026 interruption retained71 extra logged epochs before an unchanged fresh restart. Hydra leads the ten new baselines at73.4379 ±6.3244% macro-F1; prior locked CESTA/PatchTST/sparse results remain89.5615/88.4438/89.7496%. Preserve unequal parameter, loss, LR, oversampling, teacher-training and tuning-budget disclosures (HMCT uses Adam weight decay0.01, not AdamW). New baselines have no communication artifact; do not substitute zero. Combined per-seed/per-class/sample-SD, capacity, recipe and supported communication summaries are persisted; prior exposure/post-hoc2026/exploratory limitations remain. No candidate reselection, source/default changes or tests.

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

- `temporal_backend` defaults to `gru`; opt-in `patchtst` uses receiver-local patches, per-token LayerNorm (no cross-node batch statistics), interpolation to every timestep, and projection to the unchanged communication width. `patchtst_config` controls its dimensions. Shared mask-aware window standardization is external to the encoder; internal PatchTST scaling is disabled.
- Compare `config/experiments/ablations/cesta-patchtst.yaml` against `cesta-window-standardization.yaml` with seed 12 on Intel fault05/10/15/20. Both retain identical losses, standardization, graph width and corrected CRF semantics; encoder capacity differs. Firmware export currently rejects the PatchTST backend.
- Opt-in `temporal_backend: gru_dilated_conv` replaces the second bidirectional GRU layer with a residual block containing LayerNorm, two depthwise kernel-3 convolutions (dilations 1/4), GELU and a pointwise projection. It requires `num_layers: 2` and `bidirectional: true`; only one recurrent layer remains. Output/payload width and downstream modules stay unchanged. At hidden size 128 it removes 228,096 parameters. Defaults and existing GRU checkpoint keys remain unchanged; firmware export rejects this unvalidated backend.
- Opt-in `temporal_backend: patch_gru` uses trailing four-sample, stride-one patches, retaining both bidirectional GRU layers. At each timestep the patch ends at that timestep; left padding is zero, with no interpolation. Internal recurrent width is seven-eighths of `hidden_size`, rounded down to a multiple of eight (112 at hidden size 128); a learned projection restores the unchanged 256-wide payload. It requires two bidirectional layers and hidden size at least 16, and rejects encoder parameter counts above the equivalent original GRU. Masked values are zeroed, patches cannot span a missing sample, and masked encoder outputs are zero; recurrence itself still spans gaps as in the baseline. Default GRU keys and behavior remain unchanged. Firmware export rejects this unvalidated backend.
- Compare `cesta-dht-patch-gru.yaml` against `cesta-dht-gru-control.yaml` on `CESTA-DHT_1min_fault15_stuck0_drift0625`, and `cesta-patch-gru.yaml` against `cesta-window-standardization.yaml` on Intel. Only the backend differs within each pair. At hidden size 128 the patch backend saves 33,024 parameters (686,303 trainable versus 719,327 for the DHT control). Initial desktop batch-one, four-node, 60-step checks found CPU inference slightly faster but CUDA inference approximately 6% slower; this is experimental and has not met the no-latency-increase acceptance criterion. No accuracy improvement or firmware timing has been established; lock validation-selected settings before any test evaluation.
- Compare `cesta-gru-conv.yaml` with `cesta-window-standardization.yaml` on Intel fault05/10/15/20, and `cesta-dht-gru-conv.yaml` with `cesta-dht-gru-control.yaml` on adjusted DHT. DHT uses focal gamma 1/CRF weight 0.01; Intel retains gamma 2/weight 0.05. Lock these matched configs before test evaluation; do not choose variants from test scores. Desktop latency/memory measurements do not establish firmware latency.

- `patchtst_config.standalone_compatible: true` opts into standalone PatchTST internal standard-deviation scaling and BatchNorm architecture. BatchNorm running statistics are frozen even in training, preventing cross-node statistics in receiver-local features. Disable external window standardization. `temporal_transfer_checkpoint` initializes the encoder/LayerNorm strictly from a matched standalone checkpoint and embeds its head through an identity-padded payload projection. It is training-only and not required to reload the student. `scripts/run_cesta_transfer_experiment.py` screens fresh versus pretrained width32 local PatchTST, preserving the 256-wide payload and all graph communication/correction/CRF modules. The resulting model has 356,575 trainable parameters versus 719,327 original GRU; firmware export remains rejected. Historical graph provenance limitations still apply.
- Distillation computes detached ensemble probabilities once per training minibatch and reuses them for local/final losses; the cache is minibatch-local, never persisted or shared across batches.

Transfer acceptance diagnostics are recorded under `runs/cesta-patch-transfer-20261106`: full forward plus CRF is faster on single-thread CPU but slower on CUDA (6.46/7.09 ms versus GRU 11.07/5.47 ms). Do not claim latency non-regression or firmware readiness. Zero-neighbor validation diagnostics lower F1 on every confirmed checkpoint; no retraining/selection was performed. Fresh post-transfer Intel disabled-path checks preserve all 51 state tensors exactly; this is runtime invariance, not a full Intel noninferiority experiment.

The opt-in portable architecture YAML is `config/experiments/ablations/cesta-dht-patch-transfer.yaml`; supply a local matched teacher path only when opting into initialization. Compatible PatchTST internal scaling includes zero-filled missing samples; incomplete node windows are not equivalent to standalone complete-window inference. Do not silently alter trained preprocessing to address this limitation; any mask-aware redesign requires a separately validated recipe.

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
