# CESTA

Communication-Efficient Spatial-Temporal Aggregation for sensor-network fault diagnosis.

## Setup and workflow

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nghiant03/CESTA.git
cd CESTA
uv sync
uv run cesta --help
```

Transform, train, then evaluate a validation-selected checkpoint:

```bash
uv run cesta transform intel_lab data/raw/Intel/data.txt data/datasets/Intel_fault15 \
  --config config/datasets/intel-lab/fault-15.yaml
uv run cesta train config/training/Intel/cesta.yaml data/datasets/Intel_fault15
uv run cesta evaluate --model runs/cesta/<run_id> --data data/datasets/Intel_fault15
```

Use `config/training/Intel/` for preserved Intel recipes and `config/training/CESTA-DHT/` for DHT recipes. Active workflows use connectivity-chronological 70/15/15 splitting before windowing; plain chronological and 80/10/10 splits are unsupported. Lock settings on validation before test evaluation.

## Development reference

- [AGENTS.md](AGENTS.md): development rules, package boundaries, data/model contracts, and training options.
- [Firmware](firmware/README.md): ESP32-S3 configuration, model export, build/flash, runtime contracts, and telemetry.

The paper owns the architecture, baseline descriptions, and diagnostic/energy comparisons, including DHT overall accuracy and macro-F1. Keep detailed results in run artifacts rather than repeating them here.

Each training invocation creates a new `runs/<model>/<run_id>/` containing `weight.pt`, `config.json`, `history.jsonl`, and `manifest.json`. Evaluation writes classification metrics, `predictions.npz`, and `communication_metrics.json` when supported. Validation artifacts must not overwrite test artifacts. Communication estimates count directed TX+RX messages; missing communication artifacts do not mean zero cost.

## CESTA-DHT ingestion

Recorded faults are ingested without synthetic reinjection:

```bash
uv run cesta transform cesta_dht data/raw/CESTA-DHT/readings.jsonl data/datasets/CESTA-DHT --config config/datasets/cesta-dht.yaml
uv run cesta train config/training/CESTA-DHT/cesta.yaml data/datasets/CESTA-DHT
uv run python scripts/run_all_baselines.py --configs config/training/CESTA-DHT/*.yaml --datasets data/datasets/CESTA-DHT --seeds 42 --num-gpus 1 --runs-dir runs/cesta-dht-small-seed42 --keep-progress-log
```

- `esp32_00` through `esp32_03` map to nodes 1 through 4. `edges.csv` specifies unique undirected pairs, expanded to directed edges; current links are 1–3, 1–4, 3–4, and 2–3.
- Select each device's fault stream when present, otherwise normal. Paired clean values remain in `reference_temp` and `reference_humid`, excluded from model features. Graph recipes use `temp`; humidity remains in the artifact.
- Labels follow `fault_mode`: normal=0, spike=1, drift=2, stuck=3. Preserve three-second sampling and missing-node masks without interpolation, averaging streams, or filling faulty readings from clean references.
- Reject duplicate stream timestamps, invalid labels/values, unpaired fault readings, off-grid timestamps, and whole-network gaps. Split recordings with whole-network gaps into contiguous sessions.
- Static availability and 1 m fallback distances are assumptions, not measured radio reliability or geometry.

The original recording spans 2026-10-24 15:16:46 to 2026-11-01 11:08:55 UTC, with 225,444 timestamps and 1,552,994 raw rows. Its selected artifact has 889,219 rows and 15,776/563/563 train/validation/test graph windows at window length 60, training stride 10, and evaluation stride 60. These counts describe the original recording, not adjusted experimental datasets. Node 3 lacks 12,557 samples (5.57%) across two training gaps; its 5–16% clean humidity range warrants calibration review. Clean/fault pairs are paired observations, not independent samples.

Each device has one constant selected class. Device identity and topology confound chronological accuracy; rotate profiles across devices and collect held-out sessions before making generalization claims. A 60-sample window spans about three minutes on DHT, versus five hours on Intel.

## Experiment commands

Run scripts with `uv run python`; inspect supported arguments with `--help`. Experiment output directories must be new. Preserve protocol hashes, resolved configs, interrupted attempts, and historical seed 1242 artifacts. Do not select recipes, fallback candidates, or replacement seeds from test scores.

```bash
uv run python scripts/run_all_baselines.py --dry-run
uv run python scripts/run_all_baselines.py --num-gpus 4 --keep-progress-log
uv run python scripts/audit_decisive_comparison.py \
  --spec config/benchmarks/decisive-comparison.yaml \
  --runs-root runs --output runs/decisive-comparison-audit
uv run python scripts/summarize_decisive_comparison.py \
  --runs-csv runs/decisive-comparison-audit/runs.csv \
  --output runs/decisive-comparison-summary \
  --comparison <variant> <locked-reference>
uv run cesta optimize --data data/datasets/Intel_fault15 --model cnn1d --n-trials 20 --epochs 10
```

Intel radio-energy accounting:

```bash
uv run python scripts/posthoc_spatial_energy.py --output runs/posthoc-spatial-energy
```

### Runner index

The main-method constraint retains the shared BiGRU, receiver-local gate, received-set attention/correction, and CRF. Retained runners support the paper's Intel results or the training/selection lineage of the planned DHT accuracy and macro-F1 comparison. They are validation-only unless explicitly noted; retaining a selection dependency does not make its exploratory results paper results.

| Script under `scripts/` | Protocol |
|---|---|
| `run_cesta_loss_experiment.py` | Seven fixed loss variants on adjusted DHT drift125 and Intel fault05/10/15/20; seed 12, up to 100 epochs, patience 10. Eligibility requires DHT F1 gain, no Intel F1 regression, and no communication increase; lock before 42/1242. `--dry-run` previews without output. |
| `run_cesta_distillation_experiment.py` | Fresh seed-12 GRU control and four KD variants, 100 epochs without early stopping; lock a candidate beating control and teacher before 42/1242. Communication increases are reported, not vetoed. `--ensemble` is a separate two-candidate phase. |
| `run_cesta_dropout_followup.py` | Hidden/dropout 128/.5, 128/.6, 96/.5, 128/.4 with fixed three-teacher ensemble. |
| `run_cesta_seed_replacement.py` | `--replacement-seed 2026 --output runs/<new-experiment>` trains teacher then fixed student, 100 epochs each; post-hoc paired comparison, not seed search. |
| `run_cesta_optimization_followup.py` | Exact saved hidden96/dropout.5 recipe; vary LR .0005/.0015, KD temperature 1/4, half both KD weights, or communication penalty .01. Hashed preflight, then `--execute-approved`; lock before 42/2026. Temperature 1 has final-only precedent; temperature 4 is not equivalent to weaker KD. |
| `run_cesta_schedule_comparison.py` | Fresh constant/cosine pairs at 12/42/2026, 100 epochs; LR .001 through epoch 51, then cosine to .0001 at 100. Primary: paired validation-selected F1 delta; final epoch and epochs 81–100 mean/population SD are secondary. |
| `run_cesta_crf_weight_comparison.py` / `run_cesta_crf_bracket_comparison.py` | CRF training weights .005/.02 against reused sparse .01 controls at 12/42/2026; unchanged decoder. |

### Artifact index

Only result-bearing artifact references are listed here. Detailed configs, metrics, hashes, per-class results, compute accounting, and completion audits belong in run artifacts. All directories below are under `runs/` and are generated artifacts, not checked-in files. Tracked-file cleanup does not remove historical untracked or ignored outputs.

| Artifact directory | Scope |
|---|---|
| `posthoc-spatial-energy` | Intel modeled radio-energy comparison |
| `cesta-locked-test-20261106-2` | Locked DHT checkpoints evaluated for the planned comparison |
| `dht-remaining-baselines-20261106` | Baseline evaluations and consolidated comparison artifacts |

The baseline directory's `combined_results.json` and `summary.json` retain per-seed/per-class results; `recipe_comparison.json` and `communication_summary.json` retain recipes and supported communication accounting.

### Interpretation and consistency notes

- **Recording versus experiment:** the original recording described above is not the adjusted DHT experiment dataset. Match dataset hashes, selected features, split boundaries, and window recipes before comparing artifacts.
- **Offline versus on-device:** evaluating recorded DHT streams on a desktop is offline evaluation, not evidence of live ESP32-S3 diagnosis, decision delay, memory fit, or measured battery savings. The paper's planned online evaluation must remain distinct from completed recorded-stream evaluation.
- **Main method versus variants:** the DHT student uses training-only KD and auxiliary losses not included in the paper's base objective. Report these recipe differences explicitly; an encoder-transfer candidate is not the original BiGRU architecture.
- **Unequal comparisons:** capacity, losses, learning rates, oversampling, teacher-training costs, and tuning budgets differ. HMCT uses Adam with weight decay, not AdamW. Recipe-level rankings alone do not establish the contribution of gates, spatial correction, or CRF decoding; matched ablations are required.
- **Communication interpretation:** missing baseline communication artifacts do not establish zero communication. Lower payload bits from a narrower encoder do not establish selective requests. DHT fallback distances and static links are assumptions; modeled payload TX+RX is not measured device energy or full protocol overhead.
- **Payload accounting mismatch:** `communication.py` counts hidden-state and optional belief widths but excludes raw neighbor measurement features consumed by correction and carried by node deployment. Current bit/energy estimates therefore omit part of the implemented payload, as well as request handshakes, MAC overhead, and idle listening. Reconcile the accounting before treating it as complete payload cost.
- **Deployment decoding mismatch:** the exported node graph returns softmax probabilities without CRF decoding, and firmware publishes only the last timestep's argmax. This is not the paper's full-window Viterbi sequence decoder; do not claim offline/deployment equivalence.
- **Intel preprocessing omission:** the paper's dataset description does not specify voltage filtering, five-minute resampling, linear interpolation, or forward/backward filling. These steps affect reproducibility and include noncausal filling; the offline input pipeline is not an equivalent live pipeline.
- **Selection and holdout exposure:** validation was reused extensively; seed 2026 was a post-hoc replacement. Schedule timing followed observed deterioration, and the upper CRF weight followed the lower-weight comparison. Historical artifacts expose the same test target sequences from earlier CESTA/PatchTST/HMCT runs. Do not describe the comparison as a pristine holdout or claim independent/statistical superiority. Preserve negative results and seed 1242 artifacts; test scores do not authorize reselection.
- **Checkpoint epoch:** the shared helper's `best_epoch` and `first_max_f1_epoch` identify the first maximum validation-F1 epoch, not necessarily the saved checkpoint under energy tie-breaking. Verify checkpoint metadata before attributing a selected epoch. Interrupted attempts remain part of compute accounting.

## Repository layout

| Path | Contents |
| --- | --- |
| [`src/CESTA/`](src/CESTA/) | Datasets, fault injection, models, training, evaluation, CLI |
| [`config/datasets/`](config/datasets/) | Dataset transformation and fault-injection configs |
| [`config/training/`](config/training/) | Default model training configs |
| [`config/experiments/`](config/experiments/) | Intel comparison variants and supporting DHT training recipes |
| [`config/benchmarks/`](config/benchmarks/) | Retained Intel comparison specification |
| [`scripts/`](scripts/) | Benchmark runners, audits, analysis, firmware export |
| [`firmware/`](firmware/) | ESP32-S3 Rust application |
| `data/` | Raw data and transformed datasets |
| `runs/` | Generated checkpoints, metrics, and metadata |

## License

[MIT](LICENSE)
