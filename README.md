# CESTA

Communication-Efficient Spatial-Temporal Aggregation for sensor-network fault diagnosis.

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nghiant03/CESTA.git
cd CESTA
uv sync
uv run cesta --help
```

## Quickstart

```bash
# 1. Transform raw Intel Lab data into a fault-injected graph dataset
uv run cesta transform intel_lab data/raw/Intel/data.txt data/datasets/Intel_fault15 \
  --config config/datasets/intel-lab/fault-15.yaml

# 2. Train cesta
uv run cesta train config/training/Intel/cesta.yaml data/datasets/Intel_fault15

# 3. Evaluate on the test split 
uv run cesta evaluate --model runs/cesta/<run_id> --data data/datasets/Intel_fault15
```

## Paper-architecture constraint

The main CESTA method retains the shared bidirectional GRU, receiver-local request gate, received-set attention/correction, and CRF. PatchTST encoder-transfer experiments are alternative-architecture research only and **do not satisfy this main-method constraint**, despite higher validation accuracy.

Training-only staged GRU exploration uses `scripts/run_cesta_staged_experiment.py --output runs/<new-experiment>`. `local_warmup_epochs` (default 0) runs local cross entropy plus optional teacher KL on training windows only, updating only the existing GRU/classifier. It uses the terminal warmup state, with no warmup validation selection. The ordinary joint communication/CRF phase then starts with a fresh optimizer; `joint_learning_rate_scale` applies only when warmup is enabled. Defaults and inference remain unchanged. Warmup epochs are additional to configured joint `epochs` and must be included in compute accounting.

The first paper-aligned staged screen did not close the gap: 100 local warmup + 100 joint epochs scored 88.45% (seed12), or 87.96% at reduced joint learning rate. Extending ordinary local/final KD to 200 epochs reproduced its earlier three-seed 88.9882% best checkpoints exactly. A separately labeled low-learning-rate supervised refinement from those KD checkpoints reached 88.4243% mean with DRIFT-weighted CE, below the starting recipe. These negative exploratory results are preserved under `runs/cesta-gru-staged-20261106` and `runs/cesta-gru-refinement-20261106`; the main BiGRU acceptance target remains unmet. `training_initialization_checkpoint` optionally loads strict weights from an identical-architecture/data-recipe run before a fresh optimizer; defaults are null. Initialization uses validation-selected prior checkpoints, so this is additional exploratory validation reuse, not independent confirmation.

The independently reviewed exact focal20+joint80 follow-up is `scripts/run_cesta_balanced_staged_experiment.py`. It preserves local Adam moments, uses configured gamma-1 focal in both stages, and restarts Gumbel annealing at joint epoch 1. Its class weights derive only from valid overlapping training-window labels: inverse-square-root frequencies capped at 3:1, normalized to train-frequency-weighted mean 1. Here all three fault classes hit the cap, so this is all-fault balancing, not DRIFT-specific. Seed12 scores were 87.9088% (joint100 balanced), 88.0251% (20+80 balanced), and 88.9081% (20+80 unweighted), all below the 89.1309% seed12 teacher; the preregistered gate selected no candidate, so no confirmation ran. Total budget: 300 epochs, 15.60 minutes recorded training. Artifacts: `runs/cesta-balanced-staged-20261106`. Strongest confirmed original-BiGRU recipe remains 88.9882% mean versus PatchTST 89.8693%; the paper-aligned target remains unmet.

Conflict-aware training-only KD (`scripts/run_cesta_conflict_experiment.py`) has two opt-in flags: `distillation_correct_only` keeps teacher-correct training tokens and normalizes selected-token loss; `distillation_task_protection` projects only conflicting combined KD gradients on the concatenated GRU/classifier gradient, preserving the full original non-KD objective gradient. Nonshared gradients remain unchanged; this is not a guarantee about the Adam step or each individual objective. Both default false. The audited100epoch seed12 screen scored 87.4031% (correct-only) and 88.2010% (task-protected), below teacher89.1309% and prior KD90.0087%; no candidate qualified for confirmation. Projection activated on41.37% of batches; teacher-correct filtering retained98.999/95.379/85.649/86.790% of NORMAL/SPIKE/DRIFT/STUCK training tokens. Artifacts: `runs/cesta-conflict-20261106`. Main BiGRU target remains unmet; prior confirmed KD88.9882% versus teacher89.8693% mean remains the comparison.

Paper-aligned capacity/dropout screen (`scripts/run_cesta_capacity_experiment.py`) keeps two bidirectional GRU layers and all original graph/CRF components, varying only hidden width/dropout under the fixed local1/final0.3 KD recipe. Historical Optuna trials did not duplicate this matrix. Seed12: hidden96/dropout0.2 88.4321% (429,279 active parameters), hidden64/dropout0.2 89.9297% (212,959), hidden128/dropout0.4 90.3787% (719,327), hidden96/dropout0.4 89.9584% (429,279). The locked winner hidden128/dropout0.4 confirmed at 89.5237/88.3680% on seeds42/1242, yielding **89.4235 ± 1.0091%**, above prior KD88.9882% but below PatchTST89.8693%. Mean requests95.37%, modeled energy10.50183J; no success claim. Smaller variants remain single-seed only, not fallback confirmation winners. Six100epoch runs consumed33.01minutes recorded training; artifacts `runs/cesta-gru-capacity-20261106`. Intel defaults and architecture formulas remain unchanged.

The next prospectively locked dropout follow-up (`scripts/run_cesta_dropout_followup.py`) screened hidden128/dropout.5, hidden128/.6, hidden96/.5, and hidden128/.4 with fixed-three-teacher ensemble. Seed12 F1:90.6235/89.5340/90.6966/88.4111%. Hidden96/.5 won and confirmed at88.6587/90.2198% on42/1242: mean **89.8583 ± 1.0659%**, still **0.0110 percentage points below** PatchTST89.8693%, not a win. Active/total parameters429,279/466,528, payload192; every confirmed checkpoint requests100% of available edges. Bits82,575,360 and modeled energy8.25836J reflect narrower payload, **not learned selectivity**. Receiver-local/missing-edge checks pass. Six100epoch runs took34.22minutes recorded training. Artifacts `runs/cesta-gru-dropout-followup-20261106` include the prior-phase inventory; repeated validation exploration prevents independent-generalization claims.

For an explicitly requested single replacement seed, `uv run python scripts/run_cesta_seed_replacement.py --replacement-seed 2026 --output runs/<new-experiment>` trains matched standalone PatchTST first, then the locked hidden96/dropout.5 GRU using that same-seed teacher. It preserves old artifacts and writes old/new paired summaries side by side. This is **post-hoc seed replacement, not independent superiority evidence**; do not search replacement seeds until a favorable result appears. It does not change global defaults or earlier experimental seed sets.

### Fixed-width optimization followup

The prospective six-cell screen in `scripts/run_cesta_optimization_followup.py` retains hidden96/dropout0.5, both original bidirectional GRU layers, receiver-local gate, attention/correction, CRF, and gate entropy0.001. It changes only LR (0.0005/0.0015), KD temperature (1/4), both KD weights (halved), or communication penalty (0.01), one recipe at a time. The runner first writes and hashes a complete protocol; only `--execute-approved` launches its sequential100-epoch, validation-only screen and locked42/2026 confirmation. Intel defaults and inference parameter counts are unchanged:429,279 active,466,528 total,192-wide payload. Temperature1 has a prior final-only KD precedent; temperature4 is not equivalent to simply weakening KD.

Completed artifacts: `runs/cesta-gru-optimization-followup-20261106`. Seed12 macro-F1 percentages in fixed order were89.6179,89.6807,89.7053,88.8661,90.9043,91.4580. Communication penalty0.01 was locked before confirmation; its12/42/2026 results were91.4580/88.9029/89.2458%, mean89.8689 ±1.3868% (sample SD), versus incumbent89.8429 ±1.0584% and PatchTST89.8915 ±1.2654%. **PatchTST superiority remains unmet**: mean improvement over incumbent is0.0261 percentage points, but the teacher gap is−0.0225 points. No fallback was selected; original1242 results remain preserved separately.

Unlike the dense incumbent, the selected candidate requests38.5342/37.1503/41.3467% of available edges; mean transmitted bits32,212,992 and modeled TX+RX energy3.221621J are60.9896% below the same-width dense reference. These are aggregate validation communication estimates with fallback1m distances, not measured radio/firmware energy. This is exploratory repeated validation and seed reuse, not independent superiority evidence or a guaranteed gain. No test evaluation occurred. The shared helper's `best_epoch` means the first maximum validation-F1 epoch; energy tie-breaking can select a different saved checkpoint epoch. Full100-epoch and locked-config checks are in `completion_audit.json`.

The subsequent independently reviewed four-cell communication screen (`scripts/run_cesta_communication_followup.py`, artifacts `runs/cesta-gru-communication-followup-20261106`) completed100 epochs each at seed12. Penalties0.003/0.03, penalty0.01 with half KD, and penalty0.01 with entropy reward removed reached90.4508/90.4970/90.0420/90.8214% macro-F1. **None exceeded the locked sparse incumbent91.4580%**, so no42/2026 confirmation or fallback launched. Requests were56.6741/31.1310/44.2485/42.6935%, respectively; communication savings did not override accuracy eligibility. No new three-seed comparison exists for this phase. All489 locked source/reference/data/config hashes and the complete recursive Intel file inventory were independently rechecked after training. The seed12 record threshold favors seed12 specialists and is not evidence of generalization; original1242 history is unchanged. Full resolved configurations were checked against331 historical configurations before training, with no exact repeats.

### Paired late-cosine comparison

Training-only `learning_rate_schedule: late_cosine` accepts a start epoch and final LR, updates existing optimizer groups without resetting Adam, and logs actual `learning_rates`. Default `none` leaves optimizer rates untouched; warmup/nonunit joint LR scaling cannot be combined with a schedule. `scripts/run_cesta_schedule_comparison.py` ran fresh sparse constant and cosine controls for12/42/2026 in fixed paired order,100 epochs each, with no triage or fallback. LR was0.001 through51, then cosine-decayed to0.0001 at100. The decay timing was chosen after observing validation deterioration: this is exploratory reused validation, **not independent confirmation**.

Artifacts `runs/cesta-gru-schedule-comparison-20261106` show cosine F1 means89.8480 ±1.1841% versus fresh constant89.8689 ±1.3868% (sample SD), paired mean difference−0.02097 percentage points. PatchTST remains89.8915 ±1.2654%; **the primary improvement target was not met**. Final-epoch mean F1 improved87.8681→88.7333%, and mean within-run epochs81–100 population SD fell0.7651→0.4040 percentage points, but these prespecified secondaries do not replace the primary result. Selected DRIFT F1 improved in all three pairs while STUCK F1 declined in all three. Mean cosine requests38.7178%, bits31,971,328, modeled TX+RX3.197453J; constant39.0104%,32,212,992bits,3.221621J. Architecture, counts and Intel configs remain unchanged.

Three externally interrupted attempts were preserved (83,9,0 logged epochs); incomplete cells restarted from scratch using unchanged configs, not checkpoint continuation or score selection. All six final runs completed100 epochs; interrupted attempts add92 logged epochs of compute. The restarted cosine12 reproduced its original83-epoch history, and all paired histories match exactly through51. Completion audit rechecked492 locked hashes and full Intel inventory. No test evaluation or secondary-winner switching occurred.

### Reduced CRF training-weight comparison

The config-only `scripts/run_cesta_crf_weight_comparison.py` retained sparse constant-LR architecture and CRF decoding, lowering only training CRF weight0.01→0.005. All three12/42/2026 candidates completed100 epochs; controls were reused fresh constant runs from the preceding schedule study, not newly concurrent training. Candidate F1 was89.7867/89.2605/89.8231%, mean89.6234 ±0.3148% (sample SD), versus control89.8689 ±1.3868%. Primary paired mean delta−0.24550 percentage points: **no improvement or PatchTST superiority**. Mean requests fell39.0104→31.2054%, transmitted bits32,212,992→25,767,936, and modeled TX+RX energy3.221621→2.577051J, but communication cannot rescue the failed primary metric. Selected STUCK F1 fell on all three seeds; lowering a training weight is not direct inference transition scaling or a guaranteed boundary-error fix.

Artifacts `runs/cesta-gru-crf-weight-comparison-20261106` retain all configurations, per-class results, final-epoch/late-window secondary metrics, historical1242 references, and539 verified hashes plus unchanged full Intel inventory. Seed2026 was interrupted after95 logged epochs and restarted unchanged from scratch; the partial attempt is preserved, adding95 logged epochs beyond300 completed epochs. This hypothesis followed exploratory training-stage diagnostics and reused validation; no statistical superiority claim, fallback, extra validation boundary diagnostic, or test evaluation.

### Opposite-direction CRF bracket

The separately locked single CRF0.02 hypothesis (`scripts/run_cesta_crf_bracket_comparison.py`, `runs/cesta-gru-crf-bracket-comparison-20261106`) completed all three100-epoch12/42/2026 candidates against the same reused sparse0.01 controls. F1 was91.6521/89.0248/89.6136%, mean90.0968 ±1.3787% (sample SD): paired control gains+0.1941/+0.1218/+0.3678 percentage points, mean+0.22791pp. Mean exceeds PatchTST89.8915% by0.20536pp, but candidate still loses to teacher on42/2026. This is **descriptive exploratory mean superiority only**, not independent/statistical confirmation: the direction was chosen after the failed0.005 study, whose negative result remains visible above. Stronger0.05/0.1 weights were historically tested in different recipes; this is not an unprecedented mechanism or a retrospective best-of-weight selection.

The proposed STUCK mechanism was not supported: selected STUCK precision and F1 decline on all three seeds despite macro-F1 gains. Mean requests40.3051%, bits33,282,048 and modeled TX+RX3.328538J increase relative to sparse control39.0104%,32,212,992 and3.221621J. Final-epoch mean F1 improves87.8681→88.2084%, whereas late20 mean declines88.3917→88.3112%; secondary metrics do not choose a winner. Original architecture/decoder/counts remain unchanged. All590 lock hashes and full Intel inventory were verified. Seed42 was interrupted before any logged epoch and restarted unchanged; partial attempt preserved. No test evaluation, fallback, seed switching, or extra boundary diagnostic occurred.

### Locked test evaluation (after validation optimization)

User-authorized evaluation of the nine existing checkpoints completed sequentially in `runs/cesta-locked-test-20261106-2`, with no training, tuning, reselection, or seed changes. Temperature-only inputs (`temp`), the connectivity-chronological70/15/15 split,60-step windows/stride60, original decoders and graph masks were retained. The protocol locked829 input hashes, including all nine weights/configs/manifests, full data/graph, source and Intel inventory; all passed before documentation updates.

| Test macro-F1 (%) | Seed12 | Seed42 | Seed2026 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Locked CESTA CRF0.02 |90.2831|89.4790|88.9223|89.5615 ±0.6842|
| Same-seed PatchTST |89.4753|89.2598|86.5963|88.4438 ±1.6036|
| Prior sparse CRF0.01 |90.4791|89.3568|89.4129|89.7496 ±0.6324|

Primary paired candidate-minus-teacher gain is+1.1176pp (sample SD1.0871pp), positive on all three seeds. Secondary candidate-minus-sparse delta is−0.1882pp (sample SD0.3065pp): the locked candidate does **not** improve upon sparse0.01 on test. This result does not trigger reselection. Candidate mean requests42.5298%,35,119,104 bits and modeled TX+RX3.512262J exceed sparse41.3145%,34,115,584 bits and3.411900J. Energy uses assumed1m fallback distances, not measured device energy; standalone PatchTST has no inter-node communication artifact.

Cohort audit verifies exact graph-to-temporal feature/label identity:28 graph windows correspond to112 complete node windows and6,720 targets; no partial/missing node windows or unavailable edges occur in this test cohort. The fixed window recipe excludes the final12 timestamps (48 node samples). Per-seed per-class metrics, confusion matrices, predictions and communication results are preserved in `results.json` and evaluation subdirectories; means/sample SD are in `summary.json` and `communication_summary.json`.

This is a **locked evaluation after extensive validation optimization and post-hoc seed replacement**, not statistically established superiority or a pristine first exposure to this test split. The exposure audit found174 historical root/unannotated possible-test artifacts, including nine earlier CESTA/PatchTST/HMCT results on this same dataset whose saved target sequences exactly match the current test cohort. None belonged to these nine selected checkpoint directories. Original1242 and all historical artifacts remain unchanged. The unsuffixed test directory preserves a pre-inference runner indexing failure; only the cohort-check array indexing was corrected in a fresh suffixed directory, with zero evaluations before restart and no source or recipe changes. Ruff and Pyright checks on `src/CESTA` passed; no automated tests were added.

### Remaining standalone DHT baselines

The user-authorized fixed comparison in `runs/dht-remaining-baselines-20261106` completed all30 cells (ten remaining canonical DHT models × seeds12/42/2026),100 epochs each without early stopping. Historical checkpoint inspection found no exact reusable resolved recipes. Training explicitly disabled test evaluation; each validation-selected checkpoint then received one dedicated test evaluation. All30 saved target sequences match the same6,720-target graph/temporal cohort above. An interrupted DCRNN2026 attempt retained71 extra logged epochs before an unchanged full restart; no score-based retries, seed search, or test tuning occurred.

| Model | Active / total parameters | Test macro-F1 %, mean ± sample SD | Test accuracy %, mean ± sample SD |
|---|---:|---:|---:|
| Autoformer |34,436 /34,436|54.4429 ±6.9166|87.6587 ±0.7182|
| CNN-1D |12,996 /12,996|65.1675 ±2.4538|88.5516 ±0.6415|
| DCRNN |28,260 /28,260|33.2753 ±17.9535|85.6151 ±1.8128|
| HiFiNet |19,876 /19,876|22.9098 ±0.0000|84.5685 ±0.0000|
| HMCT |63,812 /63,812|39.7586 ±4.9974|85.8333 ±0.9196|
| Hydra |35,542 /35,542|73.4379 ±6.3244|90.7738 ±1.7162|
| Informer |13,188 /13,188|45.1606 ±1.8001|86.7163 ±0.2761|
| ModernTCN |37,196 /37,196|41.7378 ±6.5226|86.3393 ±0.9432|
| ST-GCN |12,004 /12,004|30.2327 ±4.6537|69.5188 ±7.1233|
| Transformer |25,892 /25,892|44.7029 ±8.5208|86.4980 ±0.8603|
| Previously locked CESTA CRF0.02 |429,279 /466,528|89.5615 ±0.6842|95.5903 ±0.3841|
| Previously locked PatchTST |25,988 /27,812|88.4438 ±1.6036|95.3819 ±0.3977|
| Secondary sparse CESTA CRF0.01 |429,279 /466,528|89.7496 ±0.6324|95.7391 ±0.2354|

Hydra has the highest mean among the ten newly run baselines; the existing locked PatchTST and CESTA scores remain higher descriptively. These are **unequal-capacity, unequal-optimization and unequal-tuning-budget recipes**, not a matched-loss or equal-compute experiment. Canonical model-specific widths, learning rates, batch sizes, dropout, focal/CE losses and oversampling were preserved; HMCT uses Adam with weight decay0.01, not AdamW. CESTA additionally uses training-only teacher distillation and auxiliary losses, with substantially more active parameters and extensive prior validation optimization. Low baseline macro-F1 must not be hidden by high majority-class accuracy: HiFiNet predicts only NORMAL on all three test runs.

`combined_results.json` and `summary.json` preserve all39 per-seed results, per-class precision/recall/F1/support and sample-SD summaries; `recipe_comparison.json` preserves full standalone recipes. Communication is unavailable in all ten new baseline artifacts and PatchTST, not measured zero. `communication_summary.json` reports supported CESTA request/bit/energy mean/sample SD: CRF0.02 requests42.5298 ±3.3944%, bits35,119,104 ±2,802,924, modeled TX+RX3.512262 ±0.280320J; secondary CRF0.01 requests41.3145 ±2.6859%, bits34,115,584 ±2,217,910, energy3.411900 ±0.221813J. The1m fallback-distance limitation remains. Protocol/source/data hashes and full Intel inventory passed completion audit; source/defaults were unchanged. Prior test exposure, exploratory validation optimization and post-hoc2026 seed-replacement limitations above still apply; no pristine-holdout or statistical-superiority claim is warranted, and the secondary sparse score does not trigger candidate reselection.

## Training-only distillation

Optional `distillation_teacher`, `distillation_weight` (default 0), and `distillation_temperature` (default 2) add frozen PatchTST soft-target supervision without changing student inference parameters, communication, or receiver-local gates. Teachers must use the same dataset hashes, features and split/window recipe. Only complete training node windows are supervised; validation/test teacher outputs are never used for optimization. Augmentation is currently rejected. The teacher is absent from student checkpoints and deployment.

Run the bounded validation-only DHT screen with `uv run python scripts/run_cesta_distillation_experiment.py --output runs/<new-experiment>`. It retains the original standardized GRU and Intel defaults, runs 100 epochs without early stopping for a fresh control and four distillation variants (including probability targets and local-plus-final supervision), and locks the best candidate beating both control and same-seed teacher before paired seed confirmation. Communication costs are reported rather than used as an eligibility veto. This exploratory feature does not imply improved accuracy or test superiority.

Ensembles are training-only: set `distillation_teachers` to a list instead of the singular teacher path; frozen teacher probabilities are averaged at the configured temperature. `distillation_local_weight` can supervise existing local logits independently of final-logit weight. Optional `distillation_drift_weight` reweights DRIFT training timesteps, normalized by total weight. `--ensemble` on the experiment script starts a separately identified exploratory phase with two fixed variants and paired-seed confirmation. Historical teacher graph/split identity is not verified when old manifests lack graph hashes; artifacts explicitly record this limitation.

Architecture-transfer experiments use `uv run python scripts/run_cesta_transfer_experiment.py --output runs/<new-experiment>`. The opt-in standalone-compatible width32 PatchTST replaces—not supplements—the GRU, preserving graph payload width, receiver-local gates, communication correction, and CRF. BatchNorm running statistics remain frozen to avoid cross-node normalization; pretrained encoder/head initialization uses a matched training checkpoint. Student parameters decrease from 719,327 to 356,575. Intel GRU defaults remain unchanged; this experimental backend is not supported by firmware export.

Validation-only transfer results (drift0625, seeds 12/42/1242): 92.1467 ± 0.9303% macro-F1 versus standalone PatchTST 89.8693 ± 1.2264%; active parameters 356,575 versus original GRU 719,327. Mean request ratio is 28.86%. Removing all available validation neighbors from the same trained checkpoints lowers macro-F1 by 1.7971/0.6723/0.7388 percentage points, demonstrating a useful communication contribution without retraining or reselection. Batch-one, four-node, 60-step full-forward-plus-CRF desktop timing (single-thread CPU, 30 warmup, three repeats of 200 iterations, synchronized CUDA) measured CPU 6.46 ms versus GRU 11.07 ms, but CUDA 7.09 ms versus 5.47 ms: **the no-CUDA-latency-increase preference is not met**. CUDA peak allocated memory was 13.59 MB versus 51.12 MB; these are desktop allocator measurements, not firmware memory/latency. Intel default-disabled runtime invariance was rechecked after transfer additions. Historical teacher graph-derived split identity remains unverified; transfer has extra pretraining compute; no test evaluation or firmware validation has been performed. Artifacts are under `runs/cesta-patch-transfer-20261106`.

The durable opt-in architecture recipe is `config/experiments/ablations/cesta-dht-patch-transfer.yaml`. It defaults `temporal_transfer_checkpoint` to null (fresh initialization); set it to your own compatible standalone PatchTST run directory to reproduce pretrained initialization. It contains no hardcoded teacher artifact path. Generic `CESTA.artifacts.load_checkpoint()` now reloads both original GRU and transferred CESTA directly, with strict weights and validation of explicitly declared descriptive metadata.

Incomplete-window limitation: the compatible PatchTST uses standalone internal scaling on its supplied window, including zero-filled missing samples. It does not reproduce the standalone complete-window cohort on incomplete node windows, and masked values may affect scaling/attention at otherwise valid timesteps. Existing node/edge masks still govern loss, decoding and communication availability. The trained candidate mathematics were intentionally not changed during integration; finite-mask checks do not establish mask-invariant scaling. Do not claim all acceptance constraints are met: CUDA latency increased and historical teacher graph-derived split identity is unverified.

## CESTA-DHT recordings

The recorded JSONL already contains injected faults. Ingest it without generating new faults:

```bash
uv run cesta transform cesta_dht data/raw/CESTA-DHT/readings.jsonl data/datasets/CESTA-DHT --config config/datasets/cesta-dht.yaml
uv run cesta train config/training/cesta.yaml data/datasets/CESTA-DHT
```

Raw recordings and topology live in `data/raw/CESTA-DHT/`; `data/datasets/CESTA-DHT/` holds the complete canonical dataset, not just training data. As with Intel, preparation splits the active timeline chronologically into **70% train, 15% validation, and 15% test** before windowing. Training stride is 10 samples; validation/test stride is 60, so window counts do not follow the same percentages.

The loader maps `esp32_00` through `esp32_03` to physical nodes 1 through 4, respectively. This is the explicit numbering convention for `edges.csv`: unweighted, undirected links 1–3, 1–4, 3–4, and 2–3. Each becomes two directed message edges internally. Static link availability and fallback distances of 1 m are assumptions, not measured radio reliability or geometry.

For each device with a fault stream, ingestion uses that stream; otherwise it uses the normal stream. Paired normal values remain in `reference_temp` and `reference_humid` for analysis, never in the model feature list. Temperature and humidity are retained, but the existing graph models train on `temp` only. Labels come directly from `fault_mode` (normal=0, spike=1, drift=2, stuck=3). The loader preserves the three-second cadence and missing-node masks without interpolation, averaging paired streams, or replacing missing faulty readings with clean ones. Duplicate stream timestamps, invalid labels/values, unpaired fault readings, off-grid timestamps, and whole-network gaps are rejected. Use separate contiguous recordings for whole-network gaps.

The original model configurations are preserved unchanged in `config/training/Intel/`. The top-level `config/training/*.yaml` configs use smaller widths (typically 32), with unchanged training schedules and splits. The Intel baseline runner defaults to the preserved configs. To train all smaller models on DHT with one seed:

```bash
uv run python scripts/run_all_baselines.py --configs config/training/*.yaml --datasets data/datasets/CESTA-DHT --seeds 42 --num-gpus 1 --runs-dir runs/cesta-dht-small-seed42 --keep-progress-log
```

### Recording analysis

The current recording spans 2026-10-24 15:16:46 through 2026-11-01 11:08:55 UTC: 225,444 three-second timestamps, 1,552,994 raw rows, no null fields, and no duplicate rows or device/path/timestamp keys. Clean and injected readings are paired observations, not independent samples.

| Physical node | Device | Selected label | Dataset rows | Temperature range (°C) |
| --- | --- | --- | ---: | ---: |
| 1 | esp32_00 | Normal | 225,444 | 26.994–32.900 |
| 2 | esp32_01 | Spike | 225,444 | 23.000–36.895 |
| 3 | esp32_02 | Drift | 212,887 | 25.100–36.900 |
| 4 | esp32_03 | Stuck | 225,444 | 27.000–32.900 |

There are 663,775 clean/fault pairs. Fault-minus-clean temperature offsets range from approximately −4 to +4 °C for spike, −2 to +2 °C for drift, and −1.9 to +1.9 °C for stuck. Node 3 is missing 12,557 samples (5.57%) across two gaps; consecutive observed readings are separated by 15,567 and 22,110 seconds. Its humidity range is 5–16%, versus 32–74% across the other nodes, so its clean reference also warrants sensor/calibration review.

The selected artifact has 889,219 rows: 25.35% each normal/spike/stuck and 23.94% drift. Connectivity-chronological 70/15/15 splitting with the existing 60-sample windows yields 15,776 training, 563 validation, and 563 test graph windows. All four classes occur in each split; both missing-data gaps are in training. A window covers about three minutes, not the five-hour span of Intel's five-minute sampling.

**Validity limitation:** each physical device has one constant selected class throughout the recording. Device identity or topology alone can predict the label; chronological accuracy is not evidence of generalization to new devices or fault transitions. Rotate normal and fault profiles across devices and collect held-out sessions before treating this dataset as a decisive diagnosis benchmark. Clean references remain excluded from training to avoid paired-reference leakage.

## Capabilities

- **Temporal models:** CNN1D, Transformer, Autoformer, Informer, PatchTST, ModernTCN, Hydra
- **Spatial models:** ST-GCN, HiFiNet, HMCT, DCRNN, CESTA
- **Faults:** Markov injection of SPIKE, DRIFT, and STUCK with per-event randomization
- **Evaluation:** classification, communication, and theoretical TX+RX radio-energy metrics
- **Reproducibility:** per-run manifests, resolved configs, checkpoints, Optuna search, benchmark audits
- **Deployment:** ESP32-S3 firmware for distributed on-device diagnosis

## Experiments

Checked-in configs cover baseline matrices and validation-tuned communication controls. The baseline matrix spans the 5–20% Intel fault datasets (see `config/datasets/intel-lab/`) with three seeds.

```bash
# Preview and launch the resumable baseline matrix
uv run python scripts/run_all_baselines.py --dry-run
uv run python scripts/run_all_baselines.py --num-gpus 4 --keep-progress-log

# Audit run artifacts; add --allow-incomplete for partial results
uv run python scripts/audit_decisive_comparison.py \
  --spec config/benchmarks/decisive-comparison.yaml \
  --runs-root runs --output runs/decisive-comparison-audit

# Paired summary against a locked reference variant
uv run python scripts/summarize_decisive_comparison.py \
  --runs-csv runs/decisive-comparison-audit/runs.csv \
  --output runs/decisive-comparison-summary \
  --comparison <variant> <locked-reference>
```

## Firmware

The ESP32-S3 firmware runs distributed diagnosis on DHT11 sensor nodes: each node processes a local 60-sample window with TensorFlow Lite Micro, exchanges neighbor payloads over ESP-NOW, and publishes readings and diagnoses over MQTT. Requires octal PSRAM, 8 MB+ flash, and a per-node exported model.

See [`firmware/README.md`](firmware/README.md) for hardware, export, build/flash, and telemetry details.

## Repository layout

| Path | Contents |
| --- | --- |
| [`src/CESTA/`](src/CESTA/) | Datasets, fault injection, models, training, evaluation, CLI |
| [`config/datasets/`](config/datasets/) | Dataset transformation and fault-injection configs |
| [`config/training/`](config/training/) | Default model training configs |
| [`config/experiments/`](config/experiments/) | Ablations, controls, and sweeps |
| [`config/benchmarks/`](config/benchmarks/) | Comparison and tuning specifications |
| [`scripts/`](scripts/) | Benchmark runners, audits, analysis, firmware export |
| [`firmware/`](firmware/) | ESP32-S3 Rust application |
| `data/` | Raw data and transformed datasets |
| `runs/` | Generated checkpoints, metrics, and metadata |

## License

[MIT](LICENSE)
