# Research

## Aim

CESTA (Communication-Efficient Spatial-Temporal Aggregation) studies sensor-network fault diagnosis under dynamic, costly radio communication. Each node encodes local temporal evidence, then selectively requests information from existing graph neighbors. The central question is whether this can exceed strong temporal-only macro-F1 while reducing TX+RX energy relative to dense spatial message passing.

### Hypotheses and success criteria

1. CESTA improves average macro-F1 over the best temporal-only model for each Intel fault ratio.
2. Receiver-local temporal state and uncertainty are sufficient to decide when neighbor information is useful, without inspecting sender embeddings before requesting them.
3. Learned request and compression policies reduce TX+RX energy relative to dense communication.
4. A learned controller outperforms validation-tuned rule-based controls at the same communication budget.
5. Gumbel-Softmax is expected to be easier to reproduce than reinforcement learning, but Pareto performance decides the final method.

The minimum target is an average macro-F1 improvement of `+0.01` over the strongest eligible temporal comparator plus Pareto-superior accuracy and energy relative to dense CESTA. The preferred accuracy improvement is `+0.03` to `+0.04`.

## Method

CESTA uses a fixed directed candidate graph from Intel connectivity data and a dynamic runtime edge mask:

```text
edge_index[0, e] = sender
edge_index[1, e] = receiver
active_edge[t, e] = link_success[t, e]
                    & node_observed[t, sender(e)]
                    & node_observed[t, receiver(e)]
```

The request gate may use only receiver-local hidden state, uncertainty, and edge metadata. It must not inspect sender embeddings before communication. Received messages are aggregated with single-head GAT-inspired attention over the received set only; an empty set produces a zero graph update. Missing labels use `-1` and are excluded from masked losses and metrics.

The first paper iteration covers Intel `fault05`, `fault10`, `fault15`, and `fault20`, temp-only input, Gumbel request gating, theoretical TX+RX energy, and measured edge energy if feasible. Unconstrained graph discovery, multi-hop protocols, and deployment claims without measurement are out of scope.

## Protocol

### Data and locked cohort

Use the canonical datasets:

```text
data/datasets/Intel_fault05
data/datasets/Intel_fault10
data/datasets/Intel_fault15
data/datasets/Intel_fault20
```

The decisive cohort uses:

```yaml
features: ["temp"]
data:
  window:
    window_size: 60
    train_stride: 10
    test_stride: 60
  split:
    strategy: connectivity-chronological
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
```

Use seeds `12`, `42`, and `1242`. Compare only runs with matching features, windowing, split boundaries, dataset hashes, test support, graph metadata, payload definition, and radio constants. Legacy temporal and ST-GCN runs use `80/10/10`; they remain descriptive but cannot support paired claims against the decisive `70/15/15` CESTA cohort.

### Baselines and controls

- All temporal families: CNN1D, LSTM, GRU, Transformer, Autoformer, Informer, PatchTST, and ModernTCN.
- Fixed CESTA temporal backbone without communication.
- Dynamic ST-GCN and HiFiNet if it is applicable and reproducible.
- Dense learned message passing over every available directed edge.
- Static top-k connectivity, random communication, and entropy-, margin-, and local-change-based controllers at matched budgets.

Lock the primary temporal family without reading test metrics. Require all 12 dataset-and-seed cells, rank families by mean checkpoint validation macro-F1, then break exact ties by lower validation standard deviation, lower active parameter count, and lexical model name. Tune controller thresholds and communication budgets on validation data only. Match controls primarily by dataset-level mean TX+RX energy across seeds within the interval `[98%, 100%]` of the learned target. Among matched candidates select the highest validation macro-F1; otherwise report the nearest candidate below target as under-budget unmatched. Keep equal combined-controller weights fixed during the first threshold sweep.

### Metrics and uncertainty

Report macro-F1, per-class F1, accuracy, confusion matrix, request ratio, requested edges, transmitted bits, TX energy, RX energy, total energy, dense-reference energy, active and total parameter counts, model size, latency, and peak memory where available.

For each dataset-and-seed matched comparison, report all 12 differences, their mean and sample standard deviation, and a deterministic two-sided 95% paired bootstrap interval with 10,000 resamples and seed `20260723`. With three seeds per dataset, treat intervals as descriptive uncertainty rather than strong significance evidence.

### Energy model

For each active sender-to-receiver message of `k` bits over distance `d`:

```text
E_tx(k, d) = E_elec * k + E_amp * k * d^n
E_rx(k) = E_elec * k
E_msg(k, d) = E_tx(k, d) + E_rx(k)
d0 = sqrt(E_fs / E_mp)
reduction = 1 - E_selective / E_dense
```

Use free-space or multipath constants according to `d0`. Count both TX and RX, and serialize constants, units, distance source, dynamic-link metadata, payload size, TX/RX shares, totals, and dense-reference reduction. Transmitted bits alone are not an energy result.

### Selection and reproducibility

Select models on the macro-F1 versus TX+RX energy Pareto frontier, not macro-F1 alone. If Pareto-tied, prefer the simpler, more reproducible design. Every complete run must contain `manifest.json`, `config.json`, `weight.pt`, `history.jsonl`, `eval_metrics.json`, and `predictions.npz`; communication-aware runs also require `communication_metrics.json`. Definitive matrices must come from one clean commit and contain no missing or duplicate cells.

Audit CESTA artifacts with:

```bash
uv run python scripts/audit_decisive_comparison.py \
  --spec config/benchmark/decisive-comparison.yaml \
  --runs-root runs \
  --output runs/decisive-comparison-audit
```

Summarize audited records with:

```bash
uv run python scripts/summarize_decisive_comparison.py \
  --runs-csv runs/decisive-comparison-audit/runs.csv \
  --output runs/decisive-comparison-summary \
  --comparison <variant> <locked-reference>
```

## Evidence

Evidence is provisional until the full protocol is complete. The current 36-run selective snapshot covers four datasets and three seeds:

| Variant | Runs | Macro-F1 mean±std | Accuracy mean±std | Request ratio | Bits | Δ vs legacy best temporal |
|---|---:|---:|---:|---:|---:|---:|
| CESTA selective, penalty `1e-3` | 12 | 0.8987±0.0248 | 0.9684±0.0115 | 0.439 | 197.5M | +0.0121 |
| `1e-3` + communication-conditioned correction | 12 | 0.8986±0.0233 | 0.9683±0.0124 | 0.423 | 190.4M | +0.0119 |
| `1e-2` + correction + VOI | 12 | 0.8922±0.0256 | 0.9666±0.0104 | 0.274 | 123.3M | +0.0056 |

These comparisons use legacy temporal references and are not eligible for the final paired accuracy claim. The strict artifact audit validates all 48 dense and selective CESTA cells and confirms serialized selective TX+RX energy savings relative to dense CESTA, but budget-matched controls are absent.

Historical single-seed `Intel_fault15` development established the following design choices:

- Dense logit correction plus CRF loss weight `0.05` reached test macro-F1 `0.9225` versus same-split GRU `0.9018`.
- Request penalty `1e-3` reached `0.9187` while reducing transmitted bits by `43.3%` against dense communication.
- Request penalty `1e-2` reached `0.9165` with `67.4%` fewer bits.
- Communication-conditioned correction preserved performance at penalty `1e-3` while reducing bits further.
- The tested standalone VOI objective was unreliable.
- A boundary head hurt DRIFT performance; CRF decoding helped persistent DRIFT and STUCK faults but can oversmooth SPIKE.

These historical values guided configurations but must not be treated as paper claims or TX+RX energy evidence.

## Work plan

### P0: decisive comparison

- [x] Complete and audit the 48-cell dense/selective CESTA matrix.
- [x] Export classification, request, and TX+RX energy records and paired summaries.
- [ ] Run all eight temporal families over four datasets and three seeds, yielding 96 unique cells from one commit.
- [ ] Use validation macro-F1 checkpointing, early stopping with patience 10 and minimum improvement `1e-4`, and no post-test configuration changes.
- [ ] Persist the validation-only family ranking before inspecting temporal test comparisons.
- [ ] Audit temporal provenance and test support, lock the primary comparator, and report paired uncertainty against CESTA.
- [ ] Publish one provisional comparison with the three-seed and missing-control limitations.

Inspect the temporal matrix with the checked-in split-matched configurations before removing `--dry-run`:

```bash
uv run python scripts/run_all_baselines.py \
  --runs-dir runs \
  --configs config/model/diagnosis/split_matched/*.yaml \
  --datasets data/datasets/Intel_fault05 data/datasets/Intel_fault10 \
             data/datasets/Intel_fault15 data/datasets/Intel_fault20 \
  --seeds 12 42 1242 \
  --early-stopping \
  --dry-run
```

Stop and investigate if a run is dirty, uses a different commit or dataset hash, has non-finite or incomplete artifacts, mismatches CESTA test support, or reuses a legacy split.

### P0: budget-matched controls

- [ ] Derive target request and energy budgets from selected CESTA variants. The validation-only, content-hashed derivation tool is implemented; clean validation artifacts are still required.
- [x] Implement random, static top-k, entropy, margin, local-change, and combined controls.
- [x] Implement predefined tuning-grid generation, validation-only evaluation and selection, content-hashed policy locking, and locked test-audit infrastructure.
- [ ] Run validation tuning, freeze the resulting lock, and compare locked policies on the macro-F1/TX+RX energy frontier.

### P1: coverage and paper readiness

- [ ] Reproduce HiFiNet if directly applicable; otherwise document the precise task or input mismatch.
- [ ] Keep dynamic ST-GCN as diagnostic evidence, not the sole spatial comparator.
- [ ] Verify accuracy uplift, dense-energy Pareto superiority, and learned-gating superiority separately.
- [ ] Report class-level failures and measure model size, parameter count, latency, and peak memory.
- [ ] Decide whether evidence supports a method paper or a narrower communication-efficiency study.

Defer compression-only CESTA, joint request/compression, RL, quantized deployment, and measured ESP32-S3 energy until both P0 sections are complete.

### Hard boundaries

- Do not claim energy savings from transmitted bits alone.
- Do not select models, controllers, thresholds, or budgets with test data.
- Do not claim learned-gating superiority without matched controls.
- Do not use ST-GCN as the only spatial comparator.
- Do not let request gates inspect sender embeddings before communication.
- Preserve missing nodes and unavailable links through node and edge masks.
- Do not call provisional evidence a final comparison.
