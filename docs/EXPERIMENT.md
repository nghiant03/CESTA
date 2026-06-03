# Experiment plan: CESTA

## Question

Can CESTA exceed temporal-only macro-F1 while reducing communication energy through receiver-side learned request and compression decisions over the existing sensor graph?

## Primary hypothesis

CESTA will improve average macro-F1 over the best temporal-only model per fault ratio by at least `+0.01`, preferably `+0.03` to `+0.04`, while reducing TX+RX communication energy relative to dense spatial message passing.

## Data

Use the harder canonical Intel graph datasets:

```text
data/canon/Intel_fault05
data/canon/Intel_fault10
data/canon/Intel_fault15
data/canon/Intel_fault20
```

Use temp-only input for comparability:

```yaml
features: ["temp"]
```

`cesta transform` serializes directed candidate edges from `connectivity.txt`, node positions from `mote_locs.txt`, edge distances, and a once-sampled bursty link-success mask:

```text
active_edge[t,e] = link_success[t,e] & node_observed[t,sender(e)] & node_observed[t,receiver(e)]
```

Missing node labels are stored as `-1` and excluded by masked loss/metrics. Complete-case timestamp filtering is not viable because no timestamp contains all 55 Intel nodes. The decisive development dataset is `data/canon/Intel_fault15`.

## Temporal targets

| Fault ratio | Best temporal baseline | Macro-F1 | Minimum target |
|---|---|---:|---:|
| fault05 | GRU | 0.8574 | 0.8674 |
| fault10 | LSTM | 0.8764 | 0.8864 |
| fault15 | GRU | 0.8999 | 0.9099 |
| fault20 | GRU | 0.9042 | 0.9142 |

Preferred target: `+0.03` to `+0.04` average macro-F1 over the best temporal baselines.

## Current results

Current dense, request-gated, communication-conditioned, VOI, CRF, and boundary-diagnosis results are kept in `docs/RESULT.md`.

Current interpretation:

1. Dense CESTA can exceed the same-split temporal baseline under `features: ["temp"]`.
2. Logit correction remains useful for both dense and request-gated paths.
3. Boundary supervision alone is not reliable and hurt DRIFT in the current experiment.
4. CRF decoding is the best dense refinement so far, but the CRF weight needs seed-robust selection.
5. Request-only Gumbel is Pareto-useful by transmitted-bit counts, but this is not yet a TX+RX energy claim.
6. Communication-conditioned correction is promising at penalty `1e-3`.
7. The tested VOI proxy is not reliable alone, though it may help in low-bit combinations.
8. The next decisive experiment is multi-seed confirmation plus budget-matched controls.

## Baselines and controls

### Temporal baselines

1. Best temporal-only model per fault ratio.
2. Fixed CESTA temporal encoder without communication.
3. GRU/LSTM/ModernTCN HPO retrain results as strong temporal references.

### Spatial and communication baselines

1. Dynamic ST-GCN.
2. HiFiNet if applicable.
3. Dense learned message passing over all currently available directed candidate edges.
4. Static top-k graph communication using strongest connectivity edges.
5. Random communication at matched average communication budget.
6. Rule-based controllers at matched budget: entropy threshold, prediction-margin threshold, local-change threshold, and combined uncertainty/change trigger.

## Metrics

Accuracy:

- macro-F1;
- per-class F1;
- accuracy;
- confusion matrix;
- average Δ macro-F1 versus best temporal-only baseline;
- average Δ macro-F1 versus fixed CESTA temporal backbone.

Communication and energy:

- theoretical TX+RX communication energy per window;
- energy reduction versus dense learned message passing;
- macro-F1 per communication-energy unit;
- active request ratio;
- requested edges per node/window;
- transmitted bits per node/window;
- compression-ratio distribution;
- receiver RX energy share;
- sender TX energy share;
- measured on-device energy if available.

Edge metrics:

- parameter count;
- serialized model size;
- inference latency on an edge-class target;
- peak memory estimate;
- quantized versus non-quantized evaluation as a secondary study only.

## Theoretical energy calculation

For every active receiver-side request from sender node `j` to receiver node `i`, count both TX and RX energy only when the directed candidate edge is available:

```text
E_tx(k, d) = E_elec · k + E_amp · k · d^n
E_rx(k) = E_elec · k
E_msg(k, d) = E_tx(k, d) + E_rx(k)
d0 = sqrt(E_fs / E_mp)
```

For CESTA:

```text
E_CESTA = Σ_windows Σ_t Σ_edges j→i available[t,j→i] · g_i,j,t · E_msg(k_i,j, d_i,j)
```

For dense learned message passing:

```text
E_dense = Σ_windows Σ_t Σ_edges j→i available[t,j→i] · E_msg(k_full, d_j,i)
reduction = 1 - E_CESTA / E_dense
```

Serialize units, constants, dynamic-link metadata, distance source, TX share, RX share, total energy, and reduction versus dense.

## Staged experiments

1. **Stage 0 dynamic-graph smoke check**: graph-batch shapes, output shape, masked loss, finite request ratio, finite communication metrics, and tiny-batch overfit on `Intel_fault15`.
2. **Fixed temporal backbone**: no-communication CESTA to separate encoder strength from spatial communication.
3. **Dense learned message passing**: upper-bound CESTA comparator over available directed edges with full embeddings.
4. **Request-only CESTA**: Gumbel request gate first; RL request policy only after Gumbel is seed-stable.
5. **Budget-matched controls**: static top-k, random, entropy, margin, local-change, and combined rule triggers.
6. **Compression-only CESTA**: fixed ratios and learned Gumbel compression selector with all available directed edges active.
7. **Full request + compression CESTA**: joint request/compression penalties and Pareto analysis.
8. **RL prototype**: request-only first, then compression only if request-only RL is reproducible.
9. **Full benchmark**: selected variants across all four Intel fault ratios.

Detailed implementation milestones live in `PLAN.md`.

## Required ablations

1. No communication.
2. Dense full communication.
3. Request-only.
4. Compression-only.
5. Request + compression.
6. Uncertainty removed from gate input.
7. Local embedding removed from gate input.
8. Fusion gate removed.
9. Static top-k neighbors.
10. Random neighbors at matched budget.
11. Gumbel-Softmax versus RL.
12. Per-window versus per-timestep decision if implementation cost permits.
13. Compression ratios `{0.25, 0.5, 1.0}` versus smaller/larger sets.
14. Different energy penalty weights.
15. Quantized versus non-quantized edge evaluation.
16. GAT single-head attention versus degree-normalized mean aggregation.
17. Attention over received set only versus softmax over padded full neighbor set.
18. Multi-head attention versus single-head if neighbor sets grow large.

## Selection rule

Do not select by macro-F1 alone. Use Pareto frontier analysis over macro-F1 and TX+RX energy:

```text
higher macro-F1 at equal/lower energy
or lower energy at equal/higher macro-F1
```

If Pareto-tied, prefer the simpler and more reproducible Gumbel design.

## Expected outcomes

Best case:

- CESTA improves average macro-F1 by `+0.03` to `+0.04` over best temporal baselines;
- communication energy falls substantially versus dense spatial communication;
- learned request/compression dominates rule-based controls;
- gate activation is higher for uncertain, DRIFT, and STUCK windows.

Minimum acceptable outcome:

- CESTA improves average macro-F1 by at least `+0.01` over best temporal baselines;
- CESTA is Pareto-superior to dense learned message passing, or at least to ST-GCN/HiFiNet if dense message passing is too costly.

Negative outcome:

- CESTA cannot exceed best temporal baselines;
- communication is only useful at all-on or near-all-on budgets;
- rule-based triggers match learned gates.

If negative, reposition the contribution as an energy-aware spatial communication study only if energy savings are strong and accuracy remains close to temporal baselines.

## Failure modes

1. Gate collapse to all-off due to an overpowered energy penalty.
2. Gate collapse to all-on because spatial messages are too useful or penalty is too weak.
3. Compression selector always chooses full embeddings.
4. RL policy instability or high variance.
5. The Intel connectivity graph lacks useful spatial signal.
6. The energy model overstates savings relative to measured ESP32-S3 behavior.
7. Dense learned message passing beats CESTA by too much.
8. HiFiNet outperforms CESTA without much extra cost.
9. Structured decoding improves DRIFT/STUCK but oversmooths SPIKE.
10. Boundary auxiliary objectives overemphasize unstable transition regions.

## Reproducibility notes

Record for every run:

- dataset path and fault ratio;
- selected features;
- graph threshold, directed edge count, node count, dynamic-link seed, and burst-simulation parameters;
- random seed;
- model config;
- training controller type;
- energy constants and distance assumptions;
- measured-energy hardware setup if used;
- communication stats;
- run manifest and git state.
