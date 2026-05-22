# Experiment plan: CESTA

## Question

Can CESTA exceed temporal-only macro-F1 while reducing communication energy through receiver-side learned request and compression decisions over the existing sensor graph?

## Primary hypothesis

CESTA will improve average macro-F1 over the best temporal-only model per fault ratio by at least +0.01, preferably +0.03 to +0.04, while reducing communication energy relative to dense spatial message passing.

## Data

Use the current harder Intel injected graph datasets:

```text
data/injected/Intel_fault05
data/injected/Intel_fault10
data/injected/Intel_fault15
data/injected/Intel_fault20
```

Use temp-only input for comparability:

```text
features: ["temp"]
```

Graph preparation should use a directed candidate edge list from `connectivity.txt` and a once-sampled bursty link-success mask. Runtime graph availability is dynamic:

```text
active_edge[t,e] = link_success[t,e] & node_observed[t,sender(e)] & node_observed[t,receiver(e)]
```

Missing node labels are stored as `-1` and excluded by masked loss/metrics. Complete-case timestamp filtering is not viable for the current Intel graph data because no timestamp contains all 55 nodes.

The first decisive development dataset is:

```text
data/injected/Intel_fault15
```

## Current temporal targets

Current HPO retrain macro-F1 targets from existing results:

| Fault ratio | Best temporal baseline | Macro-F1 |
|---|---|---:|
| fault05 | GRU | 0.8574 |
| fault10 | LSTM | 0.8764 |
| fault15 | GRU | 0.8999 |
| fault20 | GRU | 0.9042 |

Minimum paper target by fault ratio is approximately:

| Fault ratio | Minimum target |
|---|---:|
| fault05 | 0.8674 |
| fault10 | 0.8864 |
| fault15 | 0.9099 |
| fault20 | 0.9142 |

Preferred Q1-level target is +0.03 to +0.04 average macro-F1 over those best temporal baselines.

## Current Intel_fault15 diagnosis results

All results below use the hard temp-only setting:

```yaml
features: ["temp"]
```

The same-split GRU reference for the current `Intel_fault15` diagnosis runs is macro-F1 `0.9018`. The CRF strength sweep improved the dense upper bound: the best test result is dense CESTA with logit correction plus CRF at `crf_loss_weight=0.05`, with macro-F1 `0.9225`. This clears the minimum +0.01 development target by `+0.0207` over the same-split GRU and improves by `+0.0080` over dense logit correction. A validation-first rule would choose `crf_loss_weight=0.3` (`val_macro_f1=0.9301`, test `0.9198`), so the validation/test mismatch should be resolved by multi-seed confirmation before locking the dense comparator.

| Variant | Run | Val macro-F1 | Test macro-F1 | Δ vs GRU 0.9018 | Δ vs dense logit correction | Notes |
|---|---|---:|---:|---:|---:|---|
| Same-split GRU | prior run | — | 0.9018 | — | — | temporal-only reference for current split |
| Previous dense CESTA | prior run | — | 0.9126 | +0.0108 | -0.0019 | first dense spatial upper-bound candidate |
| Dense CESTA + logit correction | `runs/cesta/20260520T100015Z_cesta_seed42_a6931bb` | 0.9141 | 0.9145 | +0.0127 | — | best pre-CRF dense candidate |
| Dense CESTA + logit correction + boundary head | `runs/cesta/20260520T164518Z_cesta_seed42_a6931bb` | 0.9118 | 0.9087 | +0.0069 | -0.0058 | boundary auxiliary head and soft boundary-gated correction hurt DRIFT |
| CRF weight 0.05 | `runs/cesta/20260521T092725Z_cesta_seed42_2f16bda` | 0.9225 | 0.9225 | +0.0207 | +0.0080 | best test result; preserves SPIKE while improving DRIFT/STUCK |
| CRF weight 0.1 | `runs/cesta/20260521T154256Z_cesta_seed42_2f16bda` | 0.9266 | 0.9187 | +0.0169 | +0.0042 | similar SPIKE, weaker STUCK than 0.05 |
| CRF weight 0.2 | `runs/cesta/20260521T155537Z_cesta_seed42_2f16bda` | 0.9292 | 0.9156 | +0.0138 | +0.0011 | higher validation, lower test DRIFT |
| CRF weight 0.3 | `runs/cesta/20260521T160403Z_cesta_seed42_2f16bda` | 0.9301 | 0.9198 | +0.0180 | +0.0053 | validation-best; strongest STUCK but lower SPIKE |

Per-class test F1 for the key dense variants:

| Variant | NORMAL | SPIKE | DRIFT | STUCK | Accuracy | Transmitted bits |
|---|---:|---:|---:|---:|---:|---:|
| Dense + logit correction | 0.9831 | 0.9873 | 0.8458 | 0.8419 | 0.9691 | 449560576 |
| Boundary-aware dense | 0.9819 | 0.9856 | 0.8266 | 0.8407 | 0.9670 | 449560576 |
| CRF weight 0.05 | 0.9842 | 0.9860 | 0.8584 | 0.8615 | 0.9713 | 449560576 |
| CRF weight 0.1 | 0.9833 | 0.9856 | 0.8549 | 0.8509 | 0.9700 | 449560576 |
| CRF weight 0.2 | 0.9826 | 0.9821 | 0.8361 | 0.8618 | 0.9685 | 449560576 |
| CRF weight 0.3 | 0.9838 | 0.9772 | 0.8559 | 0.8624 | 0.9708 | 449560576 |

The sweep refines the structured-decoding tradeoff: low CRF weight (`0.05`) protects SPIKE almost as well as dense logit correction while improving both persistent-fault classes, whereas higher weights improve validation macro-F1 but increasingly risk SPIKE oversmoothing or weaker test DRIFT.

Request-only Gumbel CESTA then tested whether dense-style accuracy can be retained while reducing communication. The standalone `1e-4` anchor reached test macro-F1 `0.9261`, request ratio `0.9722`, and `437075968` transmitted bits, only a `2.8%` reduction versus dense. The stronger penalty sweep found meaningful lower-bit Pareto candidates:

| Variant | Run | Val macro-F1 | Test macro-F1 | Δ vs GRU 0.9018 | Request ratio | Transmitted bits | Bit reduction vs dense |
|---|---|---:|---:|---:|---:|---:|---:|
| Gumbel request penalty 1e-4 | `runs/cesta/20260521T171042Z_cesta_seed42_2f16bda` | 0.9273 | 0.9244 | +0.0226 | 0.9892 | 444719104 | 1.1% |
| Gumbel request penalty 5e-4 | `runs/cesta/20260521T172359Z_cesta_seed42_2f16bda` | 0.9311 | 0.9109 | +0.0091 | 0.4990 | 224321536 | 50.1% |
| Gumbel request penalty 1e-3 | `runs/cesta/20260521T174621Z_cesta_seed42_2f16bda` | 0.9257 | 0.9187 | +0.0169 | 0.5674 | 255090688 | 43.3% |
| Gumbel request penalty 3e-3 | `runs/cesta/20260521T180837Z_cesta_seed42_2f16bda` | 0.9208 | 0.9077 | +0.0059 | 0.4606 | 207085568 | 53.9% |
| Gumbel request penalty 1e-2 | `runs/cesta/20260521T182710Z_cesta_seed42_2f16bda` | 0.9256 | 0.9165 | +0.0147 | 0.3261 | 146579456 | 67.4% |

Per-class test F1 for the request-only sweep:

| Variant | NORMAL | SPIKE | DRIFT | STUCK | Accuracy |
|---|---:|---:|---:|---:|---:|
| Gumbel request penalty 1e-4 | 0.9848 | 0.9748 | 0.8623 | 0.8756 | 0.9724 |
| Gumbel request penalty 5e-4 | 0.9805 | 0.9842 | 0.8107 | 0.8680 | 0.9652 |
| Gumbel request penalty 1e-3 | 0.9828 | 0.9792 | 0.8390 | 0.8737 | 0.9694 |
| Gumbel request penalty 3e-3 | 0.9810 | 0.9833 | 0.8274 | 0.8391 | 0.9657 |
| Gumbel request penalty 1e-2 | 0.9830 | 0.9821 | 0.8335 | 0.8675 | 0.9693 |

Communication-conditioned correction and value-of-information (VOI) gate-loss ablations were then run on the two most useful request-gated penalty levels. These variants keep `communication_mode: gumbel_request`, logit correction, and CRF weight `0.05`; communication-conditioned correction augments the correction head with communicated-vs-local belief features, while VOI loss penalizes communication that does not improve the correct-class logit.

| Variant | Run | Val macro-F1 | Test macro-F1 | Δ vs matching request baseline | Δ vs GRU 0.9018 | Request ratio | Transmitted bits | Bit reduction vs dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Penalty 1e-3 + communication-conditioned correction | `runs/cesta/20260522T035943Z_cesta_seed42_2f16bda` | 0.9289 | 0.9188 | +0.0002 | +0.0170 | 0.4483 | 201555968 | 55.2% |
| Penalty 1e-3 + VOI 0.1 | `runs/cesta/20260522T042320Z_cesta_seed42_2f16bda` | 0.9203 | 0.9080 | -0.0107 | +0.0062 | 0.4173 | 187604992 | 58.3% |
| Penalty 1e-3 + correction + VOI 0.1 | `runs/cesta/20260522T044749Z_cesta_seed42_2f16bda` | 0.9288 | 0.9102 | -0.0085 | +0.0084 | 0.4002 | 179929088 | 60.0% |
| Penalty 1e-2 + communication-conditioned correction | `runs/cesta/20260522T051358Z_cesta_seed42_2f16bda` | 0.9204 | 0.9028 | -0.0137 | +0.0010 | 0.2526 | 113565696 | 74.7% |
| Penalty 1e-2 + VOI 0.1 | `runs/cesta/20260522T053929Z_cesta_seed42_2f16bda` | 0.9199 | 0.9032 | -0.0133 | +0.0014 | 0.3046 | 136953856 | 69.5% |
| Penalty 1e-2 + correction + VOI 0.1 | `runs/cesta/20260522T060603Z_cesta_seed42_2f16bda` | 0.9264 | 0.9182 | +0.0017 | +0.0164 | 0.3197 | 143720448 | 68.0% |

Per-class test F1 for the communication-conditioned and VOI ablations:

| Variant | NORMAL | SPIKE | DRIFT | STUCK | Accuracy |
|---|---:|---:|---:|---:|---:|
| Penalty 1e-3 + communication-conditioned correction | 0.9834 | 0.9838 | 0.8354 | 0.8727 | 0.9699 |
| Penalty 1e-3 + VOI 0.1 | 0.9810 | 0.9881 | 0.8166 | 0.8462 | 0.9655 |
| Penalty 1e-3 + correction + VOI 0.1 | 0.9807 | 0.9912 | 0.8101 | 0.8588 | 0.9655 |
| Penalty 1e-2 + communication-conditioned correction | 0.9799 | 0.9830 | 0.7905 | 0.8580 | 0.9635 |
| Penalty 1e-2 + VOI 0.1 | 0.9806 | 0.9829 | 0.8125 | 0.8369 | 0.9646 |
| Penalty 1e-2 + correction + VOI 0.1 | 0.9832 | 0.9843 | 0.8325 | 0.8727 | 0.9694 |

Current interpretation:

1. Dense CESTA can exceed the same-split temporal baseline under `features: ["temp"]`, and the tuned CRF result is now a stronger dense upper-bound candidate.
2. Logit correction is useful and remains part of both dense and request-gated paths.
3. Boundary supervision alone is not a reliable fix: it reduced test macro-F1 and especially DRIFT F1.
4. A lightweight CRF transition layer adds no communication overhead and is the best dense refinement so far, but `crf_loss_weight` needs a seed-robust selection rule.
5. Request-only Gumbel communication is now Pareto-useful in the development split: penalty `1e-3` keeps macro-F1 within `0.0038` of dense CRF `0.05` with `43.3%` fewer bits, while penalty `1e-2` stays above the GRU reference by `+0.0147` with `67.4%` fewer bits.
6. Communication-conditioned correction is Pareto-useful at penalty `1e-3`: it preserves macro-F1 (`0.9188` versus `0.9187`) while reducing bits from `255090688` to `201555968`, improving dense-bit reduction from `43.3%` to `55.2%`.
7. The tested VOI proxy is not yet a reliable standalone objective: at penalty `1e-3` it lowers bits but costs about `0.0107` macro-F1, and at penalty `1e-2` it nearly collapses the margin over GRU. Combined with correction at penalty `1e-2`, however, it gives the best low-bit point so far (`0.9182` macro-F1, `68.0%` fewer bits), slightly improving the old penalty `1e-2` baseline while using fewer bits.
8. The next decisive experiment should replicate dense CRF `0.05`, request penalty `1e-3`, penalty `1e-3` + communication-conditioned correction, and penalty `1e-2` + correction + VOI across seeds, then add budget-matched static/random/rule-based controls before claiming learned gating superiority.

## Baselines and controls

### Temporal baselines

1. Best temporal-only model per fault ratio.
2. Fixed CESTA temporal encoder without communication.
3. GRU/LSTM/ModernTCN HPO retrain results as strong temporal references.

### Spatial baselines

1. ST-GCN.
2. HiFiNet, if the supplied paper confirms it targets sensor/graph fault diagnosis.
3. Dense learned message passing: same encoder/aggregator as CESTA, all currently available directed candidate edges active, full embeddings transmitted.
4. Static top-k graph communication using strongest connectivity edges.
5. Random communication at matched average communication budget.

### Rule-based controls

1. Uncertainty-triggered communication using entropy or prediction margin.
2. Change-triggered communication using local change/anomaly magnitude.
3. Combined uncertainty + change trigger.

Rule-based thresholds must be tuned to match CESTA's average communication budget for fair comparison.

## Metrics

### Accuracy metrics

- macro-F1;
- per-class F1;
- accuracy;
- confusion matrix;
- average Δ macro-F1 against the best temporal-only model per fault ratio;
- average Δ macro-F1 against a fixed temporal backbone.

### Communication and energy metrics

Primary energy metrics should be based on energy consumption:

- measured on-device energy per window/inference;
- theoretical TX+RX communication energy per window;
- energy reduction versus dense learned message passing;
- macro-F1 per Joule or macro-F1 per communication-energy unit;
- active request ratio;
- requested edges per node/window;
- transmitted bits per node/window;
- compression-ratio distribution;
- receiver RX energy share;
- sender TX energy share.

### Edge metrics

- parameter count;
- serialized model size;
- inference latency on edge-class target;
- peak memory estimate;
- effect of int8/dynamic quantization as evaluation only.

## Theoretical energy calculation

For every active receiver-side request from sender node `j` to receiver node `i`, count both TX and RX energy only when the directed candidate edge is available at that timestamp/window.

```text
E_tx(k, d) = E_elec · k + E_amp · k · d^n
E_rx(k) = E_elec · k
E_msg(k, d) = E_tx(k, d) + E_rx(k)
```

Use free-space or multipath amplifier constants according to threshold distance:

```text
d0 = sqrt(E_fs / E_mp)
```

For CESTA:

```text
E_CESTA = Σ_windows Σ_t Σ_edges j→i available[t,j→i] · g_i,j,t · E_msg(k_i,j, d_i,j)
```

where `k_i,j` depends on hidden dimension, compression ratio, numeric precision, and protocol overhead if modeled.

For dense learned message passing:

```text
E_dense = Σ_windows Σ_t Σ_edges j→i available[t,j→i] · E_msg(k_full, d_j,i)
```

Report reduction:

```text
reduction = 1 - E_CESTA / E_dense
```

## Staged experiments

### Stage 0: feasibility checks

Goal: verify graph data shape, output shape, and training loop compatibility.

Run on `Intel_fault15` for a very small epoch budget.

Checks:

- graph batch carries `x`, `y`, `node_mask`, directed `edge_index`, and per-window `edge_mask`;
- logits shape is `(batch, window_size, num_nodes, num_classes)`;
- loss computes against per-node labels only where `node_mask` is true;
- communication stats are non-empty;
- requested edge ratio is not NaN;
- model can overfit a tiny batch.

### Stage 1: temporal encoder baseline

Train the CESTA temporal encoder without communication.

Purpose:

- establish the fixed backbone baseline;
- separate temporal encoder strength from spatial communication contribution.

Required outputs:

- macro-F1;
- per-class F1;
- parameter count;
- latency estimate.

### Stage 2: dense learned message passing

Train the same encoder and aggregation module with all currently available directed candidate edges active and full embeddings transmitted.

Purpose:

- establish the upper bound for the CESTA architecture without communication limits;
- provide a stronger spatial baseline than ST-GCN.

Status on `Intel_fault15`:

- dense CESTA + logit correction reached macro-F1 `0.9145`;
- boundary-aware auxiliary supervision and boundary-gated correction reached only `0.9087`, so this path is deprioritized unless redesigned as full segment-level refinement;
- dense CESTA + logit correction + CRF reached macro-F1 `0.9225` at `crf_loss_weight=0.05`, making it the current strongest test dense upper-bound candidate;
- validation-first selection currently favors `crf_loss_weight=0.3` (`val_macro_f1=0.9301`, test `0.9198`), so the next dense-stage experiment is a multi-seed confirmation of `0.05` versus `0.3`;
- all dense variants above have the same transmitted-bit estimate (`449560576`) because boundary and CRF additions are local/decoder-side and do not alter message payloads.

Required outputs:

- macro-F1;
- per-class F1;
- theoretical TX+RX energy;
- measured edge energy if available.

### Stage 3: request-only CESTA

Train receiver-side learned request gates with full embedding transmission when active.

Compare:

- Gumbel-Softmax request gate;
- RL request policy.

Purpose:

- isolate the benefit of deciding whether to communicate.

### Stage 4: compression-only CESTA

Keep all currently available directed candidate edges active but learn/select compression ratio.

Compare:

- fixed compression ratios;
- Gumbel-Softmax compression selector;
- RL compression selector if feasible.

Purpose:

- isolate the benefit of reducing payload size.

### Stage 5: full CESTA

Train receiver-side request gate and compression selector together.

Compare:

- Gumbel request + Gumbel compression;
- RL request + RL compression;
- hybrid Gumbel pretraining followed by RL fine-tuning if neither pure method dominates.

The main design is selected by Pareto dominance:

```text
higher macro-F1 at equal/lower measured energy
or lower measured energy at equal/higher macro-F1.
```

If Pareto-tied, choose the simpler Gumbel design.

### Stage 6: rule-based controls

Evaluate rule-based controllers at matched average communication budgets:

1. entropy threshold;
2. prediction-margin threshold;
3. local change magnitude threshold;
4. combined uncertainty + change threshold.

Purpose:

- prove that learned communication is better than simple triggering.

### Stage 7: full benchmark across all fault ratios

Run the selected CESTA variants across all four fault ratios.

Required comparisons:

- best temporal per fault ratio;
- fixed temporal backbone;
- ST-GCN;
- HiFiNet if applicable;
- dense learned message passing;
- static top-k;
- random budget-matched;
- best rule-based budget-matched controller.

## Ablations

Required ablations:

1. no communication;
2. dense full communication;
3. request-only;
4. compression-only;
5. request + compression;
6. uncertainty removed from gate input;
7. local embedding removed from gate input;
8. fusion gate removed;
9. static top-k neighbors;
10. random neighbors at matched budget;
11. Gumbel-Softmax versus RL;
12. per-window versus per-timestep decision if implementation cost permits;
13. compression ratios `{0.25, 0.5, 1.0}` versus smaller/larger sets;
14. different energy penalty weights;
15. quantized versus non-quantized edge evaluation;
16. GAT single-head attention versus degree-normalized mean aggregation;
17. attention over received set only versus softmax over padded full neighbor set;
18. multi-head attention versus single-head if neighbor sets grow large (>4).

## Hyperparameter sweeps

Minimum sweep axes:

- hidden size: `32`, `64`, `128`;
- gate penalty weight;
- bits/energy penalty weight;
- compression-ratio set;
- dropout;
- Gumbel temperature schedule;
- RL reward weights if RL is used.

The selection metric should not be macro-F1 alone. Use Pareto frontier analysis over macro-F1 and measured/theoretical energy.

## Expected outcomes

Best case:

- CESTA improves average macro-F1 by +0.03 to +0.04 over best temporal baselines;
- communication energy falls substantially versus dense spatial communication;
- learned request/compression dominates rule-based controls;
- gate activation is higher for uncertain, DRIFT, and STUCK windows.

Minimum acceptable outcome:

- CESTA improves average macro-F1 by at least +0.01 over best temporal baselines;
- CESTA is Pareto-superior to dense learned message passing or at least to ST-GCN/HiFiNet if dense message passing is too costly.

Negative outcome:

- CESTA cannot exceed best temporal baselines;
- communication is only useful at all-on or near-all-on budgets;
- rule-based triggers match learned gates.

If negative, reposition the contribution as an energy-aware spatial communication study only if energy savings are strong and accuracy remains close to temporal baselines.

## Failure modes

1. Gate collapse to all-off due to energy penalty overpowering classification loss.
2. Gate collapse to all-on because spatial messages are too useful or penalty is too weak.
3. Compression selector always chooses full embeddings.
4. RL policy instability or high variance.
5. The existing Intel connectivity graph lacks useful spatial signal.
6. Energy model overstates savings relative to measured ESP32-S3 behavior because radio wake/sleep overhead dominates.
7. Dense learned message passing beats CESTA by too much, weakening selective-communication claims.
8. HiFiNet outperforms CESTA without much extra cost.
9. Structured decoding improves STUCK/DRIFT but oversmooths short SPIKE events.
10. Boundary auxiliary objectives overemphasize unstable transition regions and hurt plateau classification.

## Diagnosis findings so far

The main `Intel_fault15` error analysis indicates that long fault segments are not the dominant problem. DRIFT has only a small number of test segments longer than the 60-step window and those long segments were classified correctly; STUCK has no test segments longer than the window. Errors concentrate near fault starts and ends, especially the first 10 timesteps after onset.

Observed boundary-region pattern before the boundary-head experiment:

```text
DRIFT overall accuracy: 0.8354
DRIFT first 10 steps after start: 0.7367
DRIFT away from start: 0.9202
STUCK overall accuracy: 0.7954
STUCK first 10 steps after start: 0.7510
STUCK away from start: 0.8918
```

Tested remedies:

1. **Boundary head + boundary-gated correction**: implemented auxiliary focal BCE supervision on label transitions with dilation and used predicted boundary probability to boost logit correction. Result: macro-F1 dropped from `0.9145` to `0.9087`, mainly from DRIFT F1 dropping `0.8458 -> 0.8266`. Conclusion: naive boundary gating is not sufficient and may destabilize DRIFT decisions.
2. **CRF transition layer**: added a learned linear-chain transition matrix with masked CRF negative log-likelihood and Viterbi decoding. Initial result: macro-F1 improved to `0.9160`, with STUCK F1 improving `0.8419 -> 0.8570` and DRIFT F1 improving slightly, but SPIKE F1 dropping `0.9873 -> 0.9748`.
3. **CRF loss-weight sweep**: tested `crf_loss_weight` in `{0.05, 0.1, 0.2, 0.3}`. The best test result was `0.05` at macro-F1 `0.9225`, with SPIKE `0.9860`, DRIFT `0.8584`, and STUCK `0.8615`. The best validation result was `0.3` at validation macro-F1 `0.9301` and test macro-F1 `0.9198`. Conclusion: lower CRF strength can keep the structured-decoding benefit without sacrificing SPIKE, but the validation/test ordering needs seed replication.

Promising next dense-upper-bound actions:

1. Replicate CRF weights `0.05` and `0.3` across multiple seeds and select by validation-first mean performance, reporting test only after selection.
2. Consider transition initialization from train label transition frequencies if seed variance remains high.
3. Protect SPIKE with class-aware transition regularization only if multi-seed results show persistent SPIKE oversmoothing.
4. If revisiting boundaries, use segment-level refinement or boundary-preserving smoothing rather than a simple correction multiplier.
5. Only proceed to sparse/gated communication claims once dense CESTA's upper bound is stable enough to serve as the proper comparator.

## Reproducibility notes

Record for every run:

- dataset path and fault ratio;
- selected features;
- graph threshold, directed edge count, node count, dynamic-link seed, and burst-simulation parameters;
- random seed;
- model config;
- training controller type: Gumbel, RL, or rule-based;
- energy constants and distance assumptions;
- measured-energy hardware setup;
- communication stats;
- run manifest and git state.

## First implementation checkpoint

Implement a minimal CESTA variant first:

```text
GRU temporal encoder
receiver-side local gate
Gumbel request only
full embedding when active
GAT-inspired single-head attention aggregation
communication stats logging
```

Run this on `Intel_fault15` before adding compression or RL.
