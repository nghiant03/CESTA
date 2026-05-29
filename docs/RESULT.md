# CESTA current results

## Development setting

All current diagnosis evidence is from `Intel_fault15` with temp-only input:

```yaml
features: ["temp"]
```

The same-split GRU reference for these diagnosis runs is macro-F1 `0.9018`. Current evidence is single-seed unless noted; do not treat it as a paper claim until the seed-replication plan in `PLAN.md` is complete.

## Dense upper-bound results

| Variant | Run | Val macro-F1 | Test macro-F1 | Δ vs GRU 0.9018 | Notes |
|---|---|---:|---:|---:|---|
| Same-split GRU | prior run | — | 0.9018 | — | Temporal-only reference for current split |
| Previous dense CESTA | prior run | — | 0.9126 | +0.0108 | First dense spatial upper-bound candidate |
| Dense CESTA + logit correction | `runs/cesta/20260520T100015Z_cesta_seed42_a6931bb` | 0.9141 | 0.9145 | +0.0127 | Best pre-CRF dense candidate |
| Dense CESTA + logit correction + boundary head | `runs/cesta/20260520T164518Z_cesta_seed42_a6931bb` | 0.9118 | 0.9087 | +0.0069 | Boundary auxiliary head and soft boundary-gated correction hurt DRIFT |
| Dense CESTA + logit correction + CRF, weight 0.05 | `runs/cesta/20260521T092725Z_cesta_seed42_2f16bda` | 0.9225 | 0.9225 | +0.0207 | Best test result; preserves SPIKE while improving DRIFT/STUCK |
| Dense CESTA + logit correction + CRF, weight 0.1 | `runs/cesta/20260521T154256Z_cesta_seed42_2f16bda` | 0.9266 | 0.9187 | +0.0169 | Similar SPIKE, weaker STUCK than 0.05 |
| Dense CESTA + logit correction + CRF, weight 0.2 | `runs/cesta/20260521T155537Z_cesta_seed42_2f16bda` | 0.9292 | 0.9156 | +0.0138 | Higher validation, lower test DRIFT |
| Dense CESTA + logit correction + CRF, weight 0.3 | `runs/cesta/20260521T160403Z_cesta_seed42_2f16bda` | 0.9301 | 0.9198 | +0.0180 | Validation-best; strongest STUCK but lower SPIKE |

Per-class test F1:

| Variant | NORMAL | SPIKE | DRIFT | STUCK | Accuracy | Transmitted bits |
|---|---:|---:|---:|---:|---:|---:|
| Dense + logit correction | 0.9831 | 0.9873 | 0.8458 | 0.8419 | 0.9691 | 449560576 |
| Boundary-aware dense | 0.9819 | 0.9856 | 0.8266 | 0.8407 | 0.9670 | 449560576 |
| CRF weight 0.05 | 0.9842 | 0.9860 | 0.8584 | 0.8615 | 0.9713 | 449560576 |
| CRF weight 0.1 | 0.9833 | 0.9856 | 0.8549 | 0.8509 | 0.9700 | 449560576 |
| CRF weight 0.2 | 0.9826 | 0.9821 | 0.8361 | 0.8618 | 0.9685 | 449560576 |
| CRF weight 0.3 | 0.9838 | 0.9772 | 0.8559 | 0.8624 | 0.9708 | 449560576 |

Interpretation: dense CESTA clears the minimum `Intel_fault15` target, but the best CRF weight is not yet stable. Weight `0.05` is best by test macro-F1; weight `0.3` is best by validation macro-F1. Multi-seed validation-first selection is required before locking the dense comparator.

## Request-only Gumbel results

| Variant | Run | Val macro-F1 | Test macro-F1 | Δ vs GRU 0.9018 | Request ratio | Transmitted bits | Bit reduction vs dense |
|---|---|---:|---:|---:|---:|---:|---:|
| Gumbel request penalty 1e-4 | `runs/cesta/20260521T171042Z_cesta_seed42_2f16bda` | 0.9273 | 0.9244 | +0.0226 | 0.9892 | 444719104 | 1.1% |
| Gumbel request penalty 5e-4 | `runs/cesta/20260521T172359Z_cesta_seed42_2f16bda` | 0.9311 | 0.9109 | +0.0091 | 0.4990 | 224321536 | 50.1% |
| Gumbel request penalty 1e-3 | `runs/cesta/20260521T174621Z_cesta_seed42_2f16bda` | 0.9257 | 0.9187 | +0.0169 | 0.5674 | 255090688 | 43.3% |
| Gumbel request penalty 3e-3 | `runs/cesta/20260521T180837Z_cesta_seed42_2f16bda` | 0.9208 | 0.9077 | +0.0059 | 0.4606 | 207085568 | 53.9% |
| Gumbel request penalty 1e-2 | `runs/cesta/20260521T182710Z_cesta_seed42_2f16bda` | 0.9256 | 0.9165 | +0.0147 | 0.3261 | 146579456 | 67.4% |

Per-class test F1:

| Variant | NORMAL | SPIKE | DRIFT | STUCK | Accuracy |
|---|---:|---:|---:|---:|---:|
| Gumbel request penalty 1e-4 | 0.9848 | 0.9748 | 0.8623 | 0.8756 | 0.9724 |
| Gumbel request penalty 5e-4 | 0.9805 | 0.9842 | 0.8107 | 0.8680 | 0.9652 |
| Gumbel request penalty 1e-3 | 0.9828 | 0.9792 | 0.8390 | 0.8737 | 0.9694 |
| Gumbel request penalty 3e-3 | 0.9810 | 0.9833 | 0.8274 | 0.8391 | 0.9657 |
| Gumbel request penalty 1e-2 | 0.9830 | 0.9821 | 0.8335 | 0.8675 | 0.9693 |

Interpretation: request-only Gumbel is Pareto-useful in the development split. Penalty `1e-3` stays within `0.0038` macro-F1 of dense CRF `0.05` while reducing transmitted bits by `43.3%`; penalty `1e-2` stays above GRU by `+0.0147` with `67.4%` fewer bits. These are bit-count reductions, not yet TX+RX energy claims.

## Communication-conditioned and VOI ablations

| Variant | Run | Val macro-F1 | Test macro-F1 | Δ vs matching request baseline | Δ vs GRU 0.9018 | Request ratio | Transmitted bits | Bit reduction vs dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Penalty 1e-3 + communication-conditioned correction | `runs/cesta/20260522T035943Z_cesta_seed42_2f16bda` | 0.9289 | 0.9188 | +0.0002 | +0.0170 | 0.4483 | 201555968 | 55.2% |
| Penalty 1e-3 + VOI 0.1 | `runs/cesta/20260522T042320Z_cesta_seed42_2f16bda` | 0.9203 | 0.9080 | -0.0107 | +0.0062 | 0.4173 | 187604992 | 58.3% |
| Penalty 1e-3 + correction + VOI 0.1 | `runs/cesta/20260522T044749Z_cesta_seed42_2f16bda` | 0.9288 | 0.9102 | -0.0085 | +0.0084 | 0.4002 | 179929088 | 60.0% |
| Penalty 1e-2 + communication-conditioned correction | `runs/cesta/20260522T051358Z_cesta_seed42_2f16bda` | 0.9204 | 0.9028 | -0.0137 | +0.0010 | 0.2526 | 113565696 | 74.7% |
| Penalty 1e-2 + VOI 0.1 | `runs/cesta/20260522T053929Z_cesta_seed42_2f16bda` | 0.9199 | 0.9032 | -0.0133 | +0.0014 | 0.3046 | 136953856 | 69.5% |
| Penalty 1e-2 + correction + VOI 0.1 | `runs/cesta/20260522T060603Z_cesta_seed42_2f16bda` | 0.9264 | 0.9182 | +0.0017 | +0.0164 | 0.3197 | 143720448 | 68.0% |

Interpretation:

1. Communication-conditioned correction is Pareto-useful at penalty `1e-3`: it preserves macro-F1 (`0.9188` versus `0.9187`) while reducing bits from `255090688` to `201555968`.
2. The tested VOI proxy is not reliable as a standalone objective: it reduces bits but usually costs too much macro-F1.
3. Penalty `1e-2` + correction + VOI is the best low-bit point so far (`0.9182`, `68.0%` fewer bits), but it requires multi-seed confirmation.

## Diagnosis findings

Error analysis on `Intel_fault15` found that long fault segments are not the dominant problem. DRIFT has only a few test segments longer than the 60-step window and those were classified correctly; STUCK has no test segments longer than the window. Errors concentrate near fault starts and ends, especially the first 10 timesteps after onset.

Boundary-region pattern before the boundary-head experiment:

```text
DRIFT overall accuracy: 0.8354
DRIFT first 10 steps after start: 0.7367
DRIFT away from start: 0.9202
STUCK overall accuracy: 0.7954
STUCK first 10 steps after start: 0.7510
STUCK away from start: 0.8918
```

Tested remedies:

1. Boundary head + boundary-gated correction hurt macro-F1 (`0.9145 -> 0.9087`), mainly by reducing DRIFT F1 (`0.8458 -> 0.8266`).
2. CRF transition decoding improved persistent faults; the best sweep point was `crf_loss_weight=0.05` with macro-F1 `0.9225`.
3. Higher CRF weights improve validation macro-F1 but risk SPIKE oversmoothing or weaker test DRIFT.

## Current next decision

Replicate dense CRF `0.05`, dense CRF `0.3`, request penalty `1e-3`, request penalty `1e-3` + communication-conditioned correction, and request penalty `1e-2` + correction + VOI across seeds. Then implement budget-matched controls before claiming learned gating superiority.
