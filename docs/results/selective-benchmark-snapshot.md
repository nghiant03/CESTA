# Result snapshot: selective CESTA benchmark

**Status:** Provisional evidence from a completed experiment batch, not the final paper comparison.

## Scope

This snapshot aggregates the latest completed non-CESTA baseline run per `(model, dataset, seed)` and the 36 CESTA selective ablation runs available when it was written. Datasets are `Intel_fault05`, `Intel_fault10`, `Intel_fault15`, and `Intel_fault20`; seeds are `12`, `42`, and `1242`; metric priority is test macro-F1. Dense CESTA and budget-matched controls are absent, so this snapshot cannot establish the paper's full accuracy-energy claim.

## Overall ranking across all four datasets

| Variant | Runs | Macro-F1 mean±std | Accuracy mean±std | Request ratio | Bits | Δ vs best temporal avg |
|---|---:|---:|---:|---:|---:|---:|
| cnn1d | 12 | 0.7202±0.0686 | 0.9237±0.0263 | — | — | -0.1665 |
| lstm | 12 | 0.8801±0.0226 | 0.9607±0.0150 | — | — | -0.0065 |
| gru | 12 | 0.8840±0.0219 | 0.9612±0.0134 | — | — | -0.0026 |
| transformer | 12 | 0.7944±0.0640 | 0.9465±0.0196 | — | — | -0.0923 |
| autoformer | 12 | 0.7836±0.0238 | 0.9372±0.0313 | — | — | -0.1030 |
| informer | 12 | 0.6733±0.0414 | 0.9237±0.0360 | — | — | -0.2133 |
| patchtst | 12 | 0.8548±0.0289 | 0.9514±0.0176 | — | — | -0.0318 |
| modern_tcn | 12 | 0.8695±0.0271 | 0.9569±0.0159 | — | — | -0.0172 |
| stgcn | 12 | 0.5139±0.0642 | 0.8932±0.0448 | — | — | -0.3727 |
| CESTA-Selective p1e-3 | 12 | 0.8987±0.0248 | 0.9684±0.0115 | 0.439 | 197.5M | +0.0121 |
| CESTA-Selective p1e-3 + CommCond | 12 | 0.8986±0.0233 | 0.9683±0.0124 | 0.423 | 190.4M | +0.0119 |
| CESTA-LowBit p1e-2 + CommCond + VOI | 12 | 0.8922±0.0256 | 0.9666±0.0104 | 0.274 | 123.3M | +0.0056 |

## Per-dataset winners and temporal reference

| Dataset | Best temporal | Best temporal F1 | Best CESTA selective | CESTA F1 | Δ | Request ratio | Bits |
|---|---|---:|---|---:|---:|---:|---:|
| Intel_fault05 | gru | 0.8548 | CESTA-Selective p1e-3 + CommCond | 0.8615 | +0.0068 | 0.331 | 148.8M |
| Intel_fault10 | lstm | 0.8883 | CESTA-Selective p1e-3 + CommCond | 0.9082 | +0.0199 | 0.419 | 188.5M |
| Intel_fault15 | gru | 0.9034 | CESTA-Selective p1e-3 | 0.9139 | +0.0104 | 0.477 | 214.5M |
| Intel_fault20 | gru | 0.9000 | CESTA-LowBit p1e-2 + CommCond + VOI | 0.9151 | +0.0151 | 0.314 | 141.1M |

## CESTA ablation details

| Dataset | Variant | Macro-F1 mean±std | Accuracy mean±std | Request ratio | Bits | Δ vs best temporal |
|---|---|---:|---:|---:|---:|---:|
| Intel_fault05 | CESTA-Selective p1e-3 | 0.8600±0.0150 | 0.9802±0.0024 | 0.316 | 142.1M | +0.0053 |
| Intel_fault05 | CESTA-Selective p1e-3 + CommCond | 0.8615±0.0113 | 0.9809±0.0017 | 0.331 | 148.8M | +0.0068 |
| Intel_fault05 | CESTA-LowBit p1e-2 + CommCond + VOI | 0.8549±0.0024 | 0.9799±0.0004 | 0.237 | 106.4M | +0.0001 |
| Intel_fault10 | CESTA-Selective p1e-3 | 0.9075±0.0066 | 0.9738±0.0025 | 0.448 | 201.6M | +0.0192 |
| Intel_fault10 | CESTA-Selective p1e-3 + CommCond | 0.9082±0.0040 | 0.9751±0.0013 | 0.419 | 188.5M | +0.0199 |
| Intel_fault10 | CESTA-LowBit p1e-2 + CommCond + VOI | 0.8887±0.0122 | 0.9677±0.0036 | 0.242 | 108.6M | +0.0003 |
| Intel_fault15 | CESTA-Selective p1e-3 | 0.9139±0.0067 | 0.9683±0.0030 | 0.477 | 214.5M | +0.0104 |
| Intel_fault15 | CESTA-Selective p1e-3 + CommCond | 0.9125±0.0085 | 0.9675±0.0033 | 0.454 | 203.9M | +0.0091 |
| Intel_fault15 | CESTA-LowBit p1e-2 + CommCond + VOI | 0.9103±0.0075 | 0.9663±0.0029 | 0.305 | 136.9M | +0.0069 |
| Intel_fault20 | CESTA-Selective p1e-3 | 0.9135±0.0058 | 0.9512±0.0035 | 0.516 | 231.9M | +0.0135 |
| Intel_fault20 | CESTA-Selective p1e-3 + CommCond | 0.9120±0.0023 | 0.9497±0.0020 | 0.490 | 220.3M | +0.0119 |
| Intel_fault20 | CESTA-LowBit p1e-2 + CommCond + VOI | 0.9151±0.0023 | 0.9525±0.0017 | 0.314 | 141.1M | +0.0151 |

## Interpretation

- Best average selective CESTA variant: **CESTA-Selective p1e-3** with macro-F1 `0.8987` across 12 runs.

- Primary communication-efficient representation candidate (**CESTA-Selective p1e-3 + CommCond**) averages macro-F1 `0.8986`, request ratio `0.423`, and `190.4M` transmitted bits; average Δ vs the best temporal baseline per dataset is `+0.0119`.

- Treat these as an interim benchmark snapshot. Transmitted-bit estimates are not TX+RX energy measurements, and no learned-gating superiority claim is supported until dense and budget-matched controls are included.

## Missing evidence

- Seed-replicated dense CESTA accuracy upper bound.
- TX+RX energy aggregation against dense communication.
- Static, random, and rule-based controls at matched communication budgets.
- Statistical uncertainty and significance for the primary comparisons.

## Reproducibility artifacts

- Machine-readable snapshot: `runs/selective_benchmark_snapshot.csv`
- Source run artifacts: `runs/<model>/<run_id>/eval_metrics.json`, `manifest.json`, and for CESTA `communication_metrics.json`.
