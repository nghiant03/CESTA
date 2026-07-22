# Experiment plan: decisive-comparison artifact audit

## Question

Can the current and forthcoming CESTA run artifacts support a deterministic, auditable dense-versus-selective TX+RX energy comparison without silently accepting missing, duplicate, or internally inconsistent runs?

## Why this is the next one-session task

The dense rerun itself is a 12-run compute batch and is not a reliable one-session coding task. Before spending that compute, the repository needs a strict audit path that proves each run contains comparable accuracy and energy evidence. This is the smallest task that directly advances both the serialization check and aggregation work in `PLAN.md`.

Current repository audit on 2026-07-19:

- all 36 current selective CESTA runs have `manifest.json`, `eval_metrics.json`, `communication_metrics.json`, and a serialized test energy block;
- no current CESTA manifest uses `communication_mode: dense`;
- the required dense configuration already exists at `config/model/diagnosis/cesta_graph_fusion_sweep/cesta_dense_residual0p1_logit_correction_crf_w0p05_capmatch.yaml`;
- the existing sweep runner resolves the expected dense matrix to 12 runs: four fault ratios by seeds `12`, `42`, and `1242`;
- the existing selective snapshot does not have a checked-in generator and reports transmitted bits rather than TX+RX energy.

## Scope

Implement a strict artifact auditor and normalized run-level exporter. Keep statistical summaries, confidence intervals, Pareto analysis, controller implementation, and result-memo revision out of this session.

### Inputs

- run directories under `runs/<model>/<run_id>/`;
- a checked-in benchmark specification containing datasets, seeds, model configurations, and expected communication modes;
- `manifest.json`, `eval_metrics.json`, and, for communication-aware CESTA runs, `communication_metrics.json`.

### Validation rules

For every expected run:

1. Match model configuration, dataset hashes, and seed from the manifest. Ignore only fields that are expected to vary per run, such as the seed and timing.
2. Require finite macro-F1, accuracy, and per-class F1 values.
3. Require communication mode, request counts, possible counts, transmitted bits, bits per message, graph metadata, and the complete test energy block for dense and selective CESTA.
4. Require all counts and energy values to be finite and non-negative.
5. Verify requested messages do not exceed possible messages, globally or per edge.
6. Recompute and verify:
   - active request ratio;
   - transmitted bits from requested messages and bits per message;
   - TX+RX total energy;
   - dense-reference TX+RX total energy;
   - absolute and ratio energy reductions.
7. For dense mode, require request ratio `1` within tolerance, selected energy equal to dense-reference energy, and zero reduction.
8. Reject mixed radio constants, payload sizes, distance units, dataset hashes, or graph metadata within a compared dataset.
9. Treat missing and duplicate matrix cells as explicit audit failures. Never select a duplicate by test performance.

### Outputs

Write generated files under `runs/decisive-comparison-audit/`:

- `runs.csv`: one normalized row per valid run with provenance, macro-F1, accuracy, every per-class F1, request ratio, bits, TX energy, RX energy, total energy, dense-reference energy, and energy reduction;
- `audit.json`: expected, valid, missing, duplicate, and invalid matrix cells with actionable reasons;
- a concise terminal summary and nonzero exit status in strict mode.

Generated run evidence remains outside version control. The specification, implementation, and tests are checked in.

## Expected implementation surface

- `config/benchmark/decisive-comparison.yaml`: expected datasets, seeds, and exact source configurations.
- `src/CESTA/evaluation/benchmark.py`: artifact loading, matching, validation, and normalized records.
- `scripts/audit_decisive_comparison.py`: CLI wrapper and CSV/JSON output.
- `tests/test_benchmark_audit.py`: standard-library `unittest` fixtures for valid, missing, duplicate, and malformed artifacts.
- `AGENTS.md`: document the audit command after the code exists.

Do not modify runtime energy accounting in this task unless a fixture proves that currently emitted artifacts violate an invariant. Keep runtime hardening as a separate targeted change so historical artifact validation and future serialization changes are not conflated.

## Test cases

1. Valid selective artifact passes and exports all required metrics.
2. Valid dense artifact passes with zero energy reduction.
3. Missing energy, per-class F1, or provenance fails.
4. NaN, infinity, negative values, or requested counts above possible counts fail.
5. Inconsistent message bits, TX+RX totals, or reduction identities fail.
6. Mismatched radio constants, graph metadata, or dataset hashes fail comparison.
7. Missing and duplicate expected cells appear in `audit.json` and fail strict mode.
8. Output ordering is deterministic and independent of metric values.

## Validation commands

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check src/CESTA scripts tests
uv run pyright src/CESTA scripts tests
uv run python scripts/audit_decisive_comparison.py \
  --spec config/benchmark/decisive-comparison.yaml \
  --runs-root runs \
  --output runs/decisive-comparison-audit \
  --allow-incomplete
```

The dense launch matrix remains independently verifiable with:

```bash
uv run python scripts/run_all_baselines.py \
  --runs-dir runs \
  --configs config/model/diagnosis/cesta_graph_fusion_sweep/cesta_dense_residual0p1_logit_correction_crf_w0p05_capmatch.yaml \
  --datasets data/datasets/Intel_fault05 data/datasets/Intel_fault10 data/datasets/Intel_fault15 data/datasets/Intel_fault20 \
  --seeds 12 42 1242 \
  --dry-run
```

## Completion criteria

- Fixture tests cover every validation rule above.
- The current repository audit deterministically identifies all 36 selective CESTA runs as valid or gives a precise artifact-level reason for rejection.
- Strict mode reports exactly the absent expected dense cells rather than producing a misleading complete comparison.
- The normalized CSV contains every run-level metric required by `PLAN.md` except uncertainty, which remains a separate task.
- No energy value is inferred from transmitted-bit counts alone.
- No model or run is selected using test performance.

## Follow-up, not part of this session

1. Resolve any audit failures.
2. Run the 12 dense jobs from a clean commit.
3. Re-run the audit in strict mode.
4. Add paired uncertainty and aggregate summaries.
5. Update a provisional result memo only after the full matrix passes.
