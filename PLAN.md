# CESTA research work tracker

This file tracks unresolved work only. Research intent belongs in `docs/PROPOSAL.md`, the study protocol belongs in `docs/EXPERIMENT.md`, and evidence from completed experiment batches belongs in `docs/results/`. A result memo records what was observed at that point in time; it is not a final comparison or a task tracker.

## Current evidence

- Multi-seed selective benchmark: `docs/results/selective-benchmark-snapshot.md`.
- Historical single-seed design experiments: `docs/results/intel-fault15-development.md`.
- The strongest selective variants improve average macro-F1 over the best temporal baseline by about `+0.012`, which clears the minimum accuracy target but not the preferred `+0.03` to `+0.04` target.
- The strict decisive-comparison audit now validates all 48 dense and selective CESTA cells. Selective variants save serialized TX+RX energy relative to dense CESTA, but budget-matched controls remain absent.
- Legacy temporal and ST-GCN runs use an `80/10/10` split and cannot support direct paired accuracy claims against the `70/15/15 connectivity-chronological` CESTA matrix. A split-matched temporal sweep is in progress.

## Active priorities

### P0: Complete the decisive comparison

Artifact audit implemented from `docs/experiments/decisive-comparison-artifact-audit.md`: strict mode validates all 48 dense and selective CESTA cells.

- [x] Rerun dense CESTA with CRF weight `0.05` on `Intel_fault05`, `Intel_fault10`, `Intel_fault15`, and `Intel_fault20` using seeds `12`, `42`, and `1242`.
- [x] Confirm that every dense and selective run serializes comparable TX, RX, total, and dense-reference energy fields.
- [x] Export deterministic run-level macro-F1, per-class F1, accuracy, request ratio, transmitted bits, TX+RX energy, and energy reduction from valid artifacts.
- [x] Aggregate the complete dense and selective matrix and compute paired bootstrap uncertainty.
- [ ] Complete the split-matched temporal matrix, lock the primary comparator from validation macro-F1 only, and report paired temporal-versus-CESTA uncertainty.

**Done when:** one provisional benchmark memo compares temporal, dense, and selective models across all four fault ratios without relying on transmitted-bit counts as an energy proxy.

### P0: Add budget-matched controls

- [ ] Define target budgets from the selected CESTA request ratio and TX+RX energy.
- [ ] Implement random matched-budget communication.
- [ ] Implement static top-k communication using connectivity strength.
- [ ] Implement entropy-, prediction-margin-, and local-change-threshold controllers.
- [ ] Tune every control on validation data only, then evaluate the locked policies on test data.
- [ ] Compare controls and learned gating on the macro-F1 versus TX+RX energy Pareto frontier.

**Done when:** learned request gating can be compared with simple policies at equivalent communication cost, with no test-set budget tuning.

### P1: Resolve external baseline coverage

- [ ] Determine whether HiFiNet is directly applicable to the same sensor-fault setting and graph inputs.
- [ ] Reproduce it if applicable; otherwise record a precise scope mismatch and include the closest reproducible alternative.
- [ ] Keep dynamic ST-GCN as a diagnostic baseline, not the sole spatial comparator.

**Done when:** the closest-method comparison is reproducible or its exclusion is defensible from documented task and input differences.

### P1: Check paper-readiness

- [ ] Verify the average macro-F1 uplift against the best temporal model selected independently for each fault ratio.
- [ ] Verify energy Pareto superiority against dense CESTA.
- [ ] Verify learned-gating superiority against at least random, static top-k, and the strongest rule-based matched control.
- [ ] Report failure cases by fault class, especially SPIKE oversmoothing and DRIFT/STUCK boundary errors.
- [ ] Measure parameter count, serialized size, inference latency, and peak memory for the edge-oriented claim.
- [ ] Decide whether the supported contribution is a Q1-level method paper or a narrower communication-efficiency study.

**Done when:** every primary claim maps to a completed comparison, metric, uncertainty estimate, and reproducible artifact.

## Deferred work

Do not begin these items until both P0 sections are complete:

- [ ] Compression-only CESTA with fixed ratios `{0.25, 0.5, 1.0}`.
- [ ] Joint request and compression training.
- [ ] RL request-controller prototype.
- [ ] Quantized deployment evaluation.
- [ ] Measured ESP32-S3 communication energy.

## Hard boundaries

- Do not create a final comparison document until the experiment protocol and paper-readiness checklist are complete.
- Do not claim energy savings from transmitted-bit counts alone.
- Do not select controllers, thresholds, or model variants using test results.
- Do not compare against ST-GCN alone; dense learned message passing is required.
- Do not claim learned gating superiority without budget-matched controls.
- Do not let CESTA inspect neighbor embeddings before deciding to request communication.
- Do not start compression or RL before the request-only comparison is complete and seed-robust.
- Preserve missing nodes and unavailable links through `node_mask` and `edge_mask`; do not permanently remove them.

## Documentation workflow

1. Update `docs/PROPOSAL.md` only when research questions, hypotheses, scope, or success criteria change.
2. Update `docs/EXPERIMENT.md` before changing the evaluation protocol, selection rule, baselines, or metrics.
3. Add a focused file under `docs/experiments/` when an experiment needs setup details beyond the shared protocol.
4. Add a dated or clearly scoped memo under `docs/results/` after a completed experiment batch.
5. Update this tracker when priorities or completion states change.
6. Create a paper-facing comparison only after all P0 tasks and the paper-readiness checks are complete.
