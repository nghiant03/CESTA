# Research proposal: CESTA

## Name

**CESTA**: Communication-Efficient Spatial-Temporal Aggregation.

## Motivation

Fault diagnosis in sensor networks benefits from temporal modeling, but dense spatial models can waste energy by communicating with all neighbors even when local temporal evidence is sufficient. A convincing contribution should exceed strong temporal-only macro-F1 while reducing communication energy relative to dense spatial communication; beating weak ST-GCN alone is not enough.

## Research questions

1. Can a lightweight spatial-temporal model exceed temporal-only macro-F1 by selectively requesting neighbor information?
2. Can learned request and compression decisions reduce measured and theoretical communication energy while preserving or improving accuracy?
3. Does a trainable communication controller outperform rule-based uncertainty/change-triggered communication at the same communication budget?
4. Is Gumbel-Softmax or reinforcement learning better for the communication policy under a Pareto criterion?

## Hypotheses

1. CESTA improves average macro-F1 over the best temporal-only model across Intel fault ratios.
2. CESTA reduces TX+RX communication energy compared with dense spatial message passing by requesting only useful neighbor embeddings and compressing transmitted messages.
3. Receiver-side local uncertainty and temporal state are sufficient to decide when neighbor information is useful without inspecting neighbor embeddings before communication.
4. Gumbel-Softmax will be easier to train and more reproducible than RL, but the final method should be selected by Pareto dominance.

## Success criteria

Primary success criterion:

```text
Average Δ macro-F1 >= +0.01 over the best temporal-only model per fault ratio
and Pareto-superior energy/accuracy behavior against dense spatial communication.
```

Preferred target:

```text
Average Δ macro-F1 >= +0.03 to +0.04 over the best temporal-only model per fault ratio
and substantial measured communication-energy reduction against dense spatial communication.
```

Secondary criteria:

- improve over a fixed temporal backbone used inside CESTA;
- outperform or match required spatial baselines, including HiFiNet if applicable;
- reduce theoretical TX+RX radio energy compared with dense learned message passing;
- remain lightweight enough to classify as edge-oriented, with ESP32-S3 as a loose lower target.

## Core design

CESTA is a distributed receiver-side request model. Each node encodes its local temporal window, estimates local diagnosis uncertainty, and decides which existing graph neighbors to request from and at what compression ratio. The gate must use only receiver-side local state, local uncertainty, and edge metadata; it must not inspect sender embeddings before communication.

The graph topology is a fixed directed candidate edge set derived from `connectivity.txt`, while runtime communication availability is dynamic:

```text
edge_index[0, e] = sender node
edge_index[1, e] = receiver node
edge_prob[e] = raw connectivity probability p_sender,receiver
active_edge[t,e] = link_success[t,e] & node_observed[t,sender(e)] & node_observed[t,receiver(e)]
```

Graph preparation stores directed edges, a once-sampled bursty link-success mask, node observation masks, and dynamic-link metadata. Missing labels are stored as `-1` and excluded by masked loss/metrics.

CESTA aggregates only received messages with single-head GAT-inspired attention: query from the receiver local state, key/value from received neighbor states, softmax over the received set only, and a zero vector when no neighbors are available/requested. Optional local refinements such as logit correction and CRF decoding do not add message payloads and should be treated as accuracy refinements before selective communication claims.

## Energy model

Communication claims must use energy, not transmitted-bit counts alone. For a message of `k` bits over distance `d`:

```text
E_tx(k, d) = E_elec · k + E_amp · k · d^n
E_rx(k) = E_elec · k
E_msg(k, d) = E_tx(k, d) + E_rx(k)
d0 = sqrt(E_fs / E_mp)
```

Use free-space or multipath amplifier constants according to `d0`, count both TX and RX for every active sender→receiver message, serialize constants and distance assumptions, and report energy reduction versus dense learned message passing. Measured on-device energy remains desirable but should not be assumed without measurement.

## Required baselines

1. Best temporal-only model per fault ratio.
2. Fixed CESTA temporal backbone without communication.
3. Dynamic ST-GCN.
4. HiFiNet, if it targets sensor/graph fault diagnosis and can be reproduced or bounded.
5. Dense learned message passing over all currently available directed candidate edges.
6. Static top-k graph communication using strongest connectivity edges.
7. Random communication at matched average budget.
8. Rule-based uncertainty/change-triggered communication at matched budget.

## Scope

In scope:

- Intel graph datasets across `fault05`, `fault10`, `fault15`, and `fault20`;
- temp-only input for comparability with existing baselines;
- learned receiver-side request and compression over currently available directed candidate edges;
- Gumbel and RL controller comparison once Gumbel request-only is stable;
- theoretical TX+RX energy and, if feasible, measured edge energy.

Out of scope for the first paper iteration:

- unconstrained latent graph discovery;
- multi-hop communication protocols beyond existing graph neighbors;
- quantization as the core novelty;
- assuming ESP32-S3 feasibility without measurement.

## Current evidence

Provisional evidence is recorded as scoped memos under `docs/results/`:

- `docs/results/intel-fault15-development.md` preserves the historical single-seed design experiments;
- `docs/results/selective-benchmark-snapshot.md` aggregates three seeds across four fault ratios for selected request-gated variants;
- the strongest selective variants clear the minimum average macro-F1 uplift by about `+0.012`, but remain below the preferred `+0.03` to `+0.04` target;
- the current benchmark is incomplete because it lacks seed-replicated dense CESTA, budget-matched controls, and consolidated TX+RX energy reporting.

## Main risks

1. Temporal baselines are already strong, so the preferred +0.03 to +0.04 macro-F1 target may be difficult.
2. The multi-seed selective uplift clears the minimum target but is not yet a strong Q1-level margin.
3. Dense CESTA has not yet been replicated across the full benchmark, so Pareto superiority is unresolved.
4. Learned gating has not yet been tested against budget-matched controls.
5. ST-GCN is too weak to be the only spatial comparator.
6. Communication gates may collapse all-on or all-off.
7. Energy claims require consolidated TX+RX reporting from serialized run artifacts.
8. CRF smoothing helps persistent faults but can oversmooth short SPIKE events.
9. Naive boundary supervision hurt the current dense model.
10. HiFiNet may be required or must be clearly bounded as unavailable/inapplicable.

## Research tracker

Unresolved implementation, experiment, and paper-readiness work lives in `PLAN.md`.
