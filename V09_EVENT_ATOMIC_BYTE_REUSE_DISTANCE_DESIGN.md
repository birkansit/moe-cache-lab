# v0.9 B2 — Event-atomic byte-aware LRU reuse-distance contract

## Scope

This document freezes the heterogeneous byte-capacity extension of the v0.9 B1 event-atomic LRU reuse-distance contract.

B2 remains reference-only. It does not add a public CLI/API, change cache behavior, change routing/preflight schemas, or establish any runtime or physical-memory result.

The reference implementation is:

`scripts/audit_event_atomic_lru_byte_reuse_distance.py`

Its purpose is to make the proposed weighted-stack rule falsifiable against the already-established byte-capacity LRU simulator.

## Inputs and evidence boundary

The diagnostic consumes two independent inputs:

1. one already-valid canonical routing trace;
2. one caller-supplied positive-integer byte size for every referenced canonical cache key.

Canonical identity is unchanged:

- v1: `(layer, expert_id)`;
- v2: `(routing_stage, layer, expert_id)`.

A byte-valued size is **not automatically MEASURED**. The size map may be an analysis assumption unless separate provenance establishes what those bytes physically represent.

The routing evidence boundary and size evidence boundary therefore remain separate.

## Authoritative event-atomic LRU behavior

The existing byte-cache simulator is authoritative:

1. each routing event is one atomic required-key bundle;
2. residency is inspected at event start;
3. all required keys are pinned while misses are admitted;
4. non-required victims are evicted by the minimum `(last_touch_timestamp, stable_canonical_key)`;
5. every required key receives one common event timestamp;
6. v2 unassigned events project to an empty required set and do not touch recency state.

B2 must predict this behavior. The simulator is not modified to fit the reference formula.

## Pre-event priority

For a reused key `k`, let `last(k)` be the index of its previous event touch.

Define the same B1 LRU priority:

`priority(k) = (last(k), k)`

where `k` uses the already-frozen stable canonical-key ordering.

A larger priority tuple is more recent / harder to evict.

Keys touched in the same earlier atomic event have the same timestamp. Their relative LRU priority is therefore determined only by the stable canonical key, exactly as in the simulator.

## Byte-aware stack distance

Let `size(q)` be the validated caller-supplied positive integer byte size of canonical key `q`.

For a reused key `k` immediately before event `t`, define:

`stack_distance_bytes(k) = sum(size(q) for q != k where priority(q) > priority(k))`

over all distinct canonical keys touched before event `t`.

Then define:

`required_capacity_bytes(k) = stack_distance_bytes(k) + size(k)`

Interpretation: `stack_distance_bytes` is the byte weight of the strictly higher-priority portion of the event-atomic LRU stack; adding `size(k)` gives the byte capacity threshold at which `k` itself can remain resident.

All requests in one event are evaluated against the exact same pre-event state. Only after every first-use/reuse sample in the event is evaluated are all required keys touched atomically.

Selected-expert tuple ordering inside the event is semantically inert.

## First use

A key that has not appeared in an earlier required set is a first use.

First uses:

- contribute to `first_use_count`;
- contribute to expert-request count;
- contribute to distinct referenced key count;
- produce no byte reuse-distance sample.

No infinite, zero, or sentinel distance is fabricated.

## Feasible byte capacity

For event `t` with required set `R_t`, define:

`atomic_event_bytes(t) = sum(size(k) for k in R_t)`

and:

`max_atomic_event_bytes = max_t atomic_event_bytes(t)`.

A capacity `C` is feasible for the complete trace only when:

`C >= max_atomic_event_bytes`.

This is the same atomic-fit rule already enforced by the established byte-cache simulator.

## Exact simulator parity rule

For every feasible byte capacity `C`:

`predicted_lru_hits(C) = count(reuse samples where required_capacity_bytes <= C)`.

B2 accepts the weighted-stack definition only if that value equals the established event-atomic byte-LRU simulator hit count using the exact same size map.

B2 therefore requires exact simulator parity at every tested feasible integer byte capacity; approximate agreement is insufficient.

The focused B2 suite checks every integer capacity from `max_atomic_event_bytes` through the full referenced working-set bytes for several non-uniform synthetic v1/v2 traces, including same-event timestamp ties and multi-expert atomic bundles.

Any valid counterexample is a contract blocker. If parity fails, the simple weighted-stack formulation is a bounded `NO-GO`; the simulator must not be patched and the mismatch must not be hidden.

## Uniform-size reduction to B1

If every referenced key has the same synthetic size `S`, then B2 must reduce dimensionally to B1:

`stack_distance_bytes = stack_distance_entries * S`

`required_capacity_bytes = required_capacity_entries * S`

`max_atomic_event_bytes = max_atomic_event_entries * S`

At capacities that are integer multiples of `S`, B2 predicted hits must equal B1 count-capacity predicted hits.

This uniform-size reduction is a mathematical consistency check only. The synthetic byte value is not physical extent evidence.

## Same-event stable-key example

Suppose event 0 atomically touches keys `(0,0)` and `(0,1)`, event 1 touches `(0,2)`, and a later event reuses the first pair.

Assume sizes:

- `(0,0)`: 2 bytes;
- `(0,1)`: 5 bytes;
- `(0,2)`: 3 bytes.

Before the reuse:

- `(0,2)` is newer than both event-0 keys;
- `(0,1)` outranks `(0,0)` because the earlier pair shares one timestamp and the larger stable key has higher LRU priority.

Therefore:

- `(0,1)` has `stack_distance_bytes = 3`, `required_capacity_bytes = 8`;
- `(0,0)` has `stack_distance_bytes = 5 + 3 = 8`, `required_capacity_bytes = 10`.

No serial order inside event 0 is invented.

## v2 unassigned events

A canonical v2 unassigned event has an empty required set.

It:

- contributes to canonical event count;
- contributes zero expert requests;
- contributes zero first-use/reuse samples;
- requires no size entry;
- does not change recency ordering;
- does not change byte thresholds or predicted hits, except that later displayed event indices may shift.

No preferred expert or byte size is fabricated for an unassigned event.

## Exact reference outputs

The B2 reference summary records:

- event count;
- expert-request count;
- distinct referenced key count;
- first-use count;
- reuse count;
- referenced working-set bytes;
- maximum atomic event bytes;
- deterministic histogram `required_capacity_bytes -> reuse_count`;
- minimum and maximum required byte capacities when reuses exist;
- trace-specific first feasible byte capacity with any observed LRU reuse hit;
- optional exact reuse fraction within a tested feasible byte capacity;
- per-reuse samples containing event indices, canonical key, key size, byte stack distance, and required byte capacity.

Ratios use exact `Fraction` semantics and serialize as exact numerator/denominator pairs.

The reference payload also states explicitly that sizes are caller-supplied cache-model inputs and that physical extent provenance is not inferred.

## First feasible byte capacity with any observed LRU reuse hit

When no reuse exists, the value is `None`.

Otherwise:

`max(max_atomic_event_bytes, min(required_capacity_bytes over reuse samples))`.

This is a descriptive property of the supplied trace + supplied sizes under this event-atomic LRU abstraction. It is **not a recommendation**, knee, optimum, deployment threshold, or minimum meaningful cache.

## Relationship to B1 and `ReuseGapSummary`

B2 does not modify B1.

B1 answers count-capacity LRU residency-rank questions.
B2 weights that same event-atomic LRU priority stack by caller-supplied per-key byte sizes.

`ReuseGapSummary` remains a separate event-spacing statistic and is not redefined by either contract.

## Size-map validation boundary

The reference requires:

- a positive integer size for every referenced canonical key;
- one canonical key shape per analysis (v1 pair or v2 triple);
- valid non-negative numeric identity fields;
- `encoder` / `decoder` stage names for v2;
- no booleans, zero, negative, fractional, or inferred sizes.

Extra **valid, unreferenced** entries are allowed and are result-inert, matching established byte-cache behavior.

Mixed v1/v2 identities or missing referenced sizes fail explicitly.

## Evidence and claim boundary

This diagnostic is a deterministic derivation from supplied routing evidence plus a supplied size map under the project's event-atomic LRU abstraction.

Cache replay remains **SIMULATED**.

Transfer-service quantities remain **ESTIMATED** under explicit assumptions.

B2 does not establish:

- measured or model-matched physical expert extents;
- physical residency;
- physical host/device or GPU transfer;
- allocator behavior;
- latency, throughput, tokens/s, or speedup;
- workload representativeness;
- a best cache policy;
- an optimal or recommended capacity;
- a generic model cacheability score;
- broad model/runtime compatibility.

## Deferred from B2

B2 intentionally does not define or implement:

- config-v3 physical extent provenance joins;
- shared-codebook or overlapping-storage accounting;
- LFU byte reuse-distance theory;
- real Kimi or OLMoE byte-aware numerical operating-regime results;
- adjacent-capacity or policy-delta product surfaces;
- automatic percentile/knee detection;
- public CLI/API/reporting surface;
- runtime offloading or expert swapping;
- trace v3;
- release or package publication work.

Those require separate authorization after this exact simulator-parity contract is accepted.
