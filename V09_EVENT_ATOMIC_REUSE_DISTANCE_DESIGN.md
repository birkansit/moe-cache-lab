# v0.9 B1 — Event-atomic LRU reuse-distance contract

## Scope

This document freezes the count-capacity reference semantics for one Gate B diagnostic: **event-atomic LRU reuse distance**.

It is deliberately narrower than a user-facing product feature. The reference implementation is `scripts/audit_event_atomic_lru_reuse_distance.py`; it exists to make later implementation falsifiable against the already-established cache simulator.

No cache policy, routing trace contract, preflight contract, package version, or runtime behavior changes in B1.

## Why a separate contract is necessary

`moe-cache-lab` already exposes `ReuseGapSummary`. That statistic answers a different question: how many routing **events** occur strictly between two selections of the same identity. It does not count distinct intervening cache identities and it does not encode LRU eviction rank.

The established LRU replay is also not a serial per-expert access stream. One canonical routing event is an atomic bundle:

1. residency is inspected at event start;
2. every required key is pinned while misses are admitted;
3. all required keys receive one common logical event timestamp;
4. ties at the same timestamp use the stable canonical key.

Flattening a top-k event into arbitrary serial expert accesses invents an intra-event chronology. A scalar stack distance produced from that invented order can therefore disagree with the actual event-atomic LRU semantics.

## Canonical cache identity

The diagnostic uses exactly the cache identity already established by the trace/cache contracts:

- trace v1: `(layer, expert_id)`;
- trace v2: `(routing_stage, layer, expert_id)`.

Encoder and decoder namespaces never alias. A v2 `unassigned` event has an empty required set and contributes zero expert requests.

## Authoritative chronology

Events are consumed in validated canonical order. The diagnostic never sorts, deduplicates, repairs, or reconstructs event chronology.

Let event `t` have atomic required-key set `R_t`. Every key in `R_t` is evaluated against exactly the same state that exists immediately before event `t`.

Only after all samples for `R_t` have been evaluated are all keys in `R_t` touched atomically at event index `t`.

Selected-expert tuple ordering inside the event is therefore semantically inert.

## First use

A key that has not occurred in any earlier required set is a first use.

First uses:

- contribute to `first_use_count`;
- contribute to total expert-request count;
- produce no reuse-distance sample;
- are not assigned an invented infinite, zero, or sentinel stack distance.

For any finite trace, `first_use_count` equals the number of distinct referenced canonical keys.

## Reuse priority

For a reused key `k`, let `last(k)` be the index of its previous event touch.

Immediately before the reuse event, define its LRU priority as:

`(last(k), k)`

where the second term is the exact stable canonical key used by the existing LRU tie-break.

The established simulator evicts the minimum `(timestamp, key)`. Therefore a key with a larger priority tuple is more recent / harder to evict.

Same-event peers have equal timestamps. Among them, the lexicographically larger canonical key has higher LRU priority because the smaller key is the first eviction victim when a tie must be broken.

## Event-atomic LRU stack distance

For a reused key `k`, the stack distance is the number of previously touched distinct canonical keys with strictly greater priority:

`distance(k) = |{ q != k : (last(q), q) > (last(k), k) }|`

This is the zero-based pre-event rank of `k` when previously touched keys are ordered from highest LRU priority to lowest.

The corresponding count capacity needed for `k` to be resident at the start of that event is:

`required_capacity_entries(k) = distance(k) + 1`

This definition contains no serial order among members of the current event and no serial order among members of an earlier atomic event beyond the simulator's already-frozen stable-key tie-break.

## Exact count-capacity parity rule

Define:

`max_atomic_event_entries = max_t |R_t|`.

A count capacity `C` is feasible for the complete trace only when:

`C >= max_atomic_event_entries`.

For every feasible `C`, the reference contract requires:

`predicted_lru_hits(C) = count(reuse samples with required_capacity_entries <= C)`.

That predicted hit count must equal the established event-atomic LRU replay hit count.

For trace v1 the comparison is direct against `cache.simulate(..., policy_name="lru")`.

For trace v2, B1 uses `simulate_versioned_byte_cache()` only as a parity oracle with a synthetic one-byte size assigned independently to every referenced stage-qualified key and `capacity_bytes=C`. This is a count-capacity equivalence test. The one-byte values are test units, not measured physical extents and not config-v3 evidence.

If reference rank and established simulator disagree at any tested feasible capacity, the B1 contract fails. The simulator must not be changed merely to force agreement.

## Exact reference outputs

The B1 reference summary records:

- event count;
- expert-request count;
- distinct referenced key count;
- first-use count;
- reuse count;
- maximum atomic event entries;
- deterministic histogram `required_capacity_entries -> reuse_count`;
- minimum and maximum required reuse capacity when defined;
- trace-specific first capacity with any observed LRU reuse hit;
- optional exact reuse fraction within a caller-supplied feasible capacity;
- per-reuse audit samples containing event indices, canonical key, stack distance, and required count capacity.

Ratios use exact `Fraction` semantics. JSON audit output serializes a fraction as an exact numerator/denominator pair rather than a rounded floating-point value.

## First capacity with any observed LRU reuse hit

When no reuse exists, the value is `None`.

Otherwise:

`max(max_atomic_event_entries, min(required_capacity_entries over reuse samples))`.

This is a descriptive property of **this trace under this event-atomic LRU abstraction**. It is not a recommendation and is not named a minimum meaningful cache, knee, optimal capacity, or deployment threshold.

## Relationship to `ReuseGapSummary`

The two diagnostics must remain separate.

Example: a key can have one intervening event but that event can touch several distinct canonical keys. Its event gap can therefore be `1` while its event-atomic LRU stack distance is greater than `1`.

Conversely, repeated events that touch only already-more-recent keys can change event-gap counts without introducing the same number of distinct LRU ranks.

`ReuseGapSummary` continues to answer event-spacing questions. The B1 contract answers count-capacity LRU residency-rank questions.

## Same-event tie behavior

Suppose event 0 touches keys `(0,0)` and `(0,1)` atomically, and event 1 later touches `(0,2)`.

Before a later reuse of the first pair:

- `(0,2)` is more recent than both event-0 keys;
- `(0,1)` has higher priority than `(0,0)` because their timestamps tie and `(0,0)` is the stable-key eviction victim first.

Therefore the two event-0 keys have different required capacities even though neither was touched after the other. A flattened stream can reverse that result simply by choosing the opposite serialization order for event 0; that is precisely why flattening is not the project contract.

## v2 unassigned events

An unassigned v2 event has `R_t = empty`.

It:

- increments canonical event count;
- contributes zero expert requests;
- produces no first-use or reuse sample;
- touches no recency state;
- cannot fabricate a preferred expert.

Its presence may shift displayed event indices, but it does not alter canonical-key recency ordering or capacity/hit predictions.

## Evidence class and claim boundary

Event-atomic reuse distance is a deterministic derivation from a supplied canonical routing trace under the project's LRU abstraction.

It is not by itself a runtime measurement. Its scientific interpretation inherits the provenance boundary of the routing trace but does not upgrade it.

A later cache replay remains **SIMULATED**. A later transfer-service result remains **ESTIMATED** under explicit assumptions.

This contract establishes none of the following:

- physical CPU/GPU residency;
- physical transfer traffic;
- latency, throughput, tokens/s, or speedup;
- an optimal or recommended cache size;
- a best cache policy;
- workload representativeness;
- a generic model cacheability score;
- broad model/runtime compatibility.

## Deferred from B1

B1 intentionally does not define:

- heterogeneous byte-aware reuse distance;
- model-matched physical extents;
- config-v3 extent joins;
- percentile or knee detection;
- adjacent-capacity or policy deltas;
- external Kimi/OLMoE numerical operating-regime claims;
- a public CLI or package API.

Those require separate gates after the count-capacity reference rule has exact parity with the established event-atomic LRU simulator.
