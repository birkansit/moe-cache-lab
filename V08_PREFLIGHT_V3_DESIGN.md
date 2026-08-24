# V0.8 stage-qualified pre-flight config v3 design record

Status: **GO for a new `moe-cache-lab.preflight-config` format version 3 for
single-trace canonical routing trace v2 pre-flight analysis; lifecycle extension
is DEFERRED.**

This design record freezes the smallest configuration contract needed to carry
caller-supplied expert sizes without aliasing encoder and decoder experts. It is
a contract design, not an implementation claim. At this checkpoint there is no
config-v3 parser/model, no public v2 `--preflight-config` workflow, no v2
lifecycle orchestration, and no new report schema.

The cross-format rules in [`COMPATIBILITY.md`](COMPATIBILITY.md) remain
authoritative. Routing-trace and pre-flight-config versions are independent
namespaces: config v3 does not create or require routing trace v3.

## 1. Established current behavior

The current implementation establishes these constraints:

- canonical trace v1 uses layer-qualified `(layer, expert_id)` expert identity;
- canonical trace v2 uses stage-qualified
  `(routing_stage, layer, expert_id)` identity;
- `simulate_versioned_byte_cache()` already accepts validated trace v2 with a
  stage-qualified size map and keeps unassigned events as zero-request,
  cache-inert events;
- pre-flight-config versions 1 and 2 use only `(layer_id, expert_id)` size
  records, so public pre-flight orchestration remains trace-v1-only;
- config v1/v2 allow extra expert-size entries and require sizes only for expert
  keys actually referenced by the replayed trace;
- config v2 already contains the transfer-operation-plan contract needed by the
  current serialized transfer-service estimator;
- the current single-trace transfer sensitivity helper is typed around v1
  `RoutingEvent` and `ExpertKey`, so later implementation must become
  version-aware rather than flattening v2 identity;
- the current lifecycle workload/result types and corpus loader are v1-specific:
  they require `RoutingEvent`, store two-part `ExpertKey` residency, and load
  `RoutingTrace` from the v1 corpus/manifest workflow.

These facts justify a new **pre-flight-config** version and do not justify a new
routing-trace version.

## 2. Decision summary

The accepted v3 design is:

- format family remains `moe-cache-lab.preflight-config`;
- `format_version` is exactly integer `3`;
- top-level fields are exactly the config-v2 fields;
- `capacities_bytes`, `policies`, `hardware_profiles`, and
  `transfer_operation_plans` retain config-v2 meanings unchanged;
- only the expert-size namespace changes structurally: every v3 expert-size
  record is stage-qualified;
- the v3 `expert_sizes` array may be empty so an all-unassigned trace does not
  require an irrelevant/fabricated expert-size assumption;
- for every expert actually requested by an assigned trace-v2 event, the exact
  stage-qualified size must be present;
- extra valid stage-qualified size entries remain permitted and do not affect a
  trace that does not reference them;
- config v3 is intended only for trace v2 orchestration; trace v1 + config v3
  rejects rather than inferring a decoder stage;
- trace v2 + legacy config v1/v2 continues to reject;
- single-trace v2 pre-flight is GO after later implementation gates;
- v2 lifecycle is deferred because the existing lifecycle/corpus contract is
  structurally v1-specific and cannot be widened safely under this config-only
  design.

No existing config-v1/v2 accepted or rejected input changes because of this
design.

## 3. Proposed top-level v3 contract

A v3 document is one closed JSON object with exactly these required fields:

| Field | Required value / type |
| --- | --- |
| `format` | string constant `moe-cache-lab.preflight-config` |
| `format_version` | integer constant `3`; boolean is invalid |
| `expert_sizes` | JSON array of stage-qualified expert-size records; may be empty |
| `capacities_bytes` | non-empty JSON array of positive integers |
| `policies` | non-empty JSON array containing only `lru` and/or `lfu` |
| `hardware_profiles` | non-empty JSON array using the established config-v2 profile contract |
| `transfer_operation_plans` | non-empty JSON array using the established config-v2 plan contract |

Unknown top-level fields reject. Missing fields reject. V3 parsing must dispatch
only after the exact format/version pair is established; it must not fall back
to v1/v2 parsing after a v3 validation failure.

The canonical normalized forms inherited from config v2 remain:

- capacities: ascending unique integers;
- policies: stable `lru`, then `lfu` order when selected;
- hardware profiles: ascending unique name;
- transfer-operation plans: ascending unique name.

## 4. Stage-qualified expert-size record

Each v3 `expert_sizes` entry is a closed object with exactly:

```json
{
  "routing_stage": "encoder",
  "layer_id": 3,
  "expert_id": 5,
  "size_bytes": 18874368
}
```

Rules:

- `routing_stage` is required and exactly `encoder` or `decoder`;
- `layer_id` is a non-negative integer and not a boolean;
- `expert_id` is a non-negative integer and not a boolean;
- `size_bytes` is a positive integer and not a boolean;
- identity is exactly `(routing_stage, layer_id, expert_id)`;
- duplicate identities reject even when their byte sizes are equal;
- unknown or missing record fields reject;
- deterministic canonical order is stage order `encoder`, then `decoder`, then
  ascending `layer_id`, then ascending `expert_id`.

The stage vocabulary deliberately matches canonical trace v2. This config
version does not create generic runtime-specific stage labels.

No stage is inferred from model ID, trace filename, numerical layer range,
neighboring records, or collector type. Numerical layer IDs remain local to the
stage namespace.

## 5. Why v3 is not a migration of v1/v2

V1/v2 expert-size records are two-part identities. V3 records are three-part
identities. Automatically treating every legacy record as `decoder`, offsetting
encoder layers, or deriving stage from a trace would alter the meaning of the
legacy configuration.

Therefore:

- a v3 record without `routing_stage` rejects;
- a legacy expert-size record mixed into a v3 array rejects;
- config v1/v2 are never upgraded in memory by adding an inferred stage;
- config v3 is never down-converted by discarding stage.

An arbitrary layer-offset convention is not recognized. For example, an extra
entry for `(encoder, 1003, 5)` may be syntactically well-formed, but it does not
satisfy a trace request for `(encoder, 3, 5)`. The later orchestration must
report the actual requested key as missing rather than guessing that `1003`
means layer `3`.

## 6. Existing config-v2 semantics retained exactly

### Cache capacities and policies

V3 keeps the existing requirements:

- at least one positive integer capacity;
- duplicate capacities normalize away;
- only `lru` and `lfu` are accepted;
- selected policies normalize to stable `lru`, `lfu` order.

No recommendation, auto-sizing, or hardware detection is added.

### Hardware profiles

V3 uses the same closed record as config v2:

```json
{
  "name": "fictional-link",
  "h2d_payload_bandwidth_bytes_per_second": 20000000000,
  "setup_latency_ns_per_transfer_operation": 50000000
}
```

Names remain unique. Bandwidth is a positive integer. Per-operation setup
latency is a non-negative integer. These values remain caller-supplied
assumptions unless established separately by measurement.

### Transfer-operation plans

V3 uses the same closed config-v2 record:

```json
{
  "name": "two-chunks",
  "operations_per_logical_load": 2
}
```

Names remain unique and `operations_per_logical_load` remains a positive
integer. A plan changes only the modeled setup-bearing operation count; it does
not change cache hits, misses, demand-load bytes, evictions, or residency.

No new transfer equation or overlap model is introduced by config v3.

## 7. Trace/config compatibility matrix

After the later B2/B3 implementation, the intended public single-trace matrix
is:

| Routing trace | No config | Config v1 | Config v2 | Config v3 |
| --- | --- | --- | --- | --- |
| v1 | supported descriptive analysis | supported existing pre-flight | supported existing pre-flight | **reject** |
| v2 | supported descriptive analysis | **reject** | **reject** | **supported stage-qualified pre-flight** |

Rationale:

- v1 has no canonical `routing_stage`, so v1 + v3 would require inference;
- v2 requires stage-qualified identity, so v2 + v1/v2 would require flattening
  or inferred stage;
- the accepted pair is based on identity semantics, never equal version
  numbers.

A config may still be parsed/validated independently of a trace. The matrix is
an orchestration contract for combining already validated artifacts.

## 8. Size coverage and request validation

V3 preserves the established pre-flight distinction between the caller-supplied
size universe and the observed trace request set.

Structural config validation does **not** require a complete size table for all
experts declared by every trace-v2 stage profile. A config may contain a subset,
a superset, or—when the trace makes no expert requests—an empty size array.

At orchestration/simulation time:

1. derive actual request keys only from assigned canonical trace-v2 events;
2. preserve each exact `(routing_stage, layer, expert_id)` key;
3. require a positive size for every referenced key;
4. reject if any referenced key is absent;
5. permit extra valid size records without charging or requesting them;
6. require each event's complete atomic requested working set to fit the tested
   byte capacity, preserving the existing event-atomic admission rule.

The trace validator, not the config, remains authoritative for whether an
expert ID is within the declared stage profile. The config must not reinterpret
or repair trace metadata.

### All-unassigned traces

A valid trace-v2 may contain routing events that are all explicitly unassigned.
Those events have empty expert request sets. The existing version-aware cache
core can replay such a trace with an empty size map.

Accordingly, v3 permits:

```json
"expert_sizes": []
```

when no assigned event needs a size. This avoids requiring an irrelevant caller
assumption merely to represent a zero-request workload. If any assigned event
exists, its exact requested keys still require sizes.

## 9. Unassigned-event semantics

An unassigned trace-v2 event:

- remains present in routing `event_count`;
- contributes zero expert requests;
- creates zero hits and misses;
- loads zero bytes;
- causes zero evictions;
- makes no cache-state change;
- requires no expert-size record for the dropped preference.

V3 never stores or reconstructs the pre-capacity preferred expert for an
unassigned event. Doing so would fabricate a cache request.

## 10. Single-trace pre-flight: GO boundary

**GO** for later implementation of stage-qualified single-trace pre-flight.

The existing version-aware byte-cache core already provides the critical cache
primitive. Later implementation may reuse the established exact transfer-cost
equation and config-v2 hardware/operation assumptions, provided stage identity
is preserved end-to-end.

However, the current `run_transfer_sensitivity_sweep` and `WorkloadByteContext`
are v1-typed and build two-part keys themselves. B3/B4 must introduce a
version-aware path that consumes canonical v2 requests or a validated
version-aware cache result. It must not call the v1 helper by stripping or
offsetting stage.

The required identity chain is:

```text
stage-qualified config sizes
  -> exact trace-v2 request keys
  -> version-aware byte-cache simulation
  -> hits / misses / evictions / demand bytes
  -> existing exact transfer-cost equation
  -> stage-aware workload context
  -> structured output
  -> Markdown/report
```

At every identity-bearing boundary, `(encoder, 3, 5)` and `(decoder, 3, 5)` are
different expert objects.

## 11. Lifecycle: DEFER / NO-GO for the initial v2 path

**DEFER** stage-qualified lifecycle orchestration from the initial v2 pre-flight
implementation.

This is not a cache-policy limitation. It is a contract boundary in the current
multi-workload stack:

- `ByteCacheWorkload.events` accepts only v1 `RoutingEvent` objects;
- lifecycle result residency keys are typed/stored as two-part `ExpertKey`;
- `run_preflight_lifecycle_analysis` uses the v1 corpus/manifest loader;
- `LoadedPromptTrace.trace` is a v1 `RoutingTrace`;
- current lifecycle Markdown/JSON explicitly describes layer-qualified
  identity.

Supporting v2 lifecycle safely therefore needs a separately reviewed
version-aware workload/corpus/output contract. Config v3 alone is insufficient.
B2/B3 must not silently widen lifecycle types or reinterpret existing corpus
manifests. V0.8 can ship a coherent v2 single-trace pre-flight path while the
existing v1 lifecycle workflow remains unchanged.

A later milestone may revisit lifecycle only through an explicit design gate;
absence of v2 lifecycle must be documented rather than hidden.

## 12. Output/report implications for later milestones

Current pre-flight output is v1-specific: it uses the v1 routing summary,
layer-qualified size tables, and a structured pre-flight-analysis format whose
current schema does not encode stage-qualified routing identity.

Therefore B4 must review the output contract before exposing public v2
pre-flight. If stage-qualified fields change the serialized accepted language,
the structured pre-flight-analysis output requires its own format-version bump
or a distinct version-aware format. Existing v1 report bytes and structured
output remain frozen.

B1 does not choose the final internal Python class hierarchy or final report
version number. It freezes only these requirements:

- stage must be explicit wherever an expert identity is serialized;
- no encoder/decoder aliasing is permitted;
- v1 output remains unchanged;
- v2 routing and cache counters retain assigned/unassigned distinction where
  relevant;
- no stage identity may disappear between config, simulation, context, JSON,
  and Markdown.

## 13. Synthetic valid design examples

These examples describe the proposed contract. They are not accepted by the
current parser at this design checkpoint.

### Valid: encoder and decoder use the same numerical layer/expert IDs

```json
{
  "format": "moe-cache-lab.preflight-config",
  "format_version": 3,
  "expert_sizes": [
    {"routing_stage":"encoder","layer_id":3,"expert_id":5,"size_bytes":18874368},
    {"routing_stage":"decoder","layer_id":3,"expert_id":5,"size_bytes":18874368}
  ],
  "capacities_bytes": [150994944],
  "policies": ["lru","lfu"],
  "hardware_profiles": [
    {"name":"fictional-link","h2d_payload_bandwidth_bytes_per_second":20000000000,"setup_latency_ns_per_transfer_operation":50000000}
  ],
  "transfer_operation_plans": [
    {"name":"one-operation","operations_per_logical_load":1},
    {"name":"two-chunks","operations_per_logical_load":2}
  ]
}
```

The two size records are distinct identities despite equal numerical layer and
expert IDs.

### Valid: decoder-only trace-v2 assumptions

```json
{
  "format": "moe-cache-lab.preflight-config",
  "format_version": 3,
  "expert_sizes": [
    {"routing_stage":"decoder","layer_id":1,"expert_id":0,"size_bytes":1024}
  ],
  "capacities_bytes": [1024,2048],
  "policies": ["lru"],
  "hardware_profiles": [
    {"name":"fictional-link","h2d_payload_bandwidth_bytes_per_second":1000000,"setup_latency_ns_per_transfer_operation":0}
  ],
  "transfer_operation_plans": [
    {"name":"one-operation","operations_per_logical_load":1}
  ]
}
```

### Valid: all-unassigned workload assumptions

```json
{
  "format": "moe-cache-lab.preflight-config",
  "format_version": 3,
  "expert_sizes": [],
  "capacities_bytes": [1024],
  "policies": ["lru"],
  "hardware_profiles": [
    {"name":"fictional-link","h2d_payload_bandwidth_bytes_per_second":1000000,"setup_latency_ns_per_transfer_operation":0}
  ],
  "transfer_operation_plans": [
    {"name":"one-operation","operations_per_logical_load":1}
  ]
}
```

This is compatible only with a trace whose assigned events reference no expert
keys. It does not license missing sizes for an assigned event.

## 14. Synthetic invalid design examples

### Missing stage identity

```json
{"layer_id":3,"expert_id":5,"size_bytes":18874368}
```

Reject. V3 does not infer `routing_stage`.

### Unknown stage

```json
{"routing_stage":"shared","layer_id":3,"expert_id":5,"size_bytes":18874368}
```

Reject. V3 stage vocabulary is closed to `encoder` / `decoder`.

### Duplicate identity

```json
[
  {"routing_stage":"encoder","layer_id":3,"expert_id":5,"size_bytes":100},
  {"routing_stage":"encoder","layer_id":3,"expert_id":5,"size_bytes":100}
]
```

Reject even though the sizes agree.

### Boolean or invalid integer

```json
{"routing_stage":"decoder","layer_id":true,"expert_id":5,"size_bytes":100}
```

Reject. Boolean is not accepted as an integer ID. Negative/non-integer IDs and
non-positive/non-integer byte sizes also reject.

### Unknown field

```json
{"routing_stage":"decoder","layer_id":3,"expert_id":5,"size_bytes":100,"label":"mlp"}
```

Reject. V3 expert-size records are closed.

### Legacy record mixed into v3

```json
[
  {"routing_stage":"encoder","layer_id":3,"expert_id":5,"size_bytes":100},
  {"layer_id":3,"expert_id":5,"size_bytes":100}
]
```

Reject rather than treating the second record as decoder.

### Layer-offset workaround

A config that supplies only `(encoder, 1003, 5)` cannot satisfy an actual trace
request for `(encoder, 3, 5)`. The later orchestration rejects the missing exact
requested key; it does not subtract an offset or reinterpret the record.

### Config v3 with trace v1

Reject at orchestration. Trace v1 has no canonical stage, and config v3 must not
assume `decoder`.

## 15. Evidence and claim boundary

Config v3 changes only how caller-supplied expert byte sizes are keyed. It does
not upgrade evidence quality.

- Routing output is **MEASURED** only when trace provenance establishes actual
  measurement.
- Cache hits, misses, loads, evictions, and residency remain **SIMULATED**.
- Transfer-service quantities remain **ESTIMATED** under explicit caller
  assumptions.

Even after later implementation, config v3 does not establish:

- physical expert residency;
- actual DRAM, PCIe, H2D, or D2H traffic;
- measured transfer bandwidth or setup latency;
- end-to-end latency, throughput, tokens/sec, or speedup;
- memory savings or energy benefit;
- production offload correctness;
- optimal policy or capacity;
- broad model/runtime compatibility.

## 16. Implementation gates following this design

This design permits later milestones to implement, in order:

1. a strict config-v3 parser/model while preserving v1/v2 behavior;
2. version-aware single-trace pre-flight orchestration using the existing v2
   cache primitive;
3. stage-aware workload context, transfer sensitivity, and output propagation;
4. public trace-v2 + config-v3 CLI UX and a no-download walkthrough.

Lifecycle is not part of those steps unless a separate stage-qualified
multi-workload design is accepted first.
