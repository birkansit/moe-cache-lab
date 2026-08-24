# Routing trace format family

`moe-cache-lab` consumes canonical, line-oriented routing traces so an external
producer can supply routing observations without importing or running a model
collector. The versioned family uses:

- format: `moe-cache-lab.routing-jsonl`
- version 1 schema:
  [`src/moe_cache_lab/schemas/routing-trace-v1.schema.json`](src/moe_cache_lab/schemas/routing-trace-v1.schema.json)
- version 2 schema:
  [`src/moe_cache_lab/schemas/routing-trace-v2.schema.json`](src/moe_cache_lab/schemas/routing-trace-v2.schema.json)

Both schemas are packaged with the Python module. A schema describes one JSON
record; the package validators also enforce cross-record identity, stage,
range, assignment-width, phase, and chronology rules.

## Shared file rules and version dispatch

The first non-blank line must be exactly one `metadata` record. Every later
non-blank line must be a `routing_selection` record, and at least one event is
required. JSON object key order has no meaning. Duplicate keys, unknown fields,
unknown record types, unsupported versions, and records that mix version
contracts are rejected.

`moe_cache_lab.trace_v2.read_versioned_trace` and
`validate_versioned_trace_records` inspect the metadata format/version and
dispatch to the exact v1 or v2 validator. The readers never guess a version,
migrate records, sort events, or repair invalid chronology.

The two versions coexist with intentionally different identities and phases:

| Version | Expert-cache identity | Event identity | Phases |
| --- | --- | --- | --- |
| v1 | `(layer, expert_id)` | `(token_position, layer)` | `prompt`, `generated` |
| v2 | `(routing_stage, layer, expert_id)` | `(routing_stage, token_position, layer)` | encoder `source`; decoder `decoder_prompt`, `decoder_generated` |

One routing-selection record is one atomic routing event. All unique entries in
`selected_experts` belong to that event together. A producer must not split a
top-k assignment into separate records, and downstream cache requests retain
event atomicity rather than interpreting tuple order as serial accesses.

## Version 1: layer-qualified decoder-style traces

Version 1 preserves the established layer-qualified contract:

- `format_version`: integer constant `1`
- schema: `routing-trace-v1.schema.json`
- Python reader/validator: `moe_cache_lab.trace.read_trace` and
  `validate_trace_records`

### V1 metadata

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `record_type` | string constant `metadata` | Identifies the first record. |
| `format` | string constant `moe-cache-lab.routing-jsonl` | Selects this trace family. |
| `format_version` | integer constant `1` | Selects v1. |
| `model_id` | non-empty string | Producer-supplied provenance; not a general model-support claim. |

Optional fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `num_experts` | positive integer or null | Expert count within each routed layer. When present, selected IDs must be smaller. |
| `experts_per_token` | positive integer or null | Expected event selection width. |
| `source_text` | string or null | Input text, if retained. |
| `generated_text` | string or null | Generated text, if retained. |
| `capture_method` | string | Producer description; omission loads as `unknown`. |
| `created_at` | string | Producer timestamp. |
| `transformers_version` | string or null | Transformers version, if applicable. |
| `model_revision` | string or null | Model revision, if known. |

The built-in writer emits every optional metadata field, using `null` where
appropriate. External producers may omit optional fields.

### V1 event

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `record_type` | string constant `routing_selection` | Identifies an event. |
| `phase` | `prompt` or `generated` | Prompt/prefill routing or routing for a generated token used as a decode input. |
| `token_position` | non-negative integer | Physical token position in this independent trace. |
| `layer` | non-negative integer | Routed model-layer index. |
| `selected_experts` | non-empty array of unique non-negative integers | Selected expert IDs in producer order. |

Optional fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `token_id` | non-negative integer or null | Token ID, if recorded. |
| `selected_probabilities` | array of finite numbers from 0 through 1 | Empty when unavailable or exactly as wide as `selected_experts`. |

There is no probability-sum requirement. If metadata supplies `num_experts` or
`experts_per_token`, each event must satisfy the corresponding ID range or
width. Producers must not infer expert semantics from numerical IDs or
selection frequency.

### V1 identity, atomicity, and chronology

The expert-cache identity is `(layer, expert_id)`. The same numerical expert ID
in two layers identifies two different expert objects. Input order is
authoritative:

1. all `prompt` records precede all `generated` records;
2. prompt records use ascending layer-major `(layer, token_position)` order;
3. generated records use ascending token-major `(token_position, layer)` order;
4. token positions increase strictly within each `(phase, layer)` stream;
5. `(token_position, layer)` is unique across the whole trace; and
6. every prompt token position is smaller than every generated token position.

Any violation is rejected. V1 remains supported without changing its routing,
cache, lifecycle, hardware-cost, or evidence semantics.

## Version 2: stage-qualified encoder-decoder traces

Version 2 represents encoder and decoder routing without flattening their
layer namespaces:

- `format_version`: integer constant `2`
- schema: `routing-trace-v2.schema.json`
- Python reader/writer/validator: `moe_cache_lab.trace_v2.read_trace_v2`,
  `write_trace_v2`, and `validate_trace_v2_records`

V2 has been structurally and observationally exercised only for the repository's
narrow pinned SwitchTransformers path. That does not establish general support
for all Switch checkpoints, MoE families, capacity settings, or runtimes.

### V2 metadata and routing-stage profiles

Required metadata fields are `record_type`, `format`, `format_version`,
`model_id`, and `routing_stages`. The optional text/capture/time/version/revision
fields are the same provenance fields described for v1; v2 does not use the v1
top-level `num_experts` or `experts_per_token` fields.

`routing_stages` contains one or two unique profiles in canonical order:
`encoder` before `decoder` when both are present. Each profile has:

| Field | Type | Meaning |
| --- | --- | --- |
| `routing_stage` | `encoder` or `decoder` | Separate routed namespace. |
| `num_experts` | positive integer | Expert universe for this stage. |
| `assigned_experts_per_token` | positive integer | Exact width of an assigned event in this stage. |
| `allows_unassigned` | boolean | Whether capacity-dropped/unassigned events are permitted. |

This stage-specific metadata permits different encoder and decoder expert
counts and explicitly allows an event to create zero actual expert requests.

### V2 event and assignment state

Every v2 event requires `record_type`, `routing_stage`, `phase`,
`token_position`, `layer`, `assignment_state`, `selected_experts`, and
`selected_probabilities`. `token_id` is optional and may be null.

For `assignment_state: "assigned"`:

- `selected_experts` contains unique non-negative IDs, exactly as wide as the
  stage profile's `assigned_experts_per_token`;
- every ID is smaller than that stage's `num_experts`;
- `selected_probabilities` is empty when unavailable or exactly as wide as
  `selected_experts`, with finite values from 0 through 1; and
- `unassigned_reason` must be absent.

For `assignment_state: "unassigned"`:

- both selection arrays are empty;
- `unassigned_reason` is exactly `capacity`; and
- the stage profile must set `allows_unassigned: true`.

An unassigned event faithfully records that routing reached the capacity
boundary but produced no actual expert assignment. It creates zero expert-cache
requests. V2 does not fabricate a pre-capacity expert ID or infer which expert
would otherwise have received the token. There is no probability-sum
requirement for assigned events.

### V2 identity, phases, and chronology

The expert-cache identity is `(routing_stage, layer, expert_id)`. Encoder and
decoder identities must never be flattened together or represented with
invented numerical layer offsets. Physical event identity is
`(routing_stage, token_position, layer)` and must be unique across the trace.

Stage and phase are constrained:

- encoder events use phase `source`;
- decoder events use `decoder_prompt` or `decoder_generated`;
- all encoder events precede all decoder events; and
- within the decoder, every `decoder_prompt` event precedes every
  `decoder_generated` event.

Within those blocks, input order is authoritative:

1. encoder `source` records use ascending layer-major
   `(layer, token_position)` order;
2. `decoder_prompt` records use ascending layer-major
   `(layer, token_position)` order;
3. `decoder_generated` records use ascending token-major
   `(token_position, layer)` order;
4. positions increase strictly within each
   `(routing_stage, phase, layer)` stream; and
5. every decoder-prompt position is smaller than every decoder-generated
   position.

Invalid chronology is rejected, never sorted or repaired. Assigned and
unassigned events share the same physical identity and chronology rules; only
assigned events yield expert-cache keys.

## Compatibility policy

Each version is a closed record contract. Unknown fields and record types are
rejected.

- A required-field change, field removal or rename, type/domain change, new
  record type, or identity/assignment/phase/chronology change requires a new
  `format_version` and schema.
- Even a new optional field requires a new version because existing versions
  reject unknown fields.
- Editorial clarifications may update this document without changing accepted
  records or semantics.
- Writers must choose their target version explicitly. Readers reject
  unsupported versions rather than guessing, migrating, sorting, or repairing.

The canonical cross-format policy, including independent package/trace/config
version namespaces and the rule that a new model/runtime alone does not require
a new trace version, is in [`COMPATIBILITY.md`](COMPATIBILITY.md).

## Validation command and official fixtures

Validate one canonical file without running analysis, pre-flight, cache, cost,
bundle, collector, or model code:

```text
moe-cache-lab validate-trace TRACE.jsonl
moe-cache-lab validate-trace TRACE.jsonl --json
```

Human mode writes a bounded canonical summary to stdout and exits `0` for a
valid file. Invalid input writes one bounded failure to stderr and exits `2`.
Machine mode writes exactly one newline-terminated JSON object to stdout for
both valid and invalid input, keeps stderr empty for expected validation
failures, and also exits `2` when invalid.

Machine output uses the independent family
`moe-cache-lab.trace-validation`, `format_version` 1. Its
`trace_format_version` field is separate and appears only when the declared
routing-trace version can be established without guessing. Success objects
contain `valid: true` and a bounded canonical `summary`; failure objects contain
`valid: false`, a stable `error_code`, a bounded `message`, and `line_number`
only when the authoritative failure provides a reliable line.

The version-1 error taxonomy is:

- `unsupported_format_version`, `invalid_json`, `invalid_record_shape`;
- `unknown_field`, `duplicate_event_identity`, `chronology_regression`;
- `invalid_stage_or_phase`, `invalid_assignment`,
  `invalid_expert_identity`, `invalid_trace_metadata`;
- `empty_or_missing_trace`, `io_error`.

The validator selects the existing authoritative v1/v2 reader exactly once. It
does not retry another version, sort, repair, migrate, deduplicate, or collect
additional downstream errors. The synthetic official corpus in
[`examples/trace-validation-fixtures/`](examples/trace-validation-fixtures/)
freezes valid and invalid cross-record examples that complement the schemas.

The packaged
[`examples/external-producer-no-model/`](examples/external-producer-no-model/)
workflow shows a standard-library-only script writing this external JSONL
boundary directly before invoking validation and the supported offline tools.

## External producer conformance

Passing `validate-trace` or the strict reader establishes canonical-file
validity only. It does not
by itself validate the producer's native observation boundary, chronology,
identity mapping, provenance, or non-interference. The normative requirements
and bounded Granite/Switch grounding are in
[`PRODUCER_CONFORMANCE.md`](PRODUCER_CONFORMANCE.md).

## Evidence boundary

The trace family stores routing observations. They are **MEASURED** only when
producer provenance establishes measurement. Cache outcomes derived from
assigned expert requests are **SIMULATED**, and transfer-service values are
**ESTIMATED** from declared assumptions. Schema conformance alone does not
establish measurement, representative workload coverage, expert semantics,
physical residency, native runtime caching, latency, throughput, speedup, or
broad model-family support.
