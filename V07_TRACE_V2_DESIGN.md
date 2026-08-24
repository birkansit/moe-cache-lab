# V0.7 routing trace v2 historical design record

Status: **accepted historical design basis for canonical trace v2**.

This record defined the smallest routing-trace revision able to represent the
inspected SwitchTransformers candidate without changing the meaning of
canonical trace v1. The design record itself is not a schema, validator,
collector, model-support claim, or runtime result. No model was instantiated
and no routing was measured during this design work.

## Decision summary

Trace v2 keeps the format family string `moe-cache-lab.routing-jsonl` and uses
`format_version: 2`. It adds:

- a required `routing_stage` with the closed values `encoder` and `decoder`;
- stage-specific routing metadata;
- an explicit `assignment_state` that permits an observable post-capacity
  event with no actual expert assignment; and
- encoder-decoder phase and chronology rules that do not reinterpret v1's
  decoder-only `prompt` / `generated` meanings.

The canonical v2 expert-cache identity is:

`(routing_stage, layer, expert_id)`

An unassigned event is retained as one routing event but contributes zero
expert requests and causes no simulated cache-state change.

## Source-grounded Switch semantics

The design was checked against the merged `V07_SECOND_MODEL_GATE.md`, the
canonical `TRACE_FORMAT.md`, the current v1 schema/validator and byte-cache
implementation, and official Transformers `5.12.0` source:

- [`SwitchTransformersTop1Router`](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/switch_transformers/modeling_switch_transformers.py)
  computes a softmax, chooses one argmax expert, one-hot encodes it, applies an
  expert-capacity mask, and returns the post-capacity one-hot tensor. The mask
  can make every expert bit zero for a token. The source explicitly says that
  there is no guarantee every token is processed by an expert.
- `SwitchTransformersSparseMLP` dispatches using that post-capacity tensor; its
  experts are individually addressable `ModuleDict` entries. Therefore an
  all-zero tensor means no expert is actually invoked for that token. The
  pre-capacity argmax and its maximum probability remain preferences, not an
  actual dispatch after the mask drops the token.
- `SwitchTransformersStack` constructs and runs encoder and decoder stacks
  separately, iterating each stack's layers in numerical order. The model
  executes the encoder before the decoder when encoder outputs are not already
  supplied.
- Transformers generation prepares encoder outputs before decoder inputs. The
  initial decoder input is the decoder-start token, optionally followed by a
  caller-supplied decoder prefix. With cache enabled, later decoder forwards
  consume the next decoder input position.
- The immutable
  [`google/switch-base-8` config](https://huggingface.co/google/switch-base-8/blob/92fe2d22b024d9937146fe097ba3d3a7ba146e1b/config.json)
  at revision `92fe2d22b024d9937146fe097ba3d3a7ba146e1b`
  declares an encoder-decoder model, 12 layers per stack, six sparse layers per
  stack, 8 experts, capacity 64, decoder-start token ID 0, and BF16 checkpoint
  dtype. The sparse layer indices in the inspected 5.12.0 implementation are
  1, 3, 5, 7, 9, and 11 in each stack.

The source basis was Transformers `5.12.0`. Configuration inspection retrieved
only the 1,860-byte official config in memory. No weight, tokenizer, or
generation file was requested or retained.

## Contract requirements

The following rules are contract requirements for a future v2 schema, reader,
writer, and validator. They are not optional implementation suggestions.

### 1. Version coexistence and dispatch

1. The metadata record uses `format: "moe-cache-lab.routing-jsonl"` and
   `format_version: 2`.
2. A reader examines the first non-blank JSON object and dispatches on the exact
   pair `(format, format_version)`. Version 1 is validated only by the v1
   contract; version 2 is validated only by the v2 contract. An unsupported,
   missing, boolean, or otherwise invalid version is rejected.
3. Readers must not infer a version from later fields, retry another version
   after validation fails, migrate records, sort records, or repair chronology.
4. All accepted v1 records, serialized bytes, field meanings, expert identity,
   phases, chronology, writer behavior, and rejection behavior remain frozen.
   No existing v1 record is reinterpreted as v2.
5. Existing writers continue to emit v1 unless the caller explicitly selects a
   future v2 writer. A producer may create a separate v2 trace from source
   observations, but automatic in-place v1-to-v2 migration is forbidden.
6. V2 remains a closed record contract. Unknown fields, record types, stage
   values, phase values, assignment states, and unassignment reasons are
   rejected rather than ignored.

### 2. Metadata

The first non-blank line remains one `metadata` record. The following v1
provenance fields retain their meanings and types in v2:

- required: `record_type`, `format`, `format_version`, `model_id`;
- optional: `source_text`, `generated_text`, `capture_method`, `created_at`,
  `transformers_version`, and `model_revision`.

V2 removes the v1 scalar `num_experts` and `experts_per_token` fields. Retaining
them would be misleading when stacks have independent expert namespaces and a
capacity drop can produce zero actual assignments. They are replaced by the
required `routing_stages` array. Each stage profile is a closed object with:

| Field | Type | Meaning |
| --- | --- | --- |
| `routing_stage` | `encoder` or `decoder` | Namespace described by this profile. |
| `num_experts` | positive integer | Valid expert-ID range is `0 <= id < num_experts` in this stage. |
| `assigned_experts_per_token` | positive integer | Exact number of actual post-routing assignments when an event is assigned. |
| `allows_unassigned` | boolean | Whether zero actual assignments are valid for this stage. |

Profiles have unique `routing_stage` values. When both are present their
canonical array order is `encoder`, then `decoder`. At least one profile is
required, and every event's stage must have a profile. A profile describes
configured routing semantics and need not imply that a partial trace observed
an event in that stage; evidence-coverage reporting is a later milestone.

For `google/switch-base-8`, both profiles are required and each is
`num_experts: 8`, `assigned_experts_per_token: 1`, and
`allows_unassigned: true`.

No routed-layer list, expert byte size, capacity in bytes, lifecycle setting,
hardware assumption, semantic expert label, or support claim is added to trace
metadata. Those belong in separate validated configuration/evidence layers.

### 3. Routing event and assignment state

V2 retains `record_type: "routing_selection"` so the file still contains one
record for each observed token/layer router event, including an event that
selects no expert after capacity enforcement. Each event requires:

- `record_type`;
- `routing_stage`;
- `phase`;
- `token_position`;
- `layer`;
- `assignment_state`;
- `selected_experts`; and
- `selected_probabilities`.

`token_id` remains optional and may be a non-negative integer or `null`.
Positions and layers are non-negative integers. Expert IDs are unique,
non-negative integers in producer order and within the expert range of the
event's stage profile. Probabilities are finite numbers in `[0, 1]`; no sum-to-
one rule is imposed.

The assignment state machine is exact:

| `assignment_state` | `selected_experts` | `selected_probabilities` | `unassigned_reason` |
| --- | --- | --- | --- |
| `assigned` | exactly `assigned_experts_per_token` actual post-routing expert IDs | either `[]` when unavailable or exactly one value per selected expert | prohibited |
| `unassigned` | exactly `[]` | exactly `[]` | required and exactly `capacity` |

An `unassigned` event is allowed only when its stage profile has
`allows_unassigned: true`. The initial v2 reason domain is intentionally only
`capacity`; another reason requires a later contract version. V2 does not model
partially assigned top-k routing: a future family that can retain only part of
a multi-expert assignment requires a separate contract decision rather than a
guess.

For Switch, `selected_experts` comes from the nonzero index of the returned
post-capacity one-hot tensor. If that tensor is all zero, the event is
`unassigned`. The pre-capacity argmax must not be copied into
`selected_experts`, and its maximum probability must not be placed in
`selected_probabilities` for the unassigned event because no expert consumed
that weight. The event must not be omitted.

One record remains one atomic router event. An assigned top-k event is not
split into individual records. An unassigned record is an atomic event with an
empty required expert set.

### 4. Routing stage, phase, position, and chronology

`routing_stage` is required on every v2 event and has only `encoder` or
`decoder`. Numerical layer IDs are local to their stack; they are not flattened
or offset. Encoder layer 1 and decoder layer 1 are different namespaces.

The v2 `phase` domain is deliberately different from v1:

| Stage | Phase | Meaning |
| --- | --- | --- |
| `encoder` | `source` | A source/input token processed by an encoder router. |
| `decoder` | `decoder_prompt` | Decoder-start token and any caller-supplied decoder-prefix tokens processed in the initial decoder forward. |
| `decoder` | `decoder_generated` | A model-emitted non-terminal token when it is fed back as a later decoder input. |

All other stage/phase combinations are invalid. In particular, encoder events
cannot be `decoder_prompt` or `decoder_generated`, and decoder events cannot be
`source`.

`token_position` is a physical, zero-based position in its routing stage's
token sequence. Encoder and decoder positions are separate namespaces, so an
encoder source token and a decoder token may both have position 0. Decoder
positions form one continuous position space across `decoder_prompt` and
`decoder_generated`; every decoder-prompt position must be smaller than every
decoder-generated position.

Routing events describe model inputs, not candidates that were merely
produced. An EOS candidate that is not fed and a final horizon candidate that
is not fed have no routing event. If a non-EOS candidate is fed in a subsequent
decoder forward, its event phase is `decoder_generated` at that decoder input
position.

Input order is authoritative. A future validator must enforce all of these
rules without sorting or repair:

1. if both stages have events, every `encoder/source` event precedes every
   decoder event;
2. all `decoder_prompt` events precede all `decoder_generated` events;
3. encoder/source events use ascending layer-major `(layer, token_position)`
   order;
4. decoder-prompt events use ascending layer-major `(layer, token_position)`
   order;
5. decoder-generated events use ascending token-major
   `(token_position, layer)` order;
6. positions increase strictly within every
   `(routing_stage, phase, layer)` stream;
7. `(routing_stage, token_position, layer)` is unique across the trace, so the
   same decoder position cannot appear once as prompt and again as generated;
8. stage and phase blocks cannot regress after a later block begins; and
9. an independent trace contains at least one routing event.

These orders follow the inspected 5.12.0 single-sequence execution path: each
full-sequence stack forward visits layers in order, generation computes the
encoder before decoder prefill, and cached decode forwards visit all decoder
layers for one new input position at a time. V2 does not claim a canonical
interleaving for beam search, batched independent sequences, speculative
decoding, chunked prefill, or supplied/reused encoder outputs. A future Switch
collector gate must use one independent sequence, ordinary unchunked greedy
generation, and the above call structure, or propose a new contract version.

### 5. Canonical expert identity and downstream meaning

The v2 expert-cache key is exactly
`(routing_stage, layer, expert_id)`. Thus `(encoder, 1, 3)` and
`(decoder, 1, 3)` cannot alias. Numerical IDs remain opaque; stage identity
does not attach expert semantics.

For each v2 event, expert-request extraction is:

`{(routing_stage, layer, expert_id) for expert_id in selected_experts}`

The set is replayed atomically. An unassigned event yields the empty set, adds
one to routing-event evidence counts, adds zero to expert-request counts, and
produces zero hits, misses, loads, evictions, or cache-state changes. Reports
must keep at least total event count, assigned event count, unassigned event
count, and expert request count distinguishable; a zero-request event is not a
hit or miss.

After a future version-aware normalization layer, these established concepts
can remain unchanged:

- atomic pinning of all actual requests in one event;
- deterministic LRU/LFU and fixed-policy state transitions for non-empty
  required sets;
- cold versus persistent lifecycle boundaries;
- byte-capacity feasibility based on actual requested expert objects; and
- **MEASURED** routing, **SIMULATED** cache outcomes, and **ESTIMATED** transfer
  service terminology.

At design time, implementation work remained necessary at the interfaces:
version dispatch, a v2 event/metadata representation, validation, three-part
expert keys, stage-qualified expert-size configuration, request extraction,
empty-set event accounting, lifecycle/report fields, and any view/manifest code
that assumed v1 `RoutingEvent` or `(layer, expert_id)`. The requirement was that
existing v1 Python objects, schema, validators, caches, fixtures, and results
continue to behave exactly as before.

## Synthetic JSONL-shaped examples

These examples document the proposed contract. They are not production schema
fixtures.

### Valid: encoder and decoder assignments plus one capacity drop

The encoder and decoder assignments deliberately use the same numerical layer
and expert IDs. Their cache keys are distinct because `routing_stage` differs.

```jsonl
{"record_type":"metadata","format":"moe-cache-lab.routing-jsonl","format_version":2,"model_id":"google/switch-base-8","routing_stages":[{"routing_stage":"encoder","num_experts":8,"assigned_experts_per_token":1,"allows_unassigned":true},{"routing_stage":"decoder","num_experts":8,"assigned_experts_per_token":1,"allows_unassigned":true}],"capture_method":"post-capacity router observation","transformers_version":"5.12.0","model_revision":"92fe2d22b024d9937146fe097ba3d3a7ba146e1b"}
{"record_type":"routing_selection","routing_stage":"encoder","phase":"source","token_position":0,"layer":1,"assignment_state":"assigned","selected_experts":[3],"selected_probabilities":[0.75],"token_id":10}
{"record_type":"routing_selection","routing_stage":"decoder","phase":"decoder_prompt","token_position":0,"layer":1,"assignment_state":"assigned","selected_experts":[3],"selected_probabilities":[0.60],"token_id":0}
{"record_type":"routing_selection","routing_stage":"decoder","phase":"decoder_generated","token_position":1,"layer":1,"assignment_state":"unassigned","selected_experts":[],"selected_probabilities":[],"unassigned_reason":"capacity","token_id":42}
```

The first event requests `(encoder, 1, 3)`, the second requests
`(decoder, 1, 3)`, and the third requests nothing while remaining visible as a
routing event.

### Invalid: missing stage identity

```json
{"record_type":"routing_selection","phase":"source","token_position":0,"layer":1,"assignment_state":"assigned","selected_experts":[3],"selected_probabilities":[0.75]}
```

The event is ambiguous and must be rejected; a reader may not infer `encoder`
from `phase`.

### Invalid: unassigned state with a fabricated request

```json
{"record_type":"routing_selection","routing_stage":"encoder","phase":"source","token_position":0,"layer":1,"assignment_state":"unassigned","selected_experts":[3],"selected_probabilities":[0.75],"unassigned_reason":"capacity"}
```

`unassigned` requires both arrays to be empty. Recording the pre-capacity
preference as actual dispatch would fabricate a cache request.

The inverse is also invalid: `assignment_state: "assigned"` with an empty
`selected_experts` array.

### Invalid: cross-stage chronology regression

```jsonl
{"record_type":"routing_selection","routing_stage":"decoder","phase":"decoder_prompt","token_position":0,"layer":1,"assignment_state":"assigned","selected_experts":[3],"selected_probabilities":[0.60]}
{"record_type":"routing_selection","routing_stage":"encoder","phase":"source","token_position":0,"layer":1,"assignment_state":"assigned","selected_experts":[3],"selected_probabilities":[0.75]}
```

When both stages are present, encoder events must precede decoder events. The
validator rejects this input and must not sort it.

Other mandatory rejections include a duplicate
`(routing_stage, token_position, layer)`, decoder-generated positions followed
by decoder-prompt positions, source records in token-major rather than
layer-major order, generated decoder records in layer-major rather than
token-major order, and an expert ID outside its stage profile.

## Future implementation choices, not contract latitude

The contract above fixes accepted record meanings. A later implementation gate
may choose only engineering details that do not change them, including:

- Python class names and whether v1/v2 share an internal normalized event type;
- schema-resource organization and version-dispatch table structure;
- whether post-capacity observation uses temporary hooks or validated public
  model outputs;
- how stage-qualified expert sizes are represented in separate preflight
  configuration files;
- report layout, provided assigned/unassigned event and request denominators
  remain explicit; and
- explicit conversion tooling, provided it writes a new v2 file from sufficient
  source evidence and never silently reinterprets v1 bytes.

Implementation convenience cannot relax stage identity, assignment-state
consistency, authoritative input order, empty-request handling, or v1
compatibility.

## Questions reserved for implementation or a real Switch smoke test

The following are genuine unknowns and are not resolved by config/source
inspection:

1. Which non-mutating observation surface most reliably captures both the
   returned post-capacity one-hot tensor and its associated maximum probability
   under the actual 5.12.0 model run. Public output recording alone must not be
   assumed sufficient until compared with the router tuple.
2. Whether temporary hooks preserve token/expert decisions, probabilities, and
   final model outputs bit-exactly, and whether every encoder/decoder sparse
   layer is captured exactly once per expected forward.
3. The observed hook callback order and token flattening for the bounded
   batch-size-one generation protocol. The contract order must be tested rather
   than reconstructed by sorting.
4. The actual incidence of capacity drops, including routing of padding tokens
   under the pinned config, and whether a compact synthetic input can exercise
   a real unassigned event.
5. Exact EOS, decoder-start, decoder-prefix, and last-unfed-candidate behavior
   for the future collector's bounded protocol.
6. Full-checkpoint load peak, runtime memory, and completion time on the local
   CPU machine. The prior size estimate is not a measured feasibility result.
7. Batched, beam, chunked-prefill, speculative, or reused-encoder-output traces.
   They are outside this narrow v2 contract until an observed execution model
   justifies additional sequence/call identity.

No unknown permits fabrication, omission, stage flattening, or chronology
repair. A smoke-test mismatch rejects the proposed collector path and triggers
a separate contract review.

## Subsequent implementation boundary

The design separated trace-v2 schema/reader/writer/validator implementation,
with synthetic valid and adversarial tests plus unchanged v1 compatibility and
deterministic no-download outputs, from any model execution. That implementation
boundary did not require loading Switch weights or implementing a collector.

A later, separate resource-admitted Switch collector and real smoke test could
be considered only after the canonical contract was implemented and validated.

## Evidence boundary

This note records source/config inspection and a proposed contract. It creates
no **MEASURED** Switch routing evidence, no **SIMULATED** Switch cache result,
and no **ESTIMATED** Switch transfer result. It does not establish multi-model
support, model correctness, expert semantics, physical residency, latency,
throughput, or speedup.
