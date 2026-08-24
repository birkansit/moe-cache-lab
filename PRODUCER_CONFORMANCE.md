# External routing-trace producer conformance

This document defines the evidence required to describe an external
routing-trace producer path as conforming or validated. The canonical JSONL
contract remains the primary external interoperability boundary. A producer may
write that contract directly; it does not need to import a built-in collector
or participate in a project-specific adapter framework.

The model or runtime's native routing and actual dispatch behavior is
authoritative. The canonical schema is a representation target, not permission
to redefine that behavior.

## Three separate claims

These claims are independent and must be reported separately:

| Claim | What it establishes | What it does not establish |
| --- | --- | --- |
| **Canonical trace valid** (or **schema-and-semantic reader valid**) | The JSONL records satisfy the selected closed v1 or v2 contract, including record shape, identity, assignment, phase, range, uniqueness, and chronology rules. | That the producer observed the correct native boundary, preserved real chronology and identity, or was non-interfering. |
| **Producer path semantically validated** | Bounded evidence justifies the mapping from a named native observation boundary to the declared canonical events, including chronology, identity, assignment meaning, and provenance. | That instrumentation left native routing unchanged unless the separate control comparison below also passed. |
| **Producer path non-interference validated** | A frozen producer-specific protocol compared an appropriate native/control path with the instrumented path and found the preregistered routing and output properties in agreement. | General model/runtime compatibility, representative workload coverage, or runtime-performance behavior outside the validation scope. |

A producer path is called **conforming** only for a stated validation scope in
which its output is canonical-trace valid, its observation mapping is
semantically validated, and its instrumentation is non-interference validated.
If the last claim is not established, a produced file may still be canonical
trace valid. The path may be described specifically as **semantically
validated** only if its observation mapping passed that separate gate, but it
must not be called generically validated, non-interfering, or conforming.

## Native authority and observation boundary

A producer observes native behavior. It must not replace a native routing
decision with a predictor or heuristic, invent expert IDs, reconstruct missing
dispatch from cache state or downstream analysis, or alter router logits,
probabilities, selections, or dispatch for instrumentation convenience.
Post-processed or invented expert IDs must not be described as native dispatch.

Where a runtime distinguishes them, **pre-capacity routing preference or
selection** and **post-capacity actual dispatch or assignment** are different
semantic boundaries. A producer must name the boundary it observes and must not
claim the other boundary. If only pre-capacity preference is available, it must
not be emitted as actual post-capacity assignment. If an existing trace version
cannot represent the observed native boundary without conflation or guessing,
the producer path is a bounded **NO-GO**.

## Chronology is authoritative

Canonical event order must preserve the authoritative observed chronology. A
producer must not sort events to make validation pass, reconstruct unknown order
from token or layer numbers, silently repair chronology regressions, or
deduplicate conflicting observations. Buffering is permitted only when it
preserves an already-known authoritative order.

If a concurrent or fused runtime does not expose enough ordering information to
satisfy the selected canonical contract, the producer path is NO-GO. The strict
reader's acceptance of a sorted file would not recover or prove the missing
native chronology.

## Identity must be native and lossless

Canonical v1 expert identity is exactly `(layer, expert_id)`. Canonical v2
expert identity is exactly `(routing_stage, layer, expert_id)`, with encoder
and decoder remaining separate namespaces.

A producer must not flatten encoder and decoder, encode stage with numerical
layer offsets, infer stage from a model name, layer range, filename, neighboring
event, or other indirect cue, or fabricate a missing layer, stage, or expert
identity. Expert IDs may be transformed only when the runtime itself defines an
authoritative canonical mapping and the producer documents a semantically
lossless transformation. Convenient local renumbering is not sufficient.

## Assigned and unassigned events

An explicit canonical v2 `unassigned` event records an observed capacity drop
and has **zero assigned expert requests**. Its selection and probability arrays
remain empty. A producer must not fabricate a preferred or likely expert for
that event.

Missing data alone is not evidence of a capacity drop. A producer must not infer
`unassigned` merely because assignment data is unavailable, and it must not
turn a pre-capacity preference into a claimed assigned or unassigned
post-capacity result.

## Bounded provenance

Canonical fields are used only for their documented meanings. Supported facts
should be retained when established, including:

- model identifier and immutable revision when available;
- the capture method and exact observation boundary;
- relevant producer, runtime, and library versions;
- the frozen input/workload and execution boundary used for validation; and
- explicit limitations and unavailable provenance.

The canonical trace has no generic field for every runtime fact. Additional
runtime provenance may be kept in adjacent documentation or a reproducibility
bundle rather than placed in unrelated trace fields. Provenance must exclude
credentials, environment secrets, private input content not deliberately
retained, machine-specific absolute paths, and private coordination metadata.

## Non-interference validation protocol

Non-interference evidence is producer-specific and bounded. Before running a
comparison, the protocol must freeze the model and immutable revision where
available, runtime/library versions, input and generation boundary, execution
mode, and the observation enabled/disabled paths. It must preregister:

- the native/control result and instrumented result being compared;
- exact routing/assignment identities that must agree directly;
- output semantics that must agree;
- probability or logit comparisons, reported separately when available; and
- any producer-specific numeric tolerance and why it is necessary.

Expert or assignment identity agreement must be reported directly, not hidden
inside an aggregate quality score. This contract defines no universal score and
no universal numeric tolerance across runtimes. A mismatch is recorded as a
failed bounded validation; it is not averaged away. Evidence for one frozen
path does not transfer automatically to another checkpoint, runtime, device,
precision, decoding strategy, workload, or instrumentation boundary.

## Explicit NO-GO cases

NO-GO is the correct result when faithful production cannot be established. It
is not a failed scientific result and must not be hidden behind an approximate
adapter. At minimum, record NO-GO when:

- only aggregate expert counts are available, without authoritative per-event
  chronology;
- required stage, layer, or expert identity is unavailable;
- chronology would have to be guessed, sorted, repaired, or deduplicated;
- only pre-capacity preference is visible while the claimed semantics require
  post-capacity actual assignment;
- a capacity-drop or unassigned reason would have to be inferred;
- instrumentation changes routing, dispatch, or a preregistered output
  property; or
- provenance cannot be bounded sufficiently for the claimed validation.

## External boundary and version policy

Canonical `moe-cache-lab.routing-jsonl` v1/v2 remains the external
interoperability boundary. This contract does not define plugin discovery,
entry points, a registry, a callback framework, a marketplace, or a public
producer SDK. External producers are not required to depend on internal Python
collector classes.

A new model, checkpoint, producer, or runtime alone does **not** justify routing
trace v3. Use v1 or v2 only when the existing closed contract faithfully
represents the observed native events. Genuinely new event semantics require a
separate schema/version design decision; a producer must not infer, migrate, or
auto-upgrade a version.

### Internal Python result boundary

`moe_cache_lab.producer.ProducerResult` is core-only internal plumbing that
retains an already-built canonical `RoutingTrace` or `RoutingTraceV2` unchanged.
It defines no common producer invocation API and establishes none of the three
validation claims above. External producers may continue to emit canonical
JSONL directly and are not required to import this internal abstraction.

The core-only `moe-cache-lab validate-trace` command and official synthetic
fixture corpus test that external boundary. Reader acceptance establishes only
the **canonical trace valid** claim; it does not upgrade producer semantic or
non-interference evidence.

The packaged
[`examples/external-producer-no-model/`](examples/external-producer-no-model/)
workflow demonstrates direct JSONL emission from an unrelated standard-library
script. It is synthetic contract-use evidence, not a semantically validated or
non-interference-validated runtime producer.

## Bounded grounding in existing producer paths

The following is a non-ranking account of tracked evidence. Each row is limited
to its named path and validation scope.

| Producer path | Observation boundary and identity | Bounded validation established | Not established |
| --- | --- | --- | --- |
| Granite/Transformers v1 | Forward hooks observe the native Granite router module's returned logits. The tracked extraction applies the router's selected-top-k normalization and preserves `(layer, expert_id)` in the collector's prompt/generated execution order. | The official router component test compares hooked and unhooked outputs and reconciles extracted expert IDs/probabilities with native gate outputs. The frozen CPU-FP32 V0.4 protocol compared the same pinned full-model driver with and without the exact observer; tracked results report unchanged output semantics, identical selected expert IDs, and maximum selected-probability delta `0.0` for that frozen workload and stack. | A distinct post-capacity capacity-drop boundary; v2 stage identity; other Granite checkpoints, runtimes, devices, precisions, or workloads; broad compatibility; native cache/residency or performance effects. |
| SwitchTransformers v2 | Forward hooks observe the native router's returned post-capacity assignment tensor and routing weight. Registered module identity preserves `(routing_stage, layer, expert_id)` and callback order; an all-zero post-capacity assignment becomes an empty `unassigned` event without a fabricated preference. | The pinned CPU-FP32 smoke compared no-hook and hooked generation on one frozen public input: five candidate token IDs and compared logits were exact, with maximum absolute difference `0.0`. One bounded initial forward also reconciled post-capacity nonzero assignments with expert-input rows. Tracked contract tests preserve callback order, stage identity, assigned expert extraction, and zero-request unassigned behavior. | Pre-capacity preference; an observed drop in the retained smoke (none occurred); other prompts, checkpoints, model families, runtimes, devices, precisions, batching, beam search, sampling, or broader generation shapes; native cache/residency or performance effects. |

Neither row ranks the models or producers. Evidence absent from a row is **not
established** and must not be inferred from canonical-file validity.

## Scientific claim boundary

A valid or producer-validated trace does not establish representative workload
behavior, physical CPU/GPU residency, actual transfer traffic, runtime caching
or offloading, latency, throughput, tokens per second, speedup, memory savings,
an optimal cache policy/capacity, broad model/runtime compatibility, or
production suitability. Cache results derived later remain **SIMULATED** and
transfer-service results remain **ESTIMATED** under their declared contracts.
