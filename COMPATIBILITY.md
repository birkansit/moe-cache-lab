# Compatibility and version policy

This document is the canonical cross-format compatibility policy for
`moe-cache-lab`. It defines how package releases, routing traces, pre-flight
configuration, and other serialized artifacts evolve without treating their
version numbers as one shared sequence.

## Independent version namespaces

The project has several independent version namespaces:

1. **Package release version** — for example `0.7.0`, later `0.8.0`.
2. **Routing-trace format family** — `moe-cache-lab.routing-jsonl`, currently
   `format_version` 1 and 2.
3. **Pre-flight configuration format family** —
   `moe-cache-lab.preflight-config`, currently `format_version` 1, 2, and 3.
4. **Other serialized artifacts** — reports, lifecycle outputs, experiment
   bundles, manifests, and validation artifacts keep their own explicit format
   identifiers and version contracts.

Numbers in different namespaces have no implied relationship. Routing trace v2
does not imply pre-flight-config v2. Pre-flight-config v3 does not, by itself,
imply routing trace v3. Package version changes likewise do not silently
change an artifact format.

Compatibility between two artifact families must be stated by the consuming
orchestration. It is never inferred because two format numbers happen to be
equal.

## Routing-trace contract

The routing-trace family is a strict interoperability boundary. Current readers
dispatch from the first metadata record using the exact pair
`(format, format_version)`:

- `moe-cache-lab.routing-jsonl`, version 1 -> canonical trace v1 validator;
- `moe-cache-lab.routing-jsonl`, version 2 -> canonical trace v2 validator.

Missing, boolean, non-integer, malformed, or unsupported versions are rejected.
The reader does not infer a version from model identity, collector type, runtime,
later fields, stage names, or event shape. It does not retry another validator
after the selected validator rejects the input.

Trace v1 and trace v2 are closed contracts. Their readers reject unknown fields
and unknown record types. Input chronology is authoritative: readers do not
sort, repair, normalize, or reconstruct invalid event order. Existing v1 input
is never reinterpreted as v2, and v2 input is never down-converted to v1 by
inference. Encoder and decoder namespaces are never flattened or represented by
invented layer offsets.

### When a routing-trace version changes

A new model family, checkpoint, collector, or inference runtime is **not** by
itself a reason to create routing trace v3. If that source can faithfully emit
an existing canonical contract, it should use that existing version.

A new routing-trace version is required when canonical information that must be
represented cannot be expressed faithfully by the current closed contract. A
breaking change includes, for example:

- changing required fields or their types/domains;
- removing or renaming fields;
- adding a field or enum value to a closed accepted record language;
- adding a new record type;
- changing expert or event identity;
- changing assignment meaning, stage/phase meaning, or chronology semantics;
- changing whether an observed event is represented or omitted.

The following do not require a format-version change when they preserve the
already documented accepted language and semantics:

- editorial clarification;
- additional examples;
- implementation refactoring with identical accept/reject behavior;
- a correctness fix that restores the implementation to the already specified
  contract without broadening or redefining the accepted record language.

If a proposed fix would intentionally broaden, narrow, or reinterpret the
canonical record language, it is a contract change and must be evaluated as a
format-version change rather than hidden under the existing number.

## Pre-flight-config contract

`moe-cache-lab.preflight-config` is a separate closed format family. Its current
reader accepts only its own versions 1, 2, and 3 and validates the exact field
set for the selected version. Unsupported, boolean, non-integer, or malformed
versions are rejected.

Config v1/v2 expert-size identity is layer-qualified:
`(layer_id, expert_id)`. Config v3 expert-size identity is exactly
`(routing_stage, layer_id, expert_id)`, where the stage is `encoder` or
`decoder`. V3 does not infer a stage, flatten namespaces, add numerical layer
offsets, or silently migrate legacy records. This config-language change does
not create a routing-trace v3.

Current public orchestration supports these combinations:

| Routing trace | Pre-flight config | Current behavior |
| --- | --- | --- |
| v1 | none | supported descriptive v1 analysis |
| v2 | none | supported stage-qualified descriptive v2 analysis |
| v1 | config v1 | supported v1 pre-flight workflow |
| v1 | config v2 | supported v1 pre-flight workflow |
| v1 | config v3 | rejected: trace v1 has no canonical routing stage |
| v2 | config v1 | rejected: config identity is not stage-qualified |
| v2 | config v2 | rejected: config identity is not stage-qualified |
| v2 | config v3 | supported stage-qualified single-trace pre-flight workflow |

The last row is an intentional example of why equal version numbers do not
establish compatibility.

The v2 + config-v3 workflow remains single-trace only. The existing lifecycle
command is a v1 compatibility workflow and rejects config v3; no stage or
workload identity is inferred to widen it.

## External producer boundary

An external producer may target the canonical routing-trace contract without
depending on the built-in Granite or Switch implementation. A conforming source
may rely on:

- the documented trace format/version pair;
- the packaged schemas and strict validation semantics;
- authoritative event chronology and identity rules;
- documented provenance fields;
- the accepted/rejected record language for the declared format version.

A producer is not required to depend on collector Python classes, hook layout,
Transformers internals, or another model-specific adapter. Model-specific
instrumentation remains responsible for observing native routing/dispatch
without silently changing it. Canonical-file validity, producer-path semantic
validation, and producer-path non-interference validation are separate claims;
their normative requirements are defined in
[`PRODUCER_CONFORMANCE.md`](PRODUCER_CONFORMANCE.md).

Human-readable exception wording may improve over time unless a separate
machine-readable error contract explicitly freezes it. The canonical guarantee
here is the declared format semantics and accept/reject boundary, not incidental
Python exception text.

`moe-cache-lab validate-trace` exposes that accept/reject boundary without
analysis. Its machine output is the independent
`moe-cache-lab.trace-validation` family, currently version 1. The output
`format_version` does not equal or imply the input `trace_format_version`, and
its stable error codes do not change the authoritative v1/v2 dispatch rules.

## Other artifact families

Reports, lifecycle outputs, bundles, and manifests use their own format strings
or version fields where defined. Their evolution is evaluated within those
families. A change to one artifact family does not silently migrate or renumber
another family.

## No silent migration or repair

Across the versioned boundaries covered here, unsupported input fails
explicitly. The project does not use model names, matching numeric versions,
collector identity, or best-effort field inspection to reinterpret an artifact
as another contract. Invalid chronology, missing identity, unknown fields, and
unsupported versions are not repaired into accepted input.

For routing-specific field and chronology details, see
[`TRACE_FORMAT.md`](TRACE_FORMAT.md). For public pre-flight orchestration and
config versions, see [`PREFLIGHT.md`](PREFLIGHT.md).
