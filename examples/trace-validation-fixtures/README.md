# Canonical trace validation fixtures

These tiny, deterministic JSONL files exercise the strict canonical routing
trace readers without a model or network access. All records are synthetic.

Valid fixtures:

| File | Expected result | Contract surface |
| --- | --- | --- |
| `valid/v1-minimal.jsonl` | VALID | Minimal layer-qualified trace v1. |
| `valid/v2-decoder-only.jsonl` | VALID | Decoder-only stage-qualified trace v2. |
| `valid/v2-encoder-decoder.jsonl` | VALID | Encoder and decoder both use numerical `(layer=3, expert=5)` without aliasing. |
| `valid/v2-unassigned.jsonl` | VALID | An explicit capacity-unassigned event has no selected expert or expert request. |

Invalid fixtures:

| File | Expected error code | Contract violation |
| --- | --- | --- |
| `invalid/chronology-regression.jsonl` | `chronology_regression` | A decoder-prompt event follows decoder generation. |
| `invalid/duplicate-event-identity.jsonl` | `duplicate_event_identity` | Two events reuse one canonical physical identity. |
| `invalid/stage-regression.jsonl` | `chronology_regression` | An encoder event follows a decoder event; stage cannot be reconstructed or flattened. |
| `invalid/fabricated-unassigned-selection.jsonl` | `invalid_assignment` | An unassigned event fabricates an expert selection and probability. |
| `invalid/unsupported-version.jsonl` | `unsupported_format_version` | The declared routing-trace version is unsupported. |
| `invalid/unknown-field.jsonl` | `unknown_field` | A record violates the closed contract with an unknown field. |

Passing these fixtures establishes canonical-file validity only. It does not
establish that a producer observed the correct native routing boundary or that
instrumentation was non-interfering. The fixtures do not measure or establish
runtime performance, physical residency, transfers, latency, throughput, or
speedup.
