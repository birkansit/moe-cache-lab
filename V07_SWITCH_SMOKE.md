# V0.7 SwitchTransformers observational smoke

Status: **PASS for the bounded Issue #48 routing-observation gate**.

This private-development note records one CPU observational smoke. Routing is
**MEASURED** from the native post-capacity assignment. The byte-cache outcome
below is **SIMULATED**. It is not a physical-residency, transfer-volume,
performance, model-quality, or deployment result.

## Provenance and runtime

- Exact dev base: `ecc4435795664ec4f0858fa851dfcba6cab1f4be`.
- Implementation commit: the commit containing this note; its SHA is reported
  in the branch handoff rather than embedded self-referentially here.
- Model: `google/switch-base-8`.
- Immutable revision: `92fe2d22b024d9937146fe097ba3d3a7ba146e1b`.
- Runtime: Transformers `5.12.0`, Torch `2.12.0+cpu`, CPU, float32 model,
  evaluation/inference mode, batch size 1, four Torch intra-op threads.
- Decoding: deterministic greedy, no sampling or beam search, one initial
  decoder-start input and at most four fed-back non-EOS decoder inputs.
- The collector follows the official generation execution shape: native
  encoder execution with cache disabled, then the initial and cached decoder
  forwards. This avoids forwarding decoder cache settings into the encoder and
  does not modify model code or routing.

The prompt was the public CPU usage example in the pinned Hugging Face model
card (masked-language-modeling example). Its UTF-8 SHA-256 was
`30752b0dfe1d6680cc2568d56b6a924c18fd6460ad332e8afdac8de4ddc17e77`.
The prompt text is not duplicated here. Tokenization produced 21 source tokens.
The run routed one decoder-start token and four fed-back non-EOS decoder inputs;
the final non-EOS horizon candidate was not fed and therefore has no routing
event. No second prompt was needed.

## Authorized assets

Only the following seven official files at the immutable revision were
downloaded. They total `1,242,115,371` bytes.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 1,860 | `6a1e1426873221034f8488e514ac127218c833e7d357b1e12dc002889c98fa53` |
| `generation_config.json` | 147 | `f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a` |
| `pytorch_model.bin` | 1,238,895,063 | `ff91705b718f692fa0c994a49d094154583db40b536e80f20f52e05498ff6856` |
| `special_tokens_map.json` | 2,201 | `5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f` |
| `spiece.model` | 791,656 | `d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86` |
| `tokenizer.json` | 2,422,095 | `5f0ed8ab5b8cfa9812bb73752f1d80c292e52bcf5a87a144dc9ab2d251056cbb` |
| `tokenizer_config.json` | 2,349 | `4969f8d76ef05a16553bd2b07b3501673ae8d36972aea88a0f78ad31a3ff2de9` |

No alternate revision, alternate-framework weights, training data, optimizer
state, or unrelated model asset was downloaded. Loading after download was
offline and local-only. `trust_remote_code=True` and disabled TLS verification
were not used.

## Native routing evidence

The instantiated model exposed and the trace observed sparse numerical layers
`1, 3, 5, 7, 9, 11` independently in both the encoder and decoder namespaces.
The canonical expert identity remains `(routing_stage, layer, expert_id)`; no
layer offset or flattened stage namespace was used.

Actual native router output shapes were:

- encoder source callbacks: probability `(21, 1)`, post-capacity assignment
  `(21, 1, 8)`, routing weight `(21, 1)`;
- decoder callbacks: probability `(1, 1)`, post-capacity assignment
  `(1, 1, 8)`, routing weight `(1, 1)`.

The collector derives dispatch only from nonzero entries of the middle,
post-capacity assignment tensor. The selected probability comes from the third
tuple element used by the native expert path. An all-zero assignment is encoded
as a capacity-unassigned v2 event with empty selection arrays. This smoke
observed zero such drops; it does not establish that drops cannot occur.

Event counts were:

| Stage / phase | Assigned | Unassigned | Total |
| --- | ---: | ---: | ---: |
| encoder / source | 126 | 0 | 126 |
| decoder / decoder_prompt | 6 | 0 | 6 |
| decoder / decoder_generated | 24 | 0 | 24 |
| **Combined** | **156** | **0** | **156** |

The no-hook and hooked paths produced exactly the same five greedy candidate
token IDs and identical compared last-position logits; maximum absolute logit
difference was `0.0`. Hooks returned `None` and were removed after observation.

An independent expert-input pre-hook check reconciled the post-capacity mask
with native dispatch on one bounded initial forward:

- each encoder sparse layer: 21 nonzero assignments and 21 expert input rows;
- each decoder sparse layer: 1 nonzero assignment and 1 expert input row.

No layer had a mismatch.

## Trace-v2 and byte-cache boundary

Events were emitted in callback order without sorting or chronology repair.
The temporary canonical trace contained 156 routing records and round-tripped
through `write_trace_v2`, `read_trace_v2`, and the v2 validator. Its SHA-256 was
`86a2760cca8d0fee52aa49b5089dd9a11e7f4c6870021c0e6b652cab3a4cf9b8`.
The temporary raw trace was deleted and is not committed.

The model exposed 96 stage-qualified expert objects. Each expert's float32
parameter payload was deterministically `18,874,368` bytes; combined expert
parameter payload was `1,811,939,328` bytes. These are parameter payload byte
counts, not measured physical residency.

One bounded version-aware LRU replay used capacity `150,994,944` bytes (eight
equal expert payloads). The **SIMULATED** result was:

- events / expert requests: 156 / 156;
- hits / misses: 89 / 67;
- simulated demand-load bytes: `1,264,582,656`;
- evictions / simulated evicted bytes: 59 / `1,113,587,712`;
- peak and final resident bytes: `150,994,944`.

This proves only that a real stage-qualified Switch trace enters the portable
offline byte-cache boundary without encoder/decoder aliasing.

## Limitations

- One public prompt and a four-input continuation do not characterize drop
  rates, routing distributions, workload coverage, or model quality.
- Peak process RAM was not measured with a retained, reviewed process-memory
  facility; no peak value is estimated as observation.
- The cache replay is simulated and does not measure transfers, residency,
  latency, throughput, speedup, or policy optimality.
- Beam search, sampling, batched independent sequences, chunked source input,
  external/reused encoder outputs, and other model families remain outside this
  collector gate.
- No expert meaning is inferred from IDs or routing frequency.
- This PASS completes only the Issue #48 observational gate. It does not start
  v0.7-B or authorize runtime offloading, public release, tagging, or PyPI.
