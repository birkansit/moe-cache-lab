# v0.9 — Receipted OLMoE external intake

## Scope and evidence boundary

This record covers one **NEW, separately receipted OLMoE capture**. It does not recover, repair, or backfill provenance for the historical `sqliteai/warp tests/trace_olmoe_299.jsonl @ dc079dad1c35def2ee7bb3b2de5157db8d9f08a8` artifact documented in `V09_EXTERNAL_EVIDENCE_INTAKE.md`.

The governing producer-validation contract is `PRODUCER_CONFORMANCE.md`. Its three claims remain separate:

1. **Canonical trace valid**;
2. **Producer path semantically validated**;
3. **Producer path non-interference validated**.

No external raw/canonical bytes are tracked in this repository. The external bundle remains referenced by immutable repository/commit/path and content SHA-256.

## Claim-by-claim result

| Claim or evidence check | Result |
| --- | --- |
| Artifact SHA verification | `AGREEMENT` |
| Mechanical canonical reproduction | `AGREEMENT` |
| Historical-converter regression | `PASS` |
| Raw-to-canonical preservation audit | `PASS` |
| Canonical validity | `PASS` |
| Producer semantic mapping | **`BOUNDED PASS — selected routed-expert membership on the reviewed pinned path`** |
| Native selection rank/order | `NOT PRESERVED` |
| Producer non-interference | `NOT ESTABLISHED` |
| Artifact intake decision | `ACCEPTED AS BOUNDED RECEIPTED EXTERNAL INTAKE EVIDENCE` |

The bounded semantic-mapping result is **not** generic producer validation or conformance. `PRODUCER_CONFORMANCE.md` requires the separate non-interference gate before a producer path may be called conforming. No such control-vs-instrumented result exists for this capture.

## Immutable external bundle

External contributor: `mfethe1`.

Repository / commit:

`mfethe1/warp @ 793bf319dee34cc0ed9863d57ebfd5c9d1a33811`

Directory:

`tools/moe-cache-lab/olmoe-receipted-capture-20260824/`

Independently fetched and SHA-256-verified files:

| File | Independent SHA-256 | Result |
| --- | --- | --- |
| `raw_trace.jsonl` | `323cdbb3a8a52a9f2240a914ba7c28690058e6e292ba502f3ddc999b5d5067e5` | `AGREEMENT` |
| `canonical_v1.jsonl` | `d02c69e4185dbba70305fa9f3e0dd75e23d9bc27410d3ba2aa4703240b0b0e4a` | `AGREEMENT` |
| `receipt.json` | `0da662cc23d0f4d18ff3ee80824f40a539c200f53e002257c08fc8da7a4ac1d0` | `AGREEMENT` |
| `recapture_olmoe.py` | `1495fac92ee9ff34e7ed22657921835008292161a900605538dad901dd3d3a65` | `AGREEMENT` |
| `convert_kimi_olmoe_canonical.py` | `40aca8ef5176d900353e0cc68ea8807a95db91e445cca1f20fa265b3d127b015` | `AGREEMENT` |

The independently computed values also agree with the bundle's `SHA256SUMS.txt`. Git blob identifiers were not substituted for content SHA-256.

## Superseded new-capture canonical bytes

The contributor previously reported a canonical SHA-256 of:

`28725ebb5ebcd97bb8f8d8a165e342a1a3cc5777d28befb4337b8bb45f984b97`

The contributor later withdrew/superseded that byte target because it was emitted by an inline serializer rather than the frozen converter. The contributor reports that the old and new parsed records differ only in JSON object key ordering and are semantically equal.

The exact superseded `28725ebb...` bytes are not present in the accepted immutable bundle, so this intake does **not** independently re-compare those old bytes record-for-record. The project independently establishes instead that:

- the current anchored target `d02c69e...` is reproducible from the frozen anchored raw bytes and converter;
- the canonical v1 reader/validator is independent of JSON object key order;
- exact byte/hash reproducibility is a separate requirement from semantic reader validity.

No trace-format rule requiring sorted JSON keys is introduced by this intake.

## Capture identity and recorded provenance

Model:

`allenai/OLMoE-1B-7B-0125`

Pinned/requested and resolved revision:

`9b0c1aa87e34a20052389dce1f0cf01da783f654`

The anchored receipt records the same full revision as both requested and resolved. The exact capture script passes `revision=REVISION` to both `AutoTokenizer.from_pretrained(...)` and `AutoModelForCausalLM.from_pretrained(...)`; the pinned revision is therefore part of the actual load request, not merely a later lookup.

Recorded execution provenance from the external receipt:

- OS: macOS 15.6 arm64;
- Python: 3.12.12;
- Torch: 2.13.0;
- Transformers: 5.15.1;
- device: MPS;
- dtype: `torch.bfloat16`;
- model config: 16 hidden layers, 64 experts, top-8;
- prompt tokens: 35;
- requested generation steps: 300;
- emitted raw decode rows: 299;
- sampling: multinomial, temperature 0.7;
- `torch.manual_seed(0)`;
- full re-forward each generation step.

These environment/revision fields are **recorded execution provenance**. They are not grouped under a generic `MEASURED` label.

The external receipt contains a machine-specific absolute executable path in its original command field. That path remains part of the immutable external receipt but is deliberately not copied into this project record. The evidence-relevant invocation is normalized to: capture script + output raw path + exact model ID + 300 requested steps + full pinned revision.

## Observation boundary and capture-script audit

Exact capture script:

`mfethe1/warp tools/moe-cache-lab/olmoe-receipted-capture-20260824/recapture_olmoe.py @ 793bf319dee34cc0ed9863d57ebfd5c9d1a33811`

The script is explicitly derived from:

`sqliteai/warp tools/trace_hf.py @ dc079dad1c35def2ee7bb3b2de5157db8d9f08a8`

Independent source review confirms the routing-critical structure is preserved:

- hooks are registered on each enumerated `layer.mlp.gate`;
- if the gate output is a tuple, tuple element 2 is used as `topk_index`;
- the fallback branch recomputes top-k from raw router logits, as in the historical source;
- branch counters only record which branch executed and do not feed routing or generation;
- selected IDs are stored through `sorted(idx[-1].tolist())`, so membership is retained but native top-k rank/order is destroyed;
- one full model re-forward is performed per generation step;
- per-step layer observations are associated with their enumerated native layer index and emitted in layer order.

The anchored receipt records:

- `native_topk_index_tuple`: 4,800 hook calls;
- `recomputed_from_logits`: 0 hook calls;
- active branch: `native_topk_index_tuple`;
- observed gate output type: tuple length 3.

The arithmetic reconciles with the capture shape: 300 full forwards x 16 routed layers = 4,800 hook calls. The script intentionally does not emit the first generation iteration, giving 299 raw decode rows; 299 x 16 = 4,784 canonical routing events.

## Native Transformers semantic source review

Exact reviewed implementation:

`huggingface/transformers @ v5.15.1`

`src/transformers/models/olmoe/modeling_olmoe.py`

SOURCE-READ establishes the relevant bounded native path:

1. `OlmoeTopKRouter.forward()` computes router probabilities, calls `torch.topk(..., self.top_k)`, and returns `(router_logits, router_scores, router_indices)`.
2. `OlmoeSparseMoeBlock.forward()` receives `_, top_k_weights, top_k_index = self.gate(hidden_states)`.
3. The same `top_k_index` is passed to `self.experts(hidden_states, top_k_index, top_k_weights)`.
4. `OlmoeExperts.forward()` constructs its expert mask from that `top_k_index` and dispatches to the referenced routed experts.

Combined with the receipted 4,800 native-tuple calls and 0 fallback calls, the capture's `out[2]` membership corresponds to the native top-k expert membership passed into routed-expert execution for this reviewed path.

The reviewed `OlmoeSparseMoeBlock` contains the gate and routed expert collection and does not expose a separate shared-expert execution path analogous to the reviewed Kimi path. This statement is bounded to the pinned reviewed Transformers 5.15.1 OLMoE implementation; it is not a claim about every OLMoE runtime or implementation.

The trace is still not a complete model-execution trace: it records routed-expert selection membership, not attention, residual, normalization, allocator, residency, transfer, or other model/runtime behavior.

## Converter audit and independent reproduction

Exact bundled converter:

`mfethe1/warp tools/moe-cache-lab/olmoe-receipted-capture-20260824/convert_kimi_olmoe_canonical.py @ 793bf319dee34cc0ed9863d57ebfd5c9d1a33811`

Independent content SHA-256:

`40aca8ef5176d900353e0cc68ea8807a95db91e445cca1f20fa265b3d127b015`

Relative to the previously reproduced converter, the relevant OLMoE change adds explicit capture-generation identity:

- `OLMOE_HISTORICAL_MODEL_ID` remains the historical default;
- `OLMOE_RECAPTURE_MODEL_ID` names this new capture separately;
- `convert_olmoe(..., model_id=...)` permits the recapture-specific metadata identity.

The routing conversion loop remains raw-order, raw-token, per-layer enumeration with the raw-emitted expert list unchanged.

Two independent conversions were run in the recorded reproduction environment with `OLMOE_RECAPTURE_MODEL_ID`:

- run 1 SHA-256: `d02c69e4185dbba70305fa9f3e0dd75e23d9bc27410d3ba2aa4703240b0b0e4a`;
- run 2 SHA-256: `d02c69e4185dbba70305fa9f3e0dd75e23d9bc27410d3ba2aa4703240b0b0e4a`;
- anchored canonical SHA-256: `d02c69e4185dbba70305fa9f3e0dd75e23d9bc27410d3ba2aa4703240b0b0e4a`.

The independently regenerated bytes also matched the anchored `canonical_v1.jsonl` bytes exactly.

**Mechanical canonical reproduction: `AGREEMENT`**

Recorded reproduction environment:

- GitHub Actions hosted Ubuntu 24.04.4 LTS runner;
- platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`;
- Python implementation: CPython;
- Python version: 3.10.21.

Supported conclusion:

`Deterministic reproduction established in the recorded environment.`

No cross-platform byte-determinism claim is made.

The external converter emitted Python `ResourceWarning` messages for source files opened without an explicit close. Those warnings did not change the converter output or reproduction result. This project did not modify the external converter to suppress them.

## Historical-converter regression

The new bundled converter was also run against the immutable historical OLMoE raw fixture using its historical default model identity.

Resulting SHA-256:

`83168fb10391646a97bb0c00c159094784a0ef72dc7962d18055020701f1d9b4`

This equals the already accepted historical canonical target exactly.

**Historical-converter regression: `PASS`**

This regression check prevents the new capture identity parameter from silently changing historical artifact meaning. It does not upgrade any historical provenance field.

## Independent raw-to-canonical preservation audit

The existing independent checker `scripts/audit_external_canonical_reproduction.py` was applied to the new raw fixture and independently regenerated canonical file. The checker does not import or call the external converter.

Full-fixture result:

- raw decode rows: 299;
- routed layer lists per row: 16;
- canonical events: 4,784;
- expert requests: 38,272;
- canonical layer range: `0..15`;
- canonical `token_position` equals raw `tok`;
- canonical layer equals raw layer ordinal;
- `selected_experts` equals the raw-emitted expert list exactly;
- phase is canonical v1 `generated`;
- no event is added, dropped, reordered, or chronology-repaired;
- no probabilities or native ranking information are reconstructed.

**Raw-to-canonical preservation audit: `PASS`**

List equality proves preservation of the list emitted by the producer. It does not restore the native top-k ranking destroyed upstream by `sorted(...)`.

## Canonical v1 validation

The independently regenerated trace passes the project v1 validation path with:

- trace format version: 1;
- valid: true;
- event count: 4,784;
- expert request count: 38,272;
- expert universe: 64;
- experts per token/event: 8.

The live reproduction used the current private v1 validation path. The following source files have the same Git blob identities at the accepted private base and exact public v0.8.0 source commit `e9fb09ec2bc3c39369958cd9aa9ecdc1f5c8cf25`:

- `src/moe_cache_lab/cli.py`: `eb409fbe60bb475acd5d402d8c621875b784e633`;
- `src/moe_cache_lab/trace_validation.py`: `04711b433b5a3616c2d5709d69b6926ea5516360`;
- `src/moe_cache_lab/trace.py`: `bb465665930037f31766f7c6c8329fd767a777be`;
- `src/moe_cache_lab/schemas/routing-trace-v1.schema.json`: `8405f854484081e08450a6c642eb320743fda494`.

Therefore the relevant v1 CLI/reader/validation/schema source is byte-identical to the public v0.8.0 validation source used by the external contract.

**Canonical validity: `PASS`**

Canonical validity does not establish producer semantic mapping or non-interference by itself.

## Producer semantic-mapping decision

The evidence chain is now sufficient for a narrow semantic decision:

- the exact capture script was archived and independently inspected;
- both tokenizer and model loads use the full pinned revision;
- the receipt records the active native tuple branch on every one of 4,800 routed-layer hook calls and zero fallback calls;
- exact Transformers 5.15.1 SOURCE-READ establishes tuple element 2 as the native top-k indices passed into routed expert execution;
- hook identity provides native layer identity `0..15`;
- the full raw-to-canonical preservation audit establishes token/layer chronology and expert membership preservation into canonical v1;
- no missing selection probabilities or native ranks are invented.

Decision:

**Producer path semantically validated: `BOUNDED PASS — selected routed-expert membership on the reviewed pinned path`.**

Exact scope:

`allenai/OLMoE-1B-7B-0125 @ 9b0c1aa87e34a20052389dce1f0cf01da783f654`, Transformers 5.15.1, the recorded macOS/MPS/bfloat16 capture, native tuple `topk_index` branch, batch 1, the recorded prompt, and the recorded 300-step full-re-forward generation procedure whose final 299 decode rows are emitted.

This semantic PASS establishes that the emitted/canonical expert **membership** corresponds to the selected routed experts on that path. It does not establish:

- native top-k rank/order (`NOT PRESERVED`);
- selection probabilities/weights in the trace;
- producer non-interference;
- other OLMoE revisions, runtimes, devices, dtypes, batching modes, prompts, decoding settings, or producer implementations;
- workload representativeness;
- physical residency or transfer;
- latency, throughput, tokens/sec, speedup, or memory savings;
- an optimal policy or capacity.

The path must not be called generically conforming or generally validated while non-interference remains missing.

## Producer non-interference

No frozen control-vs-instrumented experiment compares this exact receipted path with an observation-disabled control under the preregistered `PRODUCER_CONFORMANCE.md` rules.

Source inspection shows receipt counters are observational bookkeeping and do not feed routing selection, but source inspection is not a substitute for the required bounded control comparison.

**Producer path non-interference validated: `NOT ESTABLISHED`**

## Artifact intake decision

**`ACCEPTED AS BOUNDED RECEIPTED EXTERNAL INTAKE EVIDENCE`**

Permitted meaning:

- the immutable external bundle may be referenced as a reproducible receipted external producer case;
- the canonical v1 artifact may be consumed for downstream routing analysis when its exact external source/hash identity is retained;
- selected routed-expert membership may be described as semantically validated only within the exact bounded scope above;
- downstream cache outcomes remain **SIMULATED**;
- downstream transfer-service quantities remain **ESTIMATED** under explicit assumptions.

This acceptance does not authorize copying external trace bytes into project-controlled distribution surfaces, does not establish redistribution rights, does not establish producer non-interference, and does not support runtime-performance or broad-compatibility claims.

## Final offline boundary

A one-shot network-enabled evidence execution was used to fetch the immutable external bytes and produce the independent hash/reproduction/preservation/validation record above. That temporary network-enabled test was removed after the evidence was captured.

The final tracked validation surface for this record is offline and contains no external raw/canonical trace bytes.
