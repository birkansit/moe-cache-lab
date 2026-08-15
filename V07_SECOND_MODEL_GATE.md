# V0.7 second-model research gate

Status: **NO-GO for a second collector under the current trace-v1 and local
hardware constraints.**

This is a private-development selection note, not a support claim. It records
configuration and source inspection only. No candidate model was instantiated,
no model weights or tokenizer files were downloaded, and no routing was
measured.

## Scope and evidence

The gate was performed on 2026-08-13 from private-dev base
`4b1eaa92e7f1d50996b643ce748697aac775764a`. The installed implementation was
official Transformers `5.12.0` at:

`%LOCALAPPDATA%\Programs\Python\Python310\Lib\site-packages\transformers`

The corresponding upstream sources are pinned by the official `v5.12.0` tag:

- [SwitchTransformers implementation](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/switch_transformers/modeling_switch_transformers.py)
  and [configuration](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/switch_transformers/configuration_switch_transformers.py);
- [OLMoE implementation](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/olmoe/modeling_olmoe.py)
  and [configuration](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/olmoe/configuration_olmoe.py);
- [Qwen2-MoE implementation](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py)
  and [configuration](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/qwen2_moe/configuration_qwen2_moe.py).

Official Hugging Face model metadata was queried at the immutable revisions
below. Each config was parsed through the native Transformers 5.12.0
`AutoConfig` mapping, without `trust_remote_code`; all three configs have no
`auto_map` entry. The native model mappings resolved to
`SwitchTransformersForConditionalGeneration`, `OlmoeForCausalLM`, and
`Qwen2MoeForCausalLM`, respectively. No fourth candidate was added because the
required three already expose the decisive tradeoff: the locally feasible
family is not trace-v1-compatible, while the trace-v1-compatible checkpoints
are not locally feasible.

## Candidate matrix

| Candidate | Architecture and routing | Expert layout and observation in Transformers 5.12.0 | Trace-v1 assessment | Official checkpoint and local feasibility | Decision |
| --- | --- | --- | --- | --- | --- |
| `google/switch-base-8` | T5-like encoder-decoder; 8 experts, top-1; 12 encoder plus 12 decoder layers with six sparse layers in each stack (indices 1, 3, 5, 7, 9, 11); capacity 64 can drop a token's assignment | Experts are individually addressable `ModuleDict` entries. The router returns the selected maximum probability and a post-capacity one-hot assignment, so a non-mutating hook can observe actual dispatch, including an all-zero dropped assignment. | **Incompatible.** V1 has one layer namespace and cannot distinguish encoder layer 1 from decoder layer 1. It also requires a non-empty selection, while the official router explicitly provides no guarantee that every token is processed. Recording the pre-capacity argmax would misrepresent dispatch. | One BF16 `pytorch_model.bin`, 1,238,895,063 bytes (1.154 GiB). Approximate FP32 parameter payload is 2.478 GB / 2.308 GiB, so it is the only clearly plausible checkpoint on the 15.913 GiB machine. Its weight format and actual peak still require a future admitted smoke test. | Reject for v1. It is the strongest future trace-v2 candidate. |
| `allenai/OLMoE-1B-7B-0924` | Decoder-only; 64 experts, top-8; every one of 16 decoder layers is routed; no capacity/drop path in the inspected router | Routed expert weights are packed 3-D `gate_up_proj` and `down_proj` parameters, not individual expert modules. The router returns full logits, selected probabilities, and exact IDs; native output capture records logits. | **Compatible for routing selections.** Prompt and decode map to existing phases; one top-8 token/layer selection is one atomic event; layer-qualified identity and Granite-style layer-major prefill/token-major decode can be preserved. Selected weights need not sum to one, which v1 already permits. | Card: 1B active / 7B total. Three BF16 shards total 13,838,721,960 bytes (12.888 GiB). The current collector forces FP32, implying about 27.677 GB / 25.777 GiB of parameter payload before runtime overhead or load transients. | Reject on RAM feasibility; its current packed layout also adds limited layout diversity over Granite. |
| `Qwen/Qwen1.5-MoE-A2.7B` | Decoder-only; 60 routed experts, top-4; all 24 layers are sparse for this config; every token also passes through a separately gated shared expert | Routed experts are packed 3-D parameters. The top-k router returns full logits, selected probabilities, and IDs; native output capture records logits. The shared expert is an always-present MLP path, not one of the 60 router-selected IDs. | **Compatible for the sparse routing selection only.** Top-4 events, phases, identity, and chronology fit v1. The shared expert must not be invented as a selected ID; consequently a v1 cache study would cover routed experts only and would not be a complete model-residency model. | Card: 14.3B total / 2.7B active. Eight BF16 shards total 28,632,144,944 bytes (26.666 GiB). Approximate FP32 parameter payload is 57.264 GB / 53.332 GiB, far beyond physical RAM. | Reject on RAM feasibility; shared-expert accounting is an additional bounded limitation. |

## Immutable model provenance

### SwitchTransformers

- Revision: `92fe2d22b024d9937146fe097ba3d3a7ba146e1b`.
- [Model card](https://huggingface.co/google/switch-base-8/blob/92fe2d22b024d9937146fe097ba3d3a7ba146e1b/README.md),
  SHA-256 `4dcf7d0968bace568166889188491e16b80ea0099581e51ae09506c344a46896`.
- [Config](https://huggingface.co/google/switch-base-8/blob/92fe2d22b024d9937146fe097ba3d3a7ba146e1b/config.json),
  SHA-256 `6a1e1426873221034f8488e514ac127218c833e7d357b1e12dc002889c98fa53`.
- Config facts: `is_encoder_decoder=true`, 8 experts, capacity 64,
  12 layers per stack, six sparse layers per stack, BF16 checkpoint.

### OLMoE

- Revision: `6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
- [Model card](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/6d84c48581ece794365f2b8e9cfb043c68ade9c5/README.md),
  SHA-256 `a2896dce1cf6b9f1ed32c09bda64eb7be3db3ad49ac4bceef15996a197806608`.
- [Config](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/6d84c48581ece794365f2b8e9cfb043c68ade9c5/config.json),
  SHA-256 `3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335`.
- Config facts: decoder-only, 16 layers, 64 experts, top-8, BF16.

### Qwen2-MoE

- Revision: `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`.
- [Model card](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/README.md),
  SHA-256 `793525668ba598ff2c3ff8731ecc548ef212bb46ca918fd6f118b2785861c142`.
- [Config](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/config.json),
  SHA-256 `d0b1cd8f35beccb75211c06940da3274ea986363792ab0ececb60d4ec03dd8c6`.
- Config facts: decoder-only, 24 layers, sparse step 1, 60 routed
  experts, top-4, one gated shared expert path, BF16.

## Exact trace-v1 conclusion

Canonical v1 is not generally multi-family: it intentionally preserves the
Granite collector's two phases, one numerical model-layer namespace,
non-empty atomic selections, layer-major prompt order, and token-major decode
order.

OLMoE can be represented without changing those meanings. Qwen2-MoE can also
represent its sparse top-4 selections, provided reports state that the always-on
shared expert is outside the selected-expert trace and cache abstraction. No
event may be synthesized for that shared path.

SwitchTransformers cannot be represented faithfully. A future contract would
need at least an encoder/decoder routing-stage identity in the expert key and an
explicit representation of a post-capacity dropped/unassigned token. These are
semantic and field changes, so the closed v1 policy requires a new version.
Flattening the two stacks, offsetting layer numbers, omitting dropped tokens, or
recording pre-capacity preferences as assignments would silently change source
semantics and is rejected.

## Decision and possible next gate

No candidate satisfies both requirements at once:

1. faithful canonical trace-v1 routing semantics; and
2. a credible full-checkpoint CPU-FP32 collection on the known 16 GiB Windows
   machine.

Therefore Issue #40 authorizes **no second collector implementation**.

Two future paths are defensible, but each requires a new explicit issue:

- If contract portability is the priority, first design and review a trace-v2
  encoder/decoder/drop contract using synthetic records, then reconsider
  `google/switch-base-8`. Only after that contract gate should its 1.239 GB
  checkpoint be downloaded for an admitted observational smoke test.
- If the machine/runtime changes enough to provide safely more than the
  approximate 27.677 GB FP32 OLMoE parameter payload plus load/runtime
  overhead, reconsider OLMoE as the simplest v1-compatible implementation.
  That would validate 64-way top-8 routing, but not a different expert-weight
  layout in current Transformers.

Qwen2-MoE remains unsuitable for this machine even before runtime overhead.
Any later gate must separately bound the meaning of its shared expert rather
than treating it as a selected routed expert.

## Download and unresolved-risk record

- Network activity was limited to official Hugging Face/GitHub metadata,
  model-card, config, and Transformers source text. The three immutable configs
  plus three immutable model cards totaled 20,433 bytes; all retrieved content
  was held in memory or discarded rather than added to the repository/cache.
- No weight shard, tokenizer, generation config, or safetensors index was
  requested. The Hugging Face cache has no directory for any of the three
  candidates after inspection.
- `trust_remote_code=True` was never used. No package, runtime, framework, or
  dependency was installed.
- Checkpoint sizes are official repository file metadata. FP32 figures are
  arithmetic estimates from BF16 payload bytes, not measured process peaks.
- Without weights, numerical router-output equality, hook non-interference,
  actual dropped-token incidence, end-to-end chronology, and peak RAM remain
  untested. These unknowns cannot be converted into support or performance
  claims.
- No public multi-model claim, cache result, transfer estimate, runtime result,
  or expert-semantic inference follows from this note.
