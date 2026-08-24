# Project brief: moe-cache-lab

## Purpose

`moe-cache-lab` is a correctness-first, trace-driven MoE pre-flight and
evidence-analysis toolkit. The current package version is **0.8.0**. It is
research and decision-support tooling, not an inference accelerator.

[`README.md`](README.md) defines the current source capability and claim
boundary. [`V07_RELEASE_NOTES.md`](V07_RELEASE_NOTES.md) preserves the v0.7
package scope; [`V08_RELEASE_NOTES.md`](V08_RELEASE_NOTES.md) defines the v0.8
release scope and limitations.

## Current v0.8 contract

- Canonical routing trace v1 and v2 coexist. V1 preserves layer-qualified
  `(layer, expert_id)` identity and its established pre-flight behavior.
- V2 preserves stage-qualified `(routing_stage, layer, expert_id)` encoder and
  decoder identity and supports descriptive offline analysis. The
  library-level version-aware byte-cache simulator also accepts v2, preserves
  that identity, and treats unassigned events as zero-request/cache-inert
  events. Single-trace v2 pre-flight uses stage-qualified config v3; v2
  lifecycle orchestration remains unavailable rather than flattening stages.
- The lightweight base installation requires neither PyTorch nor Transformers.
  Granite remains the built-in public collection path through its optional
  dependencies.
- One pinned SwitchTransformers family/path was narrowly validated for the
  trace-v2 abstraction and research. There is no public Switch collection CLI
  and no broad Switch, Transformers, or MoE-family compatibility claim.
- Version 0.8 retains the version-aware evidence diagnostics introduced in
  v0.7 and deterministic experiment bundles with explicit integrity and
  provenance boundaries.
- Routing-derived evidence is **MEASURED** only where provenance establishes
  measurement. Cache outcomes are **SIMULATED**, and transfer-service results
  are **ESTIMATED** under explicit caller-supplied assumptions.

## Historical progression: V0.1 through V0.4

The remaining sections preserve earlier milestones, evidence, corrections, and
negative results as historical provenance. They do not replace the current
contract above.

### V0.3 and V0.4 outcome

- Stage 1 adds a conservative grouped-prefill sensitivity view, longer decode,
  three fresh deterministic repetitions, strict manifests, paired controls,
  and validated real routing evidence.
- The frozen Stage 1 continuation rule passed, but that signal is simulated and
  justified feasibility inspection rather than offloading.
- Stage 2 found no supported, meaningful per-expert residency path on the tested
  Windows/RX 6650 XT stack. The local no-go defers real offloading and tuning
  while preserving the profiler/simulator as the justified deliverable.
- `STAGE1_RESULTS.md`, `STAGE2_FEASIBILITY.md`, and
  `RELEASE_CANDIDATE.md` preserve the technical record of that progression.
- V0.4 adds a full-resident CPU observer/reference measurement. The valid
  attempt's mean paired wall-time ratio is `1.0208846178255349` (about 2.09%
  observer overhead), with exact routing IDs, probability delta `0.0`, and all
  frozen stability gates passing. See `V04_RESULTS.md`.
- This V0.4 result measures observer overhead on one pinned CPU workload. It is
  not acceleration, cache-residency, GPU, throughput, or production evidence
  and does not justify Stages 3-5.

### Confirmed V0.1 facts

- Granite configuration: 32 experts; 8 selected per token.
- Routing collection uses temporary PyTorch forward hooks. The installed
  Transformers version accepts `output_router_logits` but does not expose
  populated public router outputs for this Granite model.
- Hooks observe routing only. They must never alter model weights or routing
  decisions.
- Available policies: event-atomic LRU, online resident-frequency LFU,
  calibration-only `calibrated_static_frequency`, and the explicitly
  non-causal `offline_oracle_frequency` baseline.

### Corrected V0.1 cache abstraction

- One cache entry is exactly one layer-qualified `(layer_id, expert_id)` expert
  weight object. Equal numerical expert IDs in different layers are distinct.
- Capacity is the total number of these objects simultaneously resident across
  the whole model. Dynamic policies start empty.
- A `RoutingEvent` is replayed atomically: its unique selected experts are
  pinned and required together, and event tuple order cannot affect results.
- The oracle's fixed target set uses the full evaluation trace. Missing event
  requirements temporarily displace non-required targets within the hard
  capacity; afterward temporary entries leave and displaced targets reload.
  It is an offline, non-deployable comparison baseline.
- One chronological cache run supplies combined, prompt/prefill, and
  generated/decode counters. Generated/decode means routing for emitted tokens
  when those tokens are used as decode inputs.

Any cache-simulation results produced before this corrected abstraction are
invalidated and must not be cited as V0.1 results. Corrected real-model evidence
is tracked at `results/v0.2-corpus-v1/` and
`results/v0.2-corpus-v1-report.md`.

### Historical measurement discipline

- Never claim runtime improvement from simulator results.
- Label routing selections as **measured** only when their trace provenance
  establishes measurement; label cache results as **simulated** and transfer
  counts as **estimated**.
- Do not attach unverified semantic meaning to expert IDs.
- Keep prompt and generated-token routing distinct in collection, analysis, and
  reporting.

### V0.2 reproducible suite

- The tracked corpus is versioned and fixed at four calibration plus eight
  evaluation prompts. Categories describe workload diversity only and cannot
  support expert-semantic inference.
- One CPU-float32 Granite model/tokenizer load collects independent prompt
  contexts sequentially with deterministic greedy decode. The requested model
  revision is resolved once and its exact commit pins config, tokenizer, model,
  manifest, and traces.
- The collection manifest hashes canonical corpus content and every trace and
  records model, runtime, generation, split, prompt order, and count metadata.
- `calibrated_static_frequency` selects fixed targets only from calibration;
  all reported metrics use evaluation only. `offline_oracle_frequency` alone
  may select targets from the complete evaluation trace.
- Evaluation order is manifest order and one simulated cache persists across
  that suite for each policy/capacity. Default capacities are exactly 8, 16,
  32, 64, 128, and 256 total layer-qualified expert objects.
- V0.2 remains simulation and analysis. It does not implement expert movement
  or inference acceleration.
- Granite prefill groups all prompt-token assignments in a layer by expert.
  Per-token prompt simulation remains an explicit abstraction and cannot
  establish reuse or runtime scheduling below grouped prefill working sets.
- Final traces/manifests/reports live under `results/v0.2-corpus-v1/` with
  report `results/v0.2-corpus-v1-report.md`; these tracked artifacts are traces,
  manifests, and reports, never models or binaries.
- The completed evidence pins revision
  `0da7a48b0276d500ce5922fd2b33944091fc6c09`, corpus SHA-256
  `67144988f37ae14134aa83ca4b447f5119e96597dec4ab7158fc1f2592e6be2c`,
  and manifest SHA-256
  `d5ca0013b1dfee96b353bb964666eaf9f6335494b0279362f536b4e8bc359e30`.
  Measured counts are 1,320 calibration events / 10,560 requests and 3,144
  evaluation events (2,760 prompt plus 384 generated) / 25,152 requests.
