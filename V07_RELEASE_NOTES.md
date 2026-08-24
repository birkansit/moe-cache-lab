# v0.7 release notes

Current package version: **0.7.0**.

v0.7 extends the correctness-first, trace-driven MoE pre-flight toolkit with a
portable canonical trace contract and a release-facing offline interpretation
path. It remains research and decision-support tooling, not an inference
accelerator.

## Implemented in v0.7

1. **Canonical trace v1 and v2 coexistence.** Trace v1 preserves the established
   layer-qualified `(layer, expert_id)` decoder-only identity. Trace v2 adds the
   stage-qualified `(routing_stage, layer, expert_id)` identity needed to keep
   encoder and decoder routing distinct. Both readers enforce their canonical
   chronology and reject invalid order without sorting or repair.
2. **Lightweight offline core.** The base package has no required PyTorch or
   Transformers dependency. The public Granite collection command uses the
   `[granite]` extra. The `[switch]` extra isolates dependencies for the narrow
   reviewed Switch research collector/path; no public Switch collection CLI is
   provided.
3. **Narrow second-family validation.** The trace-v2 abstraction was
   structurally and observationally exercised on one pinned Transformers
   SwitchTransformers family/path. This is not a claim of broad Transformers,
   Switch-family, or MoE-family compatibility, and no general collector plugin
   framework was added.
4. **Evidence and coverage diagnostics.** Version-aware routing evidence reports
   expose observed coverage, warnings, configured-universe normalized entropy,
   and caller-explicit cumulative top-k selection shares. There is no hidden
   default top-k, representativeness score, quality grade, policy ranking, or
   recommendation.
5. **Bounded constructed CPU-copy validation.** A frozen SIMULATED LRU outcome
   agreed exactly with an independent constructed CPU physical-copy replay.
   The actual copy operations and content equality are **MEASURED test-replay
   observations** for that constructed replay only. They do not measure native
   inference caching, physical residency, DRAM traffic, PCIe/H2D traffic, GPU
   behavior, latency, throughput, speedup, or memory savings.
6. **Experiment bundles and canonical public CLI.** Experiment bundles record
   bounded config, tool, and environment provenance, and bundle artifacts have
   SHA-256 integrity anchors. A trace may be explicitly embedded or represented
   by an external reference plus SHA-256. External references are metadata-only
   and are never fetched by bundle creation or verification. Bundle integrity
   does not upgrade scientific evidence quality or change an artifact's
   evidence class. The canonical public mental model is: collect with a
   validated collector or import a canonical trace -> analyze offline ->
   compare/report only where that trace/config contract supports it ->
   optionally create/verify an experiment bundle. The executable
   [`WALKTHROUGH.md`](WALKTHROUGH.md) demonstrates that path on a tracked
   synthetic fixture and teaches bounded evidence interpretation.

## Compatibility and boundaries

The established `analyze --preflight-config` cache and transfer-service pipeline
remains v1-only because its expert-size configuration uses layer-qualified v1
identities rather than stage-qualified v2 identities. The library-level
`simulate_versioned_byte_cache()` API supports canonical v1 and v2 traces; its
v2 path preserves `(routing_stage, layer, expert_id)` identity and gives an
unassigned event zero expert requests and zero cache-state changes. Those
outcomes remain **SIMULATED**. V0.7 does not provide a stage-qualified v2
preflight config, transfer-cost/lifecycle orchestration, or public v2 pre-flight
workflow. Existing v1 report bytes and historical evidence remain unchanged.

Routing-derived descriptive output is **MEASURED only when trace provenance
establishes actual measurement**. Cache hits, misses, loads, evictions,
resident-byte values, and policy outcomes are **SIMULATED**. Transfer-service
values are **ESTIMATED** from simulated loads and explicit caller-supplied
serialized/no-overlap assumptions.

Neither canonical traces nor integrity bundles establish representative
coverage, scientific correctness, native caching, physical expert residency,
real transfers, end-to-end latency, throughput, tokens/sec, acceleration, an
optimal policy, or an optimal capacity. A smaller simulated or estimated value
is not a measured speedup or a deployment recommendation.
