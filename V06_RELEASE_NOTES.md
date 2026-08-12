# v0.6 release notes

Current package version: **0.6.0**.

v0.6 extends the trace-driven MoE pre-flight analysis toolkit with deterministic
offline sensitivity and lifecycle accounting. It remains research and
decision-support tooling, not an inference accelerator.

## Implemented in v0.6

1. **Modeled transfer-operation sensitivity.** Caller-supplied operation plans
   map each simulated logical demand load to a positive fixed count of modeled
   setup-bearing transfer operations while preserving the existing exact
   serialized/no-overlap transfer equation.
2. **Cache lifecycle simulation.** `cold_per_workload` starts each workload
   empty, while `persistent_sequence` preserves one cache through validated
   manifest order. Both retain the established layer-qualified, event-atomic
   LRU/LFU semantics.
3. **Descriptive sensitivity summaries.** Adjacent tested-capacity deltas,
   same-capacity LFU-minus-LRU deltas, and workload minima/maxima/ranges are
   derived deterministically from existing lifecycle rows without replay.
4. **Lifecycle transfer-service matrix.** Every aggregate and per-workload
   lifecycle row receives exact **ESTIMATED** serialized H2D service accounting
   for each configured hardware profile and transfer-operation plan, with exact
   workload-to-aggregate reconciliation.
5. **Compatibility.** Established pre-flight/config version 1 and version 2
   behavior and the historical no-download report bytes remain unchanged.

## Evidence and claim boundary

Routing observations are **MEASURED only where trace provenance establishes
measurement**. Cache hits, misses, demand bytes, evictions, residency, and
policy outcomes are **SIMULATED**. Transfer-service values are **ESTIMATED**
from caller-supplied bandwidth, setup-latency, and operation-granularity
assumptions.

Transfer-operation plans are modeled assumptions; they do not establish
Granite's physical transfer granularity. The lifecycle estimates model
serialized/no-overlap H2D service only and exclude compute, overlap,
concurrency, kernel/runtime scheduling, allocator effects, synchronization or
protocol effects, D2H writeback, and end-to-end runtime behavior.

Version 0.6.0 does not establish physical expert residency, measured model
transfer latency, end-to-end latency, throughput, tokens/sec, or speedup. A
smaller simulated or estimated value is not proof of acceleration, and no
policy, capacity, hardware profile, or operation plan is recommended or ranked.

No runtime expert swapping/offloading, GPU residency manager, asynchronous
prefetch, overlap model, second model-family collector, or additional inference
backend is implemented. Historical evidence remains unchanged.

This release document does not claim that a Git tag, GitHub Release,
PyPI publication, repository synchronization, or visibility change exists for
version 0.6.0.
