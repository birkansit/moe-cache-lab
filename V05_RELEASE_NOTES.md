# v0.5 release notes

Current package version: **0.5.0**.

v0.5 establishes `moe-cache-lab` as a trace-driven MoE pre-flight analysis
toolkit for expert caching/offloading research. It packages implemented offline
analysis, not an inference-acceleration runtime.

## Implemented in v0.5

- deterministic routing-locality analysis over compatible routing traces;
- prompt/prefill and generated/decode phase-layer summaries;
- routing-frequency concentration metrics, including exact maximum share,
  exact observed-support Gini, and Shannon entropy in bits;
- byte-aware, layer-qualified LRU/LFU cache simulation using explicit expert
  sizes;
- serialized/no-overlap transfer-service sensitivity using explicit H2D
  bandwidth and per-loaded-expert setup latency;
- one offline `analyze` workflow that emits deterministic Markdown and
  exact-rational JSON pre-flight reports;
- a tracked synthetic no-download demo.

## Evidence and claim boundary

Routing statistics are descriptive observations of the supplied trace and are
**MEASURED only when that trace's provenance establishes measurement**.

Byte-cache outcomes are **SIMULATED**. Transfer-service costs are **ESTIMATED**
from explicit assumptions. The serialized transfer model excludes GPU compute,
transfer/compute overlap, concurrency, allocator effects, kernel scheduling,
runtime synchronization, and D2H writeback for immutable expert eviction.

Version 0.5.0 does not establish actual GPU residency, end-to-end latency,
throughput, tokens/sec, or runtime speedup. It does not implement expert
swapping/offloading.

A separate Windows validation record documents fresh Granite routing,
byte-cache equivalence, workload sensitivity, and raw HIP H2D calibration.
Those HIP measurements are raw byte-transfer diagnostics, not PyTorch or Granite
GPU-runtime measurements.

## Scope limitations

The built-in collector remains Granite/Transformers-specific. Once a compatible
trace exists, the v0.5 offline analysis path is trace-format/core driven and
requires no model execution.

No second model-family collector, vLLM/llama.cpp/MoE-Infinity integration,
vendor hardware profile, hardware auto-detection, compute/transfer overlap
model, async prefetch, recommendation threshold, or cacheability score is
included.

This document does not claim that a Git tag, GitHub Release, PyPI publication,
or public repository visibility change exists for version 0.5.0.
