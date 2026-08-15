# v0.5 release notes

Current package version: **0.5.0**.

v0.5 turns the repository's corrected routing/cache foundations into a trace-driven MoE pre-flight analysis toolkit for expert caching/offloading research. It packages implemented offline analysis, not an inference-acceleration release.

## Implemented in v0.5

- deterministic routing-locality analysis over compatible routing traces;
- prompt/prefill and generated/decode phase-layer summaries;
- routing-frequency concentration metrics: exact maximum share, exact observed-support Gini, and Shannon entropy in bits;
- byte-aware, layer-qualified LRU/LFU cache simulation using explicit caller-supplied expert sizes;
- exact serialized/no-overlap transfer-service sensitivity using caller-supplied H2D bandwidth and per-loaded-expert setup latency;
- one offline `analyze` workflow that can emit deterministic Markdown and exact-rational JSON pre-flight reports;
- a tracked synthetic no-download demo under `examples/no-download-preflight/`.

## Evidence and claim boundary

Routing statistics are descriptive observations of the supplied trace and are **MEASURED only when that trace's provenance establishes measurement**. The tracked v0.5 demo is synthetic.

Byte-cache outcomes are **SIMULATED**. Transfer-service costs are **ESTIMATED** from explicit assumptions. The serialized transfer model does not include GPU compute, transfer/compute overlap, concurrency, allocator effects, kernel scheduling, runtime synchronization, or D2H writeback for immutable expert eviction.

Version 0.5.0 does not establish actual GPU residency, physical transfer timing, end-to-end latency, throughput, tokens/sec, or runtime speedup. It does not implement expert swapping/offloading.

## Scope limitations

The built-in collector remains Granite/Transformers-specific. Once a compatible trace exists, the v0.5 offline analysis path is trace-format/core driven and requires no model download or execution.

No second model-family collector, vLLM/llama.cpp/MoE-Infinity integration, vendor hardware profile, hardware auto-detection, compute/transfer overlap model, async prefetch, recommendation threshold, or cacheability score is included.

Historical v0.2/v0.3/v0.4 evidence and the historical `0.4.0rc1` checkpoint remain unchanged and retain their original interpretation.

This document does not claim that a Git tag, GitHub Release, PyPI publication, public-repository synchronization, or repository visibility change exists for version 0.5.0.
