# moe-cache-lab

`moe-cache-lab` is a correctness-first, trace-driven Mixture-of-Experts (MoE)
pre-flight analysis toolkit for expert caching/offloading research.

Given a compatible routing trace and explicit expert-size, cache-capacity, and
transfer assumptions, it helps answer a narrower engineering question:

> Is expert caching/offloading worth further runtime investigation for this
> routing workload, and what routing/cache/cost behavior is driving the result?

Current package version: **0.5.0**.

The built-in collector is currently Granite/Transformers-specific, using
`ibm-granite/granite-3.1-1b-a400m-instruct` as the reference model. Once a
compatible routing trace exists, the v0.5 analysis path is trace-format/core
driven and can run without loading a model.

> **Important:** this project is not an inference accelerator. It does not
> implement expert swapping/offloading or physical GPU residency management.
> It observes/analyzes routing, simulates cache behavior, and estimates a simple
> serialized transfer-service cost under explicit assumptions.

## What it does

`moe-cache-lab` can:

1. collect exact routing selections from the supported Granite collector;
2. store those selections in a versioned JSONL routing trace;
3. analyze global, per-layer, and prompt/decode routing locality;
4. compute routing-frequency concentration statistics;
5. replay traces through byte-capacity LRU/LFU cache simulations;
6. evaluate explicit transfer-cost assumptions over simulated demand loads;
7. emit deterministic Markdown and JSON pre-flight reports.

The project keeps evidence classes separate:

- **ROUTING OBSERVATIONS:** descriptive results over a supplied trace. Treat
  them as **MEASURED only when trace provenance establishes measurement**.
- **SIMULATED:** cache hits, misses, demand-load bytes, evictions, and residency
  accounting produced by offline replay.
- **ESTIMATED:** serialized/no-overlap transfer-service time derived from
  simulated demand loads and explicit bandwidth/setup assumptions.

A simulated or estimated result is not a measured runtime speedup, measured
end-to-end latency, throughput result, tokens/sec result, or physical-residency
measurement.

## Who this is for

This toolkit is for researchers and engineers who already have, or can collect,
a compatible MoE routing trace and want a reproducible pre-flight view before
investing in runtime integration.

## When not to use it

Do not use the pre-flight report as proof that a deployment will be faster.
Real conclusions about transfer overlap, allocator behavior, kernel scheduling,
physical data movement, latency, throughput, or tokens/sec require separate
measurement in the target runtime.

## Quick start

Requirements:

- Python 3.10 or newer;
- PyTorch 2.12.0;
- Transformers 5.12.0.

Create an environment and install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If package installation fails because of a certificate/TLS problem, repair the
certificate chain. Do not disable TLS verification as a workaround.

### No-download demo

The tracked synthetic demo requires no model download or GPU:

```powershell
moe-cache-lab analyze examples\no-download-preflight\trace.jsonl `
  --preflight-config examples\no-download-preflight\preflight-config.json `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

See [`PREFLIGHT.md`](PREFLIGHT.md) and
[`examples/no-download-preflight/README.md`](examples/no-download-preflight/README.md).

### Inspect the reference model configuration

```powershell
moe-cache-lab inspect
```

### Collect a measured Granite routing trace

The first run may download the model unless it is already cached:

```powershell
moe-cache-lab collect `
  --prompt "Explain why cache locality matters." `
  --max-new-tokens 4 `
  --output artifacts\granite-trace.jsonl
```

`--max-new-tokens 0` records prompt/prefill routing only.

### Analyze a trace

Routing-only analysis:

```powershell
moe-cache-lab analyze artifacts\granite-trace.jsonl
```

Byte-cache and transfer-assumption analysis:

```powershell
moe-cache-lab analyze artifacts\granite-trace.jsonl `
  --preflight-config preflight-config.json `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

## Validation

The v0.5 package has deterministic no-download release gates plus a separate
Windows local validation record covering fresh Granite routing, byte-cache
semantic equivalence, workload sensitivity, and raw HIP H2D calibration.

See [`V05_VALIDATION.md`](V05_VALIDATION.md).

The hardware-specific measurements are deliberately not presented as PyTorch
GPU inference, expert-offload runtime, or speedup measurements.

For source-tree tests:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

GitHub CI runs offline with respect to Hugging Face model downloads. It checks
dependencies, compiles source, runs the current unit/release suite, builds wheel
and sdist distributions, audits package metadata, performs an isolated wheel
install, and verifies the deterministic no-download demo outputs.

## Development disclosure

AI-assisted coding and review were performed using **OpenAI GPT-5.6 Sol**.
Automated tests, deterministic evidence checks, and explicit claim boundaries
are included, but AI assistance does not constitute independent peer review or
guarantee correctness. Treat this as research tooling and validate results
before relying on them for engineering or research claims.

## Current limitations

Version 0.5.0 does not provide:

- real expert swapping/offloading or GPU residency management;
- a runtime acceleration or speedup claim;
- integrated physical transfer measurement;
- hardware auto-detection or built-in vendor profiles;
- compute/transfer overlap modeling;
- asynchronous prefetch;
- a second model-family collector;
- vLLM/llama.cpp/MoE-Infinity runtime integration;
- SSD offload;
- learned expert prediction/prefetch;
- custom CUDA/ROCm kernels;
- a recommendation threshold or cacheability score;
- native-router modification.

These are future research directions only when a supported runtime and measured
bottleneck justify the added complexity.

## Documentation

- [`PREFLIGHT.md`](PREFLIGHT.md) — configuration and interpretation.
- [`V05_VALIDATION.md`](V05_VALIDATION.md) — local v0.5 validation summary.
- [`V05_RELEASE_NOTES.md`](V05_RELEASE_NOTES.md) — release scope and limits.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution and reproducibility rules.
- [`examples/no-download-preflight/README.md`](examples/no-download-preflight/README.md)
  — deterministic synthetic demo.

## License

`moe-cache-lab` is licensed under the **Apache License 2.0**. See
[`LICENSE`](LICENSE).
