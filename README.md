# moe-cache-lab

**Trace-driven pre-flight analysis for MoE routing, expert caching, and offloading research.**

`moe-cache-lab` is a correctness-first, **trace-driven MoE pre-flight analysis toolkit**
and offline research tool that sits between **routing observation** and **runtime
offloading engineering**. Give it a canonical MoE routing trace and it can
describe routing evidence, replay compatible traces through explicit byte-cache
simulations, estimate serialized transfer service under caller-supplied
assumptions, and package artifacts for reproducibility.

It is designed to answer a bounded question before deeper runtime work:

> **What does the observed routing trace establish about locality and simulated
> cache behavior, and what does it still leave unmeasured?**

Current package version: **0.7.0**.

| | |
| --- | --- |
| **Input** | A canonical, versioned MoE routing trace from a supported collector or another conforming producer |
| **Output** | Routing evidence, compatible byte-cache simulations, transfer-sensitivity estimates, and deterministic reports/bundles |
| **Base install** | Python 3.10+; offline trace analysis does not require PyTorch or Transformers |
| **Current release** | `0.7.0` |

```text
model / runtime / producer
          |
          v
canonical routing trace
          |
          v
strict read + validation
          |
          v
   routing analysis
          |
          +----------------------+
          |                      |
          v                      v
compatible pre-flight       reproducibility
cache / transfer model          bundle
```

The canonical JSONL trace is the interoperability boundary. Invalid chronology
is rejected rather than sorted or repaired, and encoder/decoder identity is not
flattened.

## One bounded result, shown without a performance claim

A tracked V0.7 observational smoke used the pinned
`google/switch-base-8` revision
`92fe2d22b024d9937146fe097ba3d3a7ba146e1b` on CPU. Native post-capacity routing
was observed without modifying routing decisions. One version-aware LRU replay
then applied an eight-expert byte capacity to that trace.

| Quantity | Result | Evidence class |
| --- | ---: | --- |
| Routing events | 156 assigned / 0 unassigned | **MEASURED routing observation** |
| Stage-qualified expert objects | 96 | model/trace structure |
| Parameter payload per expert | 18,874,368 bytes | parameter payload accounting; **not physical residency** |
| LRU cache capacity | 150,994,944 bytes | caller-selected simulation input |
| Cache hits / misses | 89 / 67 | **SIMULATED** |
| Evictions | 59 | **SIMULATED** |
| Demand-load bytes | 1,264,582,656 | **SIMULATED cache-model accounting** |
| Peak/final resident bytes | 150,994,944 | **SIMULATED cache-model accounting** |

This example demonstrates that one real stage-qualified Switch routing trace can
enter the portable offline cache model without encoder/decoder aliasing. It does
**not** establish physical GPU residency, real H2D traffic, end-to-end latency,
throughput, tokens/sec, speedup, policy optimality, or workload
representativeness. See [`V07_SWITCH_SMOKE.md`](V07_SWITCH_SMOKE.md) for the full
provenance and limitations.

## Try the no-model path

The tracked synthetic example uses fictional hardware assumptions and requires
no model download:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .

moe-cache-lab analyze examples\no-download-preflight\trace.jsonl `
  --preflight-config examples\no-download-preflight\preflight-config.json `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

The resulting cache rows are **SIMULATED** and transfer-service rows are
**ESTIMATED** from explicit assumptions. Follow the
[canonical no-model evidence walkthrough](WALKTHROUGH.md) for the executable
evidence walkthrough.

## Evidence boundaries

The project keeps evidence classes separate:

- **Routing:** descriptive, trace-derived output. Treat it as **MEASURED** only
  when trace provenance establishes measurement.
- **Cache:** **SIMULATED** event-atomic replay, never native runtime residency.
- **Transfer service:** **ESTIMATED** from simulated demand loads and explicit
  bandwidth/setup assumptions.
- **Bounded constructed CPU copy validation:** a **MEASURED test-replay CPU
  copy** result only. It did not validate native inference caching, DRAM/H2D/GPU
  movement, or performance.

A SHA-256 in an experiment bundle proves integrity of exact bytes. It does not
prove scientific correctness, representative coverage, physical residency,
speedup, latency, throughput, or an optimal policy/capacity.

> **This project is not an inference accelerator.** It does not implement
> expert swapping/offloading or physical GPU residency management.

## What the current release supports

The repository and package include:

- strict canonical routing trace v1/v2 readers and schemas;
- v1 layer-qualified and v2 encoder/decoder stage-qualified expert identity;
- explicit capacity-unassigned v2 events;
- deterministic offline routing locality analysis;
- version-aware byte-cache simulation for canonical v1/v2 traces;
- LRU/LFU byte-capacity replay with event-atomic working sets;
- exact serialized transfer-service sensitivity under explicit assumptions;
- v1 pre-flight configuration/reporting and cache lifecycle analysis;
- deterministic experiment-bundle creation and verification;
- one built-in public Granite/Transformers collection path;
- one narrowly validated pinned SwitchTransformers v2 observation path;
- tracked historical evidence and negative findings.

The public `analyze --preflight-config` orchestration in v0.7 remains **v1-only**
because its expert-size configuration is layer-qualified. Library-level v2
byte-cache simulation exists, but the public stage-qualified v2 pre-flight
pipeline is not part of v0.7. A v2 trace supplied with the legacy pre-flight flag
fails explicitly rather than flattening stage identity.

The Switch result above is one bounded model/revision/path validation. It is not
a broad Transformers, MoE-family, or runtime compatibility claim.

## Canonical trace contract

External producers can target the strict versioned JSONL contract in
[`TRACE_FORMAT.md`](TRACE_FORMAT.md):

- **v1** preserves layer-qualified decoder-only routing;
- **v2** preserves encoder/decoder stage-qualified routing and explicit
  capacity-unassigned events;
- invalid chronology is rejected without sorting or repair;
- expert IDs are identifiers, not semantic labels;
- native routing/dispatch observation remains authoritative.

The base installation imports and analyzes canonical v1/v2 traces without
PyTorch or Transformers. Model collection is optional and model-specific. The
built-in public collector remains **Granite/Transformers-specific**; other
producers must satisfy the canonical trace contract independently.

## Typical workflow

### 1. Obtain or import a canonical trace

Use an existing canonical v1/v2 JSONL trace from any producer that satisfies the
strict contract.

For the built-in Granite/Transformers collector, install the optional extra:

```powershell
python -m pip install -e ".[granite]"

moe-cache-lab collect `
  --prompt "Explain why cache locality matters." `
  --max-new-tokens 4 `
  --output artifacts\granite-trace.jsonl
```

The built-in `collect` command is Granite-specific; it is not a general
model-family collector. If your environment has a certificate/TLS problem, fix
the certificate chain rather than disabling TLS verification.

### 2. Analyze routing evidence offline

```powershell
moe-cache-lab analyze artifacts\routing-trace.jsonl
```

For canonical v2, stage-qualified coverage can be written to Markdown and JSON:

```powershell
moe-cache-lab analyze artifacts\switch-trace-v2.jsonl `
  --workload-id evaluation-01 `
  --top-k 2 1 4 `
  --output artifacts\routing-evidence.md `
  --json-output artifacts\routing-evidence.json
```

V2 locality is opt-in; no top-k ranks are inferred.

### 3. Run compatible pre-flight analysis

The v0.7 public pre-flight path is v1-only:

```powershell
moe-cache-lab analyze examples\no-download-preflight\trace.jsonl `
  --preflight-config examples\no-download-preflight\preflight-config.json `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

See [`PREFLIGHT.md`](PREFLIGHT.md) and
[`examples/no-download-preflight/README.md`](examples/no-download-preflight/README.md).

For supported v1 lifecycle inputs:

```powershell
moe-cache-lab analyze-lifecycle artifacts\corpus\manifest-v1.json `
  --preflight-config preflight-config.json `
  --output artifacts\cache-lifecycle.md `
  --json-output artifacts\cache-lifecycle.json
```

`cold_per_workload` starts each workload empty. `persistent_sequence` preserves
one simulated cache through the declared workload order. Neither mode measures
native runtime residency.

### 4. Create and verify a reproducibility bundle

```powershell
moe-cache-lab bundle-create `
  --experiment-id evaluation-01 `
  --config artifacts\experiment-config.json `
  --report-json artifacts\routing-evidence.json `
  --report-markdown artifacts\routing-evidence.md `
  --embed-trace routing artifacts\routing-trace.jsonl `
  --output-dir artifacts\evaluation-01-bundle

moe-cache-lab bundle-verify artifacts\evaluation-01-bundle
```

External trace references can also be metadata-only and are never fetched.
Embedded traces may contain prompts/generated text, so private traces should not
be published blindly.

## Who this is for

Researchers and engineers who have, or can collect, a compatible MoE routing
trace and want a reproducible offline view of routing locality, byte-cache
sensitivity, and simple transfer-cost assumptions **before** investing in a
runtime integration.

## When not to use it

Do not use a pre-flight report as proof that a deployment will be faster, as a
GPU residency monitor, or as a substitute for runtime profiling. Decisions that
depend on real transfer overlap, allocator behavior, kernel scheduling,
physical data movement, latency, throughput, or tokens/sec require separate
measurement in the target runtime.

## Historical evidence and negative findings

Historical workflows remain available for reproduction. The V0.4 experiment
measured approximately **2.09% mean added CPU wall time** for the routing
observer on its pinned workload; router selections stayed identical and the
maximum selected-probability delta was `0.0`. This is observer overhead, not
acceleration. See [`V04_RESULTS.md`](V04_RESULTS.md).

Stage 1 found simulated locality that justified a read-only runtime-feasibility
investigation under its recorded assumptions. The subsequent Stage 2 inspection
did **not** find a supported, meaningful route to real per-expert offloading on
the tested local environment, so later runtime stages were deferred rather than
replaced with an unsupported demonstration. See
[`STAGE1_RESULTS.md`](STAGE1_RESULTS.md) and
[`STAGE2_FEASIBILITY.md`](STAGE2_FEASIBILITY.md).

The original historical V0.1 cache results are invalid and must not be cited.
They aliased equal numerical expert IDs across layers and replayed selected
experts individually. The corrected implementation uses qualified expert
identity and atomic routing-event bundles.

## Privacy and development disclosure

Routing traces can include source prompts and experiment metadata. Treat traces
from private prompts as potentially sensitive data.

AI-assisted coding and review tools were used during development. Automated
tests, deterministic evidence checks, and documented limitations reduce some
failure modes but do not guarantee that the software or conclusions are
error-free. Independently validate results before relying on them for your own
engineering or research claims.

## Validation

For source-tree tests on PowerShell:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

GitHub CI does not download Hugging Face models. Release validation runs source
compilation, dependency checks, the full unit/evidence suite, wheel/sdist build,
isolated package installation, and tracked no-download snapshot checks.
Hardware-specific probing and performance benchmarks remain separate workflows.

## Documentation map

- [`WALKTHROUGH.md`](WALKTHROUGH.md) — canonical no-model workflow and evidence interpretation.
- [`TRACE_FORMAT.md`](TRACE_FORMAT.md) — canonical routing-trace contract.
- [`PREFLIGHT.md`](PREFLIGHT.md) — pre-flight configuration and transfer sensitivity.
- [`examples/no-download-preflight/README.md`](examples/no-download-preflight/README.md) — synthetic no-download demo.
- [`V07_RELEASE_NOTES.md`](V07_RELEASE_NOTES.md) — v0.7 scope and limitations.
- [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) — project purpose and invariants.
- [`DECISIONS.md`](DECISIONS.md) — architecture decisions and deferred scope.
- [`STAGE1_RESULTS.md`](STAGE1_RESULTS.md) — runtime-fidelity simulation results.
- [`STAGE2_FEASIBILITY.md`](STAGE2_FEASIBILITY.md) — local runtime/offload feasibility decision.
- [`V04_RESULTS.md`](V04_RESULTS.md) — measured CPU observer overhead.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution and reproducibility rules.

## Scope that is not implemented

The project does not provide:

- real expert swapping/offloading or GPU residency management;
- GPU acceleration or a runtime speedup claim;
- measured physical transfer timing, end-to-end latency, throughput, or tokens/sec;
- hardware auto-detection or built-in vendor profiles;
- compute/transfer overlap modeling or asynchronous prefetch;
- a general multi-family collector framework or public Switch collection CLI;
- vLLM/llama.cpp/MoE-Infinity integration;
- SSD offload;
- learned expert prediction/prefetch;
- recommendation thresholds or a cacheability score;
- native-router modification.

Those remain future research directions only when a supported runtime and direct
evidence justify them.

## License

`moe-cache-lab` is licensed under the **Apache License 2.0**. See
[`LICENSE`](LICENSE).
