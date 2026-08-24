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

Current package version: **0.8.0**. Historical v0.2/v0.3/v0.4 evidence remains
preserved with its original interpretation and hashes.

| | |
| --- | --- |
| **Input** | A canonical, versioned MoE routing trace from a supported collector or another conforming producer |
| **Output** | Routing evidence, compatible byte-cache simulations, transfer-sensitivity estimates, and deterministic reports/bundles |
| **Base install** | Python 3.10+; offline trace validation and analysis do not require PyTorch or Transformers |
| **Current release** | `0.8.0` |

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
flattened. Routing-trace and pre-flight-config version numbers are independent;
the supported combinations are defined in [`COMPATIBILITY.md`](COMPATIBILITY.md).
External producer validity, semantic mapping, and non-interference are separate
claims governed by [`PRODUCER_CONFORMANCE.md`](PRODUCER_CONFORMANCE.md).

The base installation validates, imports, and analyzes canonical v1/v2 traces
without PyTorch or Transformers. Model collection is optional and model-specific.
The built-in public collector remains Granite/Transformers-specific.

> **This project is not an inference accelerator.** It does not implement expert
> swapping/offloading or physical GPU residency management. It observes and
> analyzes routing, simulates cache behavior, and estimates a simple serialized
> transfer-service cost under explicit assumptions.

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

## Canonical current-use path

```text
produce/import canonical JSONL
    -> validate the canonical v1/v2 file offline
    -> analyze descriptive routing evidence offline
    -> run pre-flight only for a compatible trace/config pair
    -> create and verify a reproducibility bundle
```

Canonical JSONL is the external interoperability boundary. A third-party
producer may write it directly and does not need the internal `ProducerResult`
type. `validate-trace` establishes canonical-file validity only; producer
semantic mapping and non-interference require separate evidence.

Follow the executable [canonical no-model evidence walkthrough](WALKTHROUGH.md)
to reproduce the tracked report and learn which interpretations its routing,
SIMULATED cache, and ESTIMATED transfer fields do and do not support.
The packaged
[`examples/external-producer-no-model/`](examples/external-producer-no-model/)
workflow separately demonstrates how an unrelated standard-library producer
can emit canonical JSONL and run the complete offline validation, analysis,
stage-qualified pre-flight, and bundle path.

The project keeps evidence classes separate throughout this path:

- **Routing:** descriptive, trace-derived output. Treat it as **MEASURED** only
  when trace provenance establishes measurement.
- **Cache:** **SIMULATED** event-atomic replay, never native runtime residency.
- **Transfer service:** **ESTIMATED** from simulated loads and explicit caller
  assumptions.
- **Bounded constructed CPU copy validation:** a **MEASURED test-replay CPU
  copy** result only. It did not validate native inference caching,
  DRAM/H2D/GPU movement, or performance.

A SHA-256 in an experiment bundle proves integrity of exact bytes. It does not
prove scientific correctness, representative coverage, physical residency,
speedup, latency, throughput, or an optimal policy/capacity.

The external-runtime research gate inspected exact vLLM and llama.cpp
revisions and ended in a bounded **NO-GO** on the recorded no-model Windows
boundary. That **SOURCE-READ** result is not a universal rejection of either
runtime and established no runtime interoperability. See
[`EXTERNAL_RUNTIME_INTEROP.md`](EXTERNAL_RUNTIME_INTEROP.md).

## What does it do?

A Mixture-of-Experts model contains many expert weight blocks, while its router
selects only some of them for each token. `moe-cache-lab` can:

1. strictly read canonical routing trace v1/v2 without repairing chronology;
2. validate canonical traces with deterministic human or JSON output;
3. analyze v1 routing or v2 stage-qualified evidence offline;
4. replay compatible traces through byte-capacity LRU/LFU simulations;
5. compare explicit cold and persistent lifecycle scenarios for supported v1
   pre-flight inputs;
6. estimate serialized transfer service from explicit assumptions;
7. create/verify deterministic experiment-integrity bundles with embedded or
   metadata-only external trace references;
8. retain historical benchmark/evidence workflows for reproduction.

A simulated or estimated pre-flight result is not a measured speedup, measured
GPU transfer result, end-to-end latency result, throughput result, tokens/sec
result, or physical residency result.

## Current status

The repository and package include:

- strict canonical routing trace v1/v2 readers and schemas;
- one narrowly validated pinned SwitchTransformers collection path for v2;
- version-aware evidence coverage and explicit-only cross-model locality;
- deterministic offline experiment-bundle creation and verification;
- deterministic routing locality analysis;
- prompt/prefill and generated/decode phase-layer summaries;
- per-layer and phase-layer frequency concentration metrics;
- version-aware byte-capacity LRU/LFU simulation for canonical v1 and v2,
  preserving layer-qualified v1 and stage-qualified v2 identities;
- exact serialized transfer-cost sensitivity under explicit bandwidth, setup-latency, and transfer-operation assumptions;
- explicit cold-per-workload and persistent-sequence lifecycle simulation;
- deterministic capacity, policy, and workload sensitivity summaries;
- exact aggregate and per-workload lifecycle transfer-service estimates;
- deterministic offline `analyze --preflight-config` reporting for compatible
  v1/config-v1-v2 and v2/config-v3 pairs;
- tracked synthetic no-download demos under
  [`examples/no-download-preflight/`](examples/no-download-preflight/) and
  [`examples/no-download-stage-qualified-preflight/`](examples/no-download-stage-qualified-preflight/);
- a packaged, deterministic, no-model external-producer workflow under
  [`examples/external-producer-no-model/`](examples/external-producer-no-model/).

The V0.4 experiment measured approximately **2.09% mean added CPU wall time**
for the routing observer on its pinned workload. Router selections stayed
identical and the maximum selected-probability delta was `0.0`. This is observer
overhead, not acceleration. See [`V04_RESULTS.md`](V04_RESULTS.md).

Stage 2 concluded that a real per-expert residency/offload prototype was not
justified on the tested local environment: the installed PyTorch path was
CPU-only, the RX 6650 XT did not provide a supported PyTorch/HIP path for this
workflow, and Granite stores each layer's experts in packed parameters without
a supported per-expert residency interface. This is a local **no-go**, not a
claim that expert caching is impossible elsewhere. See
[`STAGE2_FEASIBILITY.md`](STAGE2_FEASIBILITY.md).

## Who this is for

This toolkit is for researchers and engineers who have, or can collect, a compatible MoE routing trace and want a reproducible pre-flight view of routing locality, byte-cache sensitivity, and simple transfer-cost assumptions before investing in runtime integration.

## When not to use it

Do not use the pre-flight report as proof that a deployment will be faster, as a GPU residency monitor, or as a substitute for runtime profiling. Decisions that depend on real transfer overlap, allocator behavior, kernel scheduling, physical data movement, latency, throughput, or tokens/sec require separate measurement in the target runtime.

## Quick start: offline canonical path

Base offline-analysis requirements:

- Python 3.10 or newer;
- no PyTorch or Transformers installation is required.

Create an environment and install the base package from this checkout. This
path needs neither PyTorch nor Transformers:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### 1. Obtain or import a canonical trace

Use an existing canonical v1/v2 JSONL trace from any producer that satisfies
the strict contract. The input path is not serialized into descriptive output.
For an executable example that imports no package internals, Torch, or
Transformers, follow
[`examples/external-producer-no-model/README.md`](examples/external-producer-no-model/README.md).

Alternatively, use the built-in Granite/Transformers collector by installing
the explicit Granite extra. It retains the validated PyTorch 2.12.0 and
Transformers 5.12.0 pins and requires enough RAM for the selected model:

```powershell
python -m pip install -e ".[granite]"
```

Without that extra, model-specific collection commands exit with a bounded
message pointing to `moe-cache-lab[granite]`; offline commands remain
available. `collect` remains Granite-specific—it is not a general model-family
collector.

If your environment has a certificate/TLS problem, fix the certificate chain.
Do **not** disable TLS verification as a workaround.

```powershell
moe-cache-lab collect `
  --prompt "Explain why cache locality matters." `
  --max-new-tokens 4 `
  --output artifacts\granite-trace.jsonl
```

### 2. Validate the canonical trace offline

Validation uses the strict version-aware reader and performs no analysis,
simulation, collection, or model execution:

```powershell
moe-cache-lab validate-trace artifacts\routing-trace.jsonl
moe-cache-lab validate-trace artifacts\routing-trace.jsonl --json
```

The official synthetic valid/invalid corpus is documented in
[`examples/trace-validation-fixtures/README.md`](examples/trace-validation-fixtures/README.md).
Passing validation proves canonical-file validity only; it does not prove that
the producer observed the correct native boundary or was non-interfering.

### 3. Analyze descriptive routing evidence offline

`analyze` dispatches strictly by the canonical trace version:

```powershell
moe-cache-lab analyze artifacts\routing-trace.jsonl
```

- v1 preserves the established routing-only Markdown/JSON behavior.
- v2 reports encoder/source, decoder/prompt, and decoder/generated coverage
  with `(routing_stage, layer)` identities intact.
- v2 locality is opt-in only; no top-k ranks are assumed:

```powershell
moe-cache-lab analyze artifacts\switch-trace-v2.jsonl `
  --workload-id evaluation-01 `
  --top-k 2 1 4 `
  --output artifacts\routing-evidence.md `
  --json-output artifacts\routing-evidence.json
```

For v2 pre-flight, pair the trace with strict config v3 so expert sizes remain
stage-qualified. Config v1/v2 remains v1-only; incompatible pairs fail
explicitly instead of flattening encoder/decoder identity. `--top-k` is not
combined with v2 pre-flight analysis.

### 4. Compare/report where a pre-flight contract applies

The tracked no-download fixture is synthetic and uses fictional hardware
assumptions:

```powershell
moe-cache-lab analyze examples\no-download-preflight\trace.jsonl `
  --preflight-config examples\no-download-preflight\preflight-config.json `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

See [`PREFLIGHT.md`](PREFLIGHT.md) and [`examples/no-download-preflight/README.md`](examples/no-download-preflight/README.md).

The separate synthetic stage-qualified example exercises trace v2 with config
v3 and preserves encoder/decoder expert identity end to end:

```powershell
moe-cache-lab analyze examples\no-download-stage-qualified-preflight\trace.jsonl `
  --preflight-config examples\no-download-stage-qualified-preflight\preflight-config.json `
  --workload-id synthetic-stage-qualified-demo `
  --output artifacts\stage-qualified-preflight-report.md `
  --json-output artifacts\stage-qualified-preflight-report.json
```

See
[`examples/no-download-stage-qualified-preflight/README.md`](examples/no-download-stage-qualified-preflight/README.md).

For v1 cache lifecycle analysis, `cold_per_workload` starts each workload empty
while `persistent_sequence` carries one simulated cache through declared order:

```powershell
moe-cache-lab analyze-lifecycle artifacts\corpus\manifest-v1.json `
  --preflight-config preflight-config.json `
  --output artifacts\cache-lifecycle.md `
  --json-output artifacts\cache-lifecycle.json
```

These rows remain **SIMULATED** cache outcomes and **ESTIMATED** serialized
transfer-service results. They do not establish physical movement or speedup.

### 5. Create and verify a reproducibility bundle

Bundle supplied config/report bytes and explicitly selected trace references:

```powershell
moe-cache-lab bundle-create `
  --experiment-id evaluation-01 `
  --config artifacts\experiment-config.json `
  --report-json artifacts\routing-evidence.json `
  --report-markdown artifacts\routing-evidence.md `
  --embed-trace routing artifacts\routing-trace.jsonl `
  --external-trace private-eval https://example.invalid/private.jsonl SHA256_HEX `
  --output-dir artifacts\evaluation-01-bundle

moe-cache-lab bundle-verify artifacts\evaluation-01-bundle
```

External references are metadata-only and never fetched. Embedded traces are
copied only when explicitly requested and may contain private prompt/generated
text. Bundle hashes prove byte integrity, not evidence quality.

## Historical compatibility and reproduction workflows

Existing commands and evidence remain available. For example, the historical
count-capacity report is reproduced with:

```powershell
moe-cache-lab benchmark artifacts\granite-trace.jsonl `
  --capacity 32 `
  --output artifacts\granite-report.md
```

The resulting hit/miss and transfer numbers describe the historical simulator,
not actual GPU memory movement.

## Reproduce the tracked historical analysis

Validated evidence is stored in the repository. Existing evidence should
normally be validated or replayed rather than recollected merely to reproduce a
report.

Examples:

```powershell
moe-cache-lab benchmark-suite `
  results\v0.2-corpus-v1\manifest-v1.json `
  --capacities 8 16 32 64 128 256 `
  --output artifacts\v0.2-report.md

moe-cache-lab validate-stage1 `
  results\stage1-runtime-fidelity-v1\stage1-set-manifest.json

moe-cache-lab validate-v04 `
  results\v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980\v04-set.json
```

## Result snapshot

The Stage 1 experiment found enough simulated locality to justify a read-only
runtime-feasibility investigation. In the conservative grouped view, LFU
estimated transfers fell from **29,937** at capacity 32 to **16,277** at
capacity 256, and capacity-256 decode-only prompt hit rates ranged from roughly
**30.1% to 48.8%**. These are simulation results and do not establish runtime
speedup. Full details are in [`STAGE1_RESULTS.md`](STAGE1_RESULTS.md).

The subsequent Stage 2 inspection did not find a supported, meaningful route to
real per-expert offloading on the tested machine, so Stages 3-5 were deliberately
deferred rather than replaced with an unsupported demonstration.

## Correctness notes

The original historical V0.1 cache results are invalid and must not be cited.
They treated equal numerical expert IDs in different layers as the same cache
object and replayed selected experts individually. The corrected implementation
uses layer-qualified expert identities and atomic routing-event bundles.

The model's native router remains authoritative. Observation hooks do not change
weights, router decisions, or selected experts.

Expert IDs are identifiers, not semantic labels. The project does not claim
that a particular expert means "math", "coding", "history", or another topic.

Granite prefill groups work differently from the simplest per-token simulation.
The project therefore retains both its original token/layer view and a more
conservative grouped-prefill sensitivity view; neither is presented as verified
physical GPU residency.

## Privacy and trace data

A routing trace can include the source prompt and experiment metadata. Treat
traces collected from private prompts as potentially sensitive data and do not
publish them blindly.

The tracked benchmark corpus and v0.5 no-download demo contain synthetic/general
content intended for reproducible testing.

## Development disclosure

AI-assisted coding and review tools were used during development. The repository
also includes automated tests, deterministic evidence checks, and documented
limitations, but neither AI assistance nor the test suite guarantees that the
software or conclusions are error-free. Treat this as research tooling and
independently validate results before relying on them for your own engineering
or research claims.

## Validation

For source-tree tests on PowerShell:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

Portable GitHub CI is intentionally offline with respect to Hugging Face model
downloads. Release validation runs source compilation, dependency checks, the
full unit/evidence suite, wheel/sdist build, an isolated wheel install, and the
tracked no-download pre-flight snapshot check. Hardware-specific HIP probing
and performance benchmarks remain separate local workflows.

The historical `0.4.0rc1` engineering checkpoint and evidence index remain in
[`RELEASE_CANDIDATE.md`](RELEASE_CANDIDATE.md).

## Documentation map

- [`PRODUCER_CONFORMANCE.md`](PRODUCER_CONFORMANCE.md) — normative external
  producer validity, semantic-mapping, non-interference, and NO-GO contract.
- [`COMPATIBILITY.md`](COMPATIBILITY.md) — canonical cross-format versioning,
  compatibility, rejection, and external-producer boundary.
- [`WALKTHROUGH.md`](WALKTHROUGH.md) — canonical no-model workflow and
  evidence-interpretation guide.
- [`PREFLIGHT.md`](PREFLIGHT.md) — offline pre-flight configuration, transfer-operation sensitivity, and interpretation.
- [`examples/no-download-preflight/README.md`](examples/no-download-preflight/README.md) — tracked synthetic no-download demo.
- [`examples/no-download-stage-qualified-preflight/README.md`](examples/no-download-stage-qualified-preflight/README.md) — tracked synthetic stage-qualified no-download demo.
- [`examples/external-producer-no-model/README.md`](examples/external-producer-no-model/README.md) — packaged standalone-producer interoperability workflow.
- [`V05_RELEASE_NOTES.md`](V05_RELEASE_NOTES.md) — v0.5 release summary and limitations.
- [`V06_RELEASE_NOTES.md`](V06_RELEASE_NOTES.md) — v0.6 release scope and limitations.
- [`V07_RELEASE_NOTES.md`](V07_RELEASE_NOTES.md) — v0.7 release scope and limitations.
- [`V08_RELEASE_NOTES.md`](V08_RELEASE_NOTES.md) — v0.8 release scope and limitations.
- [`EXTERNAL_RUNTIME_INTEROP.md`](EXTERNAL_RUNTIME_INTEROP.md) — bounded
  vLLM/llama.cpp source-inspection gate and exact-revision NO-GO.
- [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) — project purpose and key invariants.
- [`DECISIONS.md`](DECISIONS.md) — architecture decisions and deliberately
  rejected/deferred scope.
- [`PLAN.md`](PLAN.md) — completed V0.1-V0.4 engineering plan.
- [`STAGE1_RESULTS.md`](STAGE1_RESULTS.md) — runtime-fidelity simulation results.
- [`STAGE2_FEASIBILITY.md`](STAGE2_FEASIBILITY.md) — local runtime/offload
  feasibility decision.
- [`V04_RESULTS.md`](V04_RESULTS.md) — measured CPU observer overhead result.
- [`RELEASE_CANDIDATE.md`](RELEASE_CANDIDATE.md) — historical `0.4.0rc1`
  checkpoint and evidence index.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution and reproducibility rules.

## Scope that is not implemented

The project does not provide:

- real expert swapping/offloading or GPU residency management;
- GPU acceleration or a runtime speedup claim;
- measured physical transfer timing, end-to-end latency, throughput, or tokens/sec;
- hardware auto-detection or built-in vendor profiles;
- compute/transfer overlap modeling or asynchronous prefetch;
- a general multi-family collector framework, public Switch collection CLI, or
  multiple inference backends;
- vLLM/llama.cpp/MoE-Infinity integration;
- SSD offload;
- learned expert prediction/prefetch;
- RAG;
- distributed inference;
- custom CUDA/ROCm kernels;
- recommendation thresholds or a cacheability score;
- native-router modification.

Those remain future research directions only if a supported runtime and measured
bottleneck justify them.

## License

`moe-cache-lab` is licensed under the **Apache License 2.0**. See
[`LICENSE`](LICENSE).
