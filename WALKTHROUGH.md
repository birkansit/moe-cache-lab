# Canonical no-model evidence walkthrough

This walkthrough uses the repository's one tracked no-download pre-flight
fixture. The trace is **synthetic**, and every configured hardware value is a
**fictional caller-supplied assumption**. It is useful for learning the public
workflow and reproducing exact report bytes; it is not a model benchmark or a
representative production workload.

The interpretation rule throughout is:

- routing is trace-derived and is **MEASURED** only when trace provenance
  establishes actual measurement;
- cache outcomes are **SIMULATED**;
- serialized transfer-service results are **ESTIMATED** from explicit
  assumptions; and
- a successful integrity check does not change any of those evidence classes.

## 1. Install only the offline base package

From the repository root, create an environment and install the base package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Do not install the `granite` or `switch` extras for this walkthrough. The base
package has no Torch or Transformers dependency, and none of the commands below
loads or downloads a model.

## 2. Generate and identify the tracked report

Run the canonical v1 pre-flight command from the repository root:

```powershell
moe-cache-lab analyze examples\no-download-preflight\trace.jsonl `
  --preflight-config examples\no-download-preflight\preflight-config.json `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

Verify the exact bytes:

```powershell
Get-FileHash -Algorithm SHA256 `
  artifacts\preflight-report.md, `
  artifacts\preflight-report.json
```

Expected SHA-256 values:

- Markdown: `219751d3a63b97177f6d3133dd2e7b69fc941eed2416e72568125f4e2936542d`
- JSON: `464638a63f66e2c8797f4e7c59efea226f30050032ede5138255f675d0ff4a42`

Matching these values proves byte-for-byte reproduction of the tracked report.
It does **not** prove scientific correctness, representative workload coverage,
runtime behavior, speedup, or physical expert residency.

## 3. Read provenance before routing numbers

The generated report identifies:

- model ID `synthetic/moe-preflight-demo`;
- capture method `synthetic-offline-demo`;
- model revision `synthetic-fixture-v1`;
- 8 routing events and 16 layer-qualified expert assignments;
- 4 prompt/prefill events with 8 assignments; and
- 4 generated/decode events with 8 assignments.

The fixture references 6 unique `(layer, expert_id)` objects. These are exact
descriptive facts about the synthetic trace. They are not evidence of routing
in a measured production or model workload. A Transformers version is
unavailable in this generated pre-flight output, so do not infer one.

This v1 pre-flight report does not emit the separate B1 evidence-coverage and
warning surface; that field is unavailable here. When an analysis does emit
coverage or warnings, they describe which evidence was supplied and where it is
thin, mixed, or missing. Their presence or absence is not a universal quality,
confidence, or representativeness score.

**Supported reading:** “The supplied synthetic trace contains 8 events and 16
layer-qualified assignments.”

**Not supported:** “These routing frequencies represent production traffic.”

## 4. Read cache rows as SIMULATED cells

The simulator consumes each routing event atomically under the supplied expert
sizes, policies, and byte capacities. For the exact tested LRU cells:

| capacity bytes | hits | misses | demand-load bytes | evictions | evicted bytes | peak/final resident bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 2 | 14 | 45 | 12 | 37 | 8 / 8 |
| 12 | 5 | 11 | 36 | 8 | 26 | 12 / 10 |

At capacity 12, the LFU cell also has 5 hits, 11 misses, and 8 evictions, but
its simulated demand-load and evicted-byte totals are 33 and 23. That difference
is possible because expert objects have different supplied sizes even when the
number of misses is equal.

Hits, misses, demand-load bytes, eviction counts/bytes, resident-byte values,
and policy outcomes in these rows are all **SIMULATED**. A lower value in one
cell does not demonstrate native inference caching, actual GPU residency, real
data movement, or deployment speedup.

**Supported reading:** “At tested capacity 12, this simulation produced 11
misses for both LRU and LFU, with 36 LRU demand-load bytes and 33 LFU
demand-load bytes.”

**Not supported:** “LFU is the best deployment policy.”

## 5. Read transfer service as ESTIMATED

The `fictional-fast-link` assumes 20 bytes/second payload bandwidth and
50,000,000 nanoseconds of setup per simulated logical load. At capacity 12,
the generated table reports:

- LRU estimated serialized service: `47/20` seconds; and
- LFU estimated serialized service: `11/5` seconds.

Under only those fixed assumptions, the LFU estimate is `3/20` second lower
than the LRU estimate. This is arithmetic over **SIMULATED** demand loads and
fictional caller-supplied values, not measured hardware behavior or a speedup
prediction.

The current equation is serialized/no-overlap. It excludes model compute,
real compute/transfer overlap, concurrency and runtime scheduling, allocator
behavior, synchronization/protocol effects, end-to-end latency, real GPU/H2D
traffic, and D2H effects unless explicitly modeled. This v1 configuration does
not charge simulated eviction as D2H writeback.

**Supported reading:** “Under the fictional-fast-link assumptions, the tested
capacity-12 estimates are `47/20` seconds for LRU and `11/5` seconds for LFU.”

**Not supported:** “LFU will make inference `3/20` second faster.”

## 6. Describe sensitivity without choosing a winner

Compare only emitted cells at tested points:

- For LRU, moving from tested capacity 8 to tested capacity 12 changes
  simulated misses from 14 to 11 (delta `-3`) and simulated demand-load bytes
  from 45 to 36 (delta `-9`).
- For LFU, the same tested capacity change moves simulated demand-load bytes
  from 45 to 33 (delta `-12`).
- At tested capacity 12, LFU minus LRU demand-load bytes is `-3`, while both
  cells have 11 simulated misses.

These are descriptive differences between explicitly tested cells. They do not
interpolate behavior between capacities, identify an optimum, rank policies, or
recommend a deployment setting. Workload min/max/range fields are unavailable
in this single-trace pre-flight report; do not invent them.

**Supported reading:** “Between tested capacities 8 and 12, simulated LRU
misses changed by `-3` under this fixed trace and configuration.”

**Not supported:** “Capacity 12 is optimal” or “the curve proves what happens
at an untested capacity.”

## 7. Keep the CPU copy validation in its narrow context

The bounded constructed CPU copy validation froze one **SIMULATED** LRU
prediction before execution. A separate constructed CPU physical-copy executor
then replayed the canonical events. The exact result was **AGREEMENT**: the
replay's actual CPU copy operations and copied-content equality were
**MEASURED test-replay observations** matching the frozen accounting.

That supports differential accounting consistency for that constructed replay.
It does not validate native inference caching, physical residency, DRAM
traffic, PCIe/H2D or GPU behavior, latency, throughput, speedup, memory savings,
or production offloading. Logical operand bytes in that test are not a
measurement of DRAM traffic.

## 8. Bundle the exact artifacts, then verify them

Create a deterministic integrity bundle from the generated artifacts and the
explicitly embedded synthetic trace:

```powershell
moe-cache-lab bundle-create `
  --experiment-id no-download-preflight `
  --config examples\no-download-preflight\preflight-config.json `
  --report-json artifacts\preflight-report.json `
  --report-markdown artifacts\preflight-report.md `
  --embed-trace synthetic examples\no-download-preflight\trace.jsonl `
  --output-dir artifacts\no-download-preflight-bundle

moe-cache-lab bundle-verify artifacts\no-download-preflight-bundle
```

The bundle records exact artifact hashes and sizes plus bounded tool,
interpreter, platform, and explicitly supplied runtime provenance. Embedding a
trace is deliberate: real traces may contain private prompt or generated text.
For private or large traces, `--external-trace TRACE_ID REFERENCE SHA256`
records a metadata-only immutable reference; creation and verification never
fetch it.

Verification checks integrity without repairing content. Do **not** hard-code
one universal expected bundle-manifest SHA-256: environment/tool provenance is
part of the bundle, so the manifest hash may legitimately differ between
environments. Bundle integrity does not upgrade routing provenance, SIMULATED
cache evidence, ESTIMATED transfer evidence, scientific correctness,
representativeness, or runtime validity.

## 9. If the input is trace v2

This is a stage-safety note, not a second walkthrough. With an existing strict
canonical v2 trace, use the offline analyzer directly:

```powershell
moe-cache-lab analyze TRACE_V2 `
  --workload-id evaluation-01 `
  --top-k 1 2 4
```

Encoder and decoder experts retain `(routing_stage, layer)` identity; never
flatten them or manufacture numerical layer offsets. `--top-k` is
caller-explicit, preserves caller order, and has no hidden or recommended
default. Normalized configured-universe entropy and cumulative top-k shares are
descriptive routing diagnostics, not recommendations.

The legacy `--preflight-config` path intentionally rejects trace v2 because its
expert-size keys are v1 layer-qualified rather than stage-qualified. Do not
bypass that boundary by flattening or offsetting stages. This project does not
provide a public Switch collection command or a v2 cache/preflight pipeline.
