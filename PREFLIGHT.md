# Offline pre-flight analysis

The pre-flight workflow analyzes an existing compatible routing trace without running a model. It combines descriptive routing locality, byte-capacity LRU/LFU simulation, and a simple serialized transfer-service sensitivity model under explicit caller-supplied assumptions.

The built-in collector is currently Granite/Transformers-specific. After trace creation, this pre-flight path is driven by the repository routing-trace format and core analysis APIs rather than by Granite model execution.

## Trace-version boundary

The established byte-cache pre-flight configuration and report pipeline is for
canonical routing trace **v1 only**. Its expert-size identities are
`(layer_id, expert_id)`. Existing v1 inputs and outputs remain supported without
semantic changes.

Canonical trace v2 is available through the version-aware descriptive
`analyze` path, but `analyze --preflight-config` intentionally rejects v2. A v2
expert identity is stage-qualified as `(routing_stage, layer, expert_id)`;
encoder and decoder layers must never be flattened together or given invented
numeric offsets. Stage-qualified v2 pre-flight simulation is outside the
current contract.

## Tracked no-download demo

A deterministic synthetic example is tracked at [`examples/no-download-preflight/`](examples/no-download-preflight/). It requires no model download, GPU, CUDA/ROCm, or vendor SDK:

```powershell
moe-cache-lab analyze examples\no-download-preflight\trace.jsonl `
  --preflight-config examples\no-download-preflight\preflight-config.json `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

The demo trace is synthetic and its transfer profiles are deliberately fictional assumptions. The tracked `expected.sha256` file pins the deterministic Markdown and JSON output bytes.

## Config shape

For another compatible trace, create a versioned JSON config such as:

```json
{
  "format": "moe-cache-lab.preflight-config",
  "format_version": 1,
  "expert_sizes": [
    {"layer_id": 0, "expert_id": 0, "size_bytes": 1048576},
    {"layer_id": 0, "expert_id": 1, "size_bytes": 1048576}
  ],
  "capacities_bytes": [1048576, 2097152],
  "policies": ["lru", "lfu"],
  "hardware_profiles": [
    {
      "name": "example-assumption",
      "h2d_payload_bandwidth_bytes_per_second": 12000000000,
      "setup_latency_ns_per_loaded_expert": 5000
    }
  ]
}
```

Run the existing `analyze` command with the config:

```powershell
moe-cache-lab analyze artifacts\routing-trace.jsonl `
  --preflight-config preflight-config.json `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

Expert sizes, H2D bandwidth, and setup latency are caller-supplied assumptions unless they have been measured or calibrated separately. The command does not infer them from a GPU, model, filename, runtime, or environment.

Config format version 1 preserves the v0.5 transfer model exactly: each
simulated logical expert demand load is one setup-bearing transfer operation.

## Transfer-operation sensitivity (config version 2)

Config format version 2 makes transfer granularity an explicit assumption. A
transfer-operation plan assigns a fixed positive number of modeled operations
to every simulated logical expert load. For example, this compares the legacy
one-operation assumption with a two-chunk assumption:

```json
{
  "format": "moe-cache-lab.preflight-config",
  "format_version": 2,
  "expert_sizes": [
    {"layer_id": 0, "expert_id": 0, "size_bytes": 6291456},
    {"layer_id": 0, "expert_id": 1, "size_bytes": 6291456}
  ],
  "capacities_bytes": [50331648],
  "policies": ["lru", "lfu"],
  "hardware_profiles": [
    {
      "name": "example-assumption",
      "h2d_payload_bandwidth_bytes_per_second": 12000000000,
      "setup_latency_ns_per_transfer_operation": 5000
    }
  ],
  "transfer_operation_plans": [
    {"name": "one-operation-per-load", "operations_per_logical_load": 1},
    {"name": "two-chunks-per-load", "operations_per_logical_load": 2}
  ]
}
```

The byte-cache simulation is run once per capacity/policy pair. Transfer plans
do not alter hits, misses, logical demand-load bytes, eviction accounting, or
residency. They change only the modeled setup-bearing operation count and its
resulting **ESTIMATED** setup service cost. JSON format version 2 records the
logical load count, operations per load, modeled operation count, payload cost,
setup cost, and exact total separately.

This plan is generic and assumption-driven. It does not assert that a logical
expert is independently movable, that packed Granite parameters are copied in
these chunks, or that batching/coalescing occurs in a real runtime.

## Cache lifecycle comparison

The same expert-size, capacity, and policy configuration can compare two
explicit cache reset assumptions across evaluation traces in a validated corpus
manifest:

```powershell
moe-cache-lab analyze-lifecycle artifacts\corpus\manifest-v1.json `
  --preflight-config preflight-config.json `
  --output artifacts\cache-lifecycle.md `
  --json-output artifacts\cache-lifecycle.json
```

- `cold_per_workload` creates an empty cache for every workload. Its per-workload
  rows exactly match independent `simulate_byte_cache` runs.
- `persistent_sequence` creates one empty cache and preserves its state across
  workload boundaries in the exact validated manifest/corpus order.

The output records workload identity/order, aggregate and per-workload integer
counters, and stable sorted layer-qualified starting/ending resident keys. It
does not expose internal timestamps or LFU counters. Aggregate counters are
exact sums of the per-workload rows.

Lifecycle output format version 2 added descriptive summaries derived from the
existing simulation rows, without replaying the trace:

- adjacent tested-capacity comparisons use upper-minus-lower exact hit, miss,
  and simulated demand-load-byte deltas within the same lifecycle mode and
  policy; `exact_flat` means those three integer deltas are exactly zero;
- same-capacity policy comparisons use LFU-minus-LRU deltas within the same
  lifecycle mode;
- workload spread records observed minima, maxima, ranges, and all tied
  manifest workload identities in declared order.

Hit-rate deltas and workload hit-rate ranges remain exact fractions in JSON;
zero-request rates are explicitly undefined. These summaries cover tested
cells only. They do not interpolate, optimize, infer workload semantics, or
recommend a policy or capacity.

Lifecycle output format version 3 additionally applies the existing exact
serialized/no-overlap transfer-cost equation to every aggregate lifecycle row
and its per-workload attribution under every configured hardware profile and
transfer-operation plan. Workload estimates reconcile exactly to their parent
cell for logical loads, demand bytes, modeled operations, payload service,
setup service, and total serialized service. Cold and persistent modes remain
separate execution assumptions and are never summed together.

These transfer-service rows are **ESTIMATED** from caller-supplied payload
bandwidth, per-operation setup latency, and operation-granularity assumptions.
They exclude compute, overlap/concurrency, kernel/runtime scheduling, allocator
effects, synchronization/protocol effects, D2H writeback, and end-to-end
runtime behavior. They do not establish physical Granite transfer granularity,
physical expert residency, measured latency, throughput, tokens/sec, or
speedup. Established pre-flight analysis/config formats 1 and 2 and their
single-trace transfer sensitivity semantics are unchanged; lifecycle schema
evolution is isolated to the lifecycle output.

Lifecycle comparison changes reset boundaries only. It does not change
layer-qualified `(layer_id, expert_id)` identity, atomic token/layer working
sets, or LRU/LFU admission and eviction behavior. Chronology is never inferred
from workload names, prompt categories, or semantics.

Cache outcomes and demand/eviction bytes remain **SIMULATED**, not measured
runtime data movement. Hardware profiles and transfer-operation plans do not
change those cache outcomes; they are used only for the separate **ESTIMATED**
serialized H2D service matrix.

Interpret the output as three distinct evidence classes:

- **ROUTING OBSERVATIONS:** descriptive results over the supplied trace. Treat them as **MEASURED only if the trace provenance establishes that they were measured**; this command does not collect measurements.
- **SIMULATED:** byte-cache hits, misses, demand-load bytes, evictions, and residency accounting.
- **ESTIMATED:** serialized/no-overlap transfer-service time derived from the supplied bandwidth, setup-latency, and (for version 2) transfer-operation-plan assumptions.

This is a pre-flight feasibility/sensitivity analysis, not proof of runtime acceleration. It does not establish physical GPU residency, actual transfer timing, end-to-end latency, throughput, tokens/sec, or speedup. It does not implement expert swapping/offloading.
