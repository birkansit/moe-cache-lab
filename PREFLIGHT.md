# Offline pre-flight analysis

The pre-flight workflow analyzes an existing compatible routing trace without running a model. It combines descriptive routing locality, byte-capacity LRU/LFU simulation, and a simple serialized transfer-service sensitivity model under explicit caller-supplied assumptions.

The built-in collector is currently Granite/Transformers-specific. After trace creation, this pre-flight path is driven by the repository routing-trace format and core analysis APIs rather than by Granite model execution.

## Trace-version boundary

The public single-trace workflow supports two exact identity pairings:

- canonical trace v1 with config v1/v2 keeps layer-qualified
  `(layer_id, expert_id)` identity and established output bytes;
- canonical trace v2 with config v3 keeps stage-qualified
  `(routing_stage, layer_id, expert_id)` identity through routing evidence,
  cache simulation, transfer estimates, JSON, and Markdown.

All other trace/config combinations reject. Encoder and decoder layers are
never flattened or assigned invented numeric offsets. An unassigned v2 event
remains present in routing evidence but produces zero cache requests and no
cache-state change.

Routing-trace and pre-flight-config version numbers are independent namespaces;
matching numbers do not imply compatibility. The canonical cross-format rules
and current supported combinations are defined in
[`COMPATIBILITY.md`](COMPATIBILITY.md).

## Tracked no-download demos

A deterministic synthetic example is tracked at [`examples/no-download-preflight/`](examples/no-download-preflight/). It requires no model download, GPU, CUDA/ROCm, or vendor SDK:

```powershell
moe-cache-lab analyze examples\no-download-preflight\trace.jsonl `
  --preflight-config examples\no-download-preflight\preflight-config.json `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

The demo trace is synthetic and its transfer profiles are deliberately fictional assumptions. The tracked `expected.sha256` file pins the deterministic Markdown and JSON output bytes.

A separate deterministic stage-qualified fixture is tracked at
[`examples/no-download-stage-qualified-preflight/`](examples/no-download-stage-qualified-preflight/):

```powershell
moe-cache-lab analyze examples\no-download-stage-qualified-preflight\trace.jsonl `
  --preflight-config examples\no-download-stage-qualified-preflight\preflight-config.json `
  --workload-id synthetic-stage-qualified-demo `
  --output artifacts\stage-qualified-preflight-report.md `
  --json-output artifacts\stage-qualified-preflight-report.json
```

It is also synthetic and uses fictional assumptions. It deliberately preserves
different sizes for the same numerical layer/expert IDs in encoder and decoder,
and includes one explicit capacity-unassigned event. Its `expected.sha256`
pins the exact format-version-3 report bytes.

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

## Stage-qualified config version 3

Config v3 retains the config-v2 capacity, policy, hardware-profile, and
transfer-operation-plan meanings. Its expert-size records add an explicit
`routing_stage`:

```json
{
  "routing_stage": "encoder",
  "layer_id": 0,
  "expert_id": 1,
  "size_bytes": 3145728
}
```

The stage is exactly `encoder` or `decoder`; it is never inferred. V3 permits
an empty expert-size array for an all-unassigned trace, but every expert
actually requested by an assigned event must have its exact stage-qualified
size. Extra valid entries are inert. The structured pre-flight output remains
in the independent `moe-cache-lab.preflight-analysis` family and uses format
version 3 for this stage-qualified shape.

`--workload-id` labels only the descriptive routing-evidence workload. It does
not change trace events, cache replay, byte accounting, or transfer estimates.
`--top-k` cannot be combined with v2 pre-flight analysis; run the descriptive
v2 analyzer separately when caller-explicit locality ranks are needed.

## Cache lifecycle comparison (v1 only)

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

Stage-qualified v2 lifecycle orchestration remains deferred. The existing
`analyze-lifecycle` command rejects config v3 rather than flattening stages or
silently treating a v2 workload as v1.

Cache outcomes and demand/eviction bytes remain **SIMULATED**, not measured
runtime data movement. Hardware profiles and transfer-operation plans do not
change those cache outcomes; they are used only for the separate **ESTIMATED**
serialized H2D service matrix.

Interpret the output as three distinct evidence classes:

- **ROUTING OBSERVATIONS:** descriptive results over the supplied trace. Treat them as **MEASURED only if the trace provenance establishes that they were measured**; this command does not collect measurements.
- **SIMULATED:** byte-cache hits, misses, demand-load bytes, evictions, and residency accounting.
- **ESTIMATED:** serialized/no-overlap transfer-service time derived from the supplied bandwidth, setup-latency, and (for version 2) transfer-operation-plan assumptions.

This is a pre-flight feasibility/sensitivity analysis, not proof of runtime acceleration. It does not establish physical GPU residency, actual transfer timing, end-to-end latency, throughput, tokens/sec, or speedup. It does not implement expert swapping/offloading.
