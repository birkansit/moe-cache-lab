# Offline pre-flight analysis

The v0.5 pre-flight workflow analyzes an existing compatible routing trace without running a model. It combines descriptive routing locality, byte-capacity LRU/LFU simulation, and a simple serialized transfer-service sensitivity model under explicit caller-supplied assumptions.

The built-in collector is currently Granite/Transformers-specific. After trace creation, this pre-flight path is driven by the repository routing-trace format and core analysis APIs rather than by Granite model execution.

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

Interpret the output as three distinct evidence classes:

- **ROUTING OBSERVATIONS:** descriptive results over the supplied trace. Treat them as **MEASURED only if the trace provenance establishes that they were measured**; this command does not collect measurements.
- **SIMULATED:** byte-cache hits, misses, demand-load bytes, evictions, and residency accounting.
- **ESTIMATED:** serialized/no-overlap transfer-service time derived from the supplied bandwidth and setup-latency assumptions.

This is a pre-flight feasibility/sensitivity analysis, not proof of runtime acceleration. It does not establish physical GPU residency, actual transfer timing, end-to-end latency, throughput, tokens/sec, or speedup. It does not implement expert swapping/offloading.
