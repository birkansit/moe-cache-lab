# Synthetic stage-qualified no-download pre-flight

This fixture exercises the public canonical trace-v2 plus pre-flight-config-v3
path. The trace is synthetic, and both hardware profiles and transfer plans are
fictional caller-supplied assumptions. It is not a model benchmark and does not
require a model download, Torch, Transformers, a GPU, or a vendor SDK.

From the repository root:

```powershell
moe-cache-lab analyze examples\no-download-stage-qualified-preflight\trace.jsonl `
  --preflight-config examples\no-download-stage-qualified-preflight\preflight-config.json `
  --workload-id synthetic-stage-qualified-demo `
  --output artifacts\stage-qualified-preflight-report.md `
  --json-output artifacts\stage-qualified-preflight-report.json
```

Compare the generated files with `expected.sha256`. Matching hashes proves
byte-for-byte reproduction only. Routing facts describe this synthetic trace,
cache outcomes are **SIMULATED**, and transfer-service values are **ESTIMATED**
under the fictional assumptions. The outputs do not establish physical
residency, real transfers, latency, throughput, speedup, or an optimal policy or
capacity.

The fixture deliberately gives `(encoder, 0, 1)` and `(decoder, 0, 1)`
different byte sizes. They are distinct expert identities. Its explicit
capacity-unassigned encoder event is a zero-request, cache-inert event; no
preferred expert is fabricated for it.

Stage-qualified lifecycle analysis remains unsupported. The existing
`analyze-lifecycle` command is a v1 layer-qualified compatibility workflow.
