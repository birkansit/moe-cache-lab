# Synthetic no-download pre-flight demo

This directory is a tracked, deterministic **synthetic** fixture for the v0.5 offline pre-flight workflow. It is not a model benchmark and it contains no measured hardware result.

Run it from the repository root:

```powershell
moe-cache-lab analyze examples\no-download-preflight\trace.jsonl `
  --preflight-config examples\no-download-preflight\preflight-config.json `
  --output preflight-report.md `
  --json-output preflight-report.json
```

`trace.jsonl` uses fixed synthetic provenance and exercises prompt/prefill plus generated/decode routing across two layers. `preflight-config.json` uses explicit layer-qualified expert sizes, two byte capacities, both LRU and LFU, and two deliberately fictional transfer profiles. The profile values are assumptions only.

`expected.sha256` pins the exact deterministic Markdown and JSON bytes produced by the current CLI path. `tests/test_no_download_preflight_demo.py` reproduces both outputs offline and verifies those hashes.

Interpretation remains strict: routing observations in this fixture are **synthetic**, cache outcomes are **SIMULATED**, and transfer-service costs are **ESTIMATED** from fictional caller-supplied assumptions. The demo does not establish physical GPU residency, actual transfer time, end-to-end latency, throughput, tokens/sec, or runtime speedup.
