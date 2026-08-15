# Stage 1 runtime-fidelity results

Status: **validated**. This is measured routing plus simulated cache analysis
and estimated transfer counting. It is not runtime acceleration or verified
expert residency, movement, latency, throughput, or speedup.

## Immutable evidence

- Validated set: `results/stage1-runtime-fidelity-v1/`
  - 44 files, 9,202,187 bytes.
  - Set-manifest SHA-256:
    `f7e624d95368b1aa976f3e31412fc44e86dbf46439e5c15534032c14446e34d1`.
- Structured benchmark: `results/stage1-runtime-fidelity-v1-benchmark.json`
  - SHA-256:
    `dd8f0a46470e93132a63abd6d63409c02f077064c5258bb9e59c0a1f36450740`.
- Human-readable benchmark: `results/stage1-runtime-fidelity-v1-benchmark.md`
  - SHA-256:
    `f11e9b0fbbdc61d57b578eb00c4e8e8fcd2c0428ba72e60516bd2b609823633a`.
- Total finalized evidence: 46 files, 14,707,267 bytes.
- Artifact regeneration was byte-identical.

## Collection and measured routing

- Three fresh repetitions completed in 135.875 seconds of parent collection
  wall time.
- All 12 prompts reached the fixed horizon of 16 routed decode-input steps;
  none emitted terminal EOS within the horizon.
- Token IDs and selected expert IDs matched exactly across repetitions;
  maximum selected-probability difference was 0. The trusted V0.2 prompt and
  first-two-decode-step prefix gate passed.
- Calibration: 2,664 measured routing events / 21,312 layer-qualified expert
  assignments.
- Evaluation: 5,832 measured routing events / 46,656 assignments, comprising
  2,760 prompt events / 22,080 assignments and 3,072 decode events / 24,576
  assignments.
- The grouped evaluation sensitivity view derives 3,264 bundles / 29,937
  unique bundle requests: prompt 192 / 5,361 and decode 3,072 / 24,576. Its
  maximum prompt-layer union bundle is 32.

## Frozen continuation rule

The prospectively frozen confirmation rule passed in all three deterministic
repetitions:

- Grouped-view suite-persistent LFU estimated transfers fell from 29,937 at
  capacity 32 to 16,277 at capacity 256. The exact ratio is 54.370845%, below
  the frozen 90% ceiling.
- Grouped-view decode-only cold LFU at capacity 256 passed the 10% hit-rate
  requirement for every evaluation prompt in every repetition. Prompt hit
  rates ranged from 30.1107% to 48.8281%.

Passing justified **Stage 2 read-only runtime-feasibility inspection only**.
It did not justify expert offloading or a Stage 3 runtime prototype.

## Stable and view-sensitive findings

- In the conservative grouped view, both dynamic policies had zero hits at
  capacities 32, 64, and 128. This is a simulation result, not proof of Granite
  runtime residency or scheduling.
- In the token view, LRU's identical 16,719 hits at capacities 32, 64, and 128
  are a per-token abstraction artifact and must not be interpreted as verified
  grouped-prefill reuse.
- At suite-persistent capacity 256, grouped-view LFU had the fewest estimated
  transfers among evaluated policies (16,277), while token-view LRU had the
  fewest (18,530). Policy ranking is view-sensitive.
- Cache-boundary choice also matters: at grouped capacity 256, suite-persistent
  LFU beat LRU (16,277 versus 18,543 estimated transfers), while the cold
  per-prompt control ranked LRU ahead of LFU (18,771 versus 19,703).
- Fixed calibrated/oracle policies often showed higher simulated hit rates but
  worse estimated transfers than causal dynamic policies below full-model
  capacity because fixed-target prewarm/restoration costs are charged.

These findings are limited to the frozen corpus, model revision, views,
capacities, controls, and deterministic workload order. Categories and expert
IDs provide no expert-semantic evidence. Negative and null results are retained.
