# v0.5 local validation

This document summarizes the local Windows validation performed before the
public v0.5 repository was prepared.

## Claim boundary

The validation intentionally separates four evidence classes:

- **MEASURED routing:** routing selections collected from the pinned Granite
  model through observational hooks.
- **SIMULATED cache behavior:** offline LRU/LFU replay over the measured traces.
- **MEASURED raw transfer diagnostic:** direct HIP host-to-device byte copies.
- **ESTIMATED transfer service:** a serialized transfer-cost model driven by
  simulated demand loads and calibration-derived assumptions.

The raw HIP measurements are not PyTorch GPU inference, Granite runtime,
physical expert residency, expert-offload latency, throughput, or speedup
measurements.

## Source and environment

Real-model/local-hardware validation was performed against private development
checkpoint:

`d34fc5b97cc446f22f453770590d3d815a2d0eab`

The clean public export removes archived experiment-only modules and legacy CLI
surface while leaving the v0.5 routing-analysis, byte-cache, hardware-cost, and
pre-flight implementation unchanged. The clean export is then rechecked with
the no-download unit/release gates before being pushed.

Host snapshot:

- Windows 11 Pro, build 26200
- AMD Ryzen 5 5600, 6 cores / 12 logical processors
- 16 GiB physical RAM
- AMD Radeon RX 6650 XT
- Python 3.10.11
- PyTorch 2.12.0+cpu
- Transformers 5.12.0
- `torch.cuda.is_available() == False`
- `torch.version.cuda == None`
- `torch.version.hip == None`
- direct `amdhip64.dll` enumeration: 1 device
- HIP runtime version value: 50732000

Reference model:

- `ibm-granite/granite-3.1-1b-a400m-instruct`
- pinned revision:
  `0da7a48b0276d500ce5922fd2b33944091fc6c09`
- 32 experts per routed layer
- 8 selected experts per token
- 24 routed layers

## Deterministic no-download gate

The tracked synthetic pre-flight demo reproduced the expected hashes:

- Markdown:
  `219751d3a63b97177f6d3133dd2e7b69fc941eed2416e72568125f4e2936542d`
- JSON:
  `464638a63f66e2c8797f4e7c59efea226f30050032ede5138255f675d0ff4a42`

## Fresh measured Granite routing

Single-prompt prefill-only trace:

- 240 routing events
- 1,920 layer-qualified assignments
- 630 unique layer-qualified experts

Same prompt with 8 generated/decode input steps:

- 432 routing events
- 3,456 assignments
- 684 unique layer-qualified experts
- prompt/prefill: 240 events / 1,920 assignments
- generated/decode: 192 events / 1,536 assignments

The prompt/prefill routing sequence matched exactly across independent model
loads.

Fresh 12-prompt corpus:

- calibration: 4 prompts, 1,896 routing events / 15,168 requests
- evaluation: 8 prompts, 4,296 routing events / 34,368 requests
- evaluation prompt/prefill: 2,760 events
- evaluation generated/decode: 1,536 events

These are measured routing observations, not runtime performance measurements.

## Logical expert byte model

On the local CPU-FP32 model load, each logical routed expert corresponds to:

- input projection slice: 4 MiB
- output projection slice: 2 MiB
- total logical expert payload: 6 MiB

An 8-expert atomic routing event therefore has a 48 MiB logical working set.

For the 8-token decode trace:

- unique referenced experts: 684
- unique referenced logical bytes: 4,303,355,904
- maximum atomic event working set: 50,331,648 bytes (48 MiB)

This is a logical slice-size model. It does not establish independently movable
physical Parameters or real per-expert GPU residency.

## Byte-cache semantic equivalence

For equal-size 6 MiB logical experts, the byte-capacity LRU/LFU simulator was
compared directly with the existing count-capacity simulator at equivalent
capacities.

Result:

`byte_vs_count_mismatches = 0`

This validates that byte accounting preserves the established hit/miss semantics
for the equal-size case.

## Raw HIP H2D calibration

Pinned-host-memory, synchronous H2D copy measurements were collected separately
under a quiet host condition and a normal-background condition.

Quiet pinned linear fit:

- fitted bandwidth: 6,939,866,304 bytes/s
- fitted setup/intercept parameter: 66,335 ns
- R²: 0.9999995012506336

Normal-background pinned linear fit:

- fitted bandwidth: 6,942,201,481 bytes/s
- fitted setup/intercept parameter: 69,840 ns
- R²: 0.9999997892736231

Canonical raw-diagnostic SHA-256 anchors:

- quiet:
  `79785fba50e26306868be0791e5e11e5fc407f2ace4671a7d864858b310b7cff`
- normal-background:
  `af055dec33984aa72b67efad8ea6c5d48d3c2924bde8870cdcc35e64f127a28d`

The two conditions produced very similar fitted bandwidth. These measurements
are raw HIP byte-transfer diagnostics only.

## Workload sensitivity

Eight evaluation workloads were independently analyzed from cold cache with the
quiet calibration profile.

At a 1,536 MiB logical cache budget, LRU simulated hit rate ranged from:

- 0.577 on `eval-factual-01`
- to 0.712 on `eval-math-01`

The corresponding estimated serialized transfer-service values ranged from:

- 1.2930 s on `eval-factual-02`
- to 1.6909 s on `eval-summary-01`

Across these workloads, policy/capacity behavior was not constant. In
particular, LRU and LFU rankings depended on cache capacity and workload.

These service-time values are estimates from the stated model, not measured
Granite runtime latency or a speedup result.

## Evidence manifest

The local validation evidence manifest covered 9 canonical files.

Manifest SHA-256:

`ea597d5bbf1286078c13f8b4c3e0f3b599e04310341dafc0090e35bc9f39a49a`

Raw local artifacts are intentionally not copied wholesale into the clean public
repository because machine-specific logs can contain unnecessary local paths and
environment details. This document publishes the bounded technical summary and
cryptographic anchors instead.
