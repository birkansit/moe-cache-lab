# moe-cache-lab 0.4.0rc1 engineering checkpoint

Status: **historical validated checkpoint**. The public package version is now
`0.4.0`; the experiment evidence recorded here remains the technical basis for
that release.

This checkpoint documents a correctness-first routing profiler, immutable
trace/evidence workflow, two-view expert-cache simulator, deterministic
benchmark suite, a local runtime-feasibility no-go, and a measured CPU observer
overhead result. It is **not an inference accelerator**.

## Claim boundary

- Router selections are **measured** from the official Transformers model with
  observational hooks.
- Cache hits, misses, residency, and policy outcomes are **simulated**.
- Transfer counts are **estimated** from simulated expert-object loads.
- The Stage 2 HIP diagnostic measures raw byte copies only. It is not a PyTorch,
  Granite, expert-cache, latency, throughput, or speedup result.
- The conservative `prefill_layer_union_atomic` view is not verified Granite
  runtime residency or scheduling.

There is no expert offload implementation, router modification, semantic expert
labeling, acceleration claim, or hidden favorable-result tuning in this
checkpoint.

## Evidence index

| Evidence | SHA-256 |
| --- | --- |
| V0.2 manifest `results/v0.2-corpus-v1/manifest-v1.json` | `d5ca0013b1dfee96b353bb964666eaf9f6335494b0279362f536b4e8bc359e30` |
| V0.2 report `results/v0.2-corpus-v1-report.md` | `f21a338537c2001fa4bcc9d7db037beea7892262a74d73138b8d5169204bca9f` |
| Stage 1 set manifest | `f7e624d95368b1aa976f3e31412fc44e86dbf46439e5c15534032c14446e34d1` |
| Stage 1 structured benchmark | `dd8f0a46470e93132a63abd6d63409c02f077064c5258bb9e59c0a1f36450740` |
| Stage 1 Markdown benchmark | `f11e9b0fbbdc61d57b578eb00c4e8e8fcd2c0428ba72e60516bd2b609823633a` |
| Stage 2 raw HIP-copy result | `0072362ad8f38783775e84b6df0145dd6bbd2481cd58c7e89849aaeb539f76c2` |
| Stage 2 HIP-copy probe script | `3c4860b59e5d51916fa800997b87e4c133d2f8b861bf6aba5ebd24196beb0e9d` |
| V0.4 set manifest | `1f87491f3d0512912e593422a2cc0b012ebed343291726131fa51dc2e103fe60` |
| V0.4 structured result | `ed857ef581f80fbbe8e694bc869e9265e785e5c48bd6e00118b20db7d85e7b72` |
| V0.4 Markdown report | `d9b5399a125d162c3c7cdb3f847f30c196b56fd31ace9ee714f9dd4d5b81ceaf` |

The Stage 1 evidence contains three fresh CPU-FP32 offline repetitions at the
exact Granite revision
`0da7a48b0276d500ce5922fd2b33944091fc6c09`. Semantic routing was bit-identical,
the V0.2 prefix gate passed, and the repetitions validate determinism but do not
triple published denominators.

## Stage 1 result

The prospectively frozen Stage 1 continuation rule passed in every repetition:

- conservative grouped LFU suite-persistent estimated transfers were 29,937 at
  capacity 32 and 16,277 at capacity 256 (54.370845%);
- all eight grouped LFU capacity-256 decode-only-cold prompt hit rates were
  30.1107%-48.8281%, above the fixed 10% threshold.

These are simulation results. Passing the rule justified a runtime-feasibility
inspection; it did not demonstrate runtime acceleration.

## Stage 2 decision

Stage 2 returned **NO-GO for Stage 3 on the tested environment**:

- the installed PyTorch path was CPU-only and the RX 6650 XT did not provide a
  supported Windows PyTorch/HIP-SDK path for this workflow;
- Granite packs each layer's experts into two Parameters and exposes no
  supported per-expert residency/provider boundary;
- the BF16 tensor payload is 2.486 GiB (4.972 GiB in FP32) against 7.984 GiB
  reported VRAM, so a future supported GPU route should establish full
  residency before expert streaming;
- the retained raw-copy probe demonstrates byte movement only, while a
  correctness-preserving expert cache would require a new runtime path.

Stages 3-5 were therefore deferred rather than replaced with an unsupported
runtime demonstration.

## V0.4 measured CPU observer result

The V0.4 experiment compared the exact routing observer with the same
full-resident CPU-FP32 execution driver with observation disabled. Attempt
`b247eb85-eb21-4e3a-886e-6ba250cbb980` was `valid_stable`: its mean paired
wall-time ratio was `1.0208846178255349` (about 2.09% observer overhead), all
frozen stability gates passed, routing IDs were exact, and maximum selected
probability delta was `0.0`.

This is a **measured CPU observer** overhead result on one pinned workload, not
model acceleration, throughput, cache residency, GPU performance, or production
evidence.

## Supported workflows

Run with `PYTHONPATH=src` or install the package into an appropriate Python
environment:

```powershell
python -m moe_cache_lab.cli --help
python -m moe_cache_lab.cli benchmark-suite results\v0.2-corpus-v1\manifest-v1.json --output v0.2-report.md
python -m moe_cache_lab.cli validate-stage1 results\stage1-runtime-fidelity-v1\stage1-set-manifest.json --trusted-v02-manifest results\v0.2-corpus-v1\manifest-v1.json
python -m moe_cache_lab.cli benchmark-stage1 results\stage1-runtime-fidelity-v1\stage1-set-manifest.json --trusted-v02-manifest results\v0.2-corpus-v1\manifest-v1.json --json-output stage1.json --output stage1.md
python -m moe_cache_lab.cli validate-v04 results\v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980\v04-set.json
```

Collection commands remain available but load the real model. Existing tracked
evidence should normally be validated or replayed rather than recollected merely
to reproduce a report.

The optional hardware-specific raw-copy diagnostic is:

```powershell
python scripts\stage2_hip_copy_probe.py --output stage2-hip-copy-probe.json
```

It requires a compatible local `amdhip64_6.dll` and is not part of the portable
unit suite.

## Validation gate

The `0.4.0rc1` source passed 120 automated tests, compilation, dependency
consistency, CLI smoke checks, and strict Stage 1/V0.4 artifact validation. A
PEP 517 wheel (`moe_cache_lab-0.4.0rc1-py3-none-any.whl`) was built and installed
into a fresh temporary environment for import, version, CLI, and dependency
smoke checks.

The final `0.4.0` release changes package/release metadata and public-facing
documentation; it does not reinterpret the historical measured, simulated, or
estimated experiment results.

## Known limitations

- Results cover one model revision, one fixed 4/8 corpus, greedy decode, one
  prompt order, and the declared cache abstractions.
- The simulator does not observe physical memory residency, data movement,
  latency, bandwidth overlap, kernels, or output quality changes.
- The Stage 2 no-go is local to the documented Windows/RX 6650 XT/software
  stack. It is not a universal impossibility result.
- The tested model's raw weights fit the reported GPU VRAM. A future runtime
  acceleration study should use supported accelerator hardware and begin with a
  full-resident correctness/performance baseline; a larger non-fitting MoE is a
  separate and more representative offload target.
