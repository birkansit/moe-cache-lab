# Stage 2 runtime-feasibility inspection

Status: **validated local no-go**.

This report evaluates whether the Stage 1 expert-cache simulation could be
turned into a meaningful real per-expert residency/offload prototype on the
tested Windows / RX 6650 XT system using the pinned official Transformers path.
It did not implement offloading or claim a universal limitation.

## Decision

Do not start a Stage 3 expert-residency/offload prototype on the tested system.
Three facts dominate the decision:

1. The validated project environment uses CPU-only PyTorch, while the RX 6650
   XT did not provide a supported Windows PyTorch/HIP-SDK path for this
   workflow.
2. The GraniteMoE implementation stores all experts in each layer in packed
   Parameters and exposes no supported per-expert residency/provider boundary.
   Implementing the simulator's `(layer_id, expert_id)` cache would therefore
   require a new execution path rather than a safe observation hook.
3. The checkpoint's raw tensor payload is about 2.486 GiB in BF16 (4.972 GiB
   at FP32 sizing) against 7.984 GiB reported VRAM. Raw weights fit the card, so
   a supported GPU investigation should first establish a full-resident
   correctness/performance baseline before considering expert streaming.

The result is **NO-GO for Stage 3 on the tested environment**. It does not mean
expert caching is impossible or unhelpful on larger MoE models or different
hardware/runtime stacks.

## Tested system

- Windows 11 Pro 64-bit, build `26200`.
- AMD Ryzen 5 5600, 6 cores / 12 logical processors; 16 GiB physical RAM.
- AMD Radeon RX 6650 XT (`gfx1032`), approximately 7.984 GiB reported VRAM.
- Local HIP runtime can enumerate the GPU and perform raw byte copies.
- Project environment: Python 3.10.11, PyTorch `2.12.0+cpu`;
  `torch.version.cuda` and `torch.version.hip` are `None` and accelerator device
  count is zero.

Device enumeration by a HIP runtime does not make a CPU-only PyTorch build
accelerator-capable.

The release-time compatibility assessment used AMD's published Windows HIP SDK
and PyTorch compatibility information:

- [AMD Windows HIP SDK system requirements](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/shared/hipsdk/reference/system-requirements.html)
- [AMD Windows PyTorch compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)

## Model and memory layout

The pinned safetensors header was inspected without loading tensor data:

| Quantity | BF16/checkpoint | FP32 runtime sizing |
| --- | ---: | ---: |
| All 1,334,628,352 parameters | 2,669,256,704 B (2.485939 GiB) | 5,338,513,408 B (4.971878 GiB) |
| Expert parameters | 2,415,919,104 B (2.250000 GiB) | 4,831,838,208 B (4.500000 GiB) |
| Non-expert parameters | 253,337,600 B | 506,675,200 B |
| One layer's 32 experts | 100,663,296 B | 201,326,592 B |
| One layer-qualified expert | 3,145,728 B (3 MiB) | 6,291,456 B (6 MiB) |

The simulator's expert-object capacities therefore correspond to these weight
bytes only:

| Capacity | BF16 expert bytes | FP32 expert bytes |
| ---: | ---: | ---: |
| 32 | 96 MiB | 192 MiB |
| 64 | 192 MiB | 384 MiB |
| 128 | 384 MiB | 768 MiB |
| 256 | 768 MiB | 1,536 MiB |
| 512 | 1,536 MiB | 3,072 MiB |
| 768 | 2,304 MiB | 4,608 MiB |

These are parameter-byte calculations, not measured GPU peak-memory values.
Allocator overhead, operator workspaces, and a real runtime's memory behavior
were not measured.

## Why the observation hook is not an offload hook

The official Granite MoE path stores complete expert groups in packed tensors.
The routing observer runs after router computation and can record selected
experts without changing output, but it does not provide a supported mechanism
to replace individual packed expert slices with bounded GPU-resident slots.

A real per-expert cache would require at least:

- a logical expert-to-physical slot map;
- bounded GPU resident buffers;
- CPU/pinned-memory backing storage;
- miss handling and eviction;
- asynchronous transfer/synchronization rules; and
- an execution path that consumes remapped expert slots correctly.

That is runtime implementation work, not a small extension to the current
forward hook.

Reference sources:

- [Granite model card](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct)
- [Transformers license](https://github.com/huggingface/transformers/blob/main/LICENSE)
- [Accelerate big-model offload API](https://huggingface.co/docs/accelerate/package_reference/big_modeling)

## Raw transfer diagnostic

[`scripts/stage2_hip_copy_probe.py`](scripts/stage2_hip_copy_probe.py) calls the
installed HIP runtime directly through declared `ctypes` signatures. It uses
HIP-pinned host memory, synchronous `hipMemcpy`, explicit synchronization, and a
full-buffer round-trip SHA-256 check before timing.

The retained raw result is
[`results/stage2-hip-copy-probe.json`](results/stage2-hip-copy-probe.json),
SHA-256
`0072362ad8f38783775e84b6df0145dd6bbd2481cd58c7e89849aaeb539f76c2`.
The probe script SHA-256 is
`3c4860b59e5d51916fa800997b87e4c133d2f8b861bf6aba5ebd24196beb0e9d`.

Measured medians:

- H2D bulk copy: `6.889079 GB/s`;
- D2H bulk copy: `7.080894 GB/s`;
- one contiguous 3 MiB BF16-sized expert: `529.570654 us`;
- physical 2 MiB + 1 MiB copies: `634.535596 us`.

These values measure raw byte-copy capability only. They are not PyTorch/model
latency, expert-cache latency, throughput, or speedup measurements.

As a deliberately hypothetical sizing exercise, the Stage 1 grouped LFU
capacity-256 simulation has 16,277 **estimated** expert loads. At 3 MiB per
BF16 expert that maps to about 47.6865 GiB of cumulative H2D bytes. Translating
simulated loads through a raw-copy microbenchmark is useful for feasibility
bounds, but it is not a prediction of a real asynchronous inference runtime.

## Alternate runtime assessment

The release-time investigation also considered alternative paths rather than
forcing the official Transformers backend to do something it does not support.

### Native Windows ROCm/PyTorch

The tested RX 6650 XT / local software combination did not provide a supported
native Windows PyTorch accelerator path for this experiment.

### PyTorch DirectML

DirectML would replace the validated runtime and did not provide a verified
Granite per-expert residency/provider mechanism. It was not used as a Stage 3
substitute.

### vLLM

vLLM is technically relevant prior art for MoE runtime work, but the tested
machine did not provide a suitable supported vLLM GPU route. Generic/static CPU
weight offload is also different from a dynamic bounded expert cache. Expert
weight-provider/cache work in vLLM is an active runtime-engineering area rather
than something this project could safely obtain by adding a hook.

Relevant upstream references:

- [vLLM supported models](https://docs.vllm.ai/en/stable/models/supported_models/)
- [vLLM GPU installation requirements](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM incremental MoE expert-offloading RFC](https://github.com/vllm-project/vllm/issues/38256)

No third-party source code or binary was copied into this repository during
Stage 2.

## Gate summary

| Stage 2 question | Result |
| --- | --- |
| Exact packed expert layout understood? | Yes. |
| Parameter/cache sizing established? | Yes. |
| Raw byte-copy capability measured? | Yes, with explicit limitations. |
| Supported per-expert lifecycle boundary in the tested backend? | No. |
| Supported accelerator path on the tested project stack? | No. |
| Is expert streaming a meaningful first experiment when raw weights fit? | No; establish full residency first. |

## What would justify revisiting Stage 3

A future experiment should start only when there is a supported accelerator
runtime and a model/workload for which expert residency is an actual memory or
performance constraint. The first measurement should be a full-resident
correctness/performance baseline where that baseline is possible.

For studying real offload, a larger MoE whose full weights do **not** fit the
target VRAM would be a more representative test than the current Granite
reference model.
