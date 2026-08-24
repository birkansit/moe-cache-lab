# V0.7 bounded runtime-validation feasibility record

Status: **GO for one later, separately reviewed test-only CPU physical-copy
replay**. This historical research/design record does not
implement or run that replay and does not establish cache residency, H2D
transfer, latency, throughput, or speedup during model inference.

## Scope and evidence reviewed

- Prior evidence inspected: `V07_SWITCH_SMOKE.md`,
  `src/moe_cache_lab/byte_cache.py`,
  `src/moe_cache_lab/hardware_cost.py`,
  `src/moe_cache_lab/switch_collector.py`, `src/moe_cache_lab/evidence.py`,
  and their relevant tests.
- This gate changes no routing, trace, cache, lifecycle, hardware-cost,
  collector, evidence, package, test, or public-document behavior.

The decision question is deliberately narrow: can an existing offline cache
quantity be frozen, then checked against a physical operation executed and
observed independently of that offline result? The answer is **yes**, but only
for the logical demand-load count and tensor-payload bytes in the artificial
CPU replay defined below. The answer remains **no** for native inference
residency, H2D service, and the existing serialized transfer-time estimate.

## Source-grounded runtime facts

### Switch storage and execution

The official Transformers 5.12.0 Switch source defines each routed expert as a
separate `SwitchTransformersDenseActDense` child in a
`SwitchTransformersExperts(nn.ModuleDict)`. Each expert has `wi` and `wo`
linear weights. The expert forward identifies experts hit by the native
post-capacity assignment and calls only those children. The sparse MLP invokes
the router and then the expert container. See the pinned
[Transformers 5.12.0 Switch implementation](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/switch_transformers/modeling_switch_transformers.py).

That source provides a clean stage/layer/expert parameter organization for a
test payload. It does **not** provide a bounded runtime weight cache: all
registered expert parameters are ordinary model parameters and are already
resident on the model's CPU device in the validated collector path.

The earlier Switch smoke established, without retaining the raw trace, the
following bounded facts for pinned
`google/switch-base-8@92fe2d22b024d9937146fe097ba3d3a7ba146e1b`:

- CPU float32, Transformers `5.12.0`, Torch `2.12.0+cpu`;
- six sparse numerical layers in each of the encoder and decoder namespaces;
- 96 stage-qualified expert objects;
- two float32 parameter tensors per expert, totaling `18,874,368` parameter
  payload bytes per expert;
- a temporary validated 156-event trace-v2 with 156 assigned requests and no
  observed capacity drops;
- a **SIMULATED** LRU result at `150,994,944` bytes (eight equal payloads):
  89 hits, 67 misses, and `1,264,582,656` simulated demand-load bytes.

Those prior numbers demonstrate feasibility and give a reproducibility
cross-check. Because the raw trace was intentionally deleted, the future
experiment must recollect and freeze a new validated trace and prediction; it
must not present the old summary as a runtime input.

### PyTorch copy and size observability

PyTorch 2.12 documents that [`Tensor.copy_`](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.copy_.html)
copies elements from a source tensor into the destination. It documents
[`Tensor.element_size()`](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.element_size.html)
as bytes per element and [`torch.numel`](https://docs.pytorch.org/docs/2.12/generated/torch.numel.html)
as element count. Their integer product is therefore the logical tensor
payload presented to a copy. This is not a hardware-bus byte counter and need
not equal physical DRAM traffic.

The official [`torch.profiler`](https://docs.pytorch.org/docs/2.12/profiler.html)
can collect CPU operator activity and input shapes. It is suitable here as a
secondary audit that the isolated executor issued the expected `aten::copy_`
operators. It is not a byte counter and no profiler time is used as a result.

### Machine/runtime facts observed for this gate

These facts were queried locally on 2026-08-14 without loading a model:

- Windows 11 Pro `10.0.26200`, AMD64;
- AMD Ryzen 5 5600, 6 physical / 12 logical cores;
- 15.912930 GiB visible physical memory and 4.279465 GiB available at the
  inspection instant (availability is volatile);
- an AMD Radeon RX 6650 XT display adapter was enumerated;
- Python `3.10.11`, Torch `2.12.0+cpu`, Transformers `5.12.0`;
- `torch.version.cuda is None`, `torch.version.hip is None`, and
  `torch.cuda.is_available()` is false;
- Torch reported six intra-op and six inter-op threads in the inspection
  process.

The prior pinned Switch CPU smoke passed on this machine. The later experiment
must still record a fresh memory admission fact before loading the model and
must preserve an inability to load as a negative result. No CUDA/HIP execution
surface is exposed by the installed Torch build, so this gate has no H2D
observation path.

Unknowns preserved: physical DRAM traffic, cache-line write allocation,
hardware cache effects, NUMA placement, paging, allocator behavior, copy-kernel
implementation details, GPU-visible residency, PCIe/H2D bandwidth, H2D setup
latency, and end-to-end inference effects.

## Disposable synthetic observation check

One no-model, no-file synthetic check ran under Torch `2.12.0+cpu`:

1. Create CPU float32 `[3, 4]` and uint8 `[5]` sources and matching empty
   destinations.
2. Profile exactly two `destination.copy_(source)` calls with CPU activity and
   input shapes enabled.
3. Derive source payload bytes independently as `numel * element_size` and
   inspect source storage bytes.
4. Verify destination/source equality after the profile.

Observed results:

| source | logical bytes | storage bytes | profiler row | exact destination equality |
| --- | ---: | ---: | --- | --- |
| float32 `[3, 4]` | 48 | 48 | one `aten::copy_` | true |
| uint8 `[5]` | 5 | 5 | one `aten::copy_` | true |

The aggregate profiler `aten::copy_` count was exactly two. This proves only
that the installed CPU runtime exposes explicit copy operators and auditable
operand sizes for a deliberately isolated replay. It is not a throughput
measurement or model result.

## Candidate matrix

| Candidate | Independently observable fact | Circularity / validity | Decision |
| --- | --- | --- | --- |
| A. Native Switch CPU inference | Native post-capacity routing and expert execution are observable. No expert-weight demand transfer occurs: model parameters are already CPU resident. | Counting router callbacks or expert invocations would repeat routing evidence and would falsely treat execution as cache service. | **NO-GO** for cache/transfer validation. |
| B. Test-only physical-copy replay | A separate executor can issue real CPU `copy_` operations from stage-qualified expert parameters into bounded slots, audit `aten::copy_` count, operand bytes, and copied contents. | Credible only if prediction is sealed first and the executor receives no simulator decisions/counters, implements the frozen policy independently, and comparison occurs afterward. It validates the constructed replay, not native inference. | **GO**, using the exact protocol below. |
| C. Existing serialized transfer-service estimate | The formula combines simulated bytes with caller-assumed H2D bandwidth and setup latency. A CPU-copy duration could be measured separately. | CPU-copy timing does not validate H2D bandwidth or H2D setup latency. Reapplying the formula or fitting assumptions to the observation is circular. | **NO-GO** for validating the estimate on this runtime. |
| D. Built-in profiler / operand metadata | Profiler can independently count isolated CPU `aten::copy_` operators; tensor metadata identifies logical operand bytes. | Useful as a secondary audit for B. Profiler count alone says nothing about bytes, hardware-bus traffic, cache misses, or residency. | **GO only as B instrumentation**, not a standalone validation. |

## Independence and circularity boundary

Candidate B is a bounded differential validation, not a model-runtime cache
benchmark. Its independence comes from all of these requirements:

1. The offline predictor runs first and its complete inputs and result are
   serialized and SHA-256 sealed before the physical executor starts.
2. The physical executor may read only the validated canonical trace, payload
   manifest, LRU policy identifier, and byte capacity. It must not import or
   call `simulate_versioned_byte_cache`, reuse `_ByteDynamicCache` or another
   cache implementation, read the frozen predicted counters, or consume a
   simulator-generated hit/miss/load sequence.
3. The executor independently derives residency and eviction from the raw
   trace in authoritative order. A separate final comparison process reads
   both results only after execution finishes.
4. Runtime evidence includes actual destination mutations, exact post-copy
   tensor equality, and profiler-observed `aten::copy_` calls. It is not merely
   an executor-side increment of a counter.

The executor necessarily implements the same declared LRU contract over the
same trace and capacity. Exact agreement therefore validates implementation
and accounting consistency under this constructed replay; it does not provide
statistical evidence that real inference would have the simulated misses. This
limited differential independence is sufficient for the roadmap's smallest
bounded runtime check, but would be circular if relabeled as validation of a
native cache or as evidence for an optimal policy.

## Frozen next experiment

This is the complete design for a later issue. Changes to these choices require
a new review before collection.

### Inputs and ordering

1. Use exactly `google/switch-base-8` revision
   `92fe2d22b024d9937146fe097ba3d3a7ba146e1b`, Transformers `5.12.0`, Torch
   `2.12.0`, CPU, float32, evaluation/inference mode, and the existing native
   observational collector without routing changes.
2. Use the same public pinned-model-card prompt identified by UTF-8 SHA-256
   `30752b0dfe1d6680cc2568d56b6a924c18fd6460ad332e8afdac8de4ddc17e77`.
   Record the exact prompt bytes privately in the attempt manifest but do not
   publish or infer any private prompt.
3. Use one batch-size-one source and the existing maximum of four fed-back
   non-EOS decoder inputs. Preserve native encoder/source, decoder/prompt, then
   decoder/generated callback order. Do not sort or repair chronology.
4. Validate and serialize the resulting trace-v2. Record trace SHA-256 and the
   full stage-qualified event identity. Preserve assigned and unassigned
   events; an unassigned event has no expert request or copy.
5. Build the payload manifest from the actual loaded expert modules. For every
   `(routing_stage, layer, expert_id)`, record parameter component name, shape,
   dtype, device, contiguity, `numel`, `element_size`, component payload bytes,
   and summed expert payload bytes. Require exactly the native parameter set;
   do not fabricate padding or packed bytes.

### Offline prediction freeze

Run exactly one cold LRU `simulate_versioned_byte_cache` call with capacity
`150,994,944` bytes. The capacity is eight previously observed equal Switch
expert parameter payloads; the recollected manifest must independently confirm
that every expert is `18,874,368` bytes and that the capacity is exactly eight
payloads. Otherwise stop as a protocol mismatch rather than adapting capacity.

Serialize and hash, before physical replay:

- model/revision/runtime and trace/payload-manifest hashes;
- policy and capacity bytes;
- `event_count`, `expert_request_count`, `hits`, `misses`;
- `simulated_demand_load_bytes`;
- `eviction_count`, `simulated_evicted_bytes`;
- peak/final resident bytes and final stage-qualified resident keys.

All cache fields remain **SIMULATED**. The executor must not receive this result
or a derived miss/load sequence.

### Independent physical executor

Run the executor in a fresh process after the freeze file is durable:

- Load the same pinned model locally/offline and verify the payload manifest.
- Preallocate exactly eight CPU staging slots. Each slot contains destination
  tensors matching the two native `wi.weight` and `wo.weight` components. The
  slots are test storage only and are never installed into the model or used by
  an inference forward.
- Replay the canonical events exactly once. Maintain a small independent
  stage-qualified LRU slot map in Python. Do not import any project cache or
  hardware-cost implementation. An assigned resident key is a hit and performs
  no tensor copy. An assigned missing key evicts the independently selected
  LRU victim if necessary, then performs one synchronous CPU `copy_` for each
  native parameter component into that slot. Unassigned events do nothing.
- Keep profiler CPU activity scoped to only those explicit copy calls. Record
  each copy's source/destination key, component name, shape, dtype, device,
  source and destination `numel * element_size`, and profiler operator count.
  Require equal source/destination logical bytes and exact `torch.equal` after
  every completed component copy.
- Record executor-derived hits, misses, logical completed loads, evictions,
  final slot keys, and any exception for audit, but do not treat an incremented
  executor counter alone as physical evidence.

The main **MEASURED** observations are: completed destination mutations with
exact source equality, number of profiler-observed CPU `aten::copy_` operators,
and the integer logical operand bytes presented to those executed copies. They
are measurements of this CPU replay only. They are not measurements of DRAM,
PCIe, H2D, or model-inference transfers.

### Repetition, warmup, and timing

No timing result is authorized or needed. Use no timed warmup, no bandwidth
calculation, no repetition-based performance statistic, and no idle-machine
claim. One deterministic prediction/replay pair is sufficient. A second run
may be used only as a correctness-repeatability check if frozen prospectively;
it must not be selected post hoc and remains outside the minimum experiment.

### Exact agreement rule

After the executor exits, a separate comparer loads the sealed prediction and
runtime artifacts. PASS requires all of the following exact integer/equality
conditions, with no tolerance:

1. executor-derived hits, misses, evictions, and final resident keys equal the
   corresponding frozen simulation fields;
2. completed logical physical loads equal frozen `misses`;
3. the sum of actual source operand `numel * element_size` over all executed
   `copy_` calls equals frozen `simulated_demand_load_bytes`;
4. each loaded key's summed executed component bytes equals its pre-frozen
   payload-manifest size;
5. profiler-observed `aten::copy_` count equals the sum of native parameter
   component counts for the executor-derived missing keys (expected two per
   load only if the recollected native payload manifest confirms two);
6. every copy has equal source/destination logical bytes and exact destination
   content equality; and
7. all input/result hashes, trace chronology, capacity, and stage-qualified
   identities validate without mutation or repair.

Any mismatch, missing operator record, content mismatch, model/payload drift,
OOM, or incomplete attempt is a preserved **DISAGREEMENT/INVALID** result. Do
not tune capacity, rerun selectively, weaken equality, or update expected
values to force agreement.

### Artifacts and cleanup

Write an atomic attempt directory containing the canonical trace,
payload manifest, sealed prediction, runtime observation, comparison result,
environment record, and SHA-256 manifest. Never commit model weights or copied
payload tensors. Staging slots exist only in process memory and disappear when
the executor exits. Raw prompt/trace material must be treated as potentially
sensitive when deciding whether any small result artifacts are retained.

## Evidence labels and prohibited conclusions

| Quantity | Required label |
| --- | --- |
| Native post-capacity routing trace | **MEASURED routing** |
| Expert parameter `numel * element_size` manifest | **DESCRIPTIVE parameter payload bytes** |
| LRU hits/misses/load bytes/evictions/residency | **SIMULATED cache accounting** |
| Explicit CPU `copy_` calls, operand logical bytes, destination equality | **MEASURED test-replay copy observation** |
| Existing bandwidth/setup formula | **ESTIMATED serialized H2D service under caller assumptions** |

Even after a PASS, it remains forbidden to claim:

- native Switch inference uses this cache or moves weights on misses;
- observed physical CPU/GPU expert residency;
- H2D/PCIe bandwidth, setup latency, or transfer service;
- end-to-end latency, throughput, speedup, memory savings, or energy benefit;
- production offload correctness or safety;
- policy/capacity recommendation optimality;
- generalization beyond this one pinned trace, model, runtime, and artificial
  replay; or
- model quality or expert semantics.

## Decision

**GO**: Candidate B, augmented by Candidate D's operator audit, meets the
bounded gate. It can compare pre-frozen **SIMULATED** LRU miss/load-byte
accounting with an independently implemented executor's actual CPU tensor-copy
operations and operand bytes under an explicitly artificial cache replay.

Candidate A remains a native-runtime cache-validation NO-GO, and Candidate C
remains an H2D transfer-service validation NO-GO on the current CPU-only Torch
runtime. A production offload/swap engine is neither needed nor justified.

The GO decision was limited to implementing and running the frozen experiment
above. It did not justify broader runtime engineering, scenario expansion,
capacity tuning, GPU work, or production offload development.
