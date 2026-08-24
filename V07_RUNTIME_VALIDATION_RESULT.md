# V0.7 bounded Switch CPU physical-copy validation record

Status: **AGREEMENT** for the exact constructed test replay.

This historical result is deliberately narrow. Routing is **MEASURED
routing** from the native Switch post-capacity assignment. Expert parameter
sizes are **DESCRIPTIVE parameter payload bytes**. LRU accounting is
**SIMULATED cache accounting**. Explicit CPU copies, their logical operand
bytes, profiler operator count, and copied-content checks are **MEASURED
test-replay copy observations**. None of these claims describes native
inference weight movement or a production cache.

## Provenance and frozen protocol

- Protocol: `V07_RUNTIME_VALIDATION_GATE.md`.
- Model: `google/switch-base-8`.
- Immutable revision:
  `92fe2d22b024d9937146fe097ba3d3a7ba146e1b`.
- Runtime: Transformers `5.12.0`, Torch `2.12.0+cpu`, CPU, float32,
  batch size 1, deterministic greedy decode, at most four fed-back non-EOS
  decoder inputs, four Torch intra-op threads.
- Cache policy/capacity: cold LRU, `150,994,944` bytes, exactly eight
  artificial CPU staging slots.

The frozen prompt was recovered by one read of the public CPU usage example in
the [exact-revision Hugging Face model card](https://huggingface.co/google/switch-base-8/blob/92fe2d22b024d9937146fe097ba3d3a7ba146e1b/README.md).
Its 96 UTF-8 bytes hashed exactly to
`30752b0dfe1d6680cc2568d56b6a924c18fd6460ad332e8afdac8de4ddc17e77`
before model execution. The prompt text is not committed.

All seven already-cached assets from the earlier Switch smoke were checked by
size and SHA-256
before use. No model, tokenizer, generation, or weight asset was downloaded.
The largest asset, `pytorch_model.bin`, remained the pinned `1,238,895,063`
bytes with SHA-256
`ff91705b718f692fa0c994a49d094154583db40b536e80f20f52e05498ff6856`.

Immediately before the run, Windows `GlobalMemoryStatusEx` reported
`6,755,733,504` available physical bytes. The prospective safety admission was
`3,702,526,894` bytes: twice the checkpoint bytes plus the full staging
capacity plus 1 GiB of interpreter/model workspace. Admission passed. This is
an admission fact, not a peak-memory measurement.

## Fresh measured routing trace

The existing native Switch collector was used unchanged. It created one fresh
trace in native callback order, serialized it as canonical trace-v2, validated
it, and round-tripped it exactly. The raw trace was intentionally not retained
in the source tree.

- Trace SHA-256:
  `0e1ed7a4ea7ee0f627f54ce0a113a2e749ddad49c114e9d3e326a0d10ba3a310`.
- Events / actual expert requests: `156 / 156`.
- Encoder/source: `126` assigned, `0` unassigned.
- Decoder/decoder-prompt: `6` assigned, `0` unassigned.
- Decoder/decoder-generated: `24` assigned, `0` unassigned.
- Combined: `156` assigned, `0` unassigned.

No event was created for the unfed horizon candidate. The observed zero-drop
count does not imply that Switch routing cannot drop assignments.

## Descriptive real payload manifest

The predictor inspected the actual loaded stage-qualified expert modules and
sealed payload manifest SHA-256
`0362d81253cb3d0a81ccd2a9ddff6032a3cccd142d8638cf66171126c00f336a`.
It independently confirmed:

- `96` canonical `(routing_stage, layer, expert_id)` payloads;
- exactly `wi.weight` and `wo.weight` per expert;
- `wi.weight` shape `[3072, 768]`, CPU float32, `9,437,184` logical bytes;
- `wo.weight` shape `[768, 3072]`, CPU float32, `9,437,184` logical bytes;
- both components contiguous in every expert;
- `18,874,368` logical parameter payload bytes per expert.

These are `numel * element_size` descriptions of parameters, not measured
DRAM, PCIe, H2D, or native-residency bytes.

## Frozen simulated prediction

The predictor ran exactly one `simulate_versioned_byte_cache(...)` invocation
and durably sealed the result before executor startup. Frozen prediction
SHA-256 was
`d4a7e89bc820903a16ee1cecab45d4398fa3b7b07b37296715887fd005857736`.

- Events / requests: `156 / 156`.
- Hits / misses: `89 / 67`.
- Simulated demand-load bytes: `1,264,582,656`.
- Evictions / simulated evicted bytes: `59 / 1,113,587,712`.
- Peak / final resident bytes: `150,994,944 / 150,994,944`.
- Final canonical resident keys:
  - `decoder / 1 / 4`
  - `decoder / 3 / 0`
  - `decoder / 5 / 2`
  - `decoder / 7 / 7`
  - `decoder / 9 / 0`
  - `decoder / 9 / 4`
  - `decoder / 11 / 1`
  - `decoder / 11 / 4`

## Independent measured test-replay observation

The fresh executor process received only the trace and manifest identities,
model/runtime identity, cold-LRU policy, byte capacity, slot count, and output
path. Its input schema cannot contain predicted hits, misses, miss order,
eviction decisions, demand bytes, or final keys. Its module does not import or
call project cache-replay or hardware-cost code.

The executor independently replayed the raw events once. On every miss it
selected its own LRU victim, then synchronously copied both actual native
parameter components into one of eight preallocated artificial CPU slots.
Slots never replaced native parameters and never participated in inference.

- Executor hits / misses / evictions: `89 / 67 / 59`.
- Completed physical logical loads: `67`.
- Actual component copies: `134`.
- Profiler CPU `aten::copy_` operators: `134`.
- Actual source operand logical bytes: `1,264,582,656`.
- Every source/destination shape, dtype, device, and logical byte count matched.
- Every loaded key copied exactly `18,874,368` source operand bytes.
- Every destination was exactly `torch.equal` to its source after copying.
- Executor final resident bytes and keys matched the frozen prediction exactly.

The runtime observation SHA-256 was
`53497d88c9ca14ece408a107d359e2294baee37a79df70e379c95bf334207ca6`.
Profiler timing was not used and no bandwidth was calculated.

## Exact comparison and artifacts

The comparer opened the sealed prediction only after executor completion. All
integer, identity, byte, key, operator-count, and content checks used exact
equality with no tolerance or repair. Result: **AGREEMENT**.

The attempt artifact manifest SHA-256 was
`8534596a2d0da683e9d26c1e07bee407eb52f7fda4f7dbf3b0416de185c4edae`.
The attempt directory, raw trace, raw prompt, payload manifest, copied-value
observations, and process records were not retained in the source tree because
they include sensitive/raw execution material beyond this bounded record.

The desktop shell wrapper stopped polling after its short diagnostic timeout,
but the already-started parent and executor processes continued and finalized
this same single attempt. No second attempt or protocol change occurred.

## Limitations

- AGREEMENT validates only the constructed CPU test replay and the accounting
  boundary exercised by this one pinned trace.
- Native Switch inference did not use these slots and is not shown to move
  expert weights on simulated misses.
- This result does not measure DRAM traffic, GPU residency, H2D/PCIe traffic,
  latency, throughput, tokens/sec, bandwidth, speedup, memory savings, energy,
  production offload, or policy/capacity optimality.
- The existing serialized H2D service model remains **ESTIMATED** under caller
  assumptions and was not validated by this CPU-copy experiment.
- One public prompt, four fed-back decoder inputs, one model/revision, and zero
  observed drops do not establish general model-family correctness.
- No expert meaning is inferred from identifiers or routing frequency.
- The outcome is limited to the constructed replay and does not extend to
  native runtime caching or broader runtime engineering.
