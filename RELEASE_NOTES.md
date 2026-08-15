# moe-cache-lab 0.4.0

`0.4.0` is the first public release of `moe-cache-lab`.

It finalizes the validated `0.4.0rc1` engineering checkpoint. Historical
experiment evidence is retained, while the finalization changes package/release
metadata and public-facing documentation rather than reinterpreting results.

## Included

- observational Granite MoE router tracing without changing native routing;
- versioned JSONL routing traces;
- layer-qualified `(layer_id, expert_id)` cache identity;
- atomic routing-event cache replay;
- LRU, online LFU, calibrated static, and non-causal offline-oracle baselines;
- conservative grouped-prefill sensitivity analysis;
- deterministic Stage 1 benchmark/evidence workflows;
- documented Stage 2 local runtime-feasibility no-go;
- measured V0.4 CPU routing-observer overhead evidence;
- portable CI covering 120 unit/evidence tests plus package smoke checks.

## Key measured result

On the pinned V0.4 CPU workload, the routing observer added approximately 2.09%
mean wall time across the accepted paired run. Router selections were identical
and the maximum selected-probability delta was 0.0.

## Important boundaries

`moe-cache-lab 0.4.0` is **not an inference accelerator**. Cache behavior is
simulated, transfer counts are estimated, and the project does not implement
real expert CPU/GPU swapping or claim runtime speedup.

The Stage 2 no-go applies to the tested Windows / RX 6650 XT / pinned official
backend environment. It does not claim that expert caching or offloading is
impossible on other supported runtimes or hardware.

## Reproducibility

Evidence hashes and the historical `0.4.0rc1` checkpoint remain covered by the
release test suite. See `RELEASE_CANDIDATE.md`, `STAGE1_RESULTS.md`,
`STAGE2_FEASIBILITY.md`, and `V04_RESULTS.md` for the technical record.

## License

Apache License 2.0. See `LICENSE`.
