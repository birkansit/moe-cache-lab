# v0.8 release notes

`moe-cache-lab` remains correctness-first research and decision-support
tooling, not an inference accelerator.

## Release scope

1. **Stage-qualified v2 pre-flight.** Canonical trace v2 can be paired with
   strict pre-flight config v3 for single-trace cache and transfer-service
   analysis. Encoder and decoder expert identities remain exactly
   `(routing_stage, layer_id, expert_id)`. Capacity-unassigned events remain
   visible routing events but create zero expert requests. Config v1/v2 and
   trace v1 behavior remain unchanged; stage-qualified lifecycle orchestration
   is not implemented.
2. **External producer validation toolchain.** `validate-trace` exposes the
   strict canonical v1/v2 JSONL boundary with stable machine-readable failure
   categories and official valid/invalid fixtures. Passing establishes
   canonical-file validity only. Producer semantic mapping and
   non-interference are separate claims governed by
   [`PRODUCER_CONFORMANCE.md`](PRODUCER_CONFORMANCE.md).
3. **Executable no-model interoperability workflow.** The packaged
   `examples/external-producer-no-model` example uses a standalone
   standard-library producer to emit deterministic **SYNTHETIC** trace-v2
   JSONL, then validates, analyzes, runs compatible v2/config-v3 pre-flight,
   and creates/verifies a bundle. It requires neither Torch nor Transformers.
4. **Historical Python portability.** Frozen V0.4 aggregation uses an explicit
   deterministic numerical contract across supported Python 3.10-3.12 while
   preserving the original result and report bytes.
5. **Publication and distribution hygiene.** Whole-tree leakage checks,
   release-surface tests, distribution audits, and installed-wheel core checks
   protect the intended documentation, schemas, fixtures, example, and
   lightweight dependency boundary.
6. **External-runtime research gate.** Exact vLLM and llama.cpp revisions were
   inspected under a frozen contract. The result is a bounded **NO-GO** at the
   recorded no-model Windows/source boundary because real execution,
   chronology, non-interference, independent equivalence, and downstream
   acceptance were not all established. This **SOURCE-READ** result is not a
   universal runtime rejection. No runtime adapter or integration was added.

## Evidence and compatibility boundary

- Routing output is descriptive and is **MEASURED** only when trace provenance
  establishes an actual observation.
- Cache hits, misses, loads, evictions, and resident-byte accounting are
  **SIMULATED** event-atomic replay.
- Serialized transfer service is **ESTIMATED** from simulated logical loads and
  explicit caller-supplied assumptions.
- Native caching, physical CPU/GPU residency, physical DRAM/H2D traffic,
  runtime integration, latency, throughput, tokens per second, speedup, memory
  savings, representative coverage, and an optimal policy/capacity remain
  **NOT ESTABLISHED** by these workflows.

Trace v1/config v1-v2 remains layer-qualified. Trace v2/config v3 remains
stage-qualified. Invalid chronology, incompatible versions, and missing
identity are rejected rather than repaired, migrated, inferred, or flattened.
Canonical validity does not upgrade producer evidence, and bundle integrity
does not upgrade scientific evidence.

The base package has no Torch or Transformers dependency. The model-specific
`[granite]` and `[switch]` extras retain their narrow reviewed dependency pins;
no public Switch collection command, generic producer framework, external
runtime adapter, or production offloading implementation is included.

The optional extras retain PyTorch 2.12.0 with Transformers 5.12.0 to preserve
the reviewed collector and research environment. PyTorch 2.12.0 is affected by
the GitHub-reviewed low-severity advisory `GHSA-rrmf-rvhw-rf47`
(`CVE-2025-3000`, `PYSEC-2025-194`) in `torch.jit.script`; tracked supported
`moe-cache-lab` code does not call that JIT API or an equivalent indirect path.
This bounded reachability finding is not a claim that PyTorch 2.12.0 is
generally safe. The pin is retained because the exact full-model producer and
non-interference evidence is version-frozen and has not been re-established on
PyTorch 2.13.0. The base installation remains Torch-free.
