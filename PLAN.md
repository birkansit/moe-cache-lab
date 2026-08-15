# V0.1 through V0.4 engineering plan

- [x] Create the package skeleton and repository structure.
- [x] Define versioned routing-trace storage and reader.
- [x] Implement LRU, LFU, and static global-frequency cache simulation.
- [x] Add concise text/Markdown reporting and synthetic tests.
- [x] Install the project environment and run the automated suite.
- [x] Load the reference Granite model, record a real trace, simulate it, and
  inspect the report.
- [x] Repair V0.1 simulation correctness: layer-qualified identity, atomic
  routing-event replay, shared-cache phase attribution, explicit transfer
  accounting, and an explicitly non-causal offline oracle.
- [x] Repeat the corrected real Granite smoke validation and retain only
  corrected results as valid evidence.
- [x] Add the versioned 4-calibration/8-evaluation local corpus and canonical hash.
- [x] Add one-load CPU corpus collection with a versioned, hashed manifest.
- [x] Add leakage-safe calibrated static targets and evaluation-only capacity curves.
- [x] Add auditable V0.2 reporting and synthetic reproducibility/tamper tests.
- [x] Pin model/tokenizer collection to one resolved Hugging Face commit and
  validate complete per-token routed-layer structure.
- [x] Disclose grouped Granite prefill versus the per-token simulation and add
  measured prompt/layer working-set metadata.
- [x] Run the real V0.2 corpus collection and store evidence under
  `results/v0.2-corpus-v1/` with the report at
  `results/v0.2-corpus-v1-report.md`.

The original V0.1 cache-simulation results are invalid because they replayed
raw numerical expert IDs independently instead of layer-qualified routing-event
bundles. They are superseded by the corrected abstraction and must not be used.

V0.2 is a simulation/analysis milestone, not an inference accelerator.

## V0.3 runtime-fidelity and feasibility

- [x] Freeze the Stage 1 experiment before collecting the final evidence.
- [x] Add immutable token/layer and conservative grouped-prefill views.
- [x] Add EOS-aware, three-process, offline/local-only collection with strict
  repeatability and V0.2-prefix gates.
- [x] Add deterministic paired controls, fixed-plan provenance, macros, N/A
  handling, and the prospectively frozen continuation decision.
- [x] Collect, validate, and regenerate real Stage 1 evidence without tripling
  denominators.
- [x] Inspect exact model layout, local runtime support, memory, raw transfer
  capability, lifecycle hooks, licenses, and one alternate runtime route.
- [x] Record the Stage 2 local no-go; do not implement a fake or unsupported
  expert-offload path.
- [x] Complete `0.3.0rc1` packaging and evidence validation.

Stages 3-5 are deliberately inapplicable on the tested environment because no
working real runtime path passed Stage 2. The resulting deliverable is the
profiler/simulator plus measured evidence and documented local infeasibility.

## V0.4 observer-overhead measurement

- [x] Preserve the `0.3.0rc1` checkpoint as a historical baseline.
- [x] Freeze the exact CPU full-resident reference-versus-profiler overhead
  experiment and host-resource coordination rules.
- [x] Implement and synthetically validate the two-mode driver and manifests.
- [x] Run the admitted real four-child experiment and validate the immutable
  evidence. Attempt `b247eb85...` is `valid_stable`: mean paired CPU
  observer/reference ratio `1.0208846178`, with all frozen stability gates
  passed.
- [x] Prepare the bounded local `0.4.0rc1` package without treating the result as
  offloading, GPU inference, or a speed claim.
- [x] Run the release-candidate package and evidence validation gate.

V0.4 remains CPU-only observation. It does not reopen Stages 3-5 or justify
runtime offloading, tuning, GPU inference, or a speedup claim.
