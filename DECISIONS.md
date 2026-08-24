# Architectural decisions and rejected scope

This file preserves architectural decisions and rejected or deferred scope
across earlier milestones. Its historical entries remain useful provenance.
The current source capability and claim boundaries are defined by
[`README.md`](README.md) and
[`V08_RELEASE_NOTES.md`](V08_RELEASE_NOTES.md). Published v0.7 history remains
in [`V07_RELEASE_NOTES.md`](V07_RELEASE_NOTES.md); historical statements here
do not override the current source contract.

## Decisions

- Start with one backend and one model family: Hugging Face Transformers with
  Granite MoE.
- Keep V0.2 CPU-first and Windows-friendly.
- Favor correctness before optimization.
- Build the profiler and offline simulator before attempting real acceleration.
- Preserve the model's routing semantics; collection is observational only.
- Existing open-source implementations may be studied or integrated only after
  their licenses and attribution obligations are understood.
- Cache identity is `(layer_id, expert_id)`, and capacity is one global count of
  simultaneously resident layer-qualified expert-weight objects.
- Replay is atomic at `RoutingEvent` granularity. Dynamic policies start empty;
  fixed-oracle prewarm and all later loads, including fixed-target restoration,
  are reported separately without hidden transient capacity.
- `offline_oracle_frequency` is an explicitly non-causal, immutable baseline
  chosen using the complete evaluation trace; it is not a deployable policy.
- `calibrated_static_frequency` uses the same fixed-target restoration semantics
  but may select targets only from the disjoint calibration split.
- All V0.2 reported outcomes use evaluation traces only, in manifest order, and
  preserve one simulated cache across prompts. Model contexts remain independent.
- Corpus categories are workload-diversity metadata and never evidence of expert
  semantics. Preserve unfavorable policy results without tuning them away.
- Canonical corpus and per-trace SHA-256 values plus environment/generation
  metadata are required for a reproducible suite run.
- Resolve and pin one immutable Hugging Face commit for config, tokenizer, and
  weights; record it in every V0.2 manifest and trace.
- Keep per-token/layer atomic simulation while disclosing that Granite prefill
  dispatch is grouped per layer and does not validate sub-working-set runtime
  scheduling.
- Track final traces/manifests/reports under `results/v0.2-corpus-v1/` and
  `results/v0.2-corpus-v1-report.md`; never place model binaries there.

## Deferred or rejected scope

- Native-router modification.
- Kimi K3 support.
- SSD offload.
- RAG.
- Learned expert prediction.
- Multiple inference backends.

These omissions are deliberate V0.2 boundaries, not claims that the ideas lack
value. Any change requires an explicit milestone and validation plan.

## Stage 1 and Stage 2 decisions

- Preserve both `token_layer_atomic` and the conservative
  `prefill_layer_union_atomic` sensitivity view. The grouped view is primary
  for prefill policy interpretation but is not verified physical residency.
- Treat repetitions as deterministic validation checks, never additional
  statistical samples or multiplied denominators.
- The prospectively frozen Stage 1 continuation rule passed and justified only
  read-only feasibility inspection.
- Record the Stage 2 no-go for the tested Windows/RX 6650 XT/backend
  environment. Raw HIP copying is measurable, but CPU-only PyTorch and packed
  expert Parameters expose no supported per-expert residency mechanism.
- Do not implement Stage 3 merely because simulated locality exists. A future
  environment change must first establish a supported full-resident baseline;
  raw parameter bytes already fit reported VRAM.
- Stages 4 and 5 remain inapplicable without a real runtime path and a measured
  miss bottleneck. Negative feasibility is a valid release outcome.

## V0.4 observer measurement decisions

- Measure only the incremental overhead of the routing observer against the same
  full-resident CPU-FP32 execution driver with observation disabled.
  `V04_EXPERIMENT.md` freezes the design.
- Use four fresh child processes with both modes in every child, exact
  prompt/mode counterbalancing, unreported calibration warmup, strict semantic
  equivalence, trusted Stage 1 routing/probability comparison, and hierarchical
  n=4/n=8 reporting.
- Admit every child only on an idle/resource-safe host and preserve invalid
  attempts. Resource thresholds and the 20-minute ceiling are evidence rules,
  not tuning knobs.
- The accepted attempt `b247eb85-eb21-4e3a-886e-6ba250cbb980` is the V0.4
  measured CPU observer result. Across four paired fresh-process checks its mean
  ratio is `1.0208846178255349` (about 2.09% added wall time), all frozen
  stability gates pass, and exact semantic/routing IDs match with probability
  delta `0.0`. This is workload- and environment-specific observer overhead,
  not cache acceleration, offloading, GPU performance, or production throughput.
- Preserve deferred attempt `d6f7cac2-d67a-46e8-8267-d65c861ea009` as the
  retry source. The valid attempt's exact 11-file evidence tree and deterministic
  JSON/Markdown regeneration are pinned by `tests/test_v04_valid_evidence.py`.
- Do not justify Stages 3-5 from V0.4. The next release step was bounded
  `0.4.0rc1` preparation and validation.
