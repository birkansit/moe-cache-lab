# v0.9 — External evidence intake

## Purpose and evidence boundary

This record documents two historical externally hosted routing-trace artifacts as a metadata-first evidence intake:

- Kimi-Linear-48B historical raw routing fixture with the corrected canonical v2 representation;
- OLMoE-1B-7B-0125 historical raw routing fixture with the canonical v1 representation.

The mechanical reproduction evidence is recorded in `V09_EXTERNAL_CANONICAL_REPRODUCTION.md`. The governing producer-validation contract is `PRODUCER_CONFORMANCE.md`.

The three producer claims remain independent:

1. **Canonical trace valid**;
2. **Producer path semantically validated**;
3. **Producer path non-interference validated**.

Artifact intake acceptance below is a separate evidence-governance decision. It is not producer conformance and does not upgrade a producer-validation claim.

External trace bytes are not copied into this repository. The artifacts remain externally hosted and are referenced by immutable source identities and hashes.

Public provenance discussion: `https://github.com/birkansit/moe-cache-lab/issues/7`.

## Claim summary

| Claim or evidence check | Kimi historical artifact | Historical OLMoE artifact |
| --- | --- | --- |
| Mechanical reproduction | `AGREEMENT` | `AGREEMENT` |
| Raw-to-canonical preservation audit | `PASS` | `PASS` |
| Exact public v0.8.0 canonical validation | `PASS` | `PASS` |
| Canonical trace valid | `PASS` | `PASS` |
| Producer path semantically validated | `NOT ESTABLISHED` | `NOT ESTABLISHED` |
| Producer path non-interference validated | `NOT ESTABLISHED` | `NOT ESTABLISHED` |
| Native selection order/rank | `NOT PRESERVED` | `NOT PRESERVED` |
| Artifact intake decision | `ACCEPTED AS BOUNDED EXTERNAL INTAKE EVIDENCE` | `ACCEPTED AS BOUNDED HISTORICAL EXTERNAL INTAKE EVIDENCE WITH PROVENANCE GAPS` |

Canonical validity means only that the frozen canonical representation satisfies the selected closed trace contract and chronology/identity rules enforced by the v0.8.0 reader. It does not establish correct native observation semantics or non-interference.

## Kimi-Linear-48B historical intake

### Immutable artifact identity

External contributor: `mfethe1`.

Raw source owner/repository: `sqliteai/warp`.

- commit: `6e579f99689a3e7e2ea875a88d97f68c8bfd3f70`
- path: `tests/trace_kimi_300.jsonl`
- Git blob: `7487627d2cf7dec4c89d70a5ca42244e85a34823`
- content SHA-256: `06ad026c928d1c05eb59f0b1db55e1317c1899b1d3a687741ab5c49fdc014965`

Canonical converter owner/repository: `mfethe1/warp`.

- commit: `6d9800ff08e912e330efc85a2f443784857e262e`
- path: `tools/moe-cache-lab/convert_kimi_olmoe_canonical.py`
- Git blob: `b2a5d91933f45e3e6d1a03a8ee1b3e4b27e24414`
- content SHA-256: `3f3c43db566e354cd6a7f1bf34f46bf8b7f3a386f85d9d1ec757ab6e467eeb1f`

Accepted corrected canonical representation:

- trace format: `moe-cache-lab.routing-jsonl` v2
- routing stage: `decoder`
- phase: `decoder_generated`
- canonical SHA-256: `3a2367a43e6d8d3bd3aefc91ef9e294428d1444b0822e5a86dcc777c21e418ac`
- canonical events: 7,800
- expert requests: 62,400
- assigned width: 8
- expert universe per decoder stage: 256
- canonical layer range: `1..26`
- unassigned events: 0

Superseded defect lineage only:

`7f08cae05115f97b881a529177c0d345f8dbd0fb87195702aab26442a04fa756`

That earlier canonical artifact used the semantically incorrect `0..25` layer mapping. It remains defect history and is not current evidence.

### Capture and workload provenance

Capture tool/source: `sqliteai/warp` `tools/kimi_ref.py` at commit `6e579f99689a3e7e2ea875a88d97f68c8bfd3f70`.

Recorded artifact boundary:

- decode-only routing trace;
- batch 1;
- raw token positions `0..299`;
- 300 raw token rows;
- 26 routed-layer expert lists per row;
- 8 selected routed experts per event.

Canonical model identifier recorded by the reproduced trace:

`Kimi-Linear-48B/warp-3bit-oracle (warp tests/trace_kimi_300.jsonl @6e579f99)`

Capture-specific immutable model/checkpoint revision: **`NOT RECORDED`**.

Capture-specific Python version: **`NOT RECORDED`**.

Capture-specific Torch version: **`NOT RECORDED`**.

Capture-specific device/dtype: **`NOT RECORDED`**.

Capture-specific exact prompt, seed, and sampling settings: **`NOT RECORDED`** in the accepted intake evidence. Source-code defaults or examples are not substituted for missing run provenance.

The execution environment documented in `V09_EXTERNAL_CANONICAL_REPRODUCTION.md` is the later canonical-reproduction environment, not the historical capture environment.

### Native observation boundary and routed membership

The frozen producer source computes router logits, applies `sigmoid`, adds the correction bias for selection, obtains `topk_idx`, and uses those same `topk_idx` expert IDs in the routed-expert execution loop. The trace path records the selected IDs from that same tensor.

Evidence classification:

**Bounded `SOURCE-READ` support for selected routed-expert membership mapping on the reviewed producer path.**

This is meaningful semantic support for expert membership. It does not by itself satisfy the complete producer semantic-validation gate because native layer identity has a separate capture-specific limitation and the historical run provenance is incomplete.

### Native layer identity

The raw fixture contains 26 routed-layer lists per token but does not encode the native model layer number in each list. Capture-era source/config evidence supports one dense layer followed by routed model layers `1..26`, and the corrected converter maps raw routed ordinal `0..25` to canonical layer `1..26`.

Status:

**`SOURCE-SUPPORTED INFERENCE — NOT CAPTURE-PROVEN`**

The exact capture container manifest/config was not archived with the historical fixture. Mechanical reproduction of the corrected converter proves that the declared mapping is reproduced; it does not make the mapping capture-specific ground truth.

Therefore:

**Producer path semantically validated: `NOT ESTABLISHED`.**

### Chronology, membership preservation, probabilities, and rank

The A1 preservation audit establishes that:

- raw token chronology `0..299` is preserved;
- every raw routed ordinal is represented exactly once in canonical order;
- `token_position` is preserved;
- the corrected canonical layer is raw ordinal plus one;
- the accepted canonical mapping sets `routing_stage="decoder"` and `phase="decoder_generated"` without reconstructing raw probabilities or repairing chronology;
- `selected_experts` equals the raw-emitted expert list exactly;
- no event is added, dropped, reordered, or chronology-repaired.

The raw producer writes `sorted(topk_idx[-1].tolist())`. Consequently:

- routed-expert membership is retained;
- native top-k order/rank is **`NOT PRESERVED`**;
- `selected_probabilities` is empty because the raw fixture did not retain probabilities/weights; none are reconstructed.

### Execution-path coverage

The reviewed Kimi producer path executes selected routed experts and also executes a separate `shared_experts` path. The routing fixture records the routed expert selections only.

Coverage statement:

**Routed-expert selections are represented; the separate shared-expert execution path is outside this trace.**

The artifact is not a complete model execution, residency, or transfer trace.

### Expert-size status

`2,359,296 B` per expert is retained only as a **caller-supplied analysis assumption** used in external analysis. It is not measured/model-matched physical extent evidence for this historical capture and is not promoted by this intake.

### Reproduction and canonical validation

`V09_EXTERNAL_CANONICAL_REPRODUCTION.md` records:

- raw content hash: `AGREEMENT`;
- two-run canonical reproduction: `AGREEMENT`;
- deterministic reproduction in the recorded reproduction environment: `PASS`;
- raw-to-canonical preservation audit: `PASS`;
- exact public `moe-cache-lab v0.8.0` canonical validation: `PASS`;
- 7,800 canonical events and 62,400 expert requests.

These results establish the frozen artifact-to-canonical mechanical chain and canonical reader validity. They do not establish non-interference or capture-proven native layer identity.

### Producer non-interference

No frozen control-vs-instrumented comparison for this historical producer path establishes unchanged routing/output behavior under `PRODUCER_CONFORMANCE.md`.

**Producer path non-interference validated: `NOT ESTABLISHED`.**

### Redistribution status

The raw fixture is externally hosted in a public source repository described by the contributor as Apache-2.0. That public availability is not treated as explicit project permission to copy and redistribute the routing-trace bytes from a project-controlled surface.

Current intake mode: **metadata-only external reference**.

Project-controlled copying/redistribution of raw or canonical trace bytes: **`NOT ESTABLISHED`** for this intake.

### Artifact intake decision

**`ACCEPTED AS BOUNDED EXTERNAL INTAKE EVIDENCE`**

Permitted meaning:

- the immutable external raw artifact and corrected canonical v2 representation may be referenced as a reproducible external evidence case;
- downstream routing analysis may consume the canonical representation when the source and hash identity are retained;
- downstream cache outcomes must remain labeled **SIMULATED**;
- downstream transfer-service quantities must remain labeled **ESTIMATED** under explicit caller-supplied assumptions;
- the native `1..26` layer mapping must remain labeled `SOURCE-SUPPORTED INFERENCE — NOT CAPTURE-PROVEN`.

This acceptance does **not** imply producer semantic validation, producer non-interference, native-rank preservation, complete model execution coverage, measured physical expert extents, physical residency/transfer, latency, throughput, speedup, representative workload behavior, broad Kimi compatibility, or an optimal policy/capacity.

## Historical OLMoE-1B-7B-0125 intake

### Immutable artifact identity

External contributor: `mfethe1`.

Raw source owner/repository: `sqliteai/warp`.

- commit: `dc079dad1c35def2ee7bb3b2de5157db8d9f08a8`
- path: `tests/trace_olmoe_299.jsonl`
- Git blob: `77e162d8b6e3d7c15eb92b7e52654d824cd6a68b`
- content SHA-256: `7d8e7f63551aa24ddab98134b267b6bdf1b4f55d72a6fa7589042f54a4a4383b`

Canonical converter owner/repository: `mfethe1/warp`.

- commit: `6d9800ff08e912e330efc85a2f443784857e262e`
- path: `tools/moe-cache-lab/convert_kimi_olmoe_canonical.py`
- content SHA-256: `3f3c43db566e354cd6a7f1bf34f46bf8b7f3a386f85d9d1ec757ab6e467eeb1f`

Accepted canonical representation:

- trace format: `moe-cache-lab.routing-jsonl` v1
- canonical SHA-256: `83168fb10391646a97bb0c00c159094784a0ef72dc7962d18055020701f1d9b4`
- canonical events: 4,784
- expert requests: 38,272
- experts per event: 8
- expert universe: 64
- canonical layer range: `0..15`

### Capture and workload provenance

Capture tool/source: `sqliteai/warp` `tools/trace_hf.py` at commit `dc079dad1c35def2ee7bb3b2de5157db8d9f08a8`.

Recorded artifact boundary:

- decode-only routing trace;
- batch 1;
- raw token positions `1..299`;
- 299 raw token rows;
- 16 layer expert lists per row;
- 8 selected experts per event.

Canonical model identifier recorded by the reproduced trace:

`OLMoE-1B-7B-0125/warp-fixture (warp tests/trace_olmoe_299.jsonl @dc079dad)`

Contributor-supplied model identity: `allenai/OLMoE-1B-7B-0125`.

Capture-specific immutable model revision: **`NOT RECORDED`**.

Capture-specific Python version: **`NOT RECORDED`**.

Capture-specific Torch version: **`NOT RECORDED`**.

Capture-specific Transformers version: **`NOT RECORDED`**.

Capture-specific device/dtype: **`NOT RECORDED`**.

Capture-specific exact invocation: **`NOT RECORDED`**.

Capture-specific active gate-output branch: **`NOT RECORDED`**.

Capture-specific seed and sampling settings: **`NOT RECORDED`** in the accepted intake evidence. The capture script contains source defaults, but those defaults are not substituted for missing run evidence.

A later OLMoE recapture with additional receipt information is a separate artifact and is not used to backfill any historical field above.

### Observation-boundary status

The frozen historical source script supports two gate-output branches:

- when the gate output is a tuple, it reads tuple element 2 as `topk_index`;
- otherwise it recomputes top-k from the returned router logits.

The script then sorts the selected IDs before emitting the raw expert list.

Because the exact historical Transformers version and active branch are not recorded, the capture-specific native observation boundary cannot be established from the retained run provenance.

**Producer path semantically validated: `NOT ESTABLISHED`.**

Source-level structure may still be stated separately: the hook is registered on each enumerated model layer, pairs each observed list with `layer_idx`, orders the collected per-step pairs by layer index, and writes the lists for the completed decode step. Those source facts do not identify which of the two expert-selection branches produced the historical bytes.

### Chronology, layer identity, membership preservation, and rank

The A1 preservation audit establishes that:

- raw token chronology `1..299` is preserved;
- 16 raw layer lists are present per token;
- canonical `token_position` equals raw `tok`;
- canonical layer equals raw layer ordinal `0..15`;
- the accepted canonical mapping sets `phase="generated"` without reconstructing probabilities or repairing chronology;
- `selected_experts` equals the raw-emitted list exactly;
- no event is added, dropped, reordered, or chronology-repaired.

The producer emits `sorted(idx[-1].tolist())`, so native top-k order/rank is **`NOT PRESERVED`**. The accepted canonical representation preserves only the raw-emitted sorted expert list.

Selection probabilities/weights are not present in the historical raw fixture and are not reconstructed by the canonical converter.

### Shared-expert coverage

The reviewed OLMoE implementation context does not identify a separate shared-expert execution path analogous to the Kimi `shared_experts` path. The exact historical Transformers version is not recorded, so this intake does not use later source inspection to invent capture-specific structure.

Accordingly, this historical record does **not** claim that a separate OLMoE shared-expert path was omitted from the trace. The unresolved historical limitation is the active gate-output branch and environment provenance, not an assumed shared-expert omission.

### Reproduction and canonical validation

`V09_EXTERNAL_CANONICAL_REPRODUCTION.md` records:

- raw content hash: `AGREEMENT`;
- two-run canonical reproduction: `AGREEMENT`;
- deterministic reproduction in the recorded reproduction environment: `PASS`;
- raw-to-canonical preservation audit: `PASS`;
- exact public `moe-cache-lab v0.8.0` canonical validation: `PASS`;
- 4,784 canonical events and 38,272 expert requests.

Mechanical agreement does not recover the missing historical model revision, library/runtime environment, device/dtype, exact invocation, or active observation branch.

### Producer non-interference

No frozen control-vs-instrumented comparison for this historical producer path establishes unchanged routing/output behavior under `PRODUCER_CONFORMANCE.md`.

**Producer path non-interference validated: `NOT ESTABLISHED`.**

### Redistribution status

The raw fixture is externally hosted in the same public source repository described by the contributor as Apache-2.0. Public availability and repository licensing are not treated as explicit project permission to copy and redistribute the historical routing-trace bytes from a project-controlled surface.

Current intake mode: **metadata-only external reference**.

Project-controlled copying/redistribution of raw or canonical trace bytes: **`NOT ESTABLISHED`** for this intake.

### Artifact intake decision

**`ACCEPTED AS BOUNDED HISTORICAL EXTERNAL INTAKE EVIDENCE WITH PROVENANCE GAPS`**

Permitted meaning:

- the immutable external raw artifact and canonical v1 representation may be referenced as a historical external interoperability/evidence case;
- its mechanical reproducibility and canonical validity may be cited exactly;
- its unavailable historical run provenance must remain visible;
- downstream cache outcomes, if produced from this representation, must remain labeled **SIMULATED**;
- downstream transfer-service quantities, if produced, must remain labeled **ESTIMATED** under explicit caller-supplied assumptions.

This acceptance does **not** imply capture-specific producer semantic validation, producer non-interference, native-rank preservation, recovery of the historical model revision/environment/device/dtype/invocation/active branch, physical residency/transfer, latency, throughput, speedup, representative workload behavior, broad OLMoE compatibility, or an optimal policy/capacity.

## Separate later OLMoE capture

A newer OLMoE capture discussed in the public provenance thread contains additional run receipt information. It is a separate evidence candidate. It does not repair, replace, or infer missing provenance for the historical `dc079dad...` fixture, and it is not part of either intake decision in this record.

Its eventual status requires its own immutable artifact, capture-script, converter, receipt, model-revision, semantic, and non-interference review boundaries.

## Intake conclusion

The two historical external artifacts are accepted only at the bounded artifact level stated above.

Both have independently reproduced raw/canonical identities, full-fixture raw-to-canonical preservation checks, and exact public v0.8.0 canonical-validation `PASS` results. Those facts support metadata-first external evidence use and clearly labeled downstream simulation/estimation.

Neither historical producer path satisfies the complete semantic-validation gate under `PRODUCER_CONFORMANCE.md`, and neither has established non-interference. Kimi retains bounded SOURCE-READ support for routed-expert membership but its `1..26` layer mapping remains `SOURCE-SUPPORTED INFERENCE — NOT CAPTURE-PROVEN`. Historical OLMoE retains explicit missing capture provenance and an unknown active observation branch.

No claim in this intake establishes runtime performance, physical residency or transfer, broad model/runtime compatibility, workload representativeness, policy optimality, or production suitability.
