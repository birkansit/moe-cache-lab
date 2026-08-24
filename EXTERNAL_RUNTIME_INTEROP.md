# External runtime interoperability research gate

## Purpose and evidence boundary

This v0.8 D gate asks whether one external inference runtime can produce a
canonical routing trace from an authoritative native routing or actual expert
dispatch boundary without changing the decision being observed. The two
required candidates are vLLM and llama.cpp. Candidate inspection is bounded to
the exact revisions, source paths, environment, and experiment decision
recorded below.

Source inspection is **SOURCE-READ evidence**, not measured runtime evidence.
Canonical validation, if reached, would establish canonical-file validity only.
Downstream cache outcomes would remain **SIMULATED**, transfer-service results
would remain **ESTIMATED**, and all runtime behavior not directly tested would
remain **NOT ESTABLISHED**.

## Frozen decision contract

These criteria were frozen on 2026-08-23 before inspecting either candidate's
source. Findings below cannot relax or redefine them.

### Mandatory PASS criteria

A candidate is `PASS` only when direct bounded evidence establishes every item:

1. Native routing or actual expert dispatch is observable at an authoritative
   runtime boundary.
2. Observation does not change the routing or dispatch decision being observed.
3. Canonical event chronology is recoverable in native order without sorting,
   repair, deduplication, or reconstructed ordering guesses.
4. Exact layer, routing-stage where applicable, and expert identity are
   recoverable without fabricated offsets or model-name inference.
5. Pre-capacity preference and post-capacity actual dispatch are not conflated.
6. Dropped or unassigned events are emitted only when the runtime exposes that
   state; no expert or drop reason is fabricated.
7. Model, immutable model revision where available, runtime revision, capture
   boundary, device/backend, and frozen workload provenance can be bounded.
8. An independent native/runtime result source can check the observed
   selection or dispatch directly.
9. The resulting v1 or v2 canonical trace passes the existing strict
   `validate-trace` path without migration, sorting, or repair.
10. Existing downstream descriptive analysis accepts the trace.
11. Stage-qualified pre-flight runs only when the observed semantics genuinely
    fit trace v2 and config v3; otherwise it is explicitly not run.
12. Every support claim remains inside the exact demonstrated
    runtime/model/device/workload scope.

Source readability, an internal tensor name, a profiling API, or a theoretical
hook location alone cannot satisfy these criteria. Failure to establish any
mandatory item prevents `PASS`.

### Frozen NO-GO rules

A candidate is `NO-GO` for the inspected boundary when any mandatory criterion
cannot be established, including when:

- only pre-capacity scores or preferences are observable while actual dispatch
  is required;
- fused or custom execution hides assignment, exact identity, or native order;
- chronology would need sorting, repair, deduplication, or inference;
- stage, layer, expert, assigned, or dropped state would need fabrication;
- observation would modify routing, dispatch, scheduling, or the result being
  checked in an unbounded way;
- no independent native/runtime equivalence check is available;
- the existing trace v1/v2 contract cannot faithfully represent the observed
  semantics;
- exercising the production routing path requires a model, tokenizer, weights,
  checkpoint, or GGUF artifact that is not already available and is not
  authorized for download;
- the local platform/backend or available toolchain cannot execute the relevant
  path without system-wide changes; or
- integration requires a runtime/offloading rewrite outside v0.8.

`NO-GO` is bounded negative evidence, not a claim that a runtime can never
support MoE observation. No new routing trace version, approximate adapter, or
mock-only execution may be used to turn a missing criterion into a positive
result.

### Bounded execution rule

Source and local-resource analysis precede execution. A real runtime experiment
is permitted only if it needs no model/tokenizer/weight download, exercises the
native production MoE routing or dispatch implementation rather than a mock,
requires no system-wide driver/toolchain change or runtime rewrite, is bounded,
and can directly establish at least one otherwise missing PASS criterion.

## Candidate revisions and source evidence

The revisions were resolved from each official upstream repository on
2026-08-23 and cloned read-only outside this repository before source
inspection. Moving branch names are not evidence identifiers.

| Candidate | Official upstream | Exact immutable revision | Commit date | License at revision |
| --- | --- | --- | --- | --- |
| vLLM | `https://github.com/vllm-project/vllm` | `b26039b09fc97aa00f095a99eda503b7dad594ec` | 2026-08-23 | Apache License 2.0 (`LICENSE`) |
| llama.cpp | `https://github.com/ggml-org/llama.cpp` | `8144f3192e5a3131cd043f284525e6ceebf82d0f` | 2026-08-23 | MIT License (`LICENSE`) |

Exact source files, symbols, and bounded findings follow. Paths are relative to
the pinned upstream checkout.

## vLLM source inspection

### Observation boundary

| Source | Symbol | Bounded finding |
| --- | --- | --- |
| `vllm/model_executor/models/granitemoe.py` | `GraniteMoeSparseMoeBlock.forward` | The model gate produces `router_logits`, which are passed into `FusedMoEFactory` execution. This is the score boundary, not by itself dispatch evidence. |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py` | `fused_topk`, `FusedTopKRouter._compute_routing` | The standard router materializes per-token `topk_ids` and weights. Scoring may be softmax or sigmoid and may renormalize according to the configured router. |
| `vllm/model_executor/layers/fused_moe/router/base_router.py` | `BaseRouter._select_experts` | A configured `capture_fn` receives logical `topk_ids` immediately after native routing. Capture occurs before expert-parallel load-balancing (EPLB) maps logical IDs to physical replicas. |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | `MoERunner._apply_quant_method` | In the modular path, the same selected IDs returned by the router are passed to `forward_modular`; in the monolithic path, router logits instead enter the fused kernel. |
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | `RoutedExpertsCapturer`, `bind_routed_experts_capturer` | The built-in binder attaches layer-qualified callbacks to supported modular routers or supported monolithic routing-replay kernels. Unsupported routers/kernels reject rather than silently omitting data. |
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | `RoutedExpertsManager.store_batch`, `RoutedExpertsManager.get` | Scheduler-side storage keys routing by KV-cache slots and reconstructs token-order arrays from request block IDs. The returned shape is token by layer by top-k. |
| `vllm/v1/core/sched/scheduler.py` | request-output routed-expert extraction | The scheduler has distinct prompt and decode extraction paths, including speculative-decode slices, and returns per-request routing chunks. |
| `vllm/v1/engine/output_processor.py` | `RequestState.routed_experts_chunks`, output concatenation | Per-request chunks are appended and concatenated in engine-output order for the final completion. |
| `vllm/config/model.py`, `vllm/engine/arg_utils.py`, `vllm/config/vllm.py` | `enable_return_routed_experts`, `--enable-return-routed-experts` | Capture is a first-class opt-in. The pinned revision rejects pipeline parallelism, context parallelism, and KV connectors with this option. |
| `tests/model_executor/test_routed_experts_capture.py` | routed-expert capture tests | Synthetic/unit coverage checks layer binding, logical capture before EPLB mapping, DP/SP layouts, and unsupported monolithic rejection. |
| `tests/kernels/moe/test_routed_experts_capture_monolithic.py` | monolithic routing-replay tests | Supported FlashInfer-backed kernels write a replay tensor with in-range top-k IDs. These kernel tests do not establish a real model trace or observer non-interference. |

The strongest observable value is therefore a native, materialized logical
top-k selection. With EPLB disabled on a supported modular path, source flow
shows that those IDs are also the IDs supplied to modular expert execution.
With EPLB enabled, the built-in public result deliberately remains logical and
does not expose the later physical-replica mapping. Monolithic capture is
kernel-specific and rejected when routing replay is unsupported.

No capacity-limited drop state appears in the inspected standard top-k path.
Consequently, an adapter could emit assigned selections only for a demonstrated
no-capacity-drop model/path; it could not invent unassigned events, a drop
reason, or a post-capacity state. Grouping, normalization, expert parallelism,
and custom routers would each remain part of the exact demonstrated scope.

### Identity and chronology

The capture binder records the `MoERunner.layer_id`, and the result has shape
`[sequence_length, layer_count, top_k]`. The scheduler's slot-indexed manager
and prompt/decode slicing are credible source-level mechanisms for retaining
per-request token order across batching, prefix caching, preemption, and
speculative decode. They were not exercised here. The exposed identity has no
routing-stage field. A decoder-only path could therefore potentially map to
trace v1 without an invented stage; this inspection does not justify a trace
v2 mapping or any stage-qualified pre-flight run.

### Non-interference and equivalence

The callback is invoked after the router has selected logical IDs, which avoids
recomputing expert choice from logits. Capture still adds device-buffer writes,
device-to-host transfer, scheduler storage, and output transport. The inspected
unit/kernel tests validate plumbing and value ranges, not equality of routing
and model output between otherwise identical real runs with capture disabled
and enabled. The public returned array is the capture product itself, not an
independent native result source. Thus criteria 2 and 8 are not established by
the pinned source.

### vLLM bounded verdict

`NO-GO` at revision `b26039b09fc97aa00f095a99eda503b7dad594ec` on
the inspected Windows/no-model boundary. The source contains a substantially
better candidate boundary than score reconstruction, but a real supported MoE
run is required to establish non-interference, per-request chronology,
canonical emission, downstream acceptance, and an independent equivalence
check. That run was unavailable under the frozen execution rule.

## llama.cpp source inspection

### Observation boundary

| Source | Symbol | Bounded finding |
| --- | --- | --- |
| `src/llama-graph.cpp` | `llm_graph_context::build_moe_ffn` | Gate probabilities are transformed according to model-specific grouping/gating rules, then `ggml_argsort_top_k` materializes `selected_experts` with shape top-k by tokens. |
| `src/llama-graph.cpp` | callback name `ffn_moe_topk` | The selected-expert tensor is named with the exact layer number and exposed to the graph callback before weights are gathered. |
| `src/llama-graph.cpp` | `build_lora_mm_id` calls | On the ordinary path the selected IDs are passed into expert gate/up/down matrix operations, so this tensor is closer to execution than gate scores. A GROVEMOE-specific transform occurs after the named callback, preventing a broad claim across architectures. |
| `src/llama-context.cpp` | `llama_context::graph_get_cb` | Graph nodes receive names such as `ffn_moe_topk-<layer>`, preserving an exact integer layer identity. |
| `include/llama.h` | `llama_context_params.cb_eval` | The public context parameters accept a graph-evaluation callback and user data. |
| `ggml/include/ggml-backend.h` | `ggml_backend_sched_eval_callback` | The callback first declares which nodes it wants, then receives selected tensors for observation. |
| `ggml/src/ggml-backend.cpp` | callback-aware scheduler compute | Enabling the callback uses node-range graph views and backend synchronization before observation. This is read-only with respect to tensor values, but it changes scheduling/synchronization behavior and is not source-only proof of decision non-interference. |
| `src/llama-batch.cpp` | `llama_batch_allocr` split methods | Native calls may split or regroup input into microbatches. The tensor preserves microbatch token-axis order, but the evaluation callback itself does not carry request, sequence, or position identifiers. |
| `tests/test-backend-ops.cpp` | `test_topk_moe` | Synthetic backend-operation tests exercise top-k and expert-ID matrix operations. They are not a production model/GGUF interoperability experiment. |
| `docs/build.md` | Windows build instructions | Windows builds are supported through CMake plus an MSVC or clang toolchain, but the required complete local build toolchain was not present. |

The `ffn_moe_topk-<layer>` node can expose materialized selected IDs rather
than requiring score reconstruction. For many model builders those IDs feed
the ID-indexed expert operations. The generic builder also contains
architecture-specific routing transformations, so exact semantics must be
established for one concrete model path rather than inferred for all MoE
models.

The inspected path does not expose a capacity-drop record. No unassigned state
or drop reason may be synthesized. It also has no routing-stage field. A
controlled decoder-only, single-sequence run might fit trace v1, but stage,
request identity, and chronology cannot be inferred from a tensor name alone.

### Non-interference and equivalence

The public callback is observational in API intent, but source shows that its
presence changes graph batching and introduces backend synchronization. No
real-model comparison at the pinned revision established that selected IDs and
model outputs remain identical with and without observation. The selected
tensor subsequently used by expert operations is not an independent result
source; the synthetic backend test does not close that gap.

### llama.cpp bounded verdict

`NO-GO` at revision `8144f3192e5a3131cd043f284525e6ceebf82d0f` on
the inspected Windows/no-model boundary. The graph callback is a plausible
bounded research boundary, but native chronology, non-interference,
independent equivalence, canonical emission, and downstream acceptance require
a real model execution that was unavailable under the frozen execution rule.

## Local/resource feasibility

The feasibility snapshot was taken on 2026-08-23 and contains no machine
identifier:

- Windows 11 Pro 10.0.26200, 15.91 GiB physical RAM, with 6.06 GiB free at the
  snapshot;
- Python 3.10 and 3.14 available; vLLM was not installed;
- no native vLLM Windows execution path is supported by the pinned upstream
  `setup.py` and GPU installation documentation;
- the WSL command was present but no Linux distribution was installed;
- no NVIDIA runtime utility or detectable CUDA environment was available;
- llama.cpp source supports Windows, but no `llama-cli`/`llama-server`, CMake,
  Ninja, or MSVC compiler was available; clang alone was insufficient for the
  documented build path; and
- the upstream llama.cpp checkout contained tokenizer/vocabulary GGUF files,
  but no executable MoE model weights. Its synthetic `test_topk_moe` path does
  not meet the frozen production-routing criterion.

No suitable MoE weights/checkpoint/GGUF were already available inside the
authorized project scope. Obtaining one would be a forbidden model download,
and installing WSL, a system toolchain, drivers, or vLLM would exceed this
gate's system-change boundary.

## Execution decision

**Execution performed: NO.**

Neither candidate met all prerequisites of the bounded execution rule. vLLM
cannot run its relevant production path in the current native Windows
environment and is absent; llama.cpp lacks both a built runtime and real MoE
model artifact. Both real paths would require model weights, and at least one
would additionally require system/runtime setup. Running only unit operators,
mock routers, vocabulary-only GGUF files, or a fabricated miniature model would
not establish any missing mandatory criterion, so no such substitute was used.

Network activity was limited to fetching the two official source repositories
and immutable Git/GitHub metadata. No model, tokenizer, checkpoint, weight, or
generation artifact was downloaded.

## Candidate verdicts

`E` means established by direct bounded evidence. `NE` means not established
at this exact source/resource boundary. Because every criterion is mandatory,
one or more `NE` cells require the bounded `NO-GO` verdict.

| # | Criterion | vLLM | llama.cpp |
| --- | --- | --- | --- |
| 1 | Authoritative native routing or dispatch observable | E: native logical top-k capture; physical EPLB mapping excluded | E for ordinary inspected graph path: materialized selected IDs are observable; architecture-specific transforms limit scope |
| 2 | Observation does not change the decision | NE: capture plumbing adds writes/transfers and no real on/off comparison was run | NE: callback changes graph batching/synchronization and no real on/off comparison was run |
| 3 | Native chronology recoverable without repair | NE: scheduler design is credible but unexecuted | NE: microbatch tensor order exists, but callback lacks request/position identity |
| 4 | Exact layer/stage/expert identity | NE: layer/expert are explicit; no stage exists and no concrete canonical model scope was executed | NE: layer/expert are explicit on the named node; no stage exists and model-specific transforms remain untested |
| 5 | Pre/post-capacity states not conflated | NE: logical pre-EPLB selection is explicit, but no concrete capacity semantics were executed | NE: selected IDs are distinct from scores, but model-specific capacity semantics were not demonstrated |
| 6 | Unassigned emitted only when exposed | E as a negative rule: inspected paths expose no drop state, so none may be emitted | E as a negative rule: inspected path exposes no drop state, so none may be emitted |
| 7 | Provenance can be bounded | NE: revision and local environment are bounded; model/device/workload are absent | NE: revision and local environment are bounded; model/backend/workload are absent |
| 8 | Independent native/runtime equivalence check | NE: returned routing is the capture product; no independent real result | NE: observed tensor is the dispatch operand; no independent real result |
| 9 | Canonical trace validates without repair | NE: no real canonical trace was produced | NE: no real canonical trace was produced |
| 10 | Downstream descriptive analysis accepts trace | NE: no real canonical trace was produced | NE: no real canonical trace was produced |
| 11 | Stage-qualified pre-flight only when genuine | E as a gate decision: not run because stage-qualified semantics were not established | E as a gate decision: not run because stage-qualified semantics were not established |
| 12 | Claims remain within demonstrated scope | E: verdict is revision/path/environment bounded | E: verdict is revision/path/environment bounded |

| Candidate | Verdict | Decisive missing evidence |
| --- | --- | --- |
| vLLM | `NO-GO` | Real supported MoE execution, non-interference, native-result equivalence, canonical chronology/emission, and downstream validation |
| llama.cpp | `NO-GO` | Real MoE GGUF execution, request chronology, non-interference, native-result equivalence, canonical emission, and downstream validation |

## Final D verdict

**FINAL D VERDICT: NO-GO**

At the exact pinned revisions and local no-model/no-system-change boundary,
neither candidate satisfies every frozen mandatory criterion. This closes D
successfully with bounded negative evidence. It does not reject a future,
separately authorized probe on a supported host with an approved immutable MoE
artifact and an independent native equivalence oracle.

## Claims explicitly not established by source inspection

Source inspection alone does not establish measured observer overhead,
non-interference, dispatch equivalence, latency, throughput, tokens per second,
speedup, physical CPU/GPU residency, H2D or DRAM traffic, memory savings,
production offloading suitability, or broad runtime/model compatibility.
