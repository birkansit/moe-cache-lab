# Frozen Stage 1 runtime-fidelity experiment

Status: **HISTORICALLY FROZEN**. Stage 1 work may proceed only
through separately authorized, bounded work packages. Real Stage 1 collection
remains unauthorized until its preceding implementation and review gates pass.
Any material change requires a recorded protocol decision and renewed review
before results are inspected.

## 1. Purpose and claim boundary

Stage 1 tests sensitivity to Granite's grouped prefill dispatch shape and to
longer deterministic decode/repetition. It does not claim to reproduce actual
Granite expert residency, scheduling, transfer overlap, latency, or memory
movement. Routing is measured, cache outcomes are simulated, and transfers are
estimated.

## 2. Immutable inputs and two analysis views

Both views derive from the same immutable raw `RoutingTrace` JSONL files. The
trace format and recorded routing events must never be rewritten or mutated.

### `token_layer_atomic`

The existing corrected V0.2 view. Each token/layer `RoutingEvent` is one atomic
bundle of its unique layer-qualified selected experts. Prompt events remain
layer-major; generated/decode events remain token-major.

### `prefill_layer_union_atomic`

A conservative simultaneous-active-set **sensitivity view**, not verified
Granite runtime residency:

1. Treat each prompt as an independent model context.
2. For each prompt and routed layer, union the unique experts selected by all
   prompt tokens into one layer-qualified atomic bundle.
3. Sort keys only for deterministic representation; sorting has no request or
   recency semantics.
4. Emit prompt bundles in ascending layer order.
5. Leave generated/decode bundles unchanged and token-major.

Calibration traces are transformed identically for each view. Each view selects
its own `calibrated_static_frequency` targets from that view's calibration data
and its own `offline_oracle_frequency` targets from that view's evaluation data.
Targets must never cross views. Dynamic policies remain causal and receive no
calibration accesses or warming.

## 3. Frozen corpus, model, and collection protocol

- Corpus: existing corpus v1, exactly 4 calibration and 8 evaluation prompts,
  in its recorded order.
- Model: `ibm-granite/granite-3.1-1b-a400m-instruct`.
- Immutable revision:
  `0da7a48b0276d500ce5922fd2b33944091fc6c09` for config, tokenizer, and model.
- `max_decode_input_steps`: 16.
- Repetitions: 3.
- Each repetition starts a fresh collector process and cold model load, then
  processes all 12 prompts sequentially in the same order; every prompt is an
  independent context.
- Environment: the same pinned package environment, CPU, float32, greedy
  argmax decode, KV cache enabled, and explicit PyTorch thread count 4.
- Total wall-clock ceiling for all three collection repetitions: 15 minutes.
  Crossing the ceiling stops collection and fails the real-run gate; do not
  silently reduce prompts/repetitions/lengths.
- No new model, model revision, backend, or download is allowed.

Each repetition manifest must record process/repetition identity, requested and
resolved revision, Python/Transformers/PyTorch versions, device/dtype, thread
count, generation settings, corpus/order/hash, timestamps, elapsed wall time,
trace paths/hashes, and structural/count metadata.

## 4. EOS state machine and decode accounting

The collector must implement this exact `max_decode_input_steps=16` state
machine. Fixed post-EOS routing is forbidden.

1. The prompt forward produces candidate token `y0`.
2. Before every possible decode-input forward, inspect the current candidate.
3. If the candidate is EOS, record it as the emitted terminal EOS and record
   EOS status, but **do not feed or route it**. Stop immediately. If `y0` is
   EOS, actual decode-input steps are zero.
4. Otherwise record the candidate as both an emitted non-EOS token and the next
   decode-input token. Feed it exactly once, capture its routing, increment
   actual routed steps, and obtain the next candidate.
5. If that forward produces EOS, the next loop boundary applies step 3: record
   the terminal EOS and stop without routing it.
6. If 16 non-EOS decode inputs have been routed, stop at the horizon. The next
   candidate produced by the 16th forward is outside the experiment and must
   not be reported as emitted. Record `horizon_exhausted=true`,
   `eos_emitted=false`, exactly 16 routed steps, and exactly 16 emitted/routed
   non-EOS token IDs.

For every prompt/repetition the manifest must record:

- `max_decode_input_steps_requested` (16),
- `actual_decode_input_steps_routed`,
- the identical ordered lists of routed and emitted non-EOS token IDs,
- optional terminal EOS token ID and its candidate position,
- `eos_emitted` and `horizon_exhausted` flags,
- routing bundle/event and expert-request counts by phase and view.

The terminal EOS candidate position is zero-based (`y0` is position 0) and
equals the number of non-EOS decode inputs routed before it. Counts must explain
every deviation from the no-EOS expectations below. No event may exist for the
terminal EOS or for the out-of-horizon candidate.

Across all three repetitions, actual routed step counts, routed/emitted non-EOS
token IDs, optional terminal EOS ID/position/status, selected expert IDs, and
event token IDs must match exactly for each prompt. Any mismatch is a gate
failure, not variance. Selected probabilities must have maximum absolute
difference at most `1e-6` after aligning identical events. Timestamps and other
non-semantic creation metadata are excluded from repeatability comparison.
Each prompt/prefill prefix and the first two decode-input steps must also match
the tracked V0.2 trace exactly in tokens and selected experts wherever EOS
permits those steps; a mismatch fails the gate and is not explained away as a
new repetition.

## 5. Capacities and feasibility

Evaluate exactly these total layer-qualified resident-object capacities:

`8, 16, 32, 64, 128, 192, 256, 384, 512, 768`

For `prefill_layer_union_atomic`, any capacity below that evaluated view's
maximum atomic bundle is **N/A / infeasible**, never a zero-percent hit result.
Cross-view comparisons are permitted only where both views are feasible. Every
row must include the bundle/event count, expert-request denominator, and
absolute counters beside percentages.

## 6. Cache boundaries and controls

For every repetition, view, policy, and capacity:

### Primary suite-persistent evaluation

- Reset the cache at the start of the repetition.
- Replay evaluation prompts in manifest order with one cache persistent across
  the suite.
- Dynamic LRU/LFU start empty. Calibration selects calibrated fixed targets but
  never warms or accesses dynamic caches.
- Fixed calibrated/oracle targets prewarm once before evaluation; only this
  prewarm is charged in the combined scope.

### Cold per-prompt control

- Reset before every evaluation prompt and replay that prompt's prompt+decode.
- Dynamic policies start empty per prompt.
- Fixed policies prewarm once per prompt. Charge and disclose every per-prompt
  prewarm in aggregate and per-workload counters.

### Decode-only cold control

- Exclude prompt bundles from simulated accesses and reset before each prompt's
  decode sequence.
- Dynamic policies start empty at that decode boundary.
- Fixed policies prewarm once per prompt and charge it explicitly.
- This control measures only within-decode locality; it does not inherit prompt
  cache state.

No control may change target-selection data, event order within its included
phase, fixed restoration accounting, or the hard capacity definition.

## 7. Repetition and workload summaries

The three repetitions are deterministic repeatability checks, **not independent
samples**. They must not inflate sample size or be used to estimate stochastic
variance. When semantic traces match within the probability tolerance, report
one result set plus repeatability evidence; do not triple event denominators.

Workload macro statistics apply **only** to cold-per-prompt controls. For each
reported prompt-level metric use exactly 8 prompt values and report arithmetic
mean, sample standard deviation with denominator `n-1`, minimum, and maximum
(`n=8`).

Suite-persistent evaluation is order-conditioned and receives micro totals
only: exact summed bundles/events, requests, hits, misses, loads, evictions, and
estimated transfers. It must not be presented as prompt-independent macro
variance. Decode-only cold controls may report exact prompt rows and micro
totals, but are not a substitute for the declared cold-per-prompt macro slice.

Do not report confidence intervals, population/generalization claims, category
variance, or infer expert semantics from prompt categories.

## 8. Reporting contract

For every view, capacity, policy, phase/control, and feasibility state, report:

- repetition/workload count and repeatability status,
- bundle/event count and layer-qualified request denominator,
- hits, misses, hit rate, prewarm loads, demand loads, evictions, and estimated
  transfers,
- prompt/prefill, generated/decode, and combined counters where applicable,
- N/A reason for infeasible grouped capacities,
- measured routing/group metadata separately from simulated cache outcomes and
  estimated transfers,
- fixed-target size and overlap between the two views at matched capacity,
- EOS-shortened workloads and actual versus requested decode steps,
- all negative/null results without interpretation tuning.

Cross-view text must repeat that `prefill_layer_union_atomic` is a conservative
simultaneous-active-set sensitivity view, not verified runtime residency.

## 9. No-EOS expected per-repetition counts

These are predeclared invariants if no prompt emits EOS before all 16 decode
inputs. Deviations are allowed only through the recorded EOS rule and must
reconcile exactly.

- Calibration `token_layer_atomic`: 2,664 events / 21,312 requests.
- Evaluation `token_layer_atomic`: 5,832 events / 46,656 requests.
- Evaluation `prefill_layer_union_atomic`: 3,264 bundles / 29,937 requests.
  - Prompt/prefill: 192 bundles / 5,361 requests.
  - Generated/decode: 3,072 bundles / 24,576 requests.

The implementation must test these expectations with fixture metadata but must
not force counts after EOS.

## 10. Leakage, metric, and adversarial gates

Before a real run, synthetic tests and reviewer inspection must cover:

- layer-qualified identity and union de-duplication without cross-layer merges,
- immutable raw traces and deterministic derived-view ordering,
- calibration/evaluation separation in both views and view-specific fixed
  targets,
- no future access for LRU/LFU and explicit non-causal oracle labeling,
- hard-capacity atomic bundles, pinning, restoration loads/evictions, and N/A
  handling below maximum bundles,
- cache resets/persistence exactly matching primary and both controls,
- fixed prewarm charged once per suite or once per prompt as specified,
- prompt/decode phase attribution and decode-only exclusion of prompt accesses,
- EOS-before-routing behavior, contiguous boundaries, exact count reconciliation,
- repetition matching of steps/tokens/experts and probability tolerance,
- cross-view pairing integrity, denominators, absolute counters, and feasibility,
- target-overlap correctness and deterministic tie-breaking,
- manifest/path/hash/revision/thread/environment tamper detection,
- empty/one-event, capacity-edge, missing-pair, zero-variance, and all-EOS edge
  cases,
- no runtime, semantic, confidence-interval, or generalization claim from
  simulated/estimated values.

Any high-severity reviewer finding or mismatch in real repeatability fails the
Stage 1 gate and returns to a bounded fix/review cycle.

## 11. Prospectively frozen continuation rule

This rule was selected **after exploratory diagnostics on the tracked V0.2
max-step-2 data**. It is not a blind preregistration. It is a prospectively
frozen confirmation rule for newly collected `max_decode_input_steps=16`
traces, and no threshold, slice, policy, capacity, or denominator may change
after those results are visible.

Both of the following conditions are mandatory:

1. **Transfer condition**
   - View: `prefill_layer_union_atomic` only.
   - Policy: online resident-frequency `lfu` only.
   - Slice: primary suite-persistent evaluation, combined scope, micro totals
     per repetition.
   - Baseline: capacity 32 exactly.
   - Candidate: capacity 256 exactly.
   - In **each** repetition independently, using exact integer totals (never a
     pooled value):
     `estimated_transfers_lfu_cap256 <= 0.90 * estimated_transfers_lfu_cap32`.
2. **Decode stability condition**
   - Slice: decode-only cold control,
     `prefill_layer_union_atomic`, `lfu`, capacity 256.
   - Every one of the 8 evaluation prompts must have hit rate
     `hits / requests >= 10.00%` in **each** repetition.
   - No prompt may be omitted, pooled, or averaged around.
   - The semantic repeatability gate in section 4 must already pass.

No alternative policy, capacity, view, phase, or control may substitute. Both
conditions are required. Passing authorizes Stage 2 feasibility inspection
only and never authorizes offloading. If either condition fails, Stage 2 may
still document hardware/runtime limits, but Stage 3 is presumptively deferred.
