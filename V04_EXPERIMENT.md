# V0.4 / Stage 2B CPU trace-profiler overhead experiment

Status: **independently approved and frozen for bounded implementation plus
synthetic tests only**. Real model execution and collection remain unauthorized.

## Question and claim boundary

On the pinned, full-resident CPU-FP32 Granite path, what wall-time overhead is
added by the repository's current exact routing observer, relative to the
identical generation driver with observation disabled? Process-memory snapshots
are descriptive, order-conditioned context only; the experiment does not claim
an exact per-mode memory or peak-memory overhead.

The experiment may establish only a stable, measured CPU reference and exact
trace-profiler overhead on this machine, model, corpus, and software stack. It
does not measure or proxy GPU behavior, expert residency, offloading, transfer
time, cache-policy speedup, or production throughput. A pass may authorize only
continued use of the CPU profiler/reference and, at most, a separately
specified and reviewed operator-profile proposal. It never authorizes Stages
3-5, offloading, tuning, a new backend, or a performance claim derived from
simulated/estimated cache results.

## Immutable inputs and execution modes

- Corpus: existing corpus v1, in manifest order: four calibration prompts and
  eight evaluation prompts. Prompt categories remain descriptive metadata.
- Model: `ibm-granite/granite-3.1-1b-a400m-instruct`, exact local-only revision
  `0da7a48b0276d500ce5922fd2b33944091fc6c09`.
- Runtime: current official Transformers/PyTorch path, CPU FP32, exactly four
  PyTorch/OMP/MKL threads, greedy deterministic generation, no download.
- Decode protocol: Stage 1's candidate-before-feed EOS state machine with
  `max_decode_input_steps=16`. Terminal EOS is recorded but never routed; the
  candidate produced after the sixteenth routed non-EOS input is outside the
  experiment, is retained only for semantic validation, and is not reported as
  emitted or routed.
- `reference_no_observer`: the shared execution driver with no router forward
  hooks and no routing-event extraction.
- `profiled_exact`: the same driver using the current, unmodified
  `_RouterHookCapture` and `_events_from_router_logits` path. This is the only
  mode difference.

There are exactly four fresh child processes and one fresh-process local-cache
model/tokenizer load per child; OS page cache is uncontrolled. Both modes run
inside every child. The parent never loads the model.
Each prompt/mode execution is a fresh independent model context; no KV cache or
generation state crosses executions. A fresh process does not imply cold model
files: local Hugging Face cache and uncontrolled Windows OS page cache remain
in use. Neither is flushed, manipulated, or claimed to be counterbalanced; the
paired/order design and resource gate reduce but cannot eliminate that effect.

## Frozen counterbalance

Let evaluation prompts in corpus order be `E1..E8`, `R` mean
`reference_no_observer`, and `P` mean `profiled_exact`.

| Child | Evaluation prompt order | First/second mode by prompt slot |
|---:|---|---|
| 1 | E1 E2 E3 E4 E5 E6 E7 E8 | RP PR RP PR RP PR RP PR |
| 2 | E1 E2 E3 E4 E5 E6 E7 E8 | PR RP PR RP PR RP PR RP |
| 3 | E1 E2 E3 E4 E5 E6 E7 E8 | RP PR RP PR RP PR RP PR |
| 4 | E1 E2 E3 E4 E5 E6 E7 E8 | PR RP PR RP PR RP PR RP |

Public child IDs are exactly 1 through 4. Process UUID and artifact path are
unique; positive PIDs are informational and need not be unique because Windows
may reuse them. The table is data, not a seed-dependent shuffle.

Before evaluation, children 1 and 3 warm `C1 C2 C3 C4` in `R` and then
`C1 C2 C3 C4` in `P`; children 2 and 4 warm all four in `P` and then all four
in `R`. Calibration measurements and traces are validation-only and excluded
from every reported overhead aggregate. No model reload occurs between
calibration and evaluation.

## Timing boundary and exclusions

Inputs are tokenized and cloned before timing. Each prompt/mode records nested
finite, nonnegative `time.perf_counter_ns()` durations:

- total prompt primary: immediately before mode/hook setup through hook removal
  and final lifecycle assertion;
- hook setup, hook removal, and lifecycle assertion separately;
- the top-level prompt/prefill model forward;
- every top-level decode model forward separately; and
- every `_events_from_router_logits` extraction separately (`P` only).

The primary total charges hook registration/removal, capture, tensor-to-CPU
materialization performed by the existing observer, event construction, greedy
candidate handling, and the shared driver. `R` follows the identical driver
without observer work. Component intervals must nest within the primary total,
must not overlap other same-level components, and must satisfy
`accounted_ns <= primary_total_ns`; store the exact nonnegative
`unaccounted_ns = primary_total_ns - accounted_ns`. Forward timers surround
only top-level model calls. No model, MoE, expert, operator, or submodule timing
hook/profiler is allowed in the primary path.

Excluded from the primary interval: child/process startup, model/tokenizer
load, corpus/manifest parsing, tokenization, resource admission, calibration,
artifact serialization, hashing, filesystem flush/rename, report rendering,
and final cross-child validation. Record these separately where applicable;
never subtract them from raw prompt measurements.

There is no in-primary sampler. Immediately before and after each mode run,
outside its primary timer, call `GetProcessMemoryInfo` and record
`WorkingSetSize`, `PrivateUsage`, and process-lifetime `PeakWorkingSetSize`;
call `GlobalMemoryStatusEx` between runs. API status/errors and timestamps are
mandatory. Peak working set is global, monotonic, and order-conditioned, so it
cannot be assigned as a per-mode peak or used to claim exact observer-memory
overhead.

## Hook lifecycle and carryover assertions

- `R` begins and ends with zero repository router hooks registered.
- `P` registers exactly the expected routed-layer hooks, removes every handle
  in `finally`, and begins/ends with the same hook counts and identities as the
  pre-run baseline.
- A capture belongs to exactly one prompt/mode run. Captured logits, events,
  token positions, and temporary tensors are cleared before the next run.
- Reference runs produce no routing events. Profiled runs contain exactly one
  complete routed-layer set per routed token position and obey all existing
  `RoutingTrace` chronology/structure checks.
- Any hook leak, event carryover, unexpected reference event, missing layer, or
  exception invalidates the complete child. Preserve its metadata; do not use
  partial timings.

## Correctness and trusted-evidence gates

All gates are mandatory before timing is interpreted:

1. Within every prompt pair, `R` and `P` have exact tokenized input IDs and
   attention masks; every greedy candidate ID and decoded text, including the
   validation-only candidate produced by the sixteenth forward; emitted/routed
   non-EOS token IDs and decoded text; optional terminal EOS ID/text/position;
   actual steps; and EOS/horizon flags. Equality means exactly these listed
   semantic fields, including the validation-only horizon candidate; it does
   not imply unrecorded full-logit tensor equality. The observer must not
   change the listed output semantics.
2. Across all four children, each prompt's semantic fields above match exactly.
3. Every `P` trace matches the trusted Stage 1 trace for prompt token IDs,
   routed positions/layers, and selected expert IDs exactly; selected
   probabilities must be finite and have maximum absolute difference at most
   `1e-6`. The trusted prompt prefix and first `min(actual_steps, 2)` decode
   steps are checked explicitly as well.
4. Stage 1 EOS/horizon semantics and expected count reconciliation remain
   authoritative. A genuine deterministic mismatch, including a new EOS
   boundary, is a failed semantic gate rather than a timing observation.

The canonical trusted source is Stage 1 set manifest SHA-256
`f7e624d95368b1aa976f3e31412fc44e86dbf46439e5c15534032c14446e34d1`,
repetition 1 manifest SHA-256
`c752c0e3128786bb6ae40c3cdf2d67b6510b4fb369053be550b4cafdca85b94c`.
The exact prompt mapping is orders 0-11 respectively:
`cal-factual-01`, `cal-coding-01`, `cal-math-01`,
`cal-conversation-01`, `eval-factual-01`, `eval-factual-02`,
`eval-coding-01`, `eval-coding-02`, `eval-math-01`,
`eval-summary-01`, `eval-reasoning-01`, `eval-conversation-01`; each must bind
to the corresponding trace path/hash declared by that canonical repetition
manifest. Timestamps are excluded from semantic equality. Failed runs are
immutable invalid evidence and cannot be silently replaced.

## Resource admission and coordination

Before **every** child launch, use Windows PDH with English counter paths
`\Processor(_Total)\% Processor Time` and
`\PhysicalDisk(_Total)\% Disk Time`. These are the exact runtime strings with
one literal backslash at each separator. Source-language escaping is a separate
implementation concern; persisted raw paths must equal these one-backslash
strings exactly. Collect one initial baseline sample
(recorded but excluded), then exactly 15 valid samples at one-second intervals.
Use `PdhAddEnglishCounterW`; record the raw paths, raw/formatted values,
timestamps, and every API status. A missing instance/counter, API error,
nonfinite value, or value outside `[0, 100]` defers admission and is never
coerced to zero. An invalid sample is not replaced by extending the window.

For every sample, call `GlobalMemoryStatusEx` for available physical bytes and
`GetPerformanceInfo` for `CommitTotal`, `CommitLimit`, and `PageSize`. Require
positive denominators and calculate commit percentage as
`100 * CommitTotal / CommitLimit` (equivalently the byte numerator and
denominator after multiplying each by `PageSize`). Persist both page counts,
page size, derived byte numerator/denominator, available bytes, and API status.

Admission requires all of the 15 valid samples to exist and all of:

- total CPU utilization median at most 10% and maximum at most 25%;
- available RAM at least
  `max(8 GiB, FP32_parameter_payload + 2 GiB) = 8,589,934,592 bytes`;
- committed-memory percentage at most 75%; and
- disk active-time median at most 10% and maximum at most 50%.

If any admission condition fails, do not start the child: record the baseline,
samples, API statuses, and reason, then defer the complete experiment until the
host is idle. Between runs, `GlobalMemoryStatusEx` checks the 2 GiB abort floor;
no in-primary polling thread is allowed. Abort before the next mode/run if
available RAM is below 2 GiB, the parent deadline expires, or correctness is
invalid; retain atomic invalid metadata but exclude it from aggregates.

Only this experiment may perform a performance/model-heavy or pressure-sensitive
project workload during its run. Normal external activity is uncontrolled;
PDH admission reduces but does not eliminate interference or make the host a
laboratory-isolated system.

The full valid four-child run has a 20-minute wall-time ceiling beginning just
before child 1 launch and ending after child 4 finalizes. Pre-launch busy-host
deferral is outside the timer because no experiment has started. Timeout
invalidates and preserves the attempt. No reduced child/prompt set is valid.

## Raw metrics and aggregation

Store integers/raw samples before derived values:

- primary and every nested elapsed wall-nanosecond component per prompt/mode,
  plus accounted and unaccounted nanoseconds;
- process CPU user/system nanoseconds where the OS exposes them;
- before/after `WorkingSetSize`, `PrivateUsage`, process-lifetime
  `PeakWorkingSetSize`, between-run global-memory fields, timestamps, and API
  statuses/errors;
- prompt ID/order, child ID/UUID/PID, mode order, token/decode/EOS fields,
  event/assignment counts, hook counts, and resource-admission samples;
- model/package/OS/thread settings and all start/finish timestamps.

Primary prompt ratio is `P_elapsed_ns / R_elapsed_ns`; primary prompt overhead
is `P_elapsed_ns - R_elapsed_ns`. Do not clamp negative values. For each child,
sum the eight evaluation prompt times within each mode, then compute one paired
child ratio `sum(P) / sum(R)` and delta `sum(P) - sum(R)`. Report the eight
prompt-pair ratios in that child as arithmetic mean, sample standard deviation
(`n-1`), minimum, and maximum.

Across children, use the four child ratios/deltas as the repeatability level:
arithmetic mean, sample standard deviation (`n-1`), minimum, maximum, and ratio
coefficient of variation `sample_sd / mean`. Do not pool 32 prompt pairs as
independent repetitions and do not report confidence intervals. Separately,
for each of exactly eight evaluation prompt IDs, report its four paired ratios
with n=4 mean/sample-SD/min/max; report descriptive macro mean/sample-SD/min/max
across those eight prompt means. Before/after working-set/private-memory
snapshots and their signed deltas may receive the same descriptive grouping,
but process-lifetime peak and global-memory values remain raw chronological
context only: no per-mode peak attribution, n=4/n=8 peak aggregation, or exact
memory-overhead headline is permitted.

## Stability decision and one-rerun limit

For the four children, separately calculate CV (`sample_sd / mean`) for
`sum(R)`, `sum(P)`, and paired child ratio `sum(P)/sum(R)`. A nonpositive/zero
mean or denominator is invalid and should be unreachable for valid elapsed
times. The result is **inconclusive** if any of:

- CV for child `sum(R)` exceeds 10%;
- CV for child `sum(P)` exceeds 10%;
- CV for paired child ratio exceeds 10%; or
- `(max_child_ratio - min_child_ratio) / min_child_ratio` exceeds 15%.

Equality to either threshold is allowed. Correctness-gate failure is a failure,
not statistical inconclusiveness. At most one later complete four-child rerun
may be authorized after independent review of an inconclusive valid attempt.
The first attempt must remain intact and visible; thresholds, corpus, ordering,
and analysis cannot change. No third run or selective child/prompt replacement
is allowed.

Passing means all correctness gates pass and all stability conditions hold.
It makes no minimum/maximum overhead-performance promise.

## Versioned artifacts and validation

Implementation must define strict, versioned schemas for child records, the
four-child set manifest, and structured results. Use safe relative paths,
canonical UTF-8 JSON, finite numeric values, SHA-256 for corpus/spec/trusted
Stage 1 input and every artifact, exact count/size reconciliation, timezone-
aware ordered timestamps, unique UUIDs/paths, and atomic staging-to-final
rename only after validation. Markdown must render solely from the validated
structured result.

Interrupted, failed, busy-host, low-memory, timeout, semantic-mismatch, and
inconclusive attempts retain immutable status/reason/resource metadata and may
not occupy the valid final path.

Every launch/admission evaluation gets a unique UUID and immutable sibling path
`results/v0.4-cpu-profiler-attempt-<uuid>/`; staging uses that UUID and is
atomically finalized to exactly one terminal state. Paths are never reused,
overwritten, or promoted over another attempt. Frozen state transitions are:

- `PLANNED|RUNNING -> DEFERRED`: admission is missing/invalid/busy before the
  next child starts. Any already completed child data remain partial evidence.
  A later idle-host admission uses a new UUID/path; defer does not consume the
  single inconclusive rerun allowance.
- `RUNNING -> INVALID`: child/process, low-memory, timeout, semantic, hook,
  schema, hash, or correctness failure. No partial child is replaceable.
  Independent root-cause review is required before a new full attempt; the new
  attempt has a new UUID/path and is not the statistical rerun.
- `PLANNED|DEFERRED|INVALID -> BLOCKED`: a prerequisite is unavailable or a
  condition requires external/material action. A first missing-counter/API
  admission attempt is `DEFERRED`; the same availability failure on the next
  separately scheduled attempt is `BLOCKED`. An `INVALID` attempt may be
  followed only by one independently authorized full attempt after a recorded
  root-cause correction; recurrence of the same root cause is `BLOCKED`. Busy
  threshold failures remain `DEFERRED` and have no automatic sleep/retry loop.
  `BLOCKED` resumes only after the condition changes and explicit authorization.
- `RUNNING -> VALID_STABLE`: all four children/gates/stability checks pass;
  this is the sole publishable experimental result state.
- `RUNNING -> VALID_INCONCLUSIVE`: correctness passes but a stability threshold
  fails. After independent review, at most one complete rerun with a new path
  and `rerun_of=<first UUID>` may proceed. A second inconclusive result becomes
  `INCONCLUSIVE_FINAL`; it is preserved and no further run is allowed.

`DEFERRED`, `INVALID`, `BLOCKED`, and `INCONCLUSIVE_FINAL` are reported as such,
never relabeled as successful or erased. Selective prompt/child retries and
mixing attempts are forbidden.

Required no-model/network synthetic and tamper tests include: both mode orders;
all counterbalance rows; y0 EOS, EOS after intermediate steps, and horizon 16;
timer nesting/accounting/boundaries; zero hooks in `R`; hook removal on success/error;
carryover rejection; OFF/ON output mismatch; trusted Stage 1 token/expert/
probability mismatch and tolerance edges; malformed/NaN/boolean values; count,
hash, path, timestamp, UUID, mode, prompt, child, and resource-sample tampering;
PDH baseline/sample/API failures and formulas; busy admission, low-memory
abort, timeout, every state transition and immutable-path rule;
hierarchical n=4/n=8 arithmetic including zero/negative deltas; exact 10%/15%
stability boundaries; and the one-preserved-rerun limit.

## Authorization gate

Independent review approved this frozen design with no remaining high- or
medium-severity issue. Approval authorizes only bounded workflow implementation
and no-model/network synthetic and tamper tests. The implementation requires
its own independent review before any real model execution or collection; real
evidence then requires a final independent gate. Historical Terminal A at
0.3.0rc1 remains valid as an earlier release checkpoint but is not a stop
condition for this explicitly reopened, bounded mission.
