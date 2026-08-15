# V0.4 measured CPU observer result

## Decision and evidence

The V0.4 experiment is `valid_stable`. On the pinned full-resident CPU-FP32
Granite path and frozen eight-prompt evaluation workload, the repository's
current exact routing observer added approximately **2.09%** paired wall-time
overhead. This is a measured observer/reference result, not an inference
speedup, cache/offload result, GPU proxy, or production-throughput claim.

- Attempt: `b247eb85-eb21-4e3a-886e-6ba250cbb980`
- Retry source: finalized deferred attempt
  `d6f7cac2-d67a-46e8-8267-d65c861ea009`
- Set SHA-256:
  `1f87491f3d0512912e593422a2cc0b012ebed343291726131fa51dc2e103fe60`
- Structured result SHA-256:
  `ed857ef581f80fbbe8e694bc869e9265e785e5c48bd6e00118b20db7d85e7b72`
- Generated Markdown SHA-256:
  `d9b5399a125d162c3c7cdb3f847f30c196b56fd31ace9ee714f9dd4d5b81ceaf`
- Parent experiment interval: `253.29699999999866` seconds

The exact 11-file evidence tree is under
`results/v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980/`.
`tests/test_v04_valid_evidence.py` validates the set and retry lineage, checks
routing equivalence and stability, and regenerates the JSON and Markdown
byte-for-byte.

## Primary measured wall time

Each child is a fresh process and sums the same eight evaluation prompts.

| Child | Reference ns | Profiled ns | Delta ns | Ratio | Added wall time |
|---:|---:|---:|---:|---:|---:|
| 1 | 13,430,359,100 | 13,588,175,800 | 157,816,700 | 1.0117507431 | 1.175074% |
| 2 | 13,509,411,900 | 13,827,841,600 | 318,429,700 | 1.0235709520 | 2.357095% |
| 3 | 14,175,107,100 | 14,458,551,600 | 283,444,500 | 1.0199959336 | 1.999593% |
| 4 | 13,579,165,800 | 13,962,381,300 | 383,215,500 | 1.0282208425 | 2.822084% |

- Mean paired-child ratio: `1.0208846178255349`.
- Mean delta: `285,726,600 ns`; sample SD `94,762,385.0678457 ns`.
- Pooled ratio: `1.0208963594297331` from reference `54,694,043,900 ns`
  and profiled `55,836,950,300 ns`.
- Reference/profiled/paired-ratio CV:
  `0.024856668659024556 / 0.02629763658060313 / 0.006815956024537018`.
- Paired-ratio span: `0.01627881128450951`.

All frozen stability limits passed: each CV is at most 10% and the ratio span
is at most 15%. The four children are deterministic repeatability checks, not
independent statistical samples.

## Prompt and nested timing summaries

| Prompt | Mean ratio | Sample SD | Min | Max |
|---|---:|---:|---:|---:|
| `eval-factual-01` | 1.022981 | 0.008650 | 1.010494 | 1.030165 |
| `eval-factual-02` | 1.018257 | 0.006892 | 1.013792 | 1.028386 |
| `eval-coding-01` | 1.013617 | 0.038021 | 0.984948 | 1.067383 |
| `eval-coding-02` | 1.026853 | 0.020500 | 1.011486 | 1.056550 |
| `eval-math-01` | 1.031405 | 0.007733 | 1.025083 | 1.042388 |
| `eval-summary-01` | 1.021443 | 0.002392 | 1.019522 | 1.024914 |
| `eval-reasoning-01` | 1.014222 | 0.013262 | 0.998524 | 1.029739 |
| `eval-conversation-01` | 1.019737 | 0.006656 | 1.011436 | 1.027733 |

The descriptive macro across eight prompt means is `1.0210644884255045`
(sample SD `0.006049666403458691`, range
`1.0136170523493717-1.0314053722655394`). No confidence interval or workload
generalization is claimed.

Per child, summing eight evaluation prompts, the exact event-extraction timer
was `246,460,400-278,568,800 ns` (mean `257,223,400 ns`). Mean hook setup and
removal were `3,186,225 ns` and `351,950 ns`. Mean reference/profiled prefill
totals were `2,155,914,450 / 2,177,472,750 ns`; mean reference/profiled decode
totals were `11,484,797,475 / 11,491,328,325 ns`. Extraction is the largest
explicit added nested component, but it is not mechanically equal to net
overhead because paired forward and unaccounted timings vary.

## Correctness, resources, and descriptive memory

Each child reproduced exactly:

- calibration: `2,664` routing events / `21,312` assignments;
- evaluation: `5,832` routing events / `46,656` assignments;
- 16 decode-input steps per prompt, horizon exhausted, no fixed post-EOS work;
- selected token/expert IDs identical and maximum probability delta `0.0`.

All four 15-sample resource admissions passed. CPU medians were
`2.293523%-2.611904%`, CPU maxima `3.782688%-11.487138%`, disk medians
`0.028363%-0.030316%`, disk maxima `0.168035%-0.555000%`, and commit maxima
`37.214190%-37.413090%`. Available RAM ranges were
`10,075,275,264-10,321,932,288` bytes, all above the frozen
`8,589,934,592`-byte admission minimum. Minimum available memory in child
snapshots remained at least `4.1145 GiB`, above the `2 GiB` abort threshold.

Model-load elapsed times were `2,786,434,800`, `1,913,269,000`,
`1,962,185,300`, and `1,938,423,200 ns`. Model load and calibration warmup are
excluded from the primary paired evaluation timing.

Process-memory results are only signed before/after snapshots and are
order-conditioned. Across eight prompt means, working-set deltas were
`1,862,784` reference and `2,361,216` profiled bytes; private deltas were
`1,956,096` reference and `2,119,680` profiled bytes. These are not per-mode
peaks or a demonstrated memory-overhead estimate.

## Claim boundary and continuation

- The observer is observational; router decisions were not modified.
- Fresh processes/local-cache loads do not control the OS page cache.
- The corpus is small and local; prompt categories do not imply expert
  semantics.
- This result establishes stable measured CPU observer/reference overhead only.
- It does not justify Stages 3-5, offloading, cache residency, tuning, GPU
  inference, or a speedup claim.
- The next bounded step was local `0.4.0rc1` preparation and release validation.
