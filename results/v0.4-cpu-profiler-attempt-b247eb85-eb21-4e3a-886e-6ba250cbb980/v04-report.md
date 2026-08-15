# V0.4 CPU trace-profiler overhead

Measured CPU wall-time; process-memory snapshots are descriptive and order-conditioned.
No GPU, offload, cache speedup, or production-throughput claim.

Decision: **valid_stable**

| Child | Reference ns | Profiled ns | Delta ns | Ratio |
|---:|---:|---:|---:|---:|
| 1 | 13430359100 | 13588175800 | 157816700 | 1.011751 |
| 2 | 13509411900 | 13827841600 | 318429700 | 1.023571 |
| 3 | 14175107100 | 14458551600 | 283444500 | 1.019996 |
| 4 | 13579165800 | 13962381300 | 383215500 | 1.028221 |

Each child row sums all 8 evaluation prompts. Repetitions are paired repeatability checks, not independent workload samples.
Across-child ratio stats (n=4): mean 1.020885, sample SD 0.006958, min 1.011751, max 1.028221.
Across-child delta stats (n=4): mean 285726600.000 ns, sample SD 94762385.068, min 157816700.000, max 383215500.000.
CV reference/profiled/ratio: 0.024857 / 0.026298 / 0.006816
Ratio span: 0.016279


## Within-child prompt-ratio summaries

| Child | n prompts | Mean | Sample SD | Min | Max |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1.011863 | 0.012927 | 0.988635 | 1.027081 |
| 2 | 8 | 1.023516 | 0.014661 | 1.010494 | 1.056550 |
| 3 | 8 | 1.020568 | 0.017088 | 0.984948 | 1.042388 |
| 4 | 8 | 1.028310 | 0.016822 | 1.009815 | 1.067383 |

## Per-prompt paired wall-time ratio

| Prompt | n children | Mean | Sample SD | Min | Max |
|---|---:|---:|---:|---:|---:|
| eval-factual-01 | 4 | 1.022981 | 0.008650 | 1.010494 | 1.030165 |
| eval-factual-02 | 4 | 1.018257 | 0.006892 | 1.013792 | 1.028386 |
| eval-coding-01 | 4 | 1.013617 | 0.038021 | 0.984948 | 1.067383 |
| eval-coding-02 | 4 | 1.026853 | 0.020500 | 1.011486 | 1.056550 |
| eval-math-01 | 4 | 1.031405 | 0.007733 | 1.025083 | 1.042388 |
| eval-summary-01 | 4 | 1.021443 | 0.002392 | 1.019522 | 1.024914 |
| eval-reasoning-01 | 4 | 1.014222 | 0.013262 | 0.998524 | 1.029739 |
| eval-conversation-01 | 4 | 1.019737 | 0.006656 | 1.011436 | 1.027733 |

Workload macro across exactly n=8 prompt means: mean 1.021064, sample SD 0.006050, min 1.013617, max 1.031405.

## Descriptive process-memory observations

before/after signed deltas only; order-conditioned; no per-mode peak overhead. Process lifetime PeakWorkingSetSize is retained only in raw child snapshots and is not aggregated as per-mode peak overhead.

| Prompt | Metric | n children | Mean bytes | Sample SD | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| eval-factual-01 | reference_working_set_delta | 4 | -173056.000 | 687675.788 | -811008.000 | 770048.000 |
| eval-factual-01 | profiled_working_set_delta | 4 | 587776.000 | 626911.054 | -278528.000 | 1155072.000 |
| eval-factual-01 | reference_private_delta | 4 | -13312.000 | 496550.260 | -716800.000 | 397312.000 |
| eval-factual-01 | profiled_private_delta | 4 | 48128.000 | 424674.089 | -524288.000 | 454656.000 |
| eval-factual-02 | reference_working_set_delta | 4 | -336896.000 | 716791.223 | -1400832.000 | 122880.000 |
| eval-factual-02 | profiled_working_set_delta | 4 | 103424.000 | 454236.058 | -450560.000 | 520192.000 |
| eval-factual-02 | reference_private_delta | 4 | -312320.000 | 384069.626 | -786432.000 | 0.000 |
| eval-factual-02 | profiled_private_delta | 4 | -458752.000 | 582506.310 | -1048576.000 | 262144.000 |
| eval-coding-01 | reference_working_set_delta | 4 | 593920.000 | 875324.560 | -524288.000 | 1544192.000 |
| eval-coding-01 | profiled_working_set_delta | 4 | 971776.000 | 989833.945 | -131072.000 | 1929216.000 |
| eval-coding-01 | reference_private_delta | 4 | 418816.000 | 902127.146 | -749568.000 | 1376256.000 |
| eval-coding-01 | profiled_private_delta | 4 | 393216.000 | 537768.040 | -262144.000 | 1048576.000 |
| eval-coding-02 | reference_working_set_delta | 4 | -605184.000 | 505701.202 | -1097728.000 | 53248.000 |
| eval-coding-02 | profiled_working_set_delta | 4 | 109568.000 | 1153305.278 | -1069056.000 | 1236992.000 |
| eval-coding-02 | reference_private_delta | 4 | 1024.000 | 353940.777 | -491520.000 | 327680.000 |
| eval-coding-02 | profiled_private_delta | 4 | 708608.000 | 601549.815 | 32768.000 | 1351680.000 |
| eval-math-01 | reference_working_set_delta | 4 | 10183680.000 | 11777701.854 | -90112.000 | 20639744.000 |
| eval-math-01 | profiled_working_set_delta | 4 | 10615808.000 | 11743037.640 | 315392.000 | 20795392.000 |
| eval-math-01 | reference_private_delta | 4 | 10042368.000 | 11960344.957 | -593920.000 | 20709376.000 |
| eval-math-01 | profiled_private_delta | 4 | 10392576.000 | 12156998.555 | -196608.000 | 21352448.000 |
| eval-summary-01 | reference_working_set_delta | 4 | 7856128.000 | 8193828.660 | 737280.000 | 15568896.000 |
| eval-summary-01 | profiled_working_set_delta | 4 | 8372224.000 | 8647020.448 | 827392.000 | 16027648.000 |
| eval-summary-01 | reference_private_delta | 4 | 7825408.000 | 8354932.740 | 589824.000 | 15060992.000 |
| eval-summary-01 | profiled_private_delta | 4 | 7923712.000 | 8279344.746 | 720896.000 | 15126528.000 |
| eval-reasoning-01 | reference_working_set_delta | 4 | -2044928.000 | 2669928.883 | -4587520.000 | 360448.000 |
| eval-reasoning-01 | profiled_working_set_delta | 4 | -2064384.000 | 2564279.201 | -4403200.000 | 503808.000 |
| eval-reasoning-01 | reference_private_delta | 4 | -2049024.000 | 2678596.886 | -4591616.000 | 786432.000 |
| eval-reasoning-01 | profiled_private_delta | 4 | -2000896.000 | 2712662.395 | -4722688.000 | 524288.000 |
| eval-conversation-01 | reference_working_set_delta | 4 | -571392.000 | 800573.303 | -1683456.000 | 225280.000 |
| eval-conversation-01 | profiled_working_set_delta | 4 | 193536.000 | 826867.160 | -888832.000 | 1077248.000 |
| eval-conversation-01 | reference_private_delta | 4 | -264192.000 | 278698.614 | -593920.000 | 0.000 |
| eval-conversation-01 | profiled_private_delta | 4 | -49152.000 | 460321.885 | -589824.000 | 528384.000 |

Memory workload macros use exactly n=8 prompt means:

- reference_working_set_delta: mean 1862784.000, sample SD 4520382.837, min -2044928.000, max 10183680.000 (n=8)
- profiled_working_set_delta: mean 2361216.000, sample SD 4532398.523, min -2064384.000, max 10615808.000 (n=8)
- reference_private_delta: mean 1956096.000, sample SD 4408062.395, min -2049024.000, max 10042368.000 (n=8)
- profiled_private_delta: mean 2119680.000, sample SD 4467693.811, min -2000896.000, max 10392576.000 (n=8)
