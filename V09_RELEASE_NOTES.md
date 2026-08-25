# v0.9 release notes

## Release scope

v0.9 adds an exact, offline Capacity Frontier workflow and strengthens the
reproducibility foundation for canonical traces produced outside
`moe-cache-lab`. The base installation remains sufficient for validation,
analysis, frontier derivation, pre-flight analysis, and experiment bundles; no
model download is needed for those workflows.

### Exact event-atomic LRU Capacity Frontier

`moe-cache-lab capacity-frontier` derives the complete exact frontier under the
existing **SIMULATED** event-atomic LRU cache abstraction. A canonical v1 or v2
trace is sufficient for the count-capacity frontier. Supplying an existing,
compatible pre-flight config also enables a heterogeneous byte-capacity
frontier using its stage-qualified sizes where required.

The human report is bounded, while the deterministic JSON contract contains
the feasibility floor and every exact hit-changing breakpoint. The derivation
does not sweep every integer capacity. It also reports:

- fixed 25%/50%/75%/90% total-request hit-rate checkpoints as descriptive
  observations, never as targets or recommendations;
- the compulsory first-use miss fraction imposed by cold-start routing;
- the maximum reachable cold-start hit fraction; and
- an explicit `unattainable` result when a selected positive total-request
  hit-rate checkpoint cannot be reached under the supplied trace's cold-start
  event-atomic LRU frontier.

Here, `unattainable` does not mean that caching or offloading is impossible.
The output does not identify a knee, an optimum, a recommended capacity, or a
best policy.

The exact mathematics is grounded in the separately frozen B1 count-reuse and
B2 heterogeneous byte-reuse contracts. Their reference scripts provide
independent exact oracles, including deterministic parity checks against the
production simulators; those scripts are research-validation references, not
stable public APIs.

### External evidence and canonical reproducibility

The external evidence foundation records immutable source identities, hashes,
converter identities, preservation checks, and canonical-validation results
for bounded Kimi and OLMoE cases without adding external raw trace bytes to the
distribution. It keeps three questions separate: canonical-file validity,
producer semantic mapping, and producer non-interference.

Historical Kimi native-layer identity remains **SOURCE-SUPPORTED INFERENCE —
NOT CAPTURE-PROVEN**. Producer non-interference remains **NOT ESTABLISHED** for
the external cases because the required control comparison has not been
performed. The separately receipted OLMoE result supports selected
routed-expert membership only on its pinned, reviewed path; it is not broad
OLMoE or runtime support.

Mechanical reproduction establishes matching bytes in the recorded execution
environment. Canonical validation establishes schema, chronology, and identity
validity. Neither result alone establishes portable byte reproducibility,
producer semantics, non-interference, workload representativeness, or model
support.

### Offline and command-line hardening

The documented Bash/Linux installed-wheel path now exercises canonical trace
production, validation, analysis, the Capacity Frontier, compatible pre-flight
analysis, integrity hashes, and bundle verification from the packaged
no-model example. Expected input failures remain concise command-line errors
without an intended traceback.

## Evidence and interpretation boundaries

- Routing observations are **MEASURED** only when their provenance establishes
  measurement. Synthetic or externally converted routing does not receive an
  automatic evidence upgrade.
- Capacity-frontier and cache outcomes are **SIMULATED** under their exact
  event-atomic cache contract.
- Transfer-service quantities are **ESTIMATED** from simulated demand and
  explicit caller-supplied assumptions.
- Caller-supplied `size_bytes` values are cache-model inputs unless separate
  provenance establishes measured or model-matched physical extents.
- Physical residency, H2D or DRAM movement, latency, throughput, tokens/sec,
  speedup, memory savings, production offloading, and broad model/runtime
  compatibility are **NOT ESTABLISHED** by these results.

The release adds no generic cacheability score, model ranking, policy ranking,
capacity recommendation, or performance prediction.
