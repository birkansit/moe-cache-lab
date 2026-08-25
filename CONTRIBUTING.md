# Contributing to moe-cache-lab

Thanks for considering a contribution.

`moe-cache-lab` is a correctness-first research prototype. Small, reviewable
changes with explicit evidence are preferred over broad feature expansion.

## Core invariants

Contributions must preserve these boundaries:

- The model's native router is authoritative. Do not override or modify routing
  decisions in order to improve cache results.
- Routing selections are **measured** only when trace provenance establishes
  an actual observation; explicitly timed experiments are measured only inside
  their frozen workload and environment boundary.
- Cache hits, misses, residency, and policy outcomes from trace replay are
  **simulated**.
- Transfer counts derived from simulated expert loads are **estimated**.
- Do not turn simulated or estimated results into speedup, latency, throughput,
  or physical-transfer claims.
- Expert IDs are identifiers, not semantic labels.
- Preserve prompt/prefill and generated/decode distinctions.
- Preserve negative and null results; do not tune benchmarks after seeing the
  answer in order to hide an unfavorable outcome.

## Environment

The current package version is `0.9.0` and supports Python 3.10, 3.11, and
3.12. The base package requires neither PyTorch nor Transformers.
Model-specific collection dependencies are isolated behind optional extras:

- `[granite]` pins the validated PyTorch 2.12.0 and Transformers 5.12.0 pair
  used by the built-in public Granite collection CLI.
- `[switch]` pins the same validated pair for the narrow Switch research and
  validation path. It does not provide a public Switch collection CLI and makes
  no broad model-family compatibility claim.

Create a local virtual environment and install the project in editable mode:

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Bash (Linux):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Do not disable TLS or certificate verification to work around package-install
errors.

## Tests

CI separates portable core/offline coverage from the reviewed
ML/full-regression environment.

The installed-package core/offline matrix runs on Python 3.10, 3.11, and 3.12.
It installs the base package with Torch and Transformers deliberately absent,
runs `scripts/audit_core_workflow.py` against the installed package, and
exercises focused portable core contracts. Contributors working only on
core/offline behavior do not need the optional ML dependencies.

The full source-tree regression command is:

PowerShell:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

Bash (Linux):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

CI runs this command separately in the reviewed Python 3.10 ML/full-regression
lane, with Torch 2.12.0 CPU and Transformers 5.12.0.

The CI suite must not download the Granite model or run performance-sensitive
hardware benchmarks. Real model collection, Windows HIP probing, and timing
experiments are separate workflows with their own evidence requirements.

## Reproducibility

Several tracked evidence files are validated byte-for-byte with SHA-256 hashes.
Do not casually reformat, regenerate, or normalize historical evidence files.
The repository uses LF line endings through `.gitattributes` to keep those
artifacts stable across platforms.

When changing an experiment or evidence format:

1. define the new boundary before collecting results;
2. keep historical evidence intact;
3. add structural and tamper tests;
4. record the exact model/runtime/workload inputs;
5. keep measured, simulated, and estimated quantities explicit.

## Scope

Before proposing a large runtime integration, GPU/offload path, learned expert
predictor, additional model family, or additional backend, open an issue and
explain what measured bottleneck or compatibility need justifies the added
scope.

The current project does not claim production acceleration and does not include
a real expert-residency/offload implementation.

## License

By contributing, you agree that your contributions will be licensed under the
project's Apache License 2.0 license. See `LICENSE`.
