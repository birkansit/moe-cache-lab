# Contributing to moe-cache-lab

`moe-cache-lab` is a correctness-first research prototype. Small, reviewable
changes with explicit tests and claim boundaries are preferred over broad
feature expansion.

## Core invariants

Contributions must preserve these boundaries:

- The model's native router is authoritative.
- Routing observations are measured only when trace provenance establishes
  measurement.
- Cache outcomes from trace replay are simulated.
- Transfer-service outputs are estimated from explicit assumptions unless a
  separate measurement record establishes otherwise.
- Do not convert simulated or estimated results into speedup, latency,
  throughput, physical-transfer, or GPU-residency claims.
- Expert IDs are identifiers, not semantic labels.
- Preserve prompt/prefill and generated/decode distinctions.
- Preserve negative and null results rather than tuning them away.

## Environment

Version 0.5.0 uses:

- Python 3.10+
- PyTorch 2.12.0
- Transformers 5.12.0

Create an isolated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Do not disable TLS or certificate verification to work around package-install
errors.

## Tests

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

The portable CI must not download the reference model or run
performance-sensitive hardware benchmarks. Real model collection and local
hardware calibration are separate validation workflows.

## Reproducibility

When adding or changing evidence:

1. define the measurement/simulation boundary before collecting results;
2. store exact runtime/model/workload metadata;
3. preserve raw integer/rational values where practical;
4. add deterministic or tamper-detection tests where appropriate;
5. keep MEASURED, SIMULATED, and ESTIMATED quantities explicit.

Routing traces may contain prompt text and environment metadata. Do not publish
private prompts, credentials, local identity data, or unnecessary absolute
paths.

## Scope

Large runtime integrations, new model families, new backends, offload engines,
or learned predictors should be justified by a concrete compatibility need or
measured bottleneck.

The project does not claim production acceleration and does not currently
implement real expert residency/offloading.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0. See `LICENSE`.
