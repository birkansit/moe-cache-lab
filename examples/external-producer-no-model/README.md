# No-model external producer workflow

This example demonstrates the supported offline interoperability path:

```text
SYNTHETIC external event source -> canonical trace v2 -> validate ->
descriptive analyze -> stage-qualified pre-flight -> bundle create/verify
```

The producer is an independent, standard-library-only script. It writes the
canonical JSONL boundary directly and does not import `moe_cache_lab`, Torch,
Transformers, or a model-specific collector. A third-party producer likewise
does not need `ProducerResult` or internal collector APIs.

This is a **SYNTHETIC demonstration**, not a real runtime integration. Canonical
validation establishes canonical-file validity only; it does not establish
producer semantic validation or non-interference. Cache results are
**SIMULATED**, transfer-service results are **ESTIMATED** from explicit
fictional assumptions, and physical residency, real traffic, latency,
throughput, tokens/sec, speedup, memory savings, representativeness, and
production offloading are **NOT ESTABLISHED**.

## Run the packaged example

Create an isolated environment and install the base wheel. Replace the wheel
path below with the locally built or supplied base distribution; no model extra
is needed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .\dist\moe_cache_lab-0.8.0-py3-none-any.whl

$PackagedExample = Join-Path $env:VIRTUAL_ENV `
  "share\moe-cache-lab\examples\external-producer-no-model"
Copy-Item -Recurse $PackagedExample .\external-producer-no-model
Set-Location .\external-producer-no-model
New-Item -ItemType Directory -Force artifacts | Out-Null
```

The commands below use the installed `moe-cache-lab` executable. The producer
itself remains unrelated to package internals.

### 1. Produce a deterministic canonical trace

```powershell
python producer.py --output artifacts\trace.jsonl
```

The trace has encoder and decoder stages. `(encoder, 0, 1)` and
`(decoder, 0, 1)` intentionally share numeric layer/expert IDs but remain
different identities. One encoder capacity event is explicitly `unassigned`
with empty expert/probability arrays and therefore creates no expert request.
Metadata, chronology, and output bytes are fixed; no clock, hostname, current
directory, absolute path, or random identifier enters the trace.

### 2. Validate in human and machine modes

```powershell
moe-cache-lab validate-trace artifacts\trace.jsonl

python -c "import pathlib,subprocess; data=subprocess.run(['moe-cache-lab','validate-trace','artifacts/trace.jsonl','--json'],check=True,stdout=subprocess.PIPE).stdout.replace(b'\r\n',b'\n'); pathlib.Path('artifacts/validation.json').write_bytes(data)"
```

Both modes must report a valid trace-v2 file. This result does not validate a
real producer's observation boundary, native chronology mapping, or
non-interference.

### 3. Generate descriptive routing evidence

```powershell
moe-cache-lab analyze artifacts\trace.jsonl `
  --workload-id external-producer-demo `
  --output artifacts\routing-analysis.md `
  --json-output artifacts\routing-analysis.json
```

This command describes the supplied synthetic routing events. It performs no
cache simulation or transfer estimation and does not rank experts or settings.

### 4. Run compatible stage-qualified pre-flight

```powershell
moe-cache-lab analyze artifacts\trace.jsonl `
  --preflight-config preflight-config.json `
  --workload-id external-producer-demo `
  --output artifacts\preflight-report.md `
  --json-output artifacts\preflight-report.json
```

The config assigns 3 bytes to `(encoder, 0, 1)` and 5 bytes to
`(decoder, 0, 1)`. It tests capacities 5 and 9 with the existing LRU/LFU
simulator and a fictional serialized transfer profile. Different results
between tested cells are descriptive **SIMULATED**/**ESTIMATED** outcomes, not
an optimum, recommendation, or performance prediction.

### 5. Check deterministic artifact bytes

```powershell
Get-FileHash -Algorithm SHA256 `
  artifacts\trace.jsonl, `
  artifacts\validation.json, `
  artifacts\routing-analysis.md, `
  artifacts\routing-analysis.json, `
  artifacts\preflight-report.md, `
  artifacts\preflight-report.json
Get-Content expected.sha256
```

The values must match `expected.sha256`. Matching proves byte reproduction of
these synthetic artifacts only; it does not upgrade their evidence meaning.

### 6. Create and verify an integrity bundle

```powershell
moe-cache-lab bundle-create `
  --experiment-id external-producer-no-model `
  --config preflight-config.json `
  --report-json artifacts\preflight-report.json `
  --report-markdown artifacts\preflight-report.md `
  --embed-trace synthetic-external artifacts\trace.jsonl `
  --output-dir artifacts\bundle

moe-cache-lab bundle-verify artifacts\bundle
```

The verifier checks the exact embedded artifacts without fetching or repairing
content. Bundle environment provenance includes the installed tool, Python, and
platform versions, so this example intentionally does not claim one universal
bundle-manifest SHA-256. Successful integrity verification does not turn
synthetic routing into measured evidence or establish runtime behavior.

The complete workflow is offline and requires neither Torch nor Transformers,
model/tokenizer files, network access, a GPU, nor a vendor runtime.
