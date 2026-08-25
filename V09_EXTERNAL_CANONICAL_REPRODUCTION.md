# v0.9 A1 — External canonical reproduction

## Scope and claim boundary

This record covers the v0.9 external canonical reproduction evidence task only. It independently checks the historical external Kimi and historical external OLMoE canonical targets from frozen public raw fixtures and the frozen contributor converter.

No external raw trace bytes or generated canonical trace bytes are tracked in this repository. No trace schema, package version, runtime behavior, cache policy, preflight semantics, public release claim, or historical frozen project evidence is changed.

Mechanical `AGREEMENT` means only:

> frozen raw bytes + frozen converter + recorded execution conditions -> contributor-reported canonical bytes/hash reproduced

It does not establish producer non-interference, runtime performance, physical residency or transfer, model-wide representativeness, native top-k rank preservation, or capture-specific Kimi layer identity. Canonical validation establishes canonical schema/chronology validity only. The preservation audit establishes the frozen raw-emitted event/list mapping specification only.

## Frozen identities

Development base commit:

`19b1d6dd410ac240b6b0353b2fdde22ac1ee8e61`

Exact public v0.8.0 validator source commit:

`e9fb09ec2bc3c39369958cd9aa9ecdc1f5c8cf25`

Validator release identity: public `moe-cache-lab v0.8.0` source at that commit.

### Kimi raw fixture

- repository: `sqliteai/warp`
- commit: `6e579f99689a3e7e2ea875a88d97f68c8bfd3f70`
- path: `tests/trace_kimi_300.jsonl`
- Git blob: `7487627d2cf7dec4c89d70a5ca42244e85a34823`
- contributor raw SHA-256 target: `06ad026c928d1c05eb59f0b1db55e1317c1899b1d3a687741ab5c49fdc014965`
- independently computed raw SHA-256: `06ad026c928d1c05eb59f0b1db55e1317c1899b1d3a687741ab5c49fdc014965`
- raw hash result: `AGREEMENT`

### Historical OLMoE raw fixture

- repository: `sqliteai/warp`
- commit: `dc079dad1c35def2ee7bb3b2de5157db8d9f08a8`
- path: `tests/trace_olmoe_299.jsonl`
- Git blob: `77e162d8b6e3d7c15eb92b7e52654d824cd6a68b`
- contributor raw SHA-256 target: `7d8e7f63551aa24ddab98134b267b6bdf1b4f55d72a6fa7589042f54a4a4383b`
- independently computed raw SHA-256: `7d8e7f63551aa24ddab98134b267b6bdf1b4f55d72a6fa7589042f54a4a4383b`
- raw hash result: `AGREEMENT`

The Git blob identifiers are Git identities only; they were not substituted for content SHA-256.

### Contributor converter

- repository: `mfethe1/warp`
- commit: `6d9800ff08e912e330efc85a2f443784857e262e`
- path: `tools/moe-cache-lab/convert_kimi_olmoe_canonical.py`
- Git blob: `b2a5d91933f45e3e6d1a03a8ee1b3e4b27e24414`
- independently computed content SHA-256: `3f3c43db566e354cd6a7f1bf34f46bf8b7f3a386f85d9d1ec757ab6e467eeb1f`

## Converter source audit

The exact frozen converter was read directly and was not modified for reproduction. Observed properties relevant to this task:

- stdlib-only converter imports (`json`, `sys`);
- input JSONL records are consumed in file order;
- no raw-record sorting or chronology repair;
- expert lists are emitted as supplied and are not re-sorted by the converter;
- Kimi requires exactly 26 routed-layer lists per row and exits loudly otherwise;
- Kimi maps routed ordinal `i` to canonical layer `i + 1`;
- historical OLMoE enumerates each raw layer list directly as canonical layer `0..15`;
- two independent runs in the recorded environment produced identical canonical bytes for both cases.

The Kimi layer mapping remains **SOURCE-SUPPORTED INFERENCE — NOT CAPTURE-PROVEN**. Mechanical reproduction of that converter does not upgrade the mapping to capture-specific truth.

## Recorded execution conditions

Evidence execution was recorded in GitHub Actions CI run `32752290014`, job `ML/full regression (Python 3.10, Torch 2.12.0 CPU, Transformers 5.12.0)`.

- OS: Ubuntu 24.04.4 LTS
- platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`
- Python implementation: `CPython`
- Python version: `3.10.21`
- Python executable: `/opt/hostedtoolcache/Python/3.10.21/x64/bin/python`

Supported conclusion:

`Deterministic reproduction established in the recorded environment.`

No cross-platform or portable byte-determinism claim is made.

Temporary evidence paths used in the recorded run:

- converter: `/tmp/tmpevurh6r0/convert_kimi_olmoe_canonical.py`
- Kimi raw: `/tmp/tmpevurh6r0/kimi-raw.jsonl`
- Kimi outputs: `/tmp/tmpevurh6r0/kimi-work/kimi-run1.jsonl`, `/tmp/tmpevurh6r0/kimi-work/kimi-run2.jsonl`
- historical OLMoE raw: `/tmp/tmpevurh6r0/olmoe-raw.jsonl`
- historical OLMoE outputs: `/tmp/tmpevurh6r0/olmoe-work/olmoe-run1.jsonl`, `/tmp/tmpevurh6r0/olmoe-work/olmoe-run2.jsonl`

Exact conversion commands:

```text
/opt/hostedtoolcache/Python/3.10.21/x64/bin/python /tmp/tmpevurh6r0/convert_kimi_olmoe_canonical.py kimi /tmp/tmpevurh6r0/kimi-raw.jsonl /tmp/tmpevurh6r0/kimi-work/kimi-run1.jsonl
/opt/hostedtoolcache/Python/3.10.21/x64/bin/python /tmp/tmpevurh6r0/convert_kimi_olmoe_canonical.py kimi /tmp/tmpevurh6r0/kimi-raw.jsonl /tmp/tmpevurh6r0/kimi-work/kimi-run2.jsonl
/opt/hostedtoolcache/Python/3.10.21/x64/bin/python /tmp/tmpevurh6r0/convert_kimi_olmoe_canonical.py olmoe /tmp/tmpevurh6r0/olmoe-raw.jsonl /tmp/tmpevurh6r0/olmoe-work/olmoe-run1.jsonl
/opt/hostedtoolcache/Python/3.10.21/x64/bin/python /tmp/tmpevurh6r0/convert_kimi_olmoe_canonical.py olmoe /tmp/tmpevurh6r0/olmoe-raw.jsonl /tmp/tmpevurh6r0/olmoe-work/olmoe-run2.jsonl
```

## Two-run reproduction

### Kimi

- expected canonical SHA-256: `3a2367a43e6d8d3bd3aefc91ef9e294428d1444b0822e5a86dcc777c21e418ac`
- run 1 SHA-256: `3a2367a43e6d8d3bd3aefc91ef9e294428d1444b0822e5a86dcc777c21e418ac`
- run 2 SHA-256: `3a2367a43e6d8d3bd3aefc91ef9e294428d1444b0822e5a86dcc777c21e418ac`
- run 1 == run 2: `PASS`
- reproduced SHA == contributor target: `PASS`
- **Mechanical reproduction: `AGREEMENT`**

Superseded defect lineage remains recorded as:

`7f08cae05115f97b881a529177c0d345f8dbd0fb87195702aab26442a04fa756`

That old Kimi artifact is **SUPERSEDED — schema-valid but semantically mis-mapped (`0..25`)** and is not current evidence.

### Historical OLMoE

- expected canonical SHA-256: `83168fb10391646a97bb0c00c159094784a0ef72dc7962d18055020701f1d9b4`
- run 1 SHA-256: `83168fb10391646a97bb0c00c159094784a0ef72dc7962d18055020701f1d9b4`
- run 2 SHA-256: `83168fb10391646a97bb0c00c159094784a0ef72dc7962d18055020701f1d9b4`
- run 1 == run 2: `PASS`
- reproduced SHA == contributor target: `PASS`
- **Mechanical reproduction: `AGREEMENT`**

## Independent raw-to-canonical preservation audit

Tracked checker: `scripts/audit_external_canonical_reproduction.py`.

The checker does not import or call the external converter. It reads raw and canonical JSONL independently and enforces the frozen mapping specification.

### Kimi

Verified across the full fixture:

- 300 raw token rows and token chronology `0..299`;
- 26 routed-layer lists per row;
- canonical `token_position == raw tok`;
- canonical `layer == raw routed ordinal + 1`;
- `routing_stage == "decoder"`;
- `phase == "decoder_generated"`;
- v2 assigned representation remains consistent;
- `selected_experts` exactly equals the raw emitted list;
- `selected_probabilities == []`;
- no event added, dropped, reordered, or chronology-repaired;
- 7,800 canonical events;
- 62,400 expert requests;
- canonical layer range `1..26`.

**Raw-to-canonical preservation audit: `PASS`**

### Historical OLMoE

Verified across the full fixture:

- 299 raw token rows with tokens `1..299`;
- 16 layer lists per row;
- canonical `token_position == raw tok`;
- canonical `layer == raw layer ordinal`;
- `phase == "generated"`;
- `selected_experts` exactly equals the raw emitted list;
- no event added, dropped, reordered, or chronology-repaired;
- 4,784 canonical events;
- 38,272 expert requests;
- canonical layer range `0..15`.

**Raw-to-canonical preservation audit: `PASS`**

For both cases, list equality proves preservation of the list already emitted by the producer. It does not establish native top-k rank/order because the upstream producer sorted IDs before canonical conversion.

Focused offline negative tests demonstrate rejection of a wrong Kimi `0..25` mapping, changed membership, a dropped event, reordered events, and an invalid routed-layer count.

A one-shot network-enabled evidence execution was used only to acquire the frozen external bytes and produce the recorded reproduction/validator evidence above. That network-enabled test was removed after the evidence record was written; the final tracked tests are offline.

## Exact public v0.8.0 validation

The reproduced run-1 traces were validated using exact public source commit `e9fb09ec2bc3c39369958cd9aa9ecdc1f5c8cf25`.

Kimi command:

```text
/opt/hostedtoolcache/Python/3.10.21/x64/bin/python -c "from moe_cache_lab.cli import main; main()" validate-trace /tmp/tmpevurh6r0/kimi-work/kimi-run1.jsonl --json
```

Kimi machine result:

```json
{"format":"moe-cache-lab.trace-validation","format_version":1,"summary":{"assigned_event_count":7800,"event_count":7800,"expert_request_count":62400,"model_id":"Kimi-Linear-48B/warp-3bit-oracle (warp tests/trace_kimi_300.jsonl @6e579f99)","routing_stages":["decoder"],"unassigned_event_count":0},"trace_format_version":2,"valid":true}
```

**Kimi canonical validity: `PASS`**

Historical OLMoE command:

```text
/opt/hostedtoolcache/Python/3.10.21/x64/bin/python -c "from moe_cache_lab.cli import main; main()" validate-trace /tmp/tmpevurh6r0/olmoe-work/olmoe-run1.jsonl --json
```

Historical OLMoE machine result:

```json
{"format":"moe-cache-lab.trace-validation","format_version":1,"summary":{"event_count":4784,"expert_request_count":38272,"experts_per_token":8,"model_id":"OLMoE-1B-7B-0125/warp-fixture (warp tests/trace_olmoe_299.jsonl @dc079dad)","num_experts":64},"trace_format_version":1,"valid":true}
```

**Historical OLMoE canonical validity: `PASS`**

## Provenance and semantic boundaries

Historical OLMoE mechanical reproduction does not repair missing run provenance. The historical run still lacks a recorded exact model revision, exact Torch/Transformers environment, device/dtype, exact invocation, and independently recorded active hook branch. Source defaults and later recaptures are not substituted for that missing evidence.

This task does not broaden producer semantic-mapping claims beyond separately documented source evidence.

**Producer non-interference: `NOT ESTABLISHED`**

## Claim-by-claim result

| Claim | Kimi | Historical OLMoE |
| --- | --- | --- |
| Raw content hash | `AGREEMENT` | `AGREEMENT` |
| Mechanical reproduction | `AGREEMENT` | `AGREEMENT` |
| Two-run determinism in recorded environment | `PASS` | `PASS` |
| Raw-to-canonical preservation audit | `PASS` | `PASS` |
| Exact public v0.8.0 canonical validity | `PASS` | `PASS` |
| Native rank/order preservation | `NOT ESTABLISHED` | `NOT ESTABLISHED` |
| Producer non-interference | `NOT ESTABLISHED` | `NOT ESTABLISHED` |

Kimi-specific boundary:

**Kimi native-layer mapping evidence: `SOURCE-SUPPORTED INFERENCE — NOT CAPTURE-PROVEN`**

Historical OLMoE-specific boundary:

**Historical capture provenance gaps remain recorded; mechanical agreement does not fill them.**

## Bounded conclusion

Both frozen historical external cases independently reproduced the contributor-reported canonical SHA-256 targets from the exact frozen raw bytes and exact frozen converter in the recorded CI environment. Both reproduced traces passed the independent preservation audit and exact public v0.8.0 canonical validation.

The conclusion is mechanical/canonical agreement only. No runtime performance, physical transfer/residency, producer non-interference, broad representativeness, native rank preservation, or capture-proven Kimi native-layer claim follows from this result.
