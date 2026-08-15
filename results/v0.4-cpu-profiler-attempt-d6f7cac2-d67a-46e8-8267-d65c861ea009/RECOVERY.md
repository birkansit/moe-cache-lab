# V0.4 staging recovery audit

This attempt completed only the first resource-admission window. No child or
model process was launched. The immutable admission artifact was rejected
because every sample had less than the frozen 8 GiB available-RAM threshold.

The original validator incorrectly required the prime and excluded baseline to
have distinct Windows wall-clock timestamps. Their wall timestamps were equal,
while their monotonic timestamps prove the baseline occurred 184,900 ns later.
Commit `d774994` corrected validation to use nondecreasing wall time and strict
monotonic ordering.

Recovery changed only the unpublished derived `v04-set.json` disposition and
root-cause fields. It preserved the attempt UUID, timestamps, paths, admission
reference, and the admission file byte-for-byte. The obsolete derived manifest
is retained as `v04-set.pre-recovery.json`.

- `admission-1.json` SHA-256: `6dba0bea30bc7fd8ae65b699b23a305d6211ffcda87cbc40788d2a11941105cc`
- obsolete `v04-set.json` SHA-256: `285e1d5f67d56ceda7c3e7a7ed65505fcbf54636dce48af16d96be15bb577526`
- repaired `v04-set.json` SHA-256: `5bb411929ebaf0dd7b149781ac3f1a7808e3050df62b11ce9be290c8a5ec2a31`
- intended terminal state: `deferred`
- intended reason: `host busy or memory gate failed`
- intended root digest: `73b8be063b4f459233e162d6c557d5b72f7f4ae9d5129c010bf753f2ebb38980`

The repaired set manifest was strictly validated in staging before the attempt
directory was atomically renamed to its UUID-bound final path.
