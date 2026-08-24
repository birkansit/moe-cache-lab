import hashlib
import re
import unittest
from pathlib import Path

from moe_cache_lab import __version__


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_package_versions_match_release(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), __version__)
        self.assertEqual(__version__, "0.8.0")

    def test_release_evidence_hashes_and_claim_boundaries(self) -> None:
        expected = {
            "results/v0.2-corpus-v1/manifest-v1.json": "d5ca0013b1dfee96b353bb964666eaf9f6335494b0279362f536b4e8bc359e30",
            "results/v0.2-corpus-v1-report.md": "f21a338537c2001fa4bcc9d7db037beea7892262a74d73138b8d5169204bca9f",
            "results/stage1-runtime-fidelity-v1/stage1-set-manifest.json": "f7e624d95368b1aa976f3e31412fc44e86dbf46439e5c15534032c14446e34d1",
            "results/stage1-runtime-fidelity-v1-benchmark.json": "dd8f0a46470e93132a63abd6d63409c02f077064c5258bb9e59c0a1f36450740",
            "results/stage1-runtime-fidelity-v1-benchmark.md": "f11e9b0fbbdc61d57b578eb00c4e8e8fcd2c0428ba72e60516bd2b609823633a",
            "results/stage2-hip-copy-probe.json": "0072362ad8f38783775e84b6df0145dd6bbd2481cd58c7e89849aaeb539f76c2",
            "scripts/stage2_hip_copy_probe.py": "3c4860b59e5d51916fa800997b87e4c133d2f8b861bf6aba5ebd24196beb0e9d",
            "results/v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980/v04-set.json": "1f87491f3d0512912e593422a2cc0b012ebed343291726131fa51dc2e103fe60",
            "results/v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980/v04-result.json": "ed857ef581f80fbbe8e694bc869e9265e785e5c48bd6e00118b20db7d85e7b72",
            "results/v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980/v04-report.md": "d9b5399a125d162c3c7cdb3f847f30c196b56fd31ace9ee714f9dd4d5b81ceaf",
        }
        for relative, digest in expected.items():
            with self.subTest(relative=relative):
                self.assertEqual(
                    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                    digest,
                )

        checkpoint = (ROOT / "RELEASE_CANDIDATE.md").read_text(encoding="utf-8")
        for label in ("**measured**", "**simulated**", "**estimated**"):
            self.assertIn(label, checkpoint)
        self.assertIn("not an inference accelerator", checkpoint)
        self.assertIn("NO-GO for Stage 3", checkpoint)
        self.assertIn("measured CPU observer", checkpoint)
        self.assertIn("0.4.0rc1", checkpoint)
        self.assertIn("public package version is now", checkpoint)


if __name__ == "__main__":
    unittest.main()
