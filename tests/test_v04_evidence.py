from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import unittest

from moe_cache_lab.v04 import MIN_AVAILABLE_BYTES, validate_v04


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_UUID = "d6f7cac2-d67a-46e8-8267-d65c861ea009"
ATTEMPT = ROOT / "results" / f"v0.4-cpu-profiler-attempt-{ATTEMPT_UUID}"
HASHES = {
    "admission-1.json": "6dba0bea30bc7fd8ae65b699b23a305d6211ffcda87cbc40788d2a11941105cc",
    "RECOVERY.md": "ccf262fa50bb6c8fa2a7d763a8bbe822a06df06046980bc0f92429a38c784b98",
    "v04-set.json": "5bb411929ebaf0dd7b149781ac3f1a7808e3050df62b11ce9be290c8a5ec2a31",
    "v04-set.pre-recovery.json": "285e1d5f67d56ceda7c3e7a7ed65505fcbf54636dce48af16d96be15bb577526",
}
ROOT_DIGEST = "73b8be063b4f459233e162d6c557d5b72f7f4ae9d5129c010bf753f2ebb38980"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V04DeferredEvidenceTests(unittest.TestCase):
    def test_finalized_admission_only_attempt_is_byte_pinned_and_strict(self):
        self.assertTrue(ATTEMPT.is_dir())
        self.assertFalse(ATTEMPT.is_symlink())
        entries = list(ATTEMPT.rglob("*"))
        self.assertTrue(all(path.is_file() and not path.is_symlink() for path in entries))
        inventory = {path.relative_to(ATTEMPT).as_posix() for path in entries}
        self.assertEqual(inventory, set(HASHES))
        self.assertEqual(
            {name: _sha256(ATTEMPT / name) for name in sorted(inventory)},
            HASHES,
        )

        manifest_path = ATTEMPT / "v04-set.json"
        self.assertEqual(validate_v04(manifest_path), manifest_path.resolve())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["attempt_uuid"], ATTEMPT_UUID)
        self.assertEqual(manifest["state"], "deferred")
        self.assertEqual(manifest["reason"], "host busy or memory gate failed")
        self.assertEqual(manifest["root_cause"]["category"], "busy_threshold")
        self.assertEqual(manifest["root_cause"]["component"], "resource_admission")
        self.assertIsNone(manifest["root_cause"]["status"])
        self.assertEqual(manifest["root_cause"]["digest"], ROOT_DIGEST)
        self.assertEqual(manifest["children"], [])
        self.assertEqual(manifest["failures"], [])
        self.assertIsNone(manifest["experiment_started_at"])
        self.assertEqual(manifest["elapsed_seconds"], 0.0)
        self.assertEqual(len(manifest["admissions"]), 1)

        admission_ref = manifest["admissions"][0]
        admission_path = ATTEMPT / admission_ref["path"]
        self.assertEqual(admission_ref["sha256"], HASHES["admission-1.json"])
        self.assertEqual(admission_ref["size"], len(admission_path.read_bytes()))
        self.assertEqual(_sha256(admission_path), admission_ref["sha256"])
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        self.assertFalse(admission["admitted"])
        self.assertEqual(admission["reason"], "host busy or memory gate failed")
        self.assertEqual(admission["errors"], [])
        self.assertEqual(admission["audit"], {
            "open_query_status": 0,
            "add_cpu_status": 0,
            "add_disk_status": 0,
            "close_query_status": 0,
            "constructor_error": None,
        })
        self.assertEqual(len(admission["samples"]), 15)
        self.assertEqual(admission["prime"]["timestamp"], admission["baseline"]["timestamp"])
        self.assertEqual(
            admission["baseline"]["monotonic_ns"] - admission["prime"]["monotonic_ns"],
            184_900,
        )

        samples = admission["samples"]
        cpu = [sample["cpu_percent"] for sample in samples]
        disk = [sample["disk_percent"] for sample in samples]
        commit = [sample["commit_percent"] for sample in samples]
        ram = [sample["available_physical_bytes"] for sample in samples]
        self.assertLessEqual(statistics.median(cpu), 10)
        self.assertLessEqual(max(cpu), 25)
        self.assertLessEqual(statistics.median(disk), 10)
        self.assertLessEqual(max(disk), 50)
        self.assertTrue(all(value <= 75 for value in commit))
        self.assertTrue(all(value < MIN_AVAILABLE_BYTES for value in ram))
        self.assertAlmostEqual(min(ram) / 2**30, 6.22539, places=5)
        self.assertAlmostEqual(max(ram) / 2**30, 6.23562, places=5)

        obsolete = json.loads(
            (ATTEMPT / "v04-set.pre-recovery.json").read_text(encoding="utf-8")
        )
        self.assertEqual(obsolete["attempt_uuid"], ATTEMPT_UUID)
        self.assertEqual(obsolete["admissions"], manifest["admissions"])
        self.assertIn("strictly ordered/contained", obsolete["reason"])
        self.assertNotEqual(obsolete["root_cause"]["digest"], ROOT_DIGEST)
        recovery = (ATTEMPT / "RECOVERY.md").read_text(encoding="utf-8")
        normalized_recovery = " ".join(recovery.split())
        for required in (
            "No child or model process was launched",
            "184,900 ns later",
            "preserved the attempt UUID",
            HASHES["admission-1.json"],
            HASHES["v04-set.pre-recovery.json"],
            HASHES["v04-set.json"],
            ROOT_DIGEST,
        ):
            self.assertIn(required, normalized_recovery)


if __name__ == "__main__":
    unittest.main()
