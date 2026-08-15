import importlib.util
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage2_hip_copy_probe.py"
RESULT = ROOT / "results" / "stage2-hip-copy-probe.json"
RESULT_SHA256 = "0072362ad8f38783775e84b6df0145dd6bbd2481cd58c7e89849aaeb539f76c2"
SPEC = importlib.util.spec_from_file_location("stage2_hip_copy_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class Stage2HipCopyProbeTests(unittest.TestCase):
    def test_tracked_raw_result_is_structurally_and_numerically_valid(self) -> None:
        self.assertEqual(hashlib.sha256(RESULT.read_bytes()).hexdigest(), RESULT_SHA256)
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        PROBE.validate_result(result)
        self.assertEqual(
            result["environment"]["probe_script_sha256"],
            hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        )
        names = [case["name"] for case in result["cases"]]
        self.assertEqual(len(names), 4)
        self.assertEqual(len(set(names)), 4)

    def test_fill_hash_matches_direct_bytes(self) -> None:
        size = 2 * 1024 * 1024 + 17
        self.assertEqual(
            PROBE._expected_fill_sha256(size, 0xA5),
            hashlib.sha256(bytes([0xA5]) * size).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
