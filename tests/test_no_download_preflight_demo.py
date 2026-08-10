import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.cli import main
from moe_cache_lab.preflight_config import read_preflight_config
from moe_cache_lab.trace import read_trace


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "no-download-preflight"


def _expected_hashes() -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in (DEMO / "expected.sha256").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename] = digest
    return expected


class NoDownloadPreflightDemoTests(unittest.TestCase):
    def test_tracked_demo_reproduces_real_cli_outputs_offline(self) -> None:
        trace = read_trace(DEMO / "trace.jsonl")
        config = read_preflight_config(DEMO / "preflight-config.json")

        self.assertTrue(trace.model_id.startswith("synthetic/"))
        self.assertEqual(trace.capture_method, "synthetic-offline-demo")
        self.assertEqual({event.phase for event in trace.events}, {"prompt", "generated"})
        self.assertEqual({event.layer for event in trace.events}, {0, 1})
        self.assertIn((0, 0), config.expert_size_map())
        self.assertIn((1, 0), config.expert_size_map())
        self.assertEqual(config.capacities_bytes, (8, 12))
        self.assertEqual(config.policies, ("lru", "lfu"))
        self.assertTrue(all(profile.name.startswith("fictional-") for profile in config.hardware_profiles))

        expected = _expected_hashes()
        self.assertEqual(set(expected), {"preflight-report.md", "preflight-report.json"})

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            markdown = directory / "preflight-report.md"
            json_output = directory / "preflight-report.json"
            with patch(
                "moe_cache_lab.cli.collect_trace",
                side_effect=AssertionError("offline demo must not collect or download a model"),
            ), patch(
                "sys.argv",
                [
                    "moe-cache-lab",
                    "analyze",
                    str(DEMO / "trace.jsonl"),
                    "--preflight-config",
                    str(DEMO / "preflight-config.json"),
                    "--output",
                    str(markdown),
                    "--json-output",
                    str(json_output),
                ],
            ):
                main()

            outputs = {
                "preflight-report.md": markdown.read_bytes(),
                "preflight-report.json": json_output.read_bytes(),
            }

        for filename, content in outputs.items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected[filename])

        payload = json.loads(outputs["preflight-report.json"])
        self.assertEqual(payload["trace_provenance"]["model_id"], "synthetic/moe-preflight-demo")
        self.assertTrue(
            any(
                row["capacity_bytes"] == 8 and row["eviction_count"] > 0
                for row in payload["simulated_byte_cache_sensitivity"]
            )
        )
        self.assertTrue(
            all(
                row["hardware_profile_name"].startswith("fictional-")
                for row in payload["estimated_transfer_service_sensitivity"]
            )
        )


if __name__ == "__main__":
    unittest.main()
