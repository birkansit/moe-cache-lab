import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.cli import main


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "no-download-preflight"
EXPECTED_MARKDOWN_SHA256 = (
    "219751d3a63b97177f6d3133dd2e7b69fc941eed2416e72568125f4e2936542d"
)
EXPECTED_JSON_SHA256 = (
    "464638a63f66e2c8797f4e7c59efea226f30050032ede5138255f675d0ff4a42"
)


class WalkthroughTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        directory = Path(cls._temporary.name)
        cls.markdown_path = directory / "preflight-report.md"
        cls.json_path = directory / "preflight-report.json"
        with patch(
            "moe_cache_lab.cli.collect_trace",
            side_effect=AssertionError("walkthrough must remain offline and no-model"),
        ), patch(
            "sys.argv",
            [
                "moe-cache-lab",
                "analyze",
                str(DEMO / "trace.jsonl"),
                "--preflight-config",
                str(DEMO / "preflight-config.json"),
                "--output",
                str(cls.markdown_path),
                "--json-output",
                str(cls.json_path),
            ],
        ):
            main()
        cls.payload = json.loads(cls.json_path.read_text(encoding="utf-8"))
        cls.walkthrough = (ROOT / "WALKTHROUGH.md").read_text(encoding="utf-8")
        cls.walkthrough_normalized = " ".join(cls.walkthrough.split())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_walkthrough_report_hashes_and_interpreted_cells_are_exact(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.markdown_path.read_bytes()).hexdigest(),
            EXPECTED_MARKDOWN_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(self.json_path.read_bytes()).hexdigest(),
            EXPECTED_JSON_SHA256,
        )

        provenance = self.payload["trace_provenance"]
        self.assertEqual(provenance["model_id"], "synthetic/moe-preflight-demo")
        self.assertEqual(provenance["capture_method"], "synthetic-offline-demo")
        self.assertEqual(provenance["model_revision"], "synthetic-fixture-v1")
        routing = self.payload["routing_analysis"]
        self.assertEqual(routing["total_event_count"], 8)
        self.assertEqual(routing["total_assignment_count"], 16)
        self.assertEqual(routing["unique_layer_expert_count"], 6)
        self.assertEqual(
            [(row["phase"], row["event_count"], row["assignment_count"])
             for row in routing["phases"]],
            [("prompt", 4, 8), ("generated", 4, 8)],
        )

        cache_rows = {
            (row["capacity_bytes"], row["policy"]): row
            for row in self.payload["simulated_byte_cache_sensitivity"]
        }
        self.assertEqual(
            {
                key: (
                    row["hits"],
                    row["misses"],
                    row["simulated_demand_load_bytes"],
                    row["eviction_count"],
                    row["simulated_evicted_bytes"],
                    row["peak_resident_bytes"],
                    row["final_resident_bytes"],
                )
                for key, row in cache_rows.items()
            },
            {
                (8, "lru"): (2, 14, 45, 12, 37, 8, 8),
                (8, "lfu"): (2, 14, 45, 12, 37, 8, 8),
                (12, "lru"): (5, 11, 36, 8, 26, 12, 10),
                (12, "lfu"): (5, 11, 33, 8, 23, 12, 10),
            },
        )

        transfer_rows = {
            (row["cache_capacity_bytes"], row["policy"], row["hardware_profile_name"]): row
            for row in self.payload["estimated_transfer_service_sensitivity"]
        }
        lru = transfer_rows[(12, "lru", "fictional-fast-link")]
        lfu = transfer_rows[(12, "lfu", "fictional-fast-link")]
        self.assertEqual(lru["assumed_h2d_payload_bandwidth_bytes_per_second"], 20)
        self.assertEqual(lru["assumed_setup_latency_ns_per_loaded_expert"], 50_000_000)
        self.assertEqual(
            lru["estimated_serialized_transfer_service_seconds"],
            {"numerator": 47, "denominator": 20},
        )
        self.assertEqual(
            lfu["estimated_serialized_transfer_service_seconds"],
            {"numerator": 11, "denominator": 5},
        )

        for required in (
            "The trace is **synthetic**",
            "**fictional caller-supplied assumption**",
            "8 routing events and 16 layer-qualified expert assignments",
            "| 8 | 2 | 14 | 45 | 12 | 37 | 8 / 8 |",
            "| 12 | 5 | 11 | 36 | 8 | 26 | 12 / 10 |",
            "`47/20` seconds",
            "`11/5` seconds",
            "the LFU estimate is `3/20` second lower",
            "simulated misses from 14 to 11 (delta `-3`)",
            "simulated demand-load bytes from 45 to 36 (delta `-9`)",
            "from 45 to 33 (delta `-12`)",
        ):
            with self.subTest(required=required):
                self.assertIn(" ".join(required.split()), self.walkthrough_normalized)

    def test_walkthrough_preserves_evidence_and_bundle_boundaries(self) -> None:
        for label in ("**MEASURED**", "**SIMULATED**", "**ESTIMATED**"):
            self.assertIn(label, self.walkthrough)
        for boundary in (
            "not a model benchmark or a representative production workload",
            "does not demonstrate native inference caching",
            "not measured hardware behavior or a speedup prediction",
            "do not interpolate behavior between capacities",
            "Logical operand bytes in that test are not a measurement of DRAM traffic",
            "Do **not** hard-code one universal expected bundle-manifest SHA-256",
            "creation and verification never fetch it",
            "never flatten them or manufacture numerical layer offsets",
            "has no hidden or recommended default",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(" ".join(boundary.split()), self.walkthrough_normalized)

        embedded_hashes = set(re.findall(r"`([0-9a-f]{64})`", self.walkthrough))
        self.assertEqual(
            embedded_hashes,
            {EXPECTED_MARKDOWN_SHA256, EXPECTED_JSON_SHA256},
        )
        lowered = self.walkthrough.lower()
        for forbidden in (
            "c:\\users\\",
            "c:\\projects\\",
            "moe-cache-lab" + "-v05-dev",
            "private artifact",
            "worker" + "_impl",
            "reviewer " + "agent",
            "independently " + "reviewed",
            "access_token",
            "api_key",
            "bearer ",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered)

    def test_walkthrough_is_linked_and_packaged(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        audit = (ROOT / "scripts" / "audit_distributions.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("[canonical no-model evidence walkthrough](WALKTHROUGH.md)", readme)
        self.assertIn('"WALKTHROUGH.md",', pyproject)
        self.assertIn("include WALKTHROUGH.md", manifest)
        self.assertGreaterEqual(audit.count('"WALKTHROUGH.md",'), 2)


if __name__ == "__main__":
    unittest.main()
