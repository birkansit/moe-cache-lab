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

    def test_v2_transfer_operation_demo_is_offline_and_deterministic(self) -> None:
        config_payload = json.loads(
            (DEMO / "preflight-config.json").read_text(encoding="utf-8")
        )
        config_payload["format_version"] = 2
        for profile in config_payload["hardware_profiles"]:
            profile["setup_latency_ns_per_transfer_operation"] = profile.pop(
                "setup_latency_ns_per_loaded_expert"
            )
        config_payload["transfer_operation_plans"] = [
            {"name": "one-operation", "operations_per_logical_load": 1},
            {"name": "two-chunks", "operations_per_logical_load": 2},
        ]

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "preflight-v2.json"
            config_path.write_text(
                json.dumps(config_payload), encoding="utf-8"
            )
            output_pairs: list[tuple[bytes, bytes]] = []
            for run_number in (1, 2):
                markdown = directory / f"report-{run_number}.md"
                json_output = directory / f"report-{run_number}.json"
                with patch(
                    "moe_cache_lab.cli.collect_trace",
                    side_effect=AssertionError(
                        "offline demo must not collect or download a model"
                    ),
                ), patch(
                    "sys.argv",
                    [
                        "moe-cache-lab",
                        "analyze",
                        str(DEMO / "trace.jsonl"),
                        "--preflight-config",
                        str(config_path),
                        "--output",
                        str(markdown),
                        "--json-output",
                        str(json_output),
                    ],
                ):
                    main()
                output_pairs.append((markdown.read_bytes(), json_output.read_bytes()))

        self.assertEqual(output_pairs[0], output_pairs[1])
        payload = json.loads(output_pairs[0][1])
        self.assertEqual(payload["format_version"], 2)
        rows = payload["estimated_transfer_service_sensitivity"]
        one = next(
            row for row in rows
            if row["transfer_operation_plan_name"] == "one-operation"
        )
        two = next(
            row for row in rows
            if row["transfer_operation_plan_name"] == "two-chunks"
            and row["cache_capacity_bytes"] == one["cache_capacity_bytes"]
            and row["policy"] == one["policy"]
            and row["hardware_profile_name"] == one["hardware_profile_name"]
        )
        self.assertEqual(
            two["modeled_transfer_operation_count"],
            2 * one["modeled_transfer_operation_count"],
        )
        self.assertEqual(
            two["simulated_logical_demand_load_count"],
            one["simulated_logical_demand_load_count"],
        )
        self.assertEqual(
            two["simulated_demand_load_bytes"],
            one["simulated_demand_load_bytes"],
        )


if __name__ == "__main__":
    unittest.main()
