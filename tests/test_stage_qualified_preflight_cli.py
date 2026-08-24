import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.cli import main
from moe_cache_lab.preflight import run_stage_qualified_preflight_analysis
from moe_cache_lab.preflight_config import PreflightConfigV3, read_preflight_config
from moe_cache_lab.preflight_output import (
    render_stage_qualified_preflight_json,
    render_stage_qualified_preflight_report,
)
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
    read_versioned_trace,
    write_trace_v2,
)


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "no-download-stage-qualified-preflight"


def _expected_hashes() -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in (DEMO / "expected.sha256").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename] = digest
    return expected


def _invoke(arguments: list[str]) -> tuple[str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch.object(sys, "argv", ["moe-cache-lab", *arguments]), (
        contextlib.redirect_stdout(stdout)
    ), contextlib.redirect_stderr(stderr):
        main()
    return stdout.getvalue(), stderr.getvalue()


class StageQualifiedPreflightCliTests(unittest.TestCase):
    def test_tracked_fixture_runs_offline_and_reproduces_exact_outputs(self) -> None:
        trace = read_versioned_trace(DEMO / "trace.jsonl")
        config = read_preflight_config(DEMO / "preflight-config.json")
        self.assertIsInstance(trace, RoutingTraceV2)
        self.assertIsInstance(config, PreflightConfigV3)
        self.assertEqual((trace.assigned_event_count, trace.unassigned_event_count), (4, 1))
        self.assertEqual(config.expert_size_map()[("encoder", 0, 1)], 3)
        self.assertEqual(config.expert_size_map()[("decoder", 0, 1)], 5)
        self.assertEqual(config.capacities_bytes, (5, 9))
        self.assertEqual(config.policies, ("lru", "lfu"))
        self.assertTrue(all(item.name.startswith("fictional-") for item in config.hardware_profiles))

        expected = _expected_hashes()
        self.assertEqual(
            set(expected),
            {"stage-qualified-preflight-report.md", "stage-qualified-preflight-report.json"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            markdown = directory / "report.md"
            json_output = directory / "report.json"
            with patch(
                "moe_cache_lab.cli.collect_trace",
                side_effect=AssertionError("offline preflight must not collect a model"),
            ):
                _invoke([
                    "analyze",
                    str(DEMO / "trace.jsonl"),
                    "--preflight-config",
                    str(DEMO / "preflight-config.json"),
                    "--workload-id",
                    "synthetic-stage-qualified-demo",
                    "--output",
                    str(markdown),
                    "--json-output",
                    str(json_output),
                ])
            outputs = {
                "stage-qualified-preflight-report.md": markdown.read_bytes(),
                "stage-qualified-preflight-report.json": json_output.read_bytes(),
            }

        for filename, content in outputs.items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected[filename])

        direct = run_stage_qualified_preflight_analysis(
            trace,
            config,
            workload_id="synthetic-stage-qualified-demo",
        )
        self.assertEqual(
            outputs["stage-qualified-preflight-report.md"],
            render_stage_qualified_preflight_report(direct).encode(),
        )
        self.assertEqual(
            outputs["stage-qualified-preflight-report.json"],
            render_stage_qualified_preflight_json(direct).encode(),
        )
        stdout, stderr = _invoke([
            "analyze",
            str(DEMO / "trace.jsonl"),
            "--preflight-config",
            str(DEMO / "preflight-config.json"),
            "--workload-id",
            "synthetic-stage-qualified-demo",
        ])
        self.assertEqual(stderr, "")
        self.assertEqual(
            stdout.encode(), outputs["stage-qualified-preflight-report.md"]
        )

        payload = json.loads(outputs["stage-qualified-preflight-report.json"])
        self.assertEqual(payload["format"], "moe-cache-lab.preflight-analysis")
        self.assertEqual(payload["format_version"], 3)
        workload = payload["routing_evidence"]["workloads"][0]
        self.assertEqual(workload["workload_id"], "synthetic-stage-qualified-demo")
        self.assertEqual((workload["event_count"], workload["expert_request_count"]), (5, 4))
        self.assertEqual(
            payload["workload_byte_context"]["referenced_expert_keys"],
            [
                {"routing_stage": "encoder", "layer_id": 0, "expert_id": 1},
                {"routing_stage": "decoder", "layer_id": 0, "expert_id": 1},
                {"routing_stage": "decoder", "layer_id": 0, "expert_id": 2},
            ],
        )

    def test_markdown_defaults_to_stdout(self) -> None:
        stdout, stderr = _invoke([
            "analyze",
            str(DEMO / "trace.jsonl"),
            "--preflight-config",
            str(DEMO / "preflight-config.json"),
        ])
        self.assertEqual(stderr, "")
        self.assertTrue(stdout.startswith("# MoE stage-qualified pre-flight analysis report\n"))
        self.assertIn("(encoder, 0, 1), (decoder, 0, 1)", stdout)

    def test_all_unassigned_v2_cli_is_zero_request_and_zero_cost(self) -> None:
        trace = RoutingTraceV2(
            model_id="synthetic/all-unassigned-cli",
            routing_stages=(RoutingStageProfile("decoder", 4, 1, True),),
            events=(
                RoutingEventV2(
                    "decoder",
                    "decoder_prompt",
                    0,
                    0,
                    "unassigned",
                    (),
                    (),
                    unassigned_reason="capacity",
                ),
            ),
            capture_method="synthetic-test",
            created_at="fixed",
        )
        config = {
            "format": "moe-cache-lab.preflight-config",
            "format_version": 3,
            "expert_sizes": [],
            "capacities_bytes": [1],
            "policies": ["lru", "lfu"],
            "hardware_profiles": [
                {
                    "name": "fictional",
                    "h2d_payload_bandwidth_bytes_per_second": 1,
                    "setup_latency_ns_per_transfer_operation": 1,
                }
            ],
            "transfer_operation_plans": [
                {"name": "one", "operations_per_logical_load": 1}
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace_path = write_trace_v2(directory / "trace.jsonl", trace)
            config_path = directory / "config.json"
            json_path = directory / "report.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            _invoke([
                "analyze",
                str(trace_path),
                "--preflight-config",
                str(config_path),
                "--json-output",
                str(json_path),
            ])
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        workload = payload["routing_evidence"]["workloads"][0]
        self.assertEqual((workload["event_count"], workload["expert_request_count"]), (1, 0))
        for row in payload["simulated_byte_cache_sensitivity"]:
            self.assertEqual(
                (
                    row["expert_request_count"],
                    row["hits"],
                    row["misses"],
                    row["simulated_demand_load_bytes"],
                    row["eviction_count"],
                    row["final_resident_bytes"],
                ),
                (0, 0, 0, 0, 0, 0),
            )
        for row in payload["estimated_transfer_service_sensitivity"]:
            self.assertEqual(row["simulated_logical_demand_load_count"], 0)
            self.assertEqual(row["modeled_transfer_operation_count"], 0)
            self.assertEqual(
                row["estimated_serialized_transfer_service_seconds"],
                {"numerator": 0, "denominator": 1},
            )

    def test_top_k_with_v2_preflight_rejects_before_execution_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            markdown = directory / "report.md"
            json_output = directory / "report.json"
            stderr = io.StringIO()
            with patch(
                "moe_cache_lab.preflight.run_stage_qualified_preflight_analysis",
                side_effect=AssertionError("preflight must not run"),
            ) as preflight, patch.object(
                sys,
                "argv",
                [
                    "moe-cache-lab",
                    "analyze",
                    str(DEMO / "trace.jsonl"),
                    "--preflight-config",
                    str(DEMO / "preflight-config.json"),
                    "--top-k",
                    "1",
                    "--output",
                    str(markdown),
                    "--json-output",
                    str(json_output),
                ],
            ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--top-k cannot be combined with trace v2 preflight analysis", stderr.getvalue())
            preflight.assert_not_called()
            self.assertFalse(markdown.exists())
            self.assertFalse(json_output.exists())


if __name__ == "__main__":
    unittest.main()
