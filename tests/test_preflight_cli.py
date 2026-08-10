import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.analysis import analyze_routing
from moe_cache_lab.analysis_output import render_analysis_report
from moe_cache_lab.cli import main
from moe_cache_lab.trace import RoutingEvent, RoutingTrace, write_trace


def _trace() -> RoutingTrace:
    return RoutingTrace(
        model_id="example/moe",
        num_experts=3,
        experts_per_token=2,
        events=(
            RoutingEvent("prompt", 0, 0, (0, 1)),
            RoutingEvent("prompt", 1, 0, (0, 2)),
        ),
        capture_method="synthetic-test-capture",
        created_at="fixed-test-time",
        model_revision="test-revision",
    )


def _config_payload(capacities: list[int] | None = None) -> dict:
    return {
        "format": "moe-cache-lab.preflight-config",
        "format_version": 1,
        "expert_sizes": [
            {"layer_id": 0, "expert_id": 0, "size_bytes": 4},
            {"layer_id": 0, "expert_id": 1, "size_bytes": 2},
            {"layer_id": 0, "expert_id": 2, "size_bytes": 2},
        ],
        "capacities_bytes": [6, 8] if capacities is None else capacities,
        "policies": ["lru", "lfu"],
        "hardware_profiles": [
            {
                "name": "assumption-a",
                "h2d_payload_bandwidth_bytes_per_second": 8,
                "setup_latency_ns_per_loaded_expert": 1_000_000_000,
            }
        ],
    }


def _write_fixture(directory: Path, capacities: list[int] | None = None) -> tuple[Path, Path]:
    trace_path = write_trace(directory / "trace.jsonl", _trace())
    config_path = directory / "preflight.json"
    config_path.write_text(
        json.dumps(_config_payload(capacities), sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return trace_path, config_path


class PreflightCliTests(unittest.TestCase):
    def test_routing_only_analyze_remains_byte_compatible_without_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_path, _ = _write_fixture(Path(temporary))
            stdout = io.StringIO()
            with patch(
                "sys.argv", ["moe-cache-lab", "analyze", str(trace_path)]
            ), redirect_stdout(stdout):
                main()

        self.assertEqual(stdout.getvalue(), render_analysis_report(analyze_routing(_trace())))

    def test_config_enabled_analyze_requires_no_model_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_path, config_path = _write_fixture(Path(temporary))
            stdout = io.StringIO()
            with patch(
                "moe_cache_lab.cli.collect_trace",
                side_effect=AssertionError("preflight analyze must not collect a model trace"),
            ), patch(
                "sys.argv",
                [
                    "moe-cache-lab",
                    "analyze",
                    str(trace_path),
                    "--preflight-config",
                    str(config_path),
                ],
            ), redirect_stdout(stdout):
                main()

        report = stdout.getvalue()
        self.assertIn("# MoE pre-flight analysis report", report)
        self.assertIn("## SIMULATED byte-cache sensitivity", report)
        self.assertIn("## ESTIMATED serialized transfer-service sensitivity", report)
        self.assertIn("MEASURED only if trace provenance establishes that", report)

    def test_config_enabled_markdown_and_json_files_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace_path, config_path = _write_fixture(directory)
            outputs = []
            for index in (1, 2):
                markdown = directory / f"report-{index}.md"
                json_output = directory / f"report-{index}.json"
                with patch(
                    "sys.argv",
                    [
                        "moe-cache-lab",
                        "analyze",
                        str(trace_path),
                        "--preflight-config",
                        str(config_path),
                        "--output",
                        str(markdown),
                        "--json-output",
                        str(json_output),
                    ],
                ):
                    main()
                outputs.append((markdown, json_output))

            self.assertEqual(outputs[0][0].read_bytes(), outputs[1][0].read_bytes())
            self.assertEqual(outputs[0][1].read_bytes(), outputs[1][1].read_bytes())
            payload = json.loads(outputs[0][1].read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "moe-cache-lab.preflight-analysis")
            self.assertEqual(
                payload["estimated_transfer_service_sensitivity"][0][
                    "estimated_serialized_transfer_service_seconds"
                ],
                {"numerator": 4, "denominator": 1},
            )

    def test_invalid_config_and_infeasible_capacity_fail_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace_path, config_path = _write_fixture(directory)
            payload = _config_payload()
            del payload["hardware_profiles"]
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "moe-cache-lab",
                    "analyze",
                    str(trace_path),
                    "--preflight-config",
                    str(config_path),
                ],
            ), self.assertRaisesRegex(ValueError, "missing required fields"):
                main()

            _, config_path = _write_fixture(directory, capacities=[5])
            with patch(
                "sys.argv",
                [
                    "moe-cache-lab",
                    "analyze",
                    str(trace_path),
                    "--preflight-config",
                    str(config_path),
                ],
            ), self.assertRaisesRegex(
                ValueError, "required working set 6 bytes exceeds byte-cache capacity 5 bytes"
            ):
                main()


if __name__ == "__main__":
    unittest.main()
