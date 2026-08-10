import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.analysis import analyze_routing
from moe_cache_lab.analysis_output import render_analysis_json, render_analysis_report
from moe_cache_lab.cli import main
from moe_cache_lab.trace import RoutingEvent, RoutingTrace, write_trace


def _example_trace() -> RoutingTrace:
    return RoutingTrace(
        "example/moe",
        6,
        2,
        (
            RoutingEvent("prompt", 0, 0, (2, 1)),
            RoutingEvent("prompt", 1, 0, (1, 3)),
            RoutingEvent("prompt", 0, 1, (0, 4)),
            RoutingEvent("prompt", 1, 1, (0, 4)),
            RoutingEvent("generated", 2, 0, (2, 1)),
            RoutingEvent("generated", 2, 1, (4, 0)),
            RoutingEvent("generated", 3, 0, (1, 3)),
            RoutingEvent("generated", 3, 1, (0, 5)),
        ),
        created_at="fixed-test-time",
    )


class RoutingAnalysisOutputTests(unittest.TestCase):
    def test_analyze_cli_stdout_requires_no_model_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = write_trace(Path(temporary) / "trace.jsonl", _example_trace())
            stdout = io.StringIO()
            with patch(
                "moe_cache_lab.cli.collect_trace",
                side_effect=AssertionError("analyze must not collect a model trace"),
            ), patch(
                "sys.argv", ["moe-cache-lab", "analyze", str(trace_path)]
            ), redirect_stdout(stdout):
                main()

        report = stdout.getvalue()
        self.assertIn("# MoE routing analysis report", report)
        self.assertIn("Routing events: 8", report)
        self.assertIn("prompt/prefill", report)
        self.assertIn("generated/decode", report)

    def test_human_file_output_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace_path = write_trace(directory / "trace.jsonl", _example_trace())
            first = directory / "first.md"
            second = directory / "second.md"
            for output in (first, second):
                with patch(
                    "sys.argv",
                    ["moe-cache-lab", "analyze", str(trace_path), "--output", str(output)],
                ):
                    main()
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_json_output_is_exact_and_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace_path = write_trace(directory / "trace.jsonl", _example_trace())
            first_report = directory / "first.md"
            second_report = directory / "second.md"
            first_json = directory / "first.json"
            second_json = directory / "second.json"
            for report_path, json_path in (
                (first_report, first_json),
                (second_report, second_json),
            ):
                with patch(
                    "sys.argv",
                    [
                        "moe-cache-lab",
                        "analyze",
                        str(trace_path),
                        "--output",
                        str(report_path),
                        "--json-output",
                        str(json_path),
                    ],
                ):
                    main()

            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            payload = json.loads(first_json.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["consecutive_overlap"]["mean_jaccard"],
                {"numerator": 1, "denominator": 2},
            )
            prompt_layer_zero = payload["phase_layers"][0]
            self.assertEqual(
                prompt_layer_zero["consecutive_overlap"]["mean_jaccard"],
                {"numerator": 1, "denominator": 3},
            )
            self.assertNotIsInstance(
                prompt_layer_zero["consecutive_overlap"]["mean_jaccard"], float
            )

    def test_undefined_values_are_null_and_render_as_na(self) -> None:
        summary = analyze_routing(
            (RoutingEvent("prompt", 0, 0, (1, 2)),)
        )
        report = render_analysis_report(summary)
        payload = json.loads(render_analysis_json(summary))

        self.assertIn("Mean Jaccard: N/A", report)
        self.assertIn("Mean gap events: N/A", report)
        self.assertIsNone(payload["consecutive_overlap"]["mean_jaccard"])
        self.assertIsNone(payload["reuse_gap"]["mean_gap_events"])
        self.assertIsNone(
            payload["phase_layers"][0]["consecutive_overlap"]["mean_jaccard"]
        )
        self.assertIsNone(payload["phase_layers"][0]["reuse_gap"]["mean_gap_events"])

    def test_phase_layer_and_expert_ordering_is_deterministic(self) -> None:
        summary = analyze_routing(_example_trace())
        report = render_analysis_report(summary)
        payload = json.loads(render_analysis_json(summary))

        self.assertEqual(
            [(item["phase"], item["layer_id"]) for item in payload["phase_layers"]],
            [("prompt", 0), ("prompt", 1), ("generated", 0), ("generated", 1)],
        )
        self.assertEqual(
            [item["expert_id"] for item in payload["layers"][0]["expert_frequencies"]],
            [1, 2, 3],
        )
        self.assertLess(
            report.index("### prompt/prefill / layer 0"),
            report.index("### generated/decode / layer 0"),
        )
        self.assertLess(
            report.index("expert 1: 4"),
            report.index("expert 2: 2"),
        )

    def test_report_states_claim_boundaries(self) -> None:
        report = render_analysis_report(analyze_routing(_example_trace()))
        for label in ("MEASURED", "SIMULATED", "ESTIMATED"):
            self.assertIn(label, report)
        self.assertIn("descriptive routing-trace analysis", report)
        self.assertIn(
            "does not establish cache/offload speedup, latency, throughput, "
            "physical GPU residency, or transfer performance",
            report,
        )

    def test_invalid_trace_still_fails_existing_validation(self) -> None:
        valid = RoutingTrace(
            "example/moe",
            4,
            1,
            (
                RoutingEvent("prompt", 0, 0, (0,)),
                RoutingEvent("prompt", 0, 1, (1,)),
            ),
            created_at="fixed-test-time",
        )
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = write_trace(Path(temporary) / "invalid.jsonl", valid)
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            lines[1], lines[2] = lines[2], lines[1]
            trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            with patch(
                "sys.argv", ["moe-cache-lab", "analyze", str(trace_path)]
            ), self.assertRaisesRegex(ValueError, "layer-major collector order"):
                main()


if __name__ == "__main__":
    unittest.main()
