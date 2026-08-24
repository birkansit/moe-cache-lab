import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.analysis import analyze_routing
from moe_cache_lab.analysis_output import render_analysis_json, render_analysis_report
from moe_cache_lab.cli import main
from moe_cache_lab.experiment_bundle import verify_experiment_bundle
from moe_cache_lab.trace import RoutingEvent, RoutingTrace, write_trace
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
    write_trace_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def _v1_trace() -> RoutingTrace:
    return RoutingTrace(
        "synthetic/v1",
        4,
        1,
        (
            RoutingEvent("prompt", 0, 0, (1,), token_id=10),
            RoutingEvent("generated", 1, 0, (2,), token_id=11),
        ),
        capture_method="synthetic-v1",
        created_at="fixed",
    )


def _v2_trace() -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="synthetic/v2",
        routing_stages=(
            RoutingStageProfile("encoder", 4, 1, True),
            RoutingStageProfile("decoder", 4, 1, True),
        ),
        events=(
            RoutingEventV2(
                "encoder", "source", 0, 0, "assigned", (2,), (0.75,), 10
            ),
            RoutingEventV2(
                "encoder",
                "source",
                1,
                0,
                "unassigned",
                (),
                (),
                11,
                "capacity",
            ),
            RoutingEventV2(
                "decoder",
                "decoder_prompt",
                0,
                0,
                "assigned",
                (1,),
                (0.8,),
                20,
            ),
            RoutingEventV2(
                "decoder",
                "decoder_generated",
                1,
                0,
                "assigned",
                (1,),
                (0.9,),
                21,
            ),
        ),
        capture_method="synthetic-v2",
        created_at="fixed",
        transformers_version="5.12.0",
        model_revision="fixed-revision",
    )


def _v3_preflight_config_payload() -> dict:
    return {
        "format": "moe-cache-lab.preflight-config",
        "format_version": 3,
        "expert_sizes": [
            {
                "routing_stage": "encoder",
                "layer_id": 0,
                "expert_id": 1,
                "size_bytes": 16,
            }
        ],
        "capacities_bytes": [16],
        "policies": ["lru"],
        "hardware_profiles": [
            {
                "name": "synthetic",
                "h2d_payload_bandwidth_bytes_per_second": 1000,
                "setup_latency_ns_per_transfer_operation": 10,
            }
        ],
        "transfer_operation_plans": [
            {"name": "one", "operations_per_logical_load": 1}
        ],
    }


def _legacy_preflight_config_payload(version: int) -> dict:
    payload = _v3_preflight_config_payload()
    payload["format_version"] = version
    payload["expert_sizes"] = [
        {"layer_id": 0, "expert_id": 1, "size_bytes": 16}
    ]
    if version == 1:
        for profile in payload["hardware_profiles"]:
            profile["setup_latency_ns_per_loaded_expert"] = profile.pop(
                "setup_latency_ns_per_transfer_operation"
            )
        del payload["transfer_operation_plans"]
    return payload


def _invoke(arguments: list[str]) -> tuple[str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch.object(sys, "argv", ["moe-cache-lab", *arguments]), (
        contextlib.redirect_stdout(stdout)
    ), contextlib.redirect_stderr(stderr):
        main()
    return stdout.getvalue(), stderr.getvalue()


class PublicCliPathTests(unittest.TestCase):
    def test_v1_analyze_dispatch_is_byte_exact_for_markdown_and_json(self) -> None:
        trace = _v1_trace()
        expected_markdown = render_analysis_report(analyze_routing(trace))
        expected_json = render_analysis_json(analyze_routing(trace))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace_path = write_trace(directory / "trace.jsonl", trace)
            markdown_path = directory / "report.md"
            json_path = directory / "report.json"
            _invoke([
                "analyze",
                str(trace_path),
                "--output",
                str(markdown_path),
                "--json-output",
                str(json_path),
            ])
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), expected_markdown)
            self.assertEqual(json_path.read_text(encoding="utf-8"), expected_json)

            stdout, _ = _invoke(["analyze", str(trace_path)])
            self.assertEqual(stdout, expected_markdown)

    def test_established_v1_no_download_hashes_remain_exact(self) -> None:
        demo = ROOT / "examples" / "no-download-preflight"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            markdown = directory / "report.md"
            json_path = directory / "report.json"
            _invoke([
                "analyze",
                str(demo / "trace.jsonl"),
                "--preflight-config",
                str(demo / "preflight-config.json"),
                "--output",
                str(markdown),
                "--json-output",
                str(json_path),
            ])
            self.assertEqual(
                hashlib.sha256(markdown.read_bytes()).hexdigest(),
                "219751d3a63b97177f6d3133dd2e7b69fc941eed2416e72568125f4e2936542d",
            )
            self.assertEqual(
                hashlib.sha256(json_path.read_bytes()).hexdigest(),
                "464638a63f66e2c8797f4e7c59efea226f30050032ede5138255f675d0ff4a42",
            )

    def test_v1_preflight_rejects_top_k_before_analysis_or_output(self) -> None:
        demo = ROOT / "examples" / "no-download-preflight"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            markdown = directory / "report.md"
            json_path = directory / "report.json"
            stderr = io.StringIO()
            with patch(
                "moe_cache_lab.preflight.run_preflight_analysis",
                side_effect=AssertionError("preflight must not run"),
            ) as preflight, patch.object(
                sys,
                "argv",
                [
                    "moe-cache-lab",
                    "analyze",
                    str(demo / "trace.jsonl"),
                    "--preflight-config",
                    str(demo / "preflight-config.json"),
                    "--top-k",
                    "1",
                    "--output",
                    str(markdown),
                    "--json-output",
                    str(json_path),
                ],
            ), contextlib.redirect_stderr(stderr), self.assertRaises(
                SystemExit
            ) as raised:
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn(
                "--top-k is available only for trace v2 analysis",
                stderr.getvalue(),
            )
            preflight.assert_not_called()
            self.assertFalse(markdown.exists())
            self.assertFalse(json_path.exists())

    def test_v2_analyze_is_stage_qualified_and_has_no_implicit_locality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace_path = write_trace_v2(
                directory / "private-absolute-name.jsonl", _v2_trace()
            )
            markdown_path = directory / "report.md"
            json_path = directory / "report.json"
            _invoke([
                "analyze",
                str(trace_path),
                "--workload-id",
                "evaluation-01",
                "--output",
                str(markdown_path),
                "--json-output",
                str(json_path),
            ])
            markdown = markdown_path.read_text(encoding="utf-8")
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        workload = payload["evidence"]["workloads"][0]
        phases = workload["phase_coverage"]
        self.assertEqual(payload["format"], "moe-cache-lab.routing-evidence-analysis-v2")
        self.assertEqual(payload["workload_id"], "evaluation-01")
        self.assertIsNone(payload["locality"])
        self.assertNotIn("Caller-requested locality", markdown)
        self.assertEqual(workload["event_count"], 4)
        self.assertEqual(workload["expert_request_count"], 3)
        self.assertEqual(
            [(item["routing_stage"], item["layer"]) for item in workload["routed_layer_coverage"]["observed"]],
            [("encoder", 0), ("decoder", 0)],
        )
        self.assertEqual(
            [(item["routing_stage"], item["phase"], item["assigned_event_count"], item["unassigned_event_count"]) for item in phases],
            [
                ("encoder", "source", 1, 1),
                ("decoder", "decoder_prompt", 1, 0),
                ("decoder", "decoder_generated", 1, 0),
            ],
        )
        self.assertEqual(
            [item["configured_expert_count"] for item in workload["expert_support"]],
            [4, 4],
        )
        self.assertIn("single_workload", [item["code"] for item in payload["evidence"]["warnings"]])
        serialized = json.dumps(payload)
        self.assertNotIn(str(trace_path), serialized)
        for forbidden in ("quality_score", "confidence_score", "recommendation"):
            self.assertNotIn(forbidden, serialized.lower())

    def test_v2_explicit_top_k_preserves_order_and_exact_fractions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace_path = write_trace_v2(directory / "trace.jsonl", _v2_trace())
            json_path = directory / "report.json"
            _invoke([
                "analyze",
                str(trace_path),
                "--top-k",
                "2",
                "1",
                "4",
                "--json-output",
                str(json_path),
            ])
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        locality = payload["locality"]
        self.assertEqual(locality["requested_top_k"], [2, 1, 4])
        encoder = locality["layers"][0]
        self.assertEqual(encoder["layer_identity"], {"routing_stage": "encoder", "layer": 0})
        self.assertEqual(
            [item["k"] for item in encoder["cumulative_top_k_selection_shares"]],
            [2, 1, 4],
        )
        self.assertEqual(
            encoder["cumulative_top_k_selection_shares"][0]["share"],
            {"numerator": 1, "denominator": 1},
        )

    def test_v2_invalid_top_k_and_legacy_preflight_reject_boundedly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace_path = write_trace_v2(directory / "trace.jsonl", _v2_trace())
            for ranks, message in (
                (["1", "1"], "duplicate"),
                (["5"], "exceeds"),
                (["0"], "positive"),
                (["-1"], "positive"),
            ):
                stderr = io.StringIO()
                with self.subTest(ranks=ranks), patch.object(
                    sys, "argv", ["moe-cache-lab", "analyze", str(trace_path), "--top-k", *ranks]
                ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())

            for version in (1, 2):
                config_path = directory / f"preflight-v{version}.json"
                config_path.write_text(
                    json.dumps(
                        _legacy_preflight_config_payload(version), sort_keys=True
                    ),
                    encoding="utf-8",
                )
                stderr = io.StringIO()
                with self.subTest(config_version=version), patch.object(
                    sys,
                    "argv",
                    [
                        "moe-cache-lab",
                        "analyze",
                        str(trace_path),
                        "--preflight-config",
                        str(config_path),
                    ],
                ), contextlib.redirect_stderr(stderr), self.assertRaises(
                    SystemExit
                ) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(
                    "requires stage-qualified preflight config format_version 3",
                    stderr.getvalue(),
                )

    def test_stage_qualified_config_rejects_v1_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            v1_path = write_trace(directory / "trace-v1.jsonl", _v1_trace())
            config_path = directory / "preflight-v3.json"
            config_path.write_text(
                json.dumps(_v3_preflight_config_payload(), sort_keys=True),
                encoding="utf-8",
            )
            markdown = directory / "trace-v1.md"
            json_path = directory / "trace-v1.json"
            stderr = io.StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "moe-cache-lab",
                    "analyze",
                    str(v1_path),
                    "--preflight-config",
                    str(config_path),
                    "--output",
                    str(markdown),
                    "--json-output",
                    str(json_path),
                ],
            ), contextlib.redirect_stderr(stderr), self.assertRaises(
                SystemExit
            ) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("cannot be used with trace v1", stderr.getvalue())
            self.assertFalse(markdown.exists())
            self.assertFalse(json_path.exists())

    def test_stage_qualified_config_rejects_v1_lifecycle_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "preflight-v3.json"
            config_path.write_text(
                json.dumps(_v3_preflight_config_payload(), sort_keys=True),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with patch(
                "moe_cache_lab.preflight.run_preflight_lifecycle_analysis",
                side_effect=AssertionError("lifecycle must not run"),
            ) as lifecycle, patch.object(
                sys,
                "argv",
                [
                    "moe-cache-lab",
                    "analyze-lifecycle",
                    str(directory / "unused-manifest.json"),
                    "--preflight-config",
                    str(config_path),
                    "--output",
                    str(directory / "report.md"),
                    "--json-output",
                    str(directory / "report.json"),
                ],
            ), contextlib.redirect_stderr(stderr), self.assertRaises(
                SystemExit
            ) as raised:
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("cannot be used with v1 lifecycle", stderr.getvalue())
            lifecycle.assert_not_called()
            self.assertFalse((directory / "report.md").exists())
            self.assertFalse((directory / "report.json").exists())

    def test_bundle_create_maps_repeated_inputs_and_verify_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("external trace must never be fetched"),
        ):
            directory = Path(temporary)
            v1 = write_trace(directory / "source-v1.jsonl", _v1_trace())
            v2 = write_trace_v2(directory / "source-v2.jsonl", _v2_trace())
            config = directory / "config.json"
            report_json = directory / "report.json"
            report_markdown = directory / "report.md"
            config.write_text('{"z":1,"a":true}\n', encoding="utf-8")
            report_json.write_text('{"result":"ok"}\n', encoding="utf-8")
            report_markdown.write_text("# Report\n", encoding="utf-8")
            bundle = directory / "bundle"
            stdout, _ = _invoke([
                "bundle-create",
                "--experiment-id",
                "experiment-01",
                "--config",
                str(config),
                "--report-json",
                str(report_json),
                "--report-markdown",
                str(report_markdown),
                "--embed-trace",
                "granite",
                str(v1),
                "--embed-trace",
                "switch",
                str(v2),
                "--external-trace",
                "private",
                "https://example.invalid/private.jsonl",
                "a" * 64,
                "--torch-version",
                "2.12.0+cpu",
                "--transformers-version",
                "5.12.0",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--output-dir",
                str(bundle),
            ])
            created = json.loads(stdout)
            checked = verify_experiment_bundle(bundle)
            verify_stdout, _ = _invoke(["bundle-verify", str(bundle)])
            manifest = json.loads((bundle / "bundle-manifest.json").read_text())
            all_bytes = b"".join(
                path.read_bytes() for path in bundle.rglob("*") if path.is_file()
            )

        self.assertEqual(json.loads(verify_stdout), created)
        self.assertEqual(created["manifest_sha256"], checked.manifest_sha256)
        self.assertEqual(
            [(item["trace_id"], item["mode"]) for item in manifest["traces"]],
            [("granite", "embedded"), ("switch", "embedded"), ("private", "external")],
        )
        self.assertNotIn(str(v1).encode(), all_bytes)
        self.assertNotIn(str(v2).encode(), all_bytes)

    def test_bundle_tamper_fails_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = directory / "config.json"
            report_json = directory / "report.json"
            report_markdown = directory / "report.md"
            config.write_text("{}\n", encoding="utf-8")
            report_json.write_text("{}\n", encoding="utf-8")
            report_markdown.write_text("report\n", encoding="utf-8")
            bundle = directory / "bundle"
            _invoke([
                "bundle-create",
                "--experiment-id",
                "tamper-test",
                "--config",
                str(config),
                "--report-json",
                str(report_json),
                "--report-markdown",
                str(report_markdown),
                "--output-dir",
                str(bundle),
            ])
            manifest_before = (bundle / "bundle-manifest.json").read_bytes()
            (bundle / "report.md").write_bytes(b"tampered\n")
            stderr = io.StringIO()
            with patch.object(
                sys, "argv", ["moe-cache-lab", "bundle-verify", str(bundle)]
            ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("mismatch", stderr.getvalue())
            self.assertEqual((bundle / "bundle-manifest.json").read_bytes(), manifest_before)

    def test_top_level_help_exposes_canonical_flow_and_hides_children(self) -> None:
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["moe-cache-lab", "--help"]), (
            contextlib.redirect_stdout(stdout)
        ), self.assertRaises(SystemExit) as raised:
            main()
        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        for required in (
            "Canonical path:",
            "analyze",
            "bundle-create",
            "bundle-verify",
            "historical reproduction",
            "granite extra",
        ):
            self.assertIn(required, help_text)
        self.assertNotIn("_collect-stage1-child", help_text)
        self.assertNotIn("_collect-v04-child", help_text)

        for command in ("benchmark", "collect-corpus", "validate-stage1", "validate-v04"):
            with self.subTest(command=command), patch.object(
                sys, "argv", ["moe-cache-lab", command, "--help"]
            ), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as child:
                main()
            self.assertEqual(child.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
