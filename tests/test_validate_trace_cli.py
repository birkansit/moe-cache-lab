from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab import analysis, preflight
from moe_cache_lab.cli import main


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples" / "trace-validation-fixtures"
EXPECTED_INVALID = {
    "chronology-regression.jsonl": "chronology_regression",
    "duplicate-event-identity.jsonl": "duplicate_event_identity",
    "stage-regression.jsonl": "chronology_regression",
    "fabricated-unassigned-selection.jsonl": "invalid_assignment",
    "unsupported-version.jsonl": "unsupported_format_version",
    "unknown-field.jsonl": "unknown_field",
}


def _invoke(*arguments: str) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with patch.object(sys, "argv", ["moe-cache-lab", *arguments]), redirect_stdout(
        stdout
    ), redirect_stderr(stderr):
        try:
            main()
        except SystemExit as error:
            code = int(error.code)
        else:
            code = 0
    return code, stdout.getvalue(), stderr.getvalue()


class ValidateTraceHumanCliTests(unittest.TestCase):
    def test_valid_v1_human_summary_is_exact(self) -> None:
        code, stdout, stderr = _invoke(
            "validate-trace", str(FIXTURES / "valid" / "v1-minimal.jsonl")
        )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(
            stdout,
            "Trace validation: VALID\n"
            "Trace format version: 1\n"
            "Model ID: synthetic/v1-minimal\n"
            "Events: 1\n"
            "Configured experts: 8\n"
            "Experts per token: 1\n"
            "Expert requests: 1\n",
        )

    def test_all_valid_v2_human_summaries_are_bounded(self) -> None:
        expectations = {
            "v2-decoder-only.jsonl": ("Routing stages: decoder", "Events: 1"),
            "v2-encoder-decoder.jsonl": (
                "Routing stages: encoder, decoder",
                "Expert requests: 2",
            ),
            "v2-unassigned.jsonl": ("Unassigned events: 1", "Expert requests: 1"),
        }
        for name, fragments in expectations.items():
            with self.subTest(name=name):
                code, stdout, stderr = _invoke(
                    "validate-trace", str(FIXTURES / "valid" / name)
                )
                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertTrue(all(fragment in stdout for fragment in fragments))
                self.assertNotIn("cache", stdout.lower())
                self.assertNotIn("latency", stdout.lower())

    def test_invalid_human_mode_is_stderr_only_exit_two_without_traceback(self) -> None:
        for name, expected_code in EXPECTED_INVALID.items():
            with self.subTest(name=name):
                code, stdout, stderr = _invoke(
                    "validate-trace", str(FIXTURES / "invalid" / name)
                )
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn("Trace validation: INVALID\n", stderr)
                self.assertIn(f"Error code: {expected_code}\n", stderr)
                self.assertNotIn("Traceback", stderr)

    def test_bad_json_and_missing_file_fail_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            malformed = Path(temporary) / "malformed.jsonl"
            malformed.write_text("{not-json}\n", encoding="utf-8")
            cases = ((malformed, "invalid_json"), (Path(temporary) / "missing", "io_error"))
            for path, expected_code in cases:
                with self.subTest(path=path):
                    code, stdout, stderr = _invoke("validate-trace", str(path))
                    self.assertEqual(code, 2)
                    self.assertEqual(stdout, "")
                    self.assertIn(f"Error code: {expected_code}\n", stderr)
                    self.assertNotIn("Traceback", stderr)

    def test_validation_does_not_invoke_analysis_preflight_or_model_paths(self) -> None:
        with (
            patch.object(analysis, "analyze_routing") as analyze_routing,
            patch.object(preflight, "run_preflight_analysis") as run_preflight,
            patch("moe_cache_lab.cli.collect_trace") as collect,
            patch("moe_cache_lab.cli.inspect_model") as inspect,
        ):
            code, _, _ = _invoke(
                "validate-trace", str(FIXTURES / "valid" / "v1-minimal.jsonl")
            )

        self.assertEqual(code, 0)
        for mocked in (analyze_routing, run_preflight, collect, inspect):
            mocked.assert_not_called()


class ValidateTraceJsonCliTests(unittest.TestCase):
    def test_valid_v1_json_is_deterministic_with_independent_output_version(self) -> None:
        path = FIXTURES / "valid" / "v1-minimal.jsonl"
        first = _invoke("validate-trace", str(path), "--json")
        second = _invoke("validate-trace", str(path), "--json")

        self.assertEqual(first, second)
        code, stdout, stderr = first
        self.assertEqual((code, stderr), (0, ""))
        payload = json.loads(stdout)
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(payload["trace_format_version"], 1)
        self.assertEqual(payload["summary"]["expert_request_count"], 1)

    def test_valid_json_is_one_deterministic_object(self) -> None:
        path = FIXTURES / "valid" / "v2-encoder-decoder.jsonl"
        first = _invoke("validate-trace", str(path), "--json")
        second = _invoke("validate-trace", str(path), "--json")

        self.assertEqual(first, second)
        code, stdout, stderr = first
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout.count("\n"), 1)
        payload = json.loads(stdout)
        self.assertEqual(payload["format"], "moe-cache-lab.trace-validation")
        self.assertEqual(payload["format_version"], 1)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["trace_format_version"], 2)
        self.assertEqual(payload["summary"]["routing_stages"], ["encoder", "decoder"])

    def test_every_invalid_fixture_emits_one_machine_object_and_exit_two(self) -> None:
        for name, expected_code in EXPECTED_INVALID.items():
            with self.subTest(name=name):
                first = _invoke(
                    "validate-trace", str(FIXTURES / "invalid" / name), "--json"
                )
                second = _invoke(
                    "validate-trace", str(FIXTURES / "invalid" / name), "--json"
                )
                self.assertEqual(first, second)
                code, stdout, stderr = first
                self.assertEqual(code, 2)
                self.assertEqual(stderr, "")
                self.assertEqual(stdout.count("\n"), 1)
                payload = json.loads(stdout)
                self.assertFalse(payload["valid"])
                self.assertEqual(payload["error_code"], expected_code)
                self.assertEqual(payload["format_version"], 1)

    def test_trace_version_is_omitted_when_not_established(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            malformed = Path(temporary) / "malformed.jsonl"
            malformed.write_text("{not-json}\n", encoding="utf-8")
            code, stdout, stderr = _invoke(
                "validate-trace", str(malformed), "--json"
            )

        self.assertEqual(code, 2)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertNotIn("trace_format_version", payload)
        self.assertEqual(payload["error_code"], "invalid_json")

    def test_missing_file_json_is_bounded_and_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            code, stdout, stderr = _invoke(
                "validate-trace", str(Path(temporary) / "missing.jsonl"), "--json"
            )

        self.assertEqual(code, 2)
        self.assertEqual(stderr, "")
        self.assertEqual(
            json.loads(stdout),
            {
                "error_code": "io_error",
                "format": "moe-cache-lab.trace-validation",
                "format_version": 1,
                "message": "could not read trace file",
                "valid": False,
            },
        )

    def test_human_and_json_modes_agree_on_validity_and_facts(self) -> None:
        path = FIXTURES / "valid" / "v2-unassigned.jsonl"
        human_code, human_stdout, human_stderr = _invoke("validate-trace", str(path))
        json_code, json_stdout, json_stderr = _invoke(
            "validate-trace", str(path), "--json"
        )
        payload = json.loads(json_stdout)

        self.assertEqual((human_code, json_code), (0, 0))
        self.assertEqual((human_stderr, json_stderr), ("", ""))
        self.assertEqual(payload["valid"], "Trace validation: VALID" in human_stdout)
        self.assertIn(
            f"Events: {payload['summary']['event_count']}\n",
            human_stdout,
        )
        self.assertIn(
            f"Expert requests: {payload['summary']['expert_request_count']}\n",
            human_stdout,
        )


if __name__ == "__main__":
    unittest.main()
