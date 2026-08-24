import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.trace import RoutingTrace
from moe_cache_lab.trace_validation import (
    TRACE_VALIDATION_ERROR_CODES,
    TRACE_VALIDATION_FORMAT,
    TRACE_VALIDATION_FORMAT_VERSION,
    render_trace_validation_human,
    render_trace_validation_json,
    validate_trace_file,
)
from moe_cache_lab.trace_v2 import RoutingTraceV2


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


class TraceValidationTests(unittest.TestCase):
    def test_valid_v1_result_uses_canonical_trace_and_bounded_summary(self) -> None:
        result = validate_trace_file(FIXTURES / "valid" / "v1-minimal.jsonl")

        self.assertTrue(result.valid)
        self.assertIsInstance(result.trace, RoutingTrace)
        self.assertEqual(result.trace_format_version, 1)
        self.assertEqual(
            dict(result.summary),
            {
                "model_id": "synthetic/v1-minimal",
                "event_count": 1,
                "num_experts": 8,
                "experts_per_token": 1,
                "expert_request_count": 1,
            },
        )

    def test_valid_v2_result_preserves_stage_and_assignment_facts(self) -> None:
        result = validate_trace_file(
            FIXTURES / "valid" / "v2-encoder-decoder.jsonl"
        )

        self.assertTrue(result.valid)
        self.assertIsInstance(result.trace, RoutingTraceV2)
        self.assertEqual(result.trace_format_version, 2)
        self.assertEqual(
            dict(result.summary),
            {
                "model_id": "synthetic/v2-encoder-decoder",
                "event_count": 2,
                "routing_stages": ("encoder", "decoder"),
                "assigned_event_count": 2,
                "unassigned_event_count": 0,
                "expert_request_count": 2,
            },
        )

    def test_official_invalid_fixtures_map_to_stable_codes(self) -> None:
        for name, expected_code in EXPECTED_INVALID.items():
            with self.subTest(name=name):
                result = validate_trace_file(FIXTURES / "invalid" / name)
                self.assertFalse(result.valid)
                self.assertEqual(result.error_code, expected_code)
                self.assertIsNone(result.trace)
                self.assertEqual(result.summary, ())

        unknown = validate_trace_file(FIXTURES / "invalid" / "unknown-field.jsonl")
        chronology = validate_trace_file(
            FIXTURES / "invalid" / "chronology-regression.jsonl"
        )
        self.assertEqual(unknown.line_number, 2)
        self.assertIsNone(chronology.line_number)

    def test_invalid_json_empty_and_missing_file_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            malformed = root / "malformed.jsonl"
            malformed.write_text("{not-json}\n", encoding="utf-8")
            empty = root / "empty.jsonl"
            empty.write_text("\n", encoding="utf-8")

            malformed_result = validate_trace_file(malformed)
            empty_result = validate_trace_file(empty)
            missing_result = validate_trace_file(root / "missing.jsonl")

        self.assertEqual(malformed_result.error_code, "invalid_json")
        self.assertEqual(malformed_result.line_number, 1)
        self.assertIsNone(malformed_result.trace_format_version)
        self.assertEqual(empty_result.error_code, "empty_or_missing_trace")
        self.assertEqual(missing_result.error_code, "io_error")
        self.assertEqual(missing_result.message, "could not read trace file")

    def test_declared_version_is_reported_without_version_fallback(self) -> None:
        unsupported = FIXTURES / "invalid" / "unsupported-version.jsonl"
        with (
            patch("moe_cache_lab.trace_v2.validate_trace_records") as validate_v1,
            patch("moe_cache_lab.trace_v2.validate_trace_v2_records") as validate_v2,
        ):
            result = validate_trace_file(unsupported)

        self.assertEqual(result.error_code, "unsupported_format_version")
        self.assertEqual(result.trace_format_version, 3)
        validate_v1.assert_not_called()
        validate_v2.assert_not_called()

    def test_human_and_json_renderers_share_one_result(self) -> None:
        result = validate_trace_file(FIXTURES / "valid" / "v2-unassigned.jsonl")

        human = render_trace_validation_human(result)
        machine = render_trace_validation_json(result)
        payload = json.loads(machine)

        self.assertIn("Trace validation: VALID\n", human)
        self.assertIn("Assigned events: 1\n", human)
        self.assertIn("Unassigned events: 1\n", human)
        self.assertEqual(payload["valid"], result.valid)
        self.assertEqual(payload["trace_format_version"], result.trace_format_version)
        self.assertEqual(payload["summary"]["event_count"], dict(result.summary)["event_count"])

    def test_machine_format_and_taxonomy_are_frozen(self) -> None:
        self.assertEqual(TRACE_VALIDATION_FORMAT, "moe-cache-lab.trace-validation")
        self.assertEqual(TRACE_VALIDATION_FORMAT_VERSION, 1)
        self.assertEqual(
            TRACE_VALIDATION_ERROR_CODES,
            (
                "unsupported_format_version",
                "invalid_json",
                "invalid_record_shape",
                "unknown_field",
                "duplicate_event_identity",
                "chronology_regression",
                "invalid_stage_or_phase",
                "invalid_assignment",
                "invalid_expert_identity",
                "invalid_trace_metadata",
                "empty_or_missing_trace",
                "io_error",
            ),
        )


if __name__ == "__main__":
    unittest.main()
