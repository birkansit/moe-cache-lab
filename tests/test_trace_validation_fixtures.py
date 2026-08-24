from pathlib import Path
import unittest

from scripts.audit_distributions import (
    REQUIRED_SDIST_PATHS,
    TRACE_VALIDATION_FIXTURES,
)
from moe_cache_lab.trace import RoutingTrace
from moe_cache_lab.trace_v2 import RoutingTraceV2, read_versioned_trace


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples" / "trace-validation-fixtures"

VALID = {
    "v1-minimal.jsonl": RoutingTrace,
    "v2-decoder-only.jsonl": RoutingTraceV2,
    "v2-encoder-decoder.jsonl": RoutingTraceV2,
    "v2-unassigned.jsonl": RoutingTraceV2,
}
INVALID_MESSAGES = {
    "chronology-regression.jsonl": "cannot follow",
    "duplicate-event-identity.jsonl": "duplicate routing-stage/token-position/layer",
    "stage-regression.jsonl": "encoder routing events cannot follow decoder",
    "fabricated-unassigned-selection.jsonl": "unassigned events must have empty selection arrays",
    "unsupported-version.jsonl": "unsupported routing trace format version",
    "unknown-field.jsonl": "contains unknown field",
}


class TraceValidationFixtureTests(unittest.TestCase):
    def test_valid_fixtures_pass_authoritative_versioned_reader(self) -> None:
        for name, expected_type in VALID.items():
            with self.subTest(name=name):
                trace = read_versioned_trace(FIXTURES / "valid" / name)
                self.assertIsInstance(trace, expected_type)

    def test_encoder_decoder_fixture_preserves_same_numeric_identity_by_stage(self) -> None:
        trace = read_versioned_trace(FIXTURES / "valid" / "v2-encoder-decoder.jsonl")

        self.assertEqual(
            trace.expert_requests,
            (("encoder", 3, 5), ("decoder", 3, 5)),
        )

    def test_unassigned_fixture_has_zero_requests_for_unassigned_event(self) -> None:
        trace = read_versioned_trace(FIXTURES / "valid" / "v2-unassigned.jsonl")

        self.assertEqual(trace.events[-1].assignment_state, "unassigned")
        self.assertEqual(trace.events[-1].selected_experts, ())
        self.assertEqual(trace.events[-1].selected_probabilities, ())
        self.assertEqual(trace.events[-1].expert_requests, ())

    def test_invalid_fixtures_fail_the_authoritative_versioned_reader(self) -> None:
        for name, message in INVALID_MESSAGES.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                read_versioned_trace(FIXTURES / "invalid" / name)

    def test_corpus_inventory_is_exact(self) -> None:
        self.assertEqual(
            {path.name for path in (FIXTURES / "valid").glob("*.jsonl")},
            set(VALID),
        )
        self.assertEqual(
            {path.name for path in (FIXTURES / "invalid").glob("*.jsonl")},
            set(INVALID_MESSAGES),
        )

    def test_distribution_contract_requires_every_fixture(self) -> None:
        expected = {"README.md", *(
            f"valid/{name}" for name in VALID
        ), *(
            f"invalid/{name}" for name in INVALID_MESSAGES
        )}
        self.assertEqual(set(TRACE_VALIDATION_FIXTURES), expected)
        for name in expected:
            self.assertIn(
                f"examples/trace-validation-fixtures/{name}",
                REQUIRED_SDIST_PATHS,
            )


if __name__ == "__main__":
    unittest.main()
