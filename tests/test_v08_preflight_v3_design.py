from pathlib import Path
import unittest

from moe_cache_lab.preflight_config import (
    PREFLIGHT_CONFIG_FORMAT,
    PREFLIGHT_CONFIG_VERSION,
    parse_preflight_config_data,
)


ROOT = Path(__file__).resolve().parents[1]


class V08PreflightV3DesignTests(unittest.TestCase):
    def test_design_freezes_smallest_stage_qualified_contract(self) -> None:
        text = (ROOT / "V08_PREFLIGHT_V3_DESIGN.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        required = (
            "format version 3",
            "(routing_stage, layer_id, expert_id)",
            "top-level fields are exactly the config-v2 fields",
            "only the expert-size namespace changes structurally",
            "stage order `encoder`, then `decoder`",
            "config v3 does not create or require routing trace v3",
            "equal version numbers",
            "does not satisfy a trace request for `(encoder, 3, 5)`",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, normalized)

    def test_design_preserves_unassigned_zero_request_semantics(self) -> None:
        text = (ROOT / "V08_PREFLIGHT_V3_DESIGN.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for phrase in (
            "the v3 `expert_sizes` array may be empty",
            "all explicitly unassigned",
            "contributes zero expert requests",
            "loads zero bytes",
            "requires no expert-size record",
            '"expert_sizes": []',
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, normalized)

    def test_design_explicitly_defers_stage_qualified_lifecycle(self) -> None:
        text = (ROOT / "V08_PREFLIGHT_V3_DESIGN.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for phrase in (
            "Lifecycle: DEFER / NO-GO for the initial v2 path",
            "`ByteCacheWorkload.events` accepts only v1 `RoutingEvent` objects",
            "lifecycle result residency keys are typed/stored as two-part `ExpertKey`",
            "uses the v1 corpus/manifest loader",
            "Config v3 alone is insufficient",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, normalized)

    def test_design_freezes_trace_config_matrix_and_output_boundary(self) -> None:
        text = (ROOT / "V08_PREFLIGHT_V3_DESIGN.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for phrase in (
            "v1 + v3 would require inference",
            "v2 + v1/v2 would require flattening or inferred stage",
            "Current pre-flight output is v1-specific",
            "structured pre-flight-analysis output requires its own format-version bump",
            "Existing v1 report bytes and structured output remain frozen",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, normalized)

    def test_b2_parser_preserves_the_frozen_v3_design_boundary(self) -> None:
        self.assertEqual(PREFLIGHT_CONFIG_VERSION, 2)
        proposed = {
            "format": PREFLIGHT_CONFIG_FORMAT,
            "format_version": 3,
            "expert_sizes": [],
            "capacities_bytes": [1024],
            "policies": ["lru"],
            "hardware_profiles": [
                {
                    "name": "synthetic",
                    "h2d_payload_bandwidth_bytes_per_second": 1,
                    "setup_latency_ns_per_transfer_operation": 0,
                }
            ],
            "transfer_operation_plans": [
                {"name": "one", "operations_per_logical_load": 1}
            ],
        }
        config = parse_preflight_config_data(proposed)
        self.assertEqual(config.format_version, 3)
        self.assertEqual(config.expert_size_map(), {})

    def test_design_does_not_claim_runtime_or_performance_evidence(self) -> None:
        text = (ROOT / "V08_PREFLIGHT_V3_DESIGN.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for phrase in (
            "Cache hits, misses, loads, evictions, and residency remain **SIMULATED**",
            "Transfer-service quantities remain **ESTIMATED**",
            "physical expert residency",
            "end-to-end latency, throughput, tokens/sec, or speedup",
            "optimal policy or capacity",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, normalized)


if __name__ == "__main__":
    unittest.main()
