import json
import unittest
from unittest.mock import patch

from moe_cache_lab.analysis import analyze_routing
from moe_cache_lab.hardware_cost import HardwareTransferProfile
from moe_cache_lab.preflight import run_preflight_analysis
from moe_cache_lab.preflight_config import (
    ExpertSizeRecord,
    PreflightConfig,
    parse_preflight_config_data,
)
from moe_cache_lab.preflight_output import render_preflight_json, render_preflight_report
from moe_cache_lab.trace import RoutingEvent, RoutingTrace


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


def _config() -> PreflightConfig:
    return PreflightConfig(
        expert_sizes=(
            ExpertSizeRecord(0, 2, 2),
            ExpertSizeRecord(0, 0, 4),
            ExpertSizeRecord(0, 1, 2),
        ),
        capacities_bytes=(8, 6),
        policies=("lfu", "lru"),
        hardware_profiles=(
            HardwareTransferProfile("z", 16, 0),
            HardwareTransferProfile("a", 8, 1_000_000_000),
        ),
    )


def _config_payload_reordered() -> dict:
    return {
        "format": "moe-cache-lab.preflight-config",
        "format_version": 1,
        "expert_sizes": [
            {"layer_id": 0, "expert_id": 1, "size_bytes": 2},
            {"layer_id": 0, "expert_id": 2, "size_bytes": 2},
            {"layer_id": 0, "expert_id": 0, "size_bytes": 4},
        ],
        "capacities_bytes": [6, 8, 6],
        "policies": ["lru", "lfu", "lru"],
        "hardware_profiles": [
            {
                "name": "a",
                "h2d_payload_bandwidth_bytes_per_second": 8,
                "setup_latency_ns_per_loaded_expert": 1_000_000_000,
            },
            {
                "name": "z",
                "h2d_payload_bandwidth_bytes_per_second": 16,
                "setup_latency_ns_per_loaded_expert": 0,
            },
        ],
    }


class PreflightAnalysisTests(unittest.TestCase):
    def test_combined_result_reuses_existing_analysis_and_sweep_cores(self) -> None:
        routing = analyze_routing(_trace())
        sentinel_sweep = object()
        with patch("moe_cache_lab.preflight.analyze_routing", return_value=routing) as routing_core, patch(
            "moe_cache_lab.preflight.run_transfer_sensitivity_sweep",
            return_value=sentinel_sweep,
        ) as sweep_core:
            result = run_preflight_analysis(_trace(), _config())

        routing_core.assert_called_once()
        sweep_core.assert_called_once_with(
            _trace().events,
            {(0, 0): 4, (0, 1): 2, (0, 2): 2},
            (6, 8),
            ("lru", "lfu"),
            _config().hardware_profiles,
        )
        self.assertIs(result.routing, routing)
        self.assertIs(result.transfer_sensitivity, sentinel_sweep)

    def test_report_separates_evidence_classes_and_assumptions(self) -> None:
        report = render_preflight_report(run_preflight_analysis(_trace(), _config()))
        for heading in (
            "## Scope / claim boundary",
            "## Routing observations",
            "## Supplied workload-size / cache-budget context",
            "## SIMULATED byte-cache sensitivity",
            "## ESTIMATED serialized transfer-service sensitivity",
            "## Assumptions / limitations",
        ):
            self.assertIn(heading, report)
        self.assertIn("MEASURED only if trace provenance establishes that", report)
        self.assertIn("caller-supplied assumptions", report)
        self.assertIn("serialized/no-overlap", report)
        self.assertIn("not charged as D2H writeback", report)
        self.assertIn("not a runtime-benefit recommendation", report)
        self.assertNotIn("GOOD/BAD", report)
        self.assertNotIn("cacheability score", report.lower())

    def test_json_preserves_exact_routing_and_transfer_fractions(self) -> None:
        payload = json.loads(render_preflight_json(run_preflight_analysis(_trace(), _config())))
        self.assertEqual(payload["format"], "moe-cache-lab.preflight-analysis")
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(
            payload["routing_analysis"]["consecutive_overlap"]["mean_jaccard"],
            {"numerator": 1, "denominator": 3},
        )
        first_estimate = payload["estimated_transfer_service_sensitivity"][0]
        self.assertEqual(
            first_estimate["estimated_payload_service_seconds"],
            {"numerator": 1, "denominator": 1},
        )
        self.assertEqual(
            first_estimate["estimated_setup_service_seconds"],
            {"numerator": 3, "denominator": 1},
        )
        self.assertEqual(
            first_estimate["estimated_serialized_transfer_service_seconds"],
            {"numerator": 4, "denominator": 1},
        )
        self.assertNotIsInstance(
            first_estimate["estimated_serialized_transfer_service_seconds"], float
        )

    def test_canonical_sensitivity_rows_preserve_issue12_order(self) -> None:
        payload = json.loads(render_preflight_json(run_preflight_analysis(_trace(), _config())))
        rows = payload["estimated_transfer_service_sensitivity"]
        self.assertEqual(
            [
                (row["cache_capacity_bytes"], row["policy"], row["hardware_profile_name"])
                for row in rows
            ],
            [
                (6, "lru", "a"),
                (6, "lru", "z"),
                (6, "lfu", "a"),
                (6, "lfu", "z"),
                (8, "lru", "a"),
                (8, "lru", "z"),
                (8, "lfu", "a"),
                (8, "lfu", "z"),
            ],
        )
        simulated = payload["simulated_byte_cache_sensitivity"]
        self.assertEqual(
            [(row["capacity_bytes"], row["policy"]) for row in simulated],
            [(6, "lru"), (6, "lfu"), (8, "lru"), (8, "lfu")],
        )

    def test_semantically_equivalent_config_produces_identical_outputs(self) -> None:
        left = run_preflight_analysis(_trace(), _config())
        right = run_preflight_analysis(
            _trace(), parse_preflight_config_data(_config_payload_reordered())
        )
        self.assertEqual(render_preflight_report(left), render_preflight_report(right))
        self.assertEqual(render_preflight_json(left), render_preflight_json(right))

    def test_infeasible_capacity_error_remains_explicit(self) -> None:
        invalid = PreflightConfig(
            expert_sizes=_config().expert_sizes,
            capacities_bytes=(5,),
            policies=("lru",),
            hardware_profiles=_config().hardware_profiles[:1],
        )
        with self.assertRaisesRegex(
            ValueError, "required working set 6 bytes exceeds byte-cache capacity 5 bytes"
        ):
            run_preflight_analysis(_trace(), invalid)


if __name__ == "__main__":
    unittest.main()
