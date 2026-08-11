import json
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.cli import main
from moe_cache_lab.byte_cache import simulate_byte_cache_workloads
from moe_cache_lab.hardware_cost import HardwareTransferProfile, TransferOperationPlan
from moe_cache_lab.preflight import run_preflight_lifecycle_analysis
from moe_cache_lab.preflight_config import ExpertSizeRecord, PreflightConfig
from moe_cache_lab.preflight_output import (
    render_preflight_lifecycle_json,
    render_preflight_lifecycle_report,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.workflow import (
    CorpusDefinition,
    CorpusPrompt,
    LoadedPromptTrace,
    SuiteInputs,
)


def _trace(expert_id: int, created_at: str) -> RoutingTrace:
    return RoutingTrace(
        model_id="synthetic/lifecycle",
        num_experts=2,
        experts_per_token=1,
        events=(RoutingEvent("prompt", 0, 0, (expert_id,)),),
        capture_method="synthetic-lifecycle-test",
        created_at=created_at,
        model_revision="fixture-v1",
    )


def _inputs(*, descending: bool = False) -> SuiteInputs:
    first = CorpusPrompt("first", "fixture", "evaluation", "first", 4)
    second = CorpusPrompt("second", "fixture", "evaluation", "second", 5)
    records = (
        LoadedPromptTrace(first, "traces/first.jsonl", "1" * 64, _trace(0, "t1")),
        LoadedPromptTrace(second, "traces/second.jsonl", "2" * 64, _trace(0, "t2")),
    )
    if descending:
        records = tuple(reversed(records))
    return SuiteInputs(
        manifest_path=Path("suite-manifest.json"),
        manifest_sha256="a" * 64,
        manifest={},
        corpus=CorpusDefinition(
            Path("corpus.json"), "1.0", "b" * 64, (first, second)
        ),
        calibration=(),
        evaluation=records,
    )


def _config(*, version: int = 1) -> PreflightConfig:
    plans = (
        (TransferOperationPlan("one", 1), TransferOperationPlan("two", 2))
        if version == 2
        else (TransferOperationPlan("one-operation-per-logical-load", 1),)
    )
    return PreflightConfig(
        expert_sizes=(ExpertSizeRecord(0, 0, 4),),
        capacities_bytes=(4,),
        policies=("lru",),
        hardware_profiles=(HardwareTransferProfile("assumption", 100, 10),),
        transfer_operation_plans=plans,
        format_version=version,
    )


class PreflightLifecycleAnalysisTests(unittest.TestCase):
    def test_summaries_consume_each_existing_simulation_once_without_replay(self) -> None:
        with patch(
            "moe_cache_lab.preflight.load_suite_inputs", return_value=_inputs()
        ), patch(
            "moe_cache_lab.preflight.simulate_byte_cache_workloads",
            wraps=simulate_byte_cache_workloads,
        ) as simulate:
            result = run_preflight_lifecycle_analysis("ignored.json", _config())

        self.assertEqual(simulate.call_count, 2)
        self.assertEqual(
            [(row.hits, row.misses) for row in result.simulations],
            [(0, 2), (1, 1)],
        )
        self.assertEqual(len(result.sensitivity_summary.workload_sensitivity), 2)
        self.assertEqual(len(result.transfer_estimates.rows), 2)

    def test_manifest_order_drives_cold_and_persistent_scenarios(self) -> None:
        with patch("moe_cache_lab.preflight.load_suite_inputs", return_value=_inputs()):
            result = run_preflight_lifecycle_analysis("ignored.json", _config())

        self.assertEqual(
            [(item.order, item.workload_id) for item in result.workloads],
            [(4, "first"), (5, "second")],
        )
        self.assertEqual(
            [simulation.lifecycle_mode for simulation in result.simulations],
            ["cold_per_workload", "persistent_sequence"],
        )
        cold, persistent = result.simulations
        self.assertEqual((cold.hits, cold.misses), (0, 2))
        self.assertEqual((persistent.hits, persistent.misses), (1, 1))
        self.assertEqual(
            persistent.workloads[1].starting_resident_keys, ((0, 0),)
        )

    def test_json_and_markdown_are_deterministic_auditable_and_bounded(self) -> None:
        with patch("moe_cache_lab.preflight.load_suite_inputs", return_value=_inputs()):
            result = run_preflight_lifecycle_analysis("ignored.json", _config())

        left_json = render_preflight_lifecycle_json(result)
        right_json = render_preflight_lifecycle_json(result)
        self.assertEqual(left_json, right_json)
        self.assertEqual(
            render_preflight_lifecycle_report(result),
            render_preflight_lifecycle_report(result),
        )

        payload = json.loads(left_json)
        self.assertEqual(
            payload["format"],
            "moe-cache-lab.preflight-cache-lifecycle-analysis",
        )
        self.assertEqual(payload["format_version"], 3)
        self.assertEqual(
            payload["evidence_classes"],
            {
                "cache_lifecycle_scenarios": "SIMULATED",
                "descriptive_sensitivity_summaries": "SIMULATED",
                "serialized_h2d_transfer_service": "ESTIMATED",
            },
        )
        self.assertEqual(
            [(item["order"], item["workload_id"]) for item in payload["workload_order"]],
            [(4, "first"), (5, "second")],
        )
        for scenario in payload["simulated_cache_lifecycle_scenarios"]:
            for field in (
                "event_count",
                "expert_request_count",
                "hits",
                "misses",
                "simulated_demand_load_bytes",
                "eviction_count",
                "simulated_evicted_bytes",
            ):
                self.assertEqual(
                    scenario[field],
                    sum(row[field] for row in scenario["workloads"]),
                )

        summaries = payload["descriptive_sensitivity_summaries"]
        self.assertEqual(summaries["capacities_bytes"], [4])
        self.assertEqual(summaries["policies"], ["lru"])
        self.assertEqual(
            summaries["lifecycle_modes"],
            ["cold_per_workload", "persistent_sequence"],
        )
        self.assertEqual(summaries["adjacent_capacity_comparisons"], [])
        self.assertEqual(summaries["same_capacity_policy_comparisons"], [])
        self.assertEqual(len(summaries["workload_sensitivity"]), 2)
        persistent_spread = summaries["workload_sensitivity"][1]
        self.assertEqual(persistent_spread["lifecycle_mode"], "persistent_sequence")
        self.assertEqual(
            persistent_spread["metrics"]["hit_rate"]["range"],
            {"numerator": 1, "denominator": 1},
        )
        self.assertEqual(
            persistent_spread["metrics"]["hit_rate"]["minimum_workloads"],
            [{"order": 4, "workload_id": "first"}],
        )
        self.assertEqual(
            persistent_spread["metrics"]["hit_rate"]["maximum_workloads"],
            [{"order": 5, "workload_id": "second"}],
        )

        estimate_rows = payload["estimated_serialized_h2d_transfer_service"]["rows"]
        self.assertEqual(len(estimate_rows), 2)
        cold_estimate, persistent_estimate = estimate_rows
        self.assertEqual(cold_estimate["lifecycle_mode"], "cold_per_workload")
        self.assertEqual(cold_estimate["simulated_logical_demand_load_count"], 2)
        self.assertEqual(cold_estimate["simulated_demand_load_bytes"], 8)
        self.assertEqual(
            cold_estimate["estimated_payload_service_seconds"],
            {"numerator": 2, "denominator": 25},
        )
        self.assertEqual(
            cold_estimate["estimated_setup_service_seconds"],
            {"numerator": 1, "denominator": 50_000_000},
        )
        self.assertEqual(
            persistent_estimate["simulated_logical_demand_load_count"], 1
        )
        for estimate in estimate_rows:
            for field in (
                "simulated_logical_demand_load_count",
                "simulated_demand_load_bytes",
                "modeled_transfer_operation_count",
            ):
                self.assertEqual(
                    estimate[field],
                    sum(workload[field] for workload in estimate["workloads"]),
                )
            for field in (
                "estimated_payload_service_seconds",
                "estimated_setup_service_seconds",
                "estimated_serialized_transfer_service_seconds",
            ):
                aggregate = estimate[field]
                self.assertEqual(
                    Fraction(aggregate["numerator"], aggregate["denominator"]),
                    sum(
                        Fraction(
                            workload[field]["numerator"],
                            workload[field]["denominator"],
                        )
                        for workload in estimate["workloads"]
                    ),
                )

        report = render_preflight_lifecycle_report(result)
        self.assertIn("**SIMULATED:**", report)
        self.assertIn("cold_per_workload", report)
        self.assertIn("persistent_sequence", report)
        self.assertIn("not inferred from IDs, names, categories", report)
        self.assertIn("does not establish runtime latency", report)
        self.assertIn("DESCRIPTIVE adjacent tested-capacity summaries", report)
        self.assertIn("DESCRIPTIVE same-capacity policy summaries", report)
        self.assertIn("DESCRIPTIVE workload spread", report)
        self.assertIn("without rerunning simulation", report)
        self.assertIn("ESTIMATED serialized H2D transfer service", report)
        self.assertIn("caller-supplied hardware", report)
        self.assertIn("serialized/no-overlap", report)
        self.assertIn("exclude compute", report)
        self.assertIn("not proof of measured latency reduction", report)
        self.assertNotIn("recommended cache", report.lower())
        self.assertNotIn("speedup achieved", report.lower())

    def test_lifecycle_results_do_not_change_v2_transfer_plan_configuration(self) -> None:
        with patch("moe_cache_lab.preflight.load_suite_inputs", return_value=_inputs()):
            result = run_preflight_lifecycle_analysis("ignored.json", _config(version=2))

        payload = json.loads(render_preflight_lifecycle_json(result))
        self.assertEqual(payload["config"]["format_version"], 2)
        self.assertEqual(
            payload["config"]["transfer_operation_plans"],
            [
                {"name": "one", "operations_per_logical_load": 1},
                {"name": "two", "operations_per_logical_load": 2},
            ],
        )
        self.assertEqual(len(payload["simulated_cache_lifecycle_scenarios"]), 2)
        self.assertEqual(
            len(payload["estimated_serialized_h2d_transfer_service"]["rows"]),
            4,
        )

    def test_descending_manifest_workload_order_is_rejected_not_sorted(self) -> None:
        with patch(
            "moe_cache_lab.preflight.load_suite_inputs",
            return_value=_inputs(descending=True),
        ), self.assertRaisesRegex(ValueError, "ascending declared order"):
            run_preflight_lifecycle_analysis("ignored.json", _config())

    def test_cli_writes_lifecycle_outputs_without_collection(self) -> None:
        with patch("moe_cache_lab.preflight.load_suite_inputs", return_value=_inputs()):
            result = run_preflight_lifecycle_analysis("ignored.json", _config())

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "config.json"
            config_path.write_text(
                json.dumps({
                    "format": "moe-cache-lab.preflight-config",
                    "format_version": 1,
                    "expert_sizes": [
                        {"layer_id": 0, "expert_id": 0, "size_bytes": 4}
                    ],
                    "capacities_bytes": [4],
                    "policies": ["lru"],
                    "hardware_profiles": [{
                        "name": "assumption",
                        "h2d_payload_bandwidth_bytes_per_second": 100,
                        "setup_latency_ns_per_loaded_expert": 10,
                    }],
                }),
                encoding="utf-8",
            )
            markdown = directory / "lifecycle.md"
            json_output = directory / "lifecycle.json"
            with patch(
                "moe_cache_lab.preflight.run_preflight_lifecycle_analysis",
                return_value=result,
            ), patch(
                "moe_cache_lab.cli.collect_trace",
                side_effect=AssertionError("lifecycle analysis must not collect a model"),
            ), patch(
                "sys.argv",
                [
                    "moe-cache-lab",
                    "analyze-lifecycle",
                    "manifest.json",
                    "--preflight-config",
                    str(config_path),
                    "--output",
                    str(markdown),
                    "--json-output",
                    str(json_output),
                ],
            ):
                main()

            self.assertEqual(markdown.read_text(encoding="utf-8"), render_preflight_lifecycle_report(result))
            self.assertEqual(json_output.read_text(encoding="utf-8"), render_preflight_lifecycle_json(result))


if __name__ == "__main__":
    unittest.main()
