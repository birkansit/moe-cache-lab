import json
from fractions import Fraction
import unittest
from unittest.mock import patch

from moe_cache_lab.byte_cache import simulate_versioned_byte_cache
from moe_cache_lab.hardware_cost import (
    HardwareTransferProfile,
    TransferOperationPlan,
    estimate_transfer_cost,
)
from moe_cache_lab.preflight import run_stage_qualified_preflight_analysis
from moe_cache_lab.preflight_config import (
    PreflightConfigV3,
    StageQualifiedExpertSizeRecord,
)
from moe_cache_lab.preflight_output import (
    PREFLIGHT_ANALYSIS_FORMAT,
    PREFLIGHT_ANALYSIS_STAGE_QUALIFIED_VERSION,
    PREFLIGHT_ANALYSIS_VERSION,
    render_stage_qualified_preflight_json,
    render_stage_qualified_preflight_report,
)
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
)


def _encoder_decoder_trace() -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="synthetic/stage-qualified-preflight",
        routing_stages=(
            RoutingStageProfile("encoder", 8, 1, True),
            RoutingStageProfile("decoder", 8, 1, True),
        ),
        events=(
            RoutingEventV2(
                "encoder", "source", 0, 3, "assigned", (5,), (0.75,)
            ),
            RoutingEventV2(
                "decoder", "decoder_prompt", 0, 3, "assigned", (5,), (0.6,)
            ),
            RoutingEventV2(
                "decoder", "decoder_generated", 1, 3, "assigned", (5,), (0.8,)
            ),
        ),
        capture_method="synthetic-test",
        created_at="fixed",
        transformers_version="test-version",
        model_revision="test-revision",
    )


def _config() -> PreflightConfigV3:
    return PreflightConfigV3(
        expert_sizes=(
            StageQualifiedExpertSizeRecord("decoder", 3, 5, 5),
            StageQualifiedExpertSizeRecord("encoder", 7, 7, 99),
            StageQualifiedExpertSizeRecord("encoder", 3, 5, 3),
        ),
        capacities_bytes=(8, 5, 8),
        policies=("lfu", "lru"),
        hardware_profiles=(
            HardwareTransferProfile("zeta", 16, 500_000_000),
            HardwareTransferProfile("alpha", 8, 250_000_000),
        ),
        transfer_operation_plans=(
            TransferOperationPlan("two", 2),
            TransferOperationPlan("one", 1),
        ),
    )


def _all_unassigned_trace() -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="synthetic/all-unassigned",
        routing_stages=(RoutingStageProfile("decoder", 8, 1, True),),
        events=(
            RoutingEventV2(
                "decoder",
                "decoder_prompt",
                0,
                3,
                "unassigned",
                (),
                (),
                unassigned_reason="capacity",
            ),
        ),
        capture_method="synthetic-test",
        created_at="fixed",
    )


class StageQualifiedPreflightOutputTests(unittest.TestCase):
    def test_workload_id_labels_routing_evidence_only(self) -> None:
        trace = _encoder_decoder_trace()
        config = _config()

        left = run_stage_qualified_preflight_analysis(
            trace, config, workload_id="evaluation-left"
        )
        right = run_stage_qualified_preflight_analysis(
            trace, config, workload_id="evaluation-right"
        )

        self.assertEqual(left.engine, right.engine)
        self.assertEqual(left.transfer_sensitivity, right.transfer_sensitivity)
        self.assertEqual(left.routing.workload_id, "evaluation-left")
        self.assertEqual(right.routing.workload_id, "evaluation-right")

    def test_non_aliasing_survives_context_cache_cost_json_and_markdown(self) -> None:
        result = run_stage_qualified_preflight_analysis(
            _encoder_decoder_trace(), _config()
        )
        context = result.transfer_sensitivity.workload_context

        self.assertEqual(
            context.referenced_expert_keys,
            (("encoder", 3, 5), ("decoder", 3, 5)),
        )
        self.assertEqual(context.unique_referenced_expert_count, 2)
        self.assertEqual(context.unique_referenced_expert_bytes, 8)
        self.assertEqual(context.maximum_atomic_event_working_set_bytes, 5)
        self.assertNotIn(("encoder", 7, 7), context.referenced_expert_keys)

        by_cell = {
            (item.capacity_bytes, item.policy): item
            for item in result.engine.simulations
        }
        for policy in ("lru", "lfu"):
            constrained = by_cell[(5, policy)]
            self.assertEqual((constrained.hits, constrained.misses), (1, 2))
            self.assertEqual(constrained.simulated_demand_load_bytes, 8)
            self.assertEqual(constrained.eviction_count, 1)
            self.assertEqual(
                constrained.final_resident_keys,
                (("decoder", 3, 5),),
            )
            roomy = by_cell[(8, policy)]
            self.assertEqual((roomy.hits, roomy.misses), (1, 2))
            self.assertEqual(roomy.simulated_demand_load_bytes, 8)
            self.assertEqual(
                set(roomy.final_resident_keys),
                {("encoder", 3, 5), ("decoder", 3, 5)},
            )

        for row in result.transfer_sensitivity.rows:
            self.assertEqual(
                row.estimate,
                estimate_transfer_cost(
                    row.simulation,
                    row.hardware_profile,
                    row.transfer_operation_plan,
                ),
            )
            self.assertEqual(row.estimate.policy, row.simulation.policy)
            self.assertEqual(
                row.estimate.cache_capacity_bytes,
                row.simulation.capacity_bytes,
            )
            self.assertEqual(
                row.estimate.simulated_demand_load_count,
                row.simulation.misses,
            )
            self.assertEqual(
                row.estimate.simulated_demand_load_bytes,
                row.simulation.simulated_demand_load_bytes,
            )

        payload = json.loads(render_stage_qualified_preflight_json(result))
        self.assertEqual(payload["format"], PREFLIGHT_ANALYSIS_FORMAT)
        self.assertEqual(
            payload["format_version"],
            PREFLIGHT_ANALYSIS_STAGE_QUALIFIED_VERSION,
        )
        self.assertEqual(PREFLIGHT_ANALYSIS_VERSION, 2)
        self.assertTrue(
            payload["claim_boundary"][
                "config_and_output_version_namespaces_are_independent"
            ]
        )
        self.assertEqual(
            payload["workload_byte_context"]["referenced_expert_keys"],
            [
                {"routing_stage": "encoder", "layer_id": 3, "expert_id": 5},
                {"routing_stage": "decoder", "layer_id": 3, "expert_id": 5},
            ],
        )
        roomy_json = next(
            item
            for item in payload["simulated_byte_cache_sensitivity"]
            if item["capacity_bytes"] == 8 and item["policy"] == "lru"
        )
        self.assertEqual(
            roomy_json["final_resident_keys"],
            [
                {"routing_stage": "encoder", "layer_id": 3, "expert_id": 5},
                {"routing_stage": "decoder", "layer_id": 3, "expert_id": 5},
            ],
        )

        report = render_stage_qualified_preflight_report(result)
        self.assertIn("| encoder | 3 | 5 | 3 |", report)
        self.assertIn("| decoder | 3 | 5 | 5 |", report)
        self.assertIn("(encoder, 3, 5), (decoder, 3, 5)", report)
        self.assertIn("versions are independent namespaces", report)
        self.assertIn("**SIMULATED:**", report)
        self.assertIn("**ESTIMATED:**", report)
        self.assertIn("NOT ESTABLISHED", report)

    def test_reuses_exact_b3_simulations_once_and_existing_estimator(self) -> None:
        trace = _encoder_decoder_trace()
        config = _config()
        with patch(
            "moe_cache_lab.preflight.simulate_versioned_byte_cache",
            wraps=simulate_versioned_byte_cache,
        ) as cache_core, patch(
            "moe_cache_lab.hardware_cost.estimate_transfer_cost",
            wraps=estimate_transfer_cost,
        ) as estimator:
            result = run_stage_qualified_preflight_analysis(trace, config)

        self.assertEqual(cache_core.call_count, 4)
        self.assertEqual(estimator.call_count, 16)
        self.assertEqual(
            [
                (
                    row.simulation.capacity_bytes,
                    row.simulation.policy,
                    row.hardware_profile.name,
                    row.transfer_operation_plan.name,
                )
                for row in result.transfer_sensitivity.rows
            ],
            [
                (capacity, policy, profile, plan)
                for capacity in (5, 8)
                for policy in ("lru", "lfu")
                for profile in ("alpha", "zeta")
                for plan in ("one", "two")
            ],
        )
        simulation_ids = {id(item) for item in result.engine.simulations}
        self.assertTrue(
            all(
                id(row.simulation) in simulation_ids
                for row in result.transfer_sensitivity.rows
            )
        )
        self.assertTrue(
            all(
                id(call_args.args[0]) in simulation_ids
                for call_args in estimator.call_args_list
            )
        )

    def test_all_unassigned_trace_remains_zero_request_and_zero_cost(self) -> None:
        config = PreflightConfigV3(
            expert_sizes=(),
            capacities_bytes=(1,),
            policies=("lru", "lfu"),
            hardware_profiles=(HardwareTransferProfile("zero", 7, 999),),
            transfer_operation_plans=(TransferOperationPlan("two", 2),),
        )
        result = run_stage_qualified_preflight_analysis(
            _all_unassigned_trace(), config
        )

        context = result.transfer_sensitivity.workload_context
        self.assertEqual(
            (
                context.unique_referenced_expert_count,
                context.unique_referenced_expert_bytes,
                context.maximum_atomic_event_working_set_bytes,
                context.referenced_expert_keys,
            ),
            (0, 0, 0, ()),
        )
        for simulation in result.engine.simulations:
            self.assertEqual(simulation.event_count, 1)
            self.assertEqual(
                (
                    simulation.expert_request_count,
                    simulation.hits,
                    simulation.misses,
                    simulation.simulated_demand_load_bytes,
                    simulation.eviction_count,
                    simulation.simulated_evicted_bytes,
                    simulation.peak_resident_bytes,
                    simulation.final_resident_bytes,
                ),
                (0, 0, 0, 0, 0, 0, 0, 0),
            )
            self.assertEqual(simulation.final_resident_keys, ())
        for row in result.transfer_sensitivity.rows:
            self.assertEqual(row.estimate.simulated_demand_load_count, 0)
            self.assertEqual(row.estimate.modeled_transfer_operation_count, 0)
            self.assertEqual(
                row.estimate.estimated_payload_service_seconds, Fraction(0, 1)
            )
            self.assertEqual(
                row.estimate.estimated_setup_service_seconds, Fraction(0, 1)
            )
            self.assertEqual(
                row.estimate.estimated_serialized_transfer_service_seconds,
                Fraction(0, 1),
            )

        payload = json.loads(render_stage_qualified_preflight_json(result))
        workload = payload["routing_evidence"]["workloads"][0]
        self.assertEqual(workload["event_count"], 1)
        self.assertEqual(workload["expert_request_count"], 0)
        self.assertEqual(payload["config"]["expert_sizes"], [])
        self.assertEqual(
            payload["workload_byte_context"]["referenced_expert_keys"], []
        )
        self.assertTrue(
            all(
                item["final_resident_keys"] == []
                for item in payload["simulated_byte_cache_sensitivity"]
            )
        )
        report = render_stage_qualified_preflight_report(result)
        self.assertIn("No expert-size assumptions were supplied.", report)
        self.assertIn("Referenced expert keys: empty", report)
        self.assertNotIn("(decoder, 3,", report)

    def test_decoder_only_extra_sizes_are_not_referenced_or_charged(self) -> None:
        trace = RoutingTraceV2(
            model_id="synthetic/decoder-only",
            routing_stages=(RoutingStageProfile("decoder", 4, 1, True),),
            events=(
                RoutingEventV2(
                    "decoder", "decoder_prompt", 0, 2, "assigned", (1,), ()
                ),
                RoutingEventV2(
                    "decoder", "decoder_generated", 1, 2, "assigned", (1,), ()
                ),
            ),
            capture_method="synthetic-test",
            created_at="fixed",
        )
        config = PreflightConfigV3(
            expert_sizes=(
                StageQualifiedExpertSizeRecord("encoder", 9, 9, 999),
                StageQualifiedExpertSizeRecord("decoder", 2, 1, 4),
            ),
            capacities_bytes=(4,),
            policies=("lru",),
            hardware_profiles=(HardwareTransferProfile("p", 8, 0),),
            transfer_operation_plans=(TransferOperationPlan("one", 1),),
        )

        result = run_stage_qualified_preflight_analysis(trace, config)

        self.assertEqual(
            result.transfer_sensitivity.workload_context.referenced_expert_keys,
            (("decoder", 2, 1),),
        )
        self.assertEqual(
            result.transfer_sensitivity.workload_context.unique_referenced_expert_bytes,
            4,
        )
        simulation = result.engine.simulations[0]
        self.assertEqual((simulation.hits, simulation.misses), (1, 1))
        self.assertEqual(simulation.simulated_demand_load_bytes, 4)
        self.assertEqual(
            result.transfer_sensitivity.rows[0].estimate.estimated_payload_service_seconds,
            Fraction(1, 2),
        )

    def test_json_and_markdown_are_deterministic_and_contain_no_process_leakage(self) -> None:
        result = run_stage_qualified_preflight_analysis(
            _encoder_decoder_trace(), _config()
        )
        left_json = render_stage_qualified_preflight_json(result)
        right_json = render_stage_qualified_preflight_json(result)
        left_markdown = render_stage_qualified_preflight_report(result)
        right_markdown = render_stage_qualified_preflight_report(result)

        self.assertEqual(left_json, right_json)
        self.assertEqual(left_markdown, right_markdown)
        for output in (left_json, left_markdown):
            lowered = output.lower()
            for forbidden in (
                "c:\\\\",
                "birkan" + "sit",
                "[acti" + "ve]",
                "private repo" + "sitory",
                "direc" + "tor",
                "work" + "er",
            ):
                self.assertNotIn(forbidden, lowered)

    def test_wrong_types_reject_without_falling_back_to_legacy(self) -> None:
        with self.assertRaisesRegex(TypeError, "RoutingTraceV2"):
            run_stage_qualified_preflight_analysis(object(), _config())  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "PreflightConfigV3"):
            run_stage_qualified_preflight_analysis(  # type: ignore[arg-type]
                _encoder_decoder_trace(), object()
            )


if __name__ == "__main__":
    unittest.main()
