import unittest
from unittest.mock import call, patch

from moe_cache_lab.byte_cache import simulate_versioned_byte_cache
from moe_cache_lab.hardware_cost import HardwareTransferProfile, TransferOperationPlan
from moe_cache_lab.preflight import run_versioned_preflight_engine
from moe_cache_lab.preflight_config import (
    ExpertSizeRecord,
    PreflightConfig,
    PreflightConfigV3,
    StageQualifiedExpertSizeRecord,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
)


_PROFILE = HardwareTransferProfile("synthetic", 1_000, 0)
_PLAN = TransferOperationPlan("one", 1)


def _encoder_decoder_trace() -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="synthetic/v2",
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
        ),
        capture_method="synthetic-test",
        created_at="fixed",
        model_revision="test-revision",
    )


def _decoder_trace(events: tuple[RoutingEventV2, ...], *, width: int = 1) -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="synthetic/v2-decoder",
        routing_stages=(RoutingStageProfile("decoder", 8, width, True),),
        events=events,
        capture_method="synthetic-test",
        created_at="fixed",
    )


def _v1_trace() -> RoutingTrace:
    return RoutingTrace(
        model_id="synthetic/v1",
        num_experts=2,
        experts_per_token=1,
        events=(RoutingEvent("prompt", 0, 0, (1,)),),
        capture_method="synthetic-test",
        created_at="fixed",
    )


def _v3_config(
    expert_sizes: tuple[StageQualifiedExpertSizeRecord, ...],
    *,
    capacities: tuple[int, ...] = (7,),
    policies: tuple[str, ...] = ("lru",),
) -> PreflightConfigV3:
    return PreflightConfigV3(
        expert_sizes=expert_sizes,
        capacities_bytes=capacities,
        policies=policies,
        hardware_profiles=(_PROFILE,),
        transfer_operation_plans=(_PLAN,),
    )


def _legacy_config() -> PreflightConfig:
    return PreflightConfig(
        expert_sizes=(ExpertSizeRecord(0, 1, 3),),
        capacities_bytes=(3,),
        policies=("lru",),
        hardware_profiles=(_PROFILE,),
    )


class VersionedPreflightEngineTests(unittest.TestCase):
    def test_v2_sweep_reuses_versioned_cache_once_per_normalized_cell(self) -> None:
        trace = _encoder_decoder_trace()
        config = _v3_config(
            (
                StageQualifiedExpertSizeRecord("decoder", 3, 5, 4),
                StageQualifiedExpertSizeRecord("encoder", 3, 5, 3),
            ),
            capacities=(7, 4, 7),
            policies=("lfu", "lru"),
        )
        size_map = config.expert_size_map()
        expected = tuple(
            simulate_versioned_byte_cache(trace, capacity, size_map, policy)
            for capacity in config.capacities_bytes
            for policy in config.policies
        )

        with patch(
            "moe_cache_lab.preflight.simulate_versioned_byte_cache",
            wraps=simulate_versioned_byte_cache,
        ) as cache_core:
            result = run_versioned_preflight_engine(trace, config)

        self.assertIs(result.trace, trace)
        self.assertIs(result.config, config)
        self.assertEqual(result.simulations, expected)
        self.assertEqual(
            [(item.capacity_bytes, item.policy) for item in result.simulations],
            [(4, "lru"), (4, "lfu"), (7, "lru"), (7, "lfu")],
        )
        self.assertEqual(
            cache_core.call_args_list,
            [
                call(trace, 4, size_map, "lru"),
                call(trace, 4, size_map, "lfu"),
                call(trace, 7, size_map, "lru"),
                call(trace, 7, size_map, "lfu"),
            ],
        )
        for simulation in result.simulations[2:]:
            self.assertEqual(
                set(simulation.final_resident_keys),
                {("encoder", 3, 5), ("decoder", 3, 5)},
            )
            self.assertEqual(simulation.simulated_demand_load_bytes, 7)

    def test_decoder_only_extra_size_entries_are_inert(self) -> None:
        trace = _decoder_trace(
            (
                RoutingEventV2(
                    "decoder", "decoder_prompt", 0, 2, "assigned", (1,), ()
                ),
                RoutingEventV2(
                    "decoder", "decoder_generated", 1, 2, "assigned", (1,), ()
                ),
            )
        )
        exact = _v3_config(
            (StageQualifiedExpertSizeRecord("decoder", 2, 1, 2),),
            capacities=(2,),
            policies=("lfu",),
        )
        extra = _v3_config(
            (
                StageQualifiedExpertSizeRecord("encoder", 7, 7, 99),
                StageQualifiedExpertSizeRecord("decoder", 2, 1, 2),
            ),
            capacities=(2,),
            policies=("lfu",),
        )

        exact_result = run_versioned_preflight_engine(trace, exact).simulations
        extra_result = run_versioned_preflight_engine(trace, extra).simulations

        self.assertEqual(extra_result, exact_result)
        self.assertEqual((extra_result[0].hits, extra_result[0].misses), (1, 1))
        self.assertEqual(extra_result[0].simulated_demand_load_bytes, 2)

    def test_all_unassigned_trace_accepts_empty_sizes_and_is_cache_inert(self) -> None:
        trace = _decoder_trace(
            (
                RoutingEventV2(
                    "decoder",
                    "decoder_prompt",
                    0,
                    0,
                    "unassigned",
                    (),
                    (),
                    unassigned_reason="capacity",
                ),
            )
        )
        config = _v3_config((), capacities=(1,), policies=("lru", "lfu"))

        result = run_versioned_preflight_engine(trace, config)

        self.assertEqual(len(result.simulations), 2)
        for simulation in result.simulations:
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

    def test_engine_enforces_exact_trace_config_compatibility(self) -> None:
        v1_trace = _v1_trace()
        legacy = _legacy_config()
        v3 = _v3_config(
            (
                StageQualifiedExpertSizeRecord("encoder", 3, 5, 3),
                StageQualifiedExpertSizeRecord("decoder", 3, 5, 4),
            )
        )

        v1_result = run_versioned_preflight_engine(v1_trace, legacy)
        self.assertEqual(
            v1_result.simulations,
            (
                simulate_versioned_byte_cache(
                    v1_trace, 3, legacy.expert_size_map(), "lru"
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "routing trace v1 requires"):
            run_versioned_preflight_engine(v1_trace, v3)
        with self.assertRaisesRegex(ValueError, "routing trace v2 requires"):
            run_versioned_preflight_engine(_encoder_decoder_trace(), legacy)
        with self.assertRaisesRegex(TypeError, "RoutingTrace or RoutingTraceV2"):
            run_versioned_preflight_engine(object(), legacy)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "PreflightConfigV3"):
            run_versioned_preflight_engine(  # type: ignore[arg-type]
                _encoder_decoder_trace(), object()
            )

    def test_missing_exact_stage_or_unoffset_layer_keys_reject(self) -> None:
        trace = _encoder_decoder_trace()
        cases = (
            (
                (StageQualifiedExpertSizeRecord("decoder", 3, 5, 4),),
                "('encoder', 3, 5)",
            ),
            (
                (StageQualifiedExpertSizeRecord("encoder", 3, 5, 3),),
                "('decoder', 3, 5)",
            ),
            (
                (
                    StageQualifiedExpertSizeRecord("encoder", 1003, 5, 3),
                    StageQualifiedExpertSizeRecord("decoder", 3, 5, 4),
                ),
                "('encoder', 3, 5)",
            ),
        )
        for expert_sizes, missing_key in cases:
            with self.subTest(missing_key=missing_key), self.assertRaisesRegex(
                ValueError, "missing referenced stage-qualified experts"
            ) as raised:
                run_versioned_preflight_engine(
                    trace,
                    _v3_config(expert_sizes),
                )
            self.assertIn(missing_key, str(raised.exception))

    def test_atomic_event_larger_than_capacity_rejects_in_cache_core(self) -> None:
        trace = _decoder_trace(
            (
                RoutingEventV2(
                    "decoder", "decoder_prompt", 0, 0, "assigned", (0, 1), ()
                ),
            ),
            width=2,
        )
        config = _v3_config(
            (
                StageQualifiedExpertSizeRecord("decoder", 0, 0, 3),
                StageQualifiedExpertSizeRecord("decoder", 0, 1, 3),
            ),
            capacities=(5,),
        )

        with self.assertRaisesRegex(
            ValueError, "working set 6 bytes exceeds byte-cache capacity 5 bytes"
        ):
            run_versioned_preflight_engine(trace, config)


if __name__ == "__main__":
    unittest.main()
