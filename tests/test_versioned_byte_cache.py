from dataclasses import fields
import unittest

from moe_cache_lab.byte_cache import (
    ByteCacheSimulation,
    simulate_byte_cache,
    simulate_versioned_byte_cache,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
)


def _v1_trace() -> RoutingTrace:
    return RoutingTrace(
        model_id="synthetic/v1",
        num_experts=5,
        experts_per_token=None,
        events=(
            RoutingEvent("prompt", 0, 0, (0, 1)),
            RoutingEvent("prompt", 1, 0, (1,)),
            RoutingEvent("prompt", 2, 0, (2,)),
            RoutingEvent("prompt", 3, 0, (0,)),
        ),
        created_at="fixed",
    )


def _decoder_trace(
    events: tuple[RoutingEventV2, ...],
    *,
    width: int = 1,
) -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="synthetic/v2",
        routing_stages=(RoutingStageProfile("decoder", 8, width, True),),
        events=events,
        created_at="fixed",
    )


def _result_without_event_count(result: ByteCacheSimulation) -> tuple[object, ...]:
    return tuple(
        getattr(result, item.name)
        for item in fields(ByteCacheSimulation)
        if item.name != "event_count"
    )


class VersionedByteCacheTests(unittest.TestCase):
    def test_v1_versioned_path_is_exactly_existing_lru_and_lfu(self) -> None:
        trace = _v1_trace()
        sizes = {(0, 0): 3, (0, 1): 2, (0, 2): 3, (9, 9): 99}

        for policy in ("lru", "lfu"):
            with self.subTest(policy=policy):
                self.assertEqual(
                    simulate_versioned_byte_cache(trace, 5, sizes, policy),
                    simulate_byte_cache(trace.events, 5, sizes, policy),
                )

    def test_v2_encoder_decoder_numerical_ids_do_not_alias(self) -> None:
        trace = RoutingTraceV2(
            model_id="synthetic/v2",
            routing_stages=(
                RoutingStageProfile("encoder", 8, 1, True),
                RoutingStageProfile("decoder", 8, 1, True),
            ),
            events=(
                RoutingEventV2(
                    "encoder", "source", 0, 1, "assigned", (3,), (0.75,)
                ),
                RoutingEventV2(
                    "decoder",
                    "decoder_prompt",
                    0,
                    1,
                    "assigned",
                    (3,),
                    (0.6,),
                ),
            ),
            created_at="fixed",
        )
        sizes = {("encoder", 1, 3): 3, ("decoder", 1, 3): 4}

        result = simulate_versioned_byte_cache(trace, 7, sizes, "lru")

        self.assertEqual((result.event_count, result.expert_request_count), (2, 2))
        self.assertEqual((result.hits, result.misses), (0, 2))
        self.assertEqual(result.simulated_demand_load_bytes, 7)
        self.assertEqual(
            result.final_resident_keys,
            (("decoder", 1, 3), ("encoder", 1, 3)),
        )

    def test_v2_unassigned_event_has_zero_requests_and_no_cache_effect(self) -> None:
        event = RoutingEventV2(
            "decoder",
            "decoder_generated",
            0,
            1,
            "unassigned",
            (),
            (),
            unassigned_reason="capacity",
        )

        for policy in ("lru", "lfu"):
            with self.subTest(policy=policy):
                result = simulate_versioned_byte_cache(
                    _decoder_trace((event,)), 8, {}, policy
                )
                self.assertEqual((result.event_count, result.expert_request_count), (1, 0))
                self.assertEqual((result.hits, result.misses), (0, 0))
                self.assertEqual(result.simulated_demand_load_bytes, 0)
                self.assertEqual((result.eviction_count, result.simulated_evicted_bytes), (0, 0))
                self.assertEqual((result.peak_resident_bytes, result.final_resident_bytes), (0, 0))
                self.assertEqual(result.final_resident_keys, ())

    def test_inserting_unassigned_event_does_not_change_lru_or_lfu(self) -> None:
        first = RoutingEventV2(
            "decoder", "decoder_prompt", 0, 1, "assigned", (0,), ()
        )
        second = RoutingEventV2(
            "decoder", "decoder_generated", 1, 1, "assigned", (1,), ()
        )
        empty = RoutingEventV2(
            "decoder",
            "decoder_generated",
            2,
            1,
            "unassigned",
            (),
            (),
            unassigned_reason="capacity",
        )
        third = RoutingEventV2(
            "decoder", "decoder_generated", 3, 1, "assigned", (2,), ()
        )
        fourth = RoutingEventV2(
            "decoder", "decoder_generated", 4, 1, "assigned", (0,), ()
        )
        sizes = {("decoder", 1, expert): 2 for expert in range(3)}

        for policy in ("lru", "lfu"):
            with self.subTest(policy=policy):
                baseline = simulate_versioned_byte_cache(
                    _decoder_trace((first, second, third, fourth)),
                    4,
                    sizes,
                    policy,
                )
                inserted = simulate_versioned_byte_cache(
                    _decoder_trace((first, second, empty, third, fourth)),
                    4,
                    sizes,
                    policy,
                )
                self.assertEqual(inserted.event_count, baseline.event_count + 1)
                self.assertEqual(
                    _result_without_event_count(inserted),
                    _result_without_event_count(baseline),
                )

    def test_v2_multi_expert_bundle_is_atomic_and_order_insensitive(self) -> None:
        sizes = {
            ("decoder", 1, 0): 3,
            ("decoder", 1, 1): 2,
            ("decoder", 1, 2): 2,
            ("decoder", 1, 3): 3,
        }
        left = _decoder_trace(
            (
                RoutingEventV2(
                    "decoder", "decoder_prompt", 0, 1, "assigned", (0, 1, 2), ()
                ),
                RoutingEventV2(
                    "decoder", "decoder_generated", 1, 1, "assigned", (1, 2, 3), ()
                ),
            ),
            width=3,
        )
        right = _decoder_trace(
            (
                RoutingEventV2(
                    "decoder", "decoder_prompt", 0, 1, "assigned", (2, 0, 1), ()
                ),
                RoutingEventV2(
                    "decoder", "decoder_generated", 1, 1, "assigned", (3, 2, 1), ()
                ),
            ),
            width=3,
        )

        for policy in ("lru", "lfu"):
            with self.subTest(policy=policy):
                result = simulate_versioned_byte_cache(left, 7, sizes, policy)
                self.assertEqual(
                    result,
                    simulate_versioned_byte_cache(right, 7, sizes, policy),
                )
                self.assertEqual((result.hits, result.misses), (2, 4))
                self.assertEqual(result.final_resident_bytes, 7)

    def test_v2_missing_size_names_stage_qualified_identity(self) -> None:
        trace = _decoder_trace(
            (
                RoutingEventV2(
                    "decoder", "decoder_prompt", 0, 1, "assigned", (3,), ()
                ),
            )
        )

        with self.assertRaisesRegex(
            ValueError, r"stage-qualified experts: \('decoder', 1, 3\)"
        ):
            simulate_versioned_byte_cache(trace, 8, {}, "lru")

    def test_v2_size_map_rejects_malformed_keys_values_and_mixed_identity(self) -> None:
        trace = _decoder_trace(
            (
                RoutingEventV2(
                    "decoder", "decoder_prompt", 0, 1, "assigned", (0,), ()
                ),
            )
        )
        malformed_maps = (
            {(1, 0): 1},
            {("decoder", 1, 0): 1, (1, 0): 1},
            {("other", 1, 0): 1},
            {("decoder", True, 0): 1},
            {("decoder", 1, -1): 1},
            {("decoder", 1, 0): True},
            {("decoder", 1, 0): 1.5},
        )
        for sizes in malformed_maps:
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                simulate_versioned_byte_cache(trace, 8, sizes, "lru")
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            simulate_versioned_byte_cache(trace, 8, [], "lru")  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "layer_id, expert_id"):
            simulate_versioned_byte_cache(
                _v1_trace(), 8, {("decoder", 0, 0): 1}, "lru"
            )

    def test_v2_required_working_set_larger_than_capacity_rejects_before_replay(self) -> None:
        trace = _decoder_trace(
            (
                RoutingEventV2(
                    "decoder", "decoder_prompt", 0, 1, "assigned", (0, 1), ()
                ),
            ),
            width=2,
        )
        sizes = {("decoder", 1, 0): 3, ("decoder", 1, 1): 2}

        with self.assertRaisesRegex(
            ValueError, "working set 5 bytes exceeds.*4 bytes"
        ):
            simulate_versioned_byte_cache(trace, 4, sizes, "lfu")

    def test_v2_extra_valid_size_entries_do_not_change_result(self) -> None:
        trace = _decoder_trace(
            (
                RoutingEventV2(
                    "decoder", "decoder_prompt", 0, 1, "assigned", (0,), ()
                ),
                RoutingEventV2(
                    "decoder", "decoder_generated", 1, 1, "assigned", (1,), ()
                ),
            )
        )
        exact = {("decoder", 1, 0): 2, ("decoder", 1, 1): 3}
        extra = {("encoder", 7, 7): 99, **exact}

        self.assertEqual(
            simulate_versioned_byte_cache(trace, 5, exact, "lfu"),
            simulate_versioned_byte_cache(trace, 5, extra, "lfu"),
        )

    def test_versioned_api_rejects_non_trace_objects(self) -> None:
        with self.assertRaisesRegex(TypeError, "RoutingTrace or RoutingTraceV2"):
            simulate_versioned_byte_cache(object(), 8, {}, "lru")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
