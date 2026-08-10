import unittest

from moe_cache_lab.byte_cache import ByteCacheSimulation, simulate_byte_cache
from moe_cache_lab.cache import simulate
from moe_cache_lab.trace import RoutingEvent


def event(
    layer: int,
    experts: tuple[int, ...],
    position: int,
    phase: str = "prompt",
) -> RoutingEvent:
    return RoutingEvent(phase, position, layer, experts)


class ByteCacheSimulationTests(unittest.TestCase):
    def test_layer_qualified_identity_uses_independent_sizes_and_residency(self) -> None:
        result = simulate_byte_cache(
            [
                event(0, (1,), 0),
                event(1, (1,), 1),
            ],
            capacity_bytes=3,
            expert_sizes_bytes={(0, 1): 2, (1, 1): 3},
            policy_name="lru",
        )

        self.assertEqual((result.expert_request_count, result.hits, result.misses), (2, 0, 2))
        self.assertEqual(result.simulated_demand_load_bytes, 5)
        self.assertEqual((result.eviction_count, result.simulated_evicted_bytes), (1, 2))
        self.assertEqual(result.final_resident_keys, ((1, 1),))
        self.assertEqual(result.final_resident_bytes, 3)

    def test_heterogeneous_sizes_change_capacity_behavior_from_count_slots(self) -> None:
        events = [
            event(0, (0,), 0),
            event(0, (1,), 1),
            event(0, (2,), 2),
            event(0, (3,), 3),
        ]
        byte_result = simulate_byte_cache(
            events,
            capacity_bytes=5,
            expert_sizes_bytes={(0, 0): 4, (0, 1): 1, (0, 2): 1, (0, 3): 1},
            policy_name="lru",
        )
        count_result = simulate(events, capacity=2, policy_name="lru").combined

        self.assertEqual(byte_result.eviction_count, 1)
        self.assertEqual(count_result.evictions, 2)
        self.assertEqual(
            byte_result.final_resident_keys,
            ((0, 1), (0, 2), (0, 3)),
        )

    def test_atomic_multi_expert_event_pins_required_hits_and_ignores_tuple_order(self) -> None:
        sizes = {(0, 0): 3, (0, 1): 2, (0, 2): 2, (0, 3): 3}
        left = simulate_byte_cache(
            [
                event(0, (0, 1, 2), 0),
                event(0, (1, 2, 3), 1),
            ],
            7,
            sizes,
            "lru",
        )
        right = simulate_byte_cache(
            [
                event(0, (2, 0, 1), 0),
                event(0, (3, 2, 1), 1),
            ],
            7,
            sizes,
            "lru",
        )

        self.assertEqual(left, right)
        self.assertEqual((left.hits, left.misses), (2, 4))
        self.assertEqual((left.eviction_count, left.simulated_evicted_bytes), (1, 3))
        self.assertEqual(left.final_resident_keys, ((0, 1), (0, 2), (0, 3)))
        self.assertEqual(left.final_resident_bytes, 7)

    def test_event_working_set_equal_to_capacity_succeeds_and_larger_rejects(self) -> None:
        events = [event(0, (0, 1), 0)]
        sizes = {(0, 0): 2, (0, 1): 3}

        exact = simulate_byte_cache(events, 5, sizes, "lru")
        self.assertEqual((exact.peak_resident_bytes, exact.final_resident_bytes), (5, 5))

        with self.assertRaisesRegex(ValueError, "working set 5 bytes exceeds.*4 bytes"):
            simulate_byte_cache(events, 4, sizes, "lru")

    def test_lru_stable_key_tie_break_handles_multiple_evictions(self) -> None:
        result = simulate_byte_cache(
            [
                event(0, (0, 1, 2), 0),
                event(0, (3,), 1),
            ],
            capacity_bytes=6,
            expert_sizes_bytes={
                (0, 0): 2,
                (0, 1): 2,
                (0, 2): 2,
                (0, 3): 4,
            },
            policy_name="lru",
        )

        self.assertEqual((result.eviction_count, result.simulated_evicted_bytes), (2, 4))
        self.assertEqual(result.final_resident_keys, ((0, 2), (0, 3)))
        self.assertEqual(result.final_resident_bytes, 6)

    def test_lfu_frequency_oldest_and_stable_key_order_handles_multiple_evictions(self) -> None:
        result = simulate_byte_cache(
            [
                event(0, (0, 1, 2), 0),
                event(0, (2,), 1),
                event(0, (3,), 2),
            ],
            capacity_bytes=6,
            expert_sizes_bytes={
                (0, 0): 2,
                (0, 1): 2,
                (0, 2): 2,
                (0, 3): 4,
            },
            policy_name="lfu",
        )

        self.assertEqual(result.hits, 1)
        self.assertEqual((result.eviction_count, result.simulated_evicted_bytes), (2, 4))
        self.assertEqual(result.final_resident_keys, ((0, 2), (0, 3)))
        self.assertEqual(result.final_resident_bytes, 6)

    def test_small_trace_accounting_reconciles_by_hand(self) -> None:
        result = simulate_byte_cache(
            [
                event(0, (0,), 0),
                event(0, (1,), 1),
                event(0, (0,), 2),
                event(0, (2,), 3),
            ],
            capacity_bytes=5,
            expert_sizes_bytes={(0, 0): 3, (0, 1): 2, (0, 2): 2},
            policy_name="lru",
        )

        self.assertEqual(result.policy, "lru")
        self.assertEqual(result.capacity_bytes, 5)
        self.assertEqual((result.event_count, result.expert_request_count), (4, 4))
        self.assertEqual((result.hits, result.misses, result.hit_rate), (1, 3, 0.25))
        self.assertEqual(result.simulated_demand_load_bytes, 7)
        self.assertEqual((result.eviction_count, result.simulated_evicted_bytes), (1, 2))
        self.assertEqual((result.peak_resident_bytes, result.final_resident_bytes), (5, 5))
        self.assertEqual(result.final_resident_keys, ((0, 0), (0, 2)))

    def test_size_map_insertion_order_and_extra_valid_entries_do_not_change_result(self) -> None:
        events = [event(0, (0,), 0), event(0, (1,), 1), event(0, (0,), 2)]
        left = {(0, 0): 2, (0, 1): 3, (9, 9): 99}
        right = {(9, 9): 99, (0, 1): 3, (0, 0): 2}

        self.assertEqual(
            simulate_byte_cache(events, 5, left, "lfu"),
            simulate_byte_cache(events, 5, right, "lfu"),
        )

    def test_invalid_capacity_policy_events_and_size_metadata_are_rejected(self) -> None:
        events = [event(0, (0,), 0)]
        sizes = {(0, 0): 1}

        for capacity in (0, -1, 1.5, True):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                simulate_byte_cache(events, capacity, sizes, "lru")

        for invalid_size in (0, -1, 1.5, True):
            with self.subTest(size=invalid_size), self.assertRaisesRegex(
                ValueError, "positive integer byte"
            ):
                simulate_byte_cache(events, 1, {(0, 0): invalid_size}, "lru")

        with self.assertRaisesRegex(ValueError, "missing referenced"):
            simulate_byte_cache(events, 1, {}, "lru")
        with self.assertRaisesRegex(ValueError, "unknown byte-cache policy"):
            simulate_byte_cache(events, 1, sizes, "offline_oracle_frequency")
        with self.assertRaisesRegex(TypeError, "RoutingEvent"):
            simulate_byte_cache([(0, 0)], 1, sizes, "lru")
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            simulate_byte_cache(events, 1, [((0, 0), 1)], "lru")
        with self.assertRaisesRegex(ValueError, "layer_id, expert_id"):
            simulate_byte_cache(events, 1, {(0, True): 1, (0, 0): 1}, "lru")

    def test_empty_sequence_has_zero_accounting_and_zero_hit_rate(self) -> None:
        for policy in ("lru", "lfu"):
            with self.subTest(policy=policy):
                result = simulate_byte_cache([], 8, {}, policy)
                self.assertEqual(result.event_count, 0)
                self.assertEqual(result.expert_request_count, 0)
                self.assertEqual((result.hits, result.misses, result.hit_rate), (0, 0, 0.0))
                self.assertEqual(result.simulated_demand_load_bytes, 0)
                self.assertEqual(result.simulated_evicted_bytes, 0)
                self.assertEqual((result.peak_resident_bytes, result.final_resident_bytes), (0, 0))
                self.assertEqual(result.final_resident_keys, ())

    def test_result_names_and_docs_keep_simulated_claim_boundary_explicit(self) -> None:
        fields = ByteCacheSimulation.__dataclass_fields__
        self.assertIn("simulated_demand_load_bytes", fields)
        self.assertIn("simulated_evicted_bytes", fields)
        documentation = " ".join((ByteCacheSimulation.__doc__ or "").lower().split())
        self.assertIn("simulated", documentation)
        self.assertIn("cache-model accounting", documentation)
        self.assertIn("not represent measured physical transfers", documentation)
        self.assertIn("speedup", documentation)


if __name__ == "__main__":
    unittest.main()
