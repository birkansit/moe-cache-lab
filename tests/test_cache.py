import unittest

from moe_cache_lab.cache import LFUCache, LRUCache, simulate
from moe_cache_lab.trace import RoutingEvent


def event(layer: int, experts: tuple[int, ...], phase: str = "prompt", position: int = 0) -> RoutingEvent:
    return RoutingEvent(phase, position, layer, experts)


class CacheTests(unittest.TestCase):
    def test_layer_qualified_identity_prevents_numerical_id_collision(self) -> None:
        events = [event(0, (1,)), event(1, (1,), position=1)]
        for policy in ("lru", "lfu"):
            with self.subTest(policy=policy):
                result = simulate(events, capacity=1, policy_name=policy).combined
                self.assertEqual((result.requests, result.hits, result.misses, result.evictions), (2, 0, 2, 1))
        oracle = simulate(events, capacity=1, policy_name="offline_oracle_frequency").combined
        self.assertEqual((oracle.requests, oracle.hits, oracle.misses), (2, 1, 1))
        self.assertEqual((oracle.demand_loads, oracle.evictions), (2, 2))

    def test_atomic_bundle_is_order_invariant_and_pins_required_hits(self) -> None:
        prefix = [event(0, (0, 1)), event(0, (2,), position=1)]
        left = simulate(prefix + [event(0, (1, 3), position=2), event(0, (0,), position=3)], 3, "lru")
        right = simulate(prefix + [event(0, (3, 1), position=2), event(0, (0,), position=3)], 3, "lru")
        self.assertEqual(left.combined, right.combined)
        self.assertEqual((left.combined.hits, left.combined.misses, left.combined.evictions), (1, 5, 2))

    def test_rejects_capacity_below_largest_atomic_bundle(self) -> None:
        for policy in ("lru", "lfu", "offline_oracle_frequency"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "largest routing-event bundle 2"):
                simulate([event(0, (1, 2))], capacity=1, policy_name=policy)

    def test_capacity_equal_to_bundle_handles_disjoint_events(self) -> None:
        for policy in ("lru", "lfu"):
            with self.subTest(policy=policy):
                result = simulate(
                    [event(0, (0, 1)), event(0, (2, 3), position=1)], 2, policy
                ).combined
                self.assertEqual((result.hits, result.misses, result.evictions), (0, 4, 2))

        oracle = simulate(
            [event(0, (0, 1)), event(0, (2, 3), position=1)],
            2,
            "offline_oracle_frequency",
        )
        self.assertEqual(oracle.fixed_entries, frozenset({(0, 0), (0, 1)}))
        self.assertEqual(
            (oracle.combined.hits, oracle.combined.misses, oracle.combined.demand_loads, oracle.combined.evictions),
            (2, 2, 4, 4),
        )

    def test_oracle_capacity_may_exceed_distinct_fixed_targets(self) -> None:
        oracle = simulate([event(0, (0, 1))], 4, "offline_oracle_frequency")
        self.assertEqual(oracle.fixed_entries, frozenset({(0, 0), (0, 1)}))
        self.assertEqual(
            (oracle.combined.prewarm_loads, oracle.combined.hits, oracle.combined.demand_loads, oracle.combined.evictions),
            (2, 2, 0, 0),
        )

    def test_lru_same_event_tie_uses_stable_layer_expert_key(self) -> None:
        cache = LRUCache(2)
        cache.access_bundle(frozenset({(0, 1), (0, 0)}))
        cache.access_bundle(frozenset({(0, 2)}))
        self.assertEqual(set(cache.entries), {(0, 1), (0, 2)})

    def test_lfu_updates_once_per_event_pins_and_resets_evicted_history(self) -> None:
        cache = LFUCache(2)
        cache.access_bundle(frozenset({(0, 0), (0, 1)}))
        cache.access_bundle(frozenset({(0, 1)}))
        cache.access_bundle(frozenset({(0, 0), (0, 2)}))
        self.assertEqual(cache.entries[(0, 0)][0], 2)
        self.assertNotIn((0, 1), cache.entries)  # Required key 0 was pinned despite equal frequency.
        self.assertEqual(cache.entries[(0, 2)][0], 1)
        cache.access_bundle(frozenset({(0, 1)}))
        self.assertNotIn((0, 2), cache.entries)
        self.assertEqual(cache.entries[(0, 1)][0], 1)  # Evicted frequency history did not survive.

    def test_lfu_ties_use_oldest_then_stable_key(self) -> None:
        same_event = LFUCache(2)
        same_event.access_bundle(frozenset({(0, 1), (0, 0)}))
        same_event.access_bundle(frozenset({(0, 2)}))
        self.assertNotIn((0, 0), same_event.entries)

        different_events = LFUCache(2)
        different_events.access_bundle(frozenset({(0, 3)}))
        different_events.access_bundle(frozenset({(0, 2)}))
        different_events.access_bundle(frozenset({(0, 4)}))
        self.assertNotIn((0, 3), different_events.entries)

    def test_offline_oracle_frequency_uses_future_data_and_is_explicitly_labeled(self) -> None:
        events = [
            event(0, (0,)),
            event(0, (1,), position=1),
            event(0, (1,), position=2),
            event(0, (1,), position=3),
        ]
        simulation = simulate(events, 1, "offline_oracle_frequency")
        self.assertEqual(simulation.policy, "offline_oracle_frequency")
        self.assertEqual(simulation.fixed_entries, frozenset({(0, 1)}))
        self.assertEqual((simulation.combined.hits, simulation.combined.misses), (3, 1))

    def test_phase_rows_share_cache_chronology_instead_of_cold_reruns(self) -> None:
        simulation = simulate(
            [event(0, (2,), "prompt"), event(0, (2,), "generated", 1)],
            1,
            "lru",
        )
        self.assertEqual((simulation.combined.samples, simulation.combined.hits), (2, 1))
        self.assertEqual((simulation.prompt.samples, simulation.prompt.hits, simulation.prompt.misses), (1, 0, 1))
        self.assertEqual((simulation.generated.samples, simulation.generated.hits, simulation.generated.misses), (1, 1, 0))

    def test_metric_accounting_for_dynamic_and_fixed_policies(self) -> None:
        events = [event(0, (0,)), event(0, (1,), position=1), event(0, (0,), position=2)]
        dynamic = simulate(events, 1, "lru").combined
        self.assertEqual(dynamic.prewarm_loads, 0)
        self.assertEqual(dynamic.demand_loads, dynamic.misses)
        self.assertEqual(dynamic.estimated_expert_transfers, dynamic.misses)

        oracle = simulate(events, 1, "offline_oracle_frequency")
        self.assertEqual((oracle.combined.prewarm_loads, oracle.combined.demand_loads), (1, 2))
        self.assertEqual(oracle.combined.estimated_expert_transfers, 3)
        self.assertEqual(oracle.combined.evictions, 2)
        self.assertEqual(
            (oracle.prompt.prewarm_loads, oracle.prompt.demand_loads, oracle.prompt.evictions),
            (0, 2, 2),
        )

    def test_empty_event_sequence_has_zero_rates_and_oracle_prewarm(self) -> None:
        for policy in ("lru", "lfu", "offline_oracle_frequency"):
            simulation = simulate([], capacity=1, policy_name=policy)
            self.assertEqual((simulation.combined.requests, simulation.combined.hits), (0, 0))
            self.assertEqual(simulation.combined.hit_rate, 0.0)
            self.assertEqual(simulation.combined.prewarm_loads, 0)

    def test_rejects_invalid_capacity(self) -> None:
        for capacity in (0, -1, 1.5, True):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, "positive integer"):
                simulate([event(0, (1,))], capacity=capacity, policy_name="lru")

    def test_rejects_non_event_requests_instead_of_flat_replay(self) -> None:
        with self.assertRaisesRegex(TypeError, "RoutingEvent"):
            simulate([(0, 1)], capacity=1, policy_name="lru")
