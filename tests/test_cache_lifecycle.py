import unittest

from moe_cache_lab.byte_cache import (
    ByteCacheWorkload,
    simulate_byte_cache,
    simulate_byte_cache_workloads,
)
from moe_cache_lab.trace import RoutingEvent


def _event(layer: int, experts: tuple[int, ...], position: int = 0) -> RoutingEvent:
    return RoutingEvent("prompt", position, layer, experts)


def _workload(
    workload_id: str,
    order: int,
    *events: RoutingEvent,
) -> ByteCacheWorkload:
    return ByteCacheWorkload(workload_id, order, tuple(events))


class ByteCacheLifecycleTests(unittest.TestCase):
    def test_cold_mode_exactly_matches_independent_empty_cache_simulations(self) -> None:
        workloads = (
            _workload("first", 0, _event(0, (0,)), _event(0, (1,), 1)),
            _workload("second", 1, _event(0, (0,))),
        )
        sizes = {(0, 0): 4, (0, 1): 4}

        result = simulate_byte_cache_workloads(
            workloads, 8, sizes, "lfu", "cold_per_workload"
        )

        for workload, attributed in zip(workloads, result.workloads):
            independent = simulate_byte_cache(workload.events, 8, sizes, "lfu")
            self.assertEqual(attributed.event_count, independent.event_count)
            self.assertEqual(
                attributed.expert_request_count, independent.expert_request_count
            )
            self.assertEqual(attributed.hits, independent.hits)
            self.assertEqual(attributed.misses, independent.misses)
            self.assertEqual(
                attributed.simulated_demand_load_bytes,
                independent.simulated_demand_load_bytes,
            )
            self.assertEqual(attributed.eviction_count, independent.eviction_count)
            self.assertEqual(
                attributed.ending_resident_keys, independent.final_resident_keys
            )
            self.assertEqual(attributed.starting_resident_keys, ())
        self.assertIsNone(result.shared_sequence_final_resident_keys)
        self.assertIsNone(result.shared_sequence_final_resident_bytes)

    def test_persistent_boundary_reuse_differs_from_cold_intentionally(self) -> None:
        workloads = (
            _workload("first", 0, _event(0, (0,))),
            _workload("second", 1, _event(0, (0,))),
        )
        sizes = {(0, 0): 4}

        cold = simulate_byte_cache_workloads(
            workloads, 4, sizes, "lru", "cold_per_workload"
        )
        persistent = simulate_byte_cache_workloads(
            workloads, 4, sizes, "lru", "persistent_sequence"
        )

        self.assertEqual((cold.hits, cold.misses), (0, 2))
        self.assertEqual((persistent.hits, persistent.misses), (1, 1))
        self.assertEqual(persistent.workloads[1].starting_resident_keys, ((0, 0),))
        self.assertEqual(persistent.workloads[1].hits, 1)
        self.assertEqual(persistent.shared_sequence_final_resident_keys, ((0, 0),))

    def test_persistent_aggregate_equals_one_concatenated_replay(self) -> None:
        workloads = (
            _workload("alpha", 3, _event(0, (0,)), _event(0, (1,), 1)),
            _workload("beta", 7, _event(0, (0,)), _event(0, (2,), 1)),
        )
        sizes = {(0, 0): 2, (0, 1): 3, (0, 2): 4}

        for policy in ("lru", "lfu"):
            with self.subTest(policy=policy):
                result = simulate_byte_cache_workloads(
                    workloads, 6, sizes, policy, "persistent_sequence"
                )
                concatenated = simulate_byte_cache(
                    tuple(event for item in workloads for event in item.events),
                    6,
                    sizes,
                    policy,
                )
                self.assertEqual(result.event_count, concatenated.event_count)
                self.assertEqual(
                    result.expert_request_count, concatenated.expert_request_count
                )
                self.assertEqual(result.hits, concatenated.hits)
                self.assertEqual(result.misses, concatenated.misses)
                self.assertEqual(
                    result.simulated_demand_load_bytes,
                    concatenated.simulated_demand_load_bytes,
                )
                self.assertEqual(result.eviction_count, concatenated.eviction_count)
                self.assertEqual(
                    result.simulated_evicted_bytes,
                    concatenated.simulated_evicted_bytes,
                )
                self.assertEqual(
                    result.shared_sequence_final_resident_keys,
                    concatenated.final_resident_keys,
                )

    def test_aggregate_counters_reconcile_exactly_with_workload_rows(self) -> None:
        workloads = (
            _workload("one", 0, _event(0, (0, 1))),
            _workload("two", 1, _event(0, (1, 2))),
        )
        result = simulate_byte_cache_workloads(
            workloads,
            6,
            {(0, 0): 2, (0, 1): 2, (0, 2): 2},
            "lru",
            "persistent_sequence",
        )

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
                getattr(result, field),
                sum(getattr(row, field) for row in result.workloads),
            )
        self.assertEqual(result.expert_request_count, result.hits + result.misses)

    def test_zero_request_hit_rates_are_explicitly_undefined(self) -> None:
        result = simulate_byte_cache_workloads(
            (_workload("empty", 0),),
            1,
            {},
            "lru",
            "persistent_sequence",
        )

        self.assertIsNone(result.hit_rate)
        self.assertIsNone(result.workloads[0].hit_rate)
        self.assertEqual(result.workloads[0].starting_resident_keys, ())
        self.assertEqual(result.workloads[0].ending_resident_keys, ())

    def test_layer_identity_and_atomic_working_set_remain_unchanged(self) -> None:
        layer_workloads = (
            _workload("layer-zero", 0, _event(0, (0,))),
            _workload("layer-one", 1, _event(1, (0,))),
        )
        result = simulate_byte_cache_workloads(
            layer_workloads,
            8,
            {(0, 0): 4, (1, 0): 4},
            "lru",
            "persistent_sequence",
        )
        self.assertEqual((result.hits, result.misses), (0, 2))
        self.assertEqual(
            result.shared_sequence_final_resident_keys, ((0, 0), (1, 0))
        )

        atomic = (_workload("atomic", 0, _event(0, (0, 1))),)
        with self.assertRaisesRegex(ValueError, "required working set 6 bytes"):
            simulate_byte_cache_workloads(
                atomic,
                5,
                {(0, 0): 2, (0, 1): 4},
                "lru",
                "persistent_sequence",
            )

    def test_invalid_identity_order_mode_and_inputs_reject(self) -> None:
        first = _workload("first", 0, _event(0, (0,)))
        second = _workload("second", 1, _event(0, (0,)))
        sizes = {(0, 0): 1}

        invalid_collections = (
            ((second, first), "ascending declared order"),
            ((first, _workload("first", 1)), "IDs must be unique"),
            ((first, _workload("second", 0)), "orders must be unique"),
        )
        for workloads, message in invalid_collections:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                simulate_byte_cache_workloads(
                    workloads, 1, sizes, "lru", "cold_per_workload"
                )

        with self.assertRaisesRegex(ValueError, "at least one workload"):
            simulate_byte_cache_workloads(
                (), 1, sizes, "lru", "cold_per_workload"
            )
        with self.assertRaisesRegex(ValueError, "unknown byte-cache lifecycle mode"):
            simulate_byte_cache_workloads(
                (first,), 1, sizes, "lru", "inferred-from-name"
            )
        with self.assertRaisesRegex(TypeError, "ByteCacheWorkload"):
            simulate_byte_cache_workloads(
                (object(),), 1, sizes, "lru", "cold_per_workload"
            )

        for workload_id in ("", " leading", "trailing ", "line\nbreak", None):
            with self.subTest(workload_id=workload_id), self.assertRaisesRegex(
                ValueError, "workload_id"
            ):
                ByteCacheWorkload(workload_id, 0, ())
        for order in (-1, True, 1.5):
            with self.subTest(order=order), self.assertRaisesRegex(
                ValueError, "workload order"
            ):
                ByteCacheWorkload("id", order, ())


if __name__ == "__main__":
    unittest.main()
