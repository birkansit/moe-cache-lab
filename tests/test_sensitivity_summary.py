from dataclasses import replace
from fractions import Fraction
import unittest

from moe_cache_lab.byte_cache import (
    CACHE_LIFECYCLE_MODES,
    ByteCacheWorkload,
    simulate_byte_cache_workloads,
)
from moe_cache_lab.sensitivity_summary import summarize_lifecycle_sensitivity
from moe_cache_lab.trace import RoutingEvent


def _events(*expert_ids: int) -> tuple[RoutingEvent, ...]:
    return tuple(
        RoutingEvent("generated", position, 0, (expert_id,))
        for position, expert_id in enumerate(expert_ids)
    )


def _grid(
    *,
    capacities: tuple[int, ...] = (1, 2, 3, 4),
    policies: tuple[str, ...] = ("lru", "lfu"),
    workloads: tuple[ByteCacheWorkload, ...] | None = None,
):
    if workloads is None:
        workloads = (
            ByteCacheWorkload("long", 3, _events(0, 1, 0, 2, 1, 2)),
            ByteCacheWorkload("short", 7, _events(0, 0)),
        )
    sizes = {(0, expert_id): 1 for expert_id in range(3)}
    return tuple(
        simulate_byte_cache_workloads(workloads, capacity, sizes, policy, mode)
        for capacity in capacities
        for policy in policies
        for mode in CACHE_LIFECYCLE_MODES
    )


class LifecycleSensitivitySummaryTests(unittest.TestCase):
    def test_adjacent_flat_and_nonflat_rows_are_exact_and_canonical(self) -> None:
        rows = _grid()
        summary = summarize_lifecycle_sensitivity(reversed(rows))

        self.assertEqual(summary.capacities_bytes, (1, 2, 3, 4))
        self.assertEqual(summary.policies, ("lru", "lfu"))
        self.assertEqual(summary.lifecycle_modes, CACHE_LIFECYCLE_MODES)
        self.assertEqual(
            [
                (item.lifecycle_mode, item.policy, item.lower_capacity_bytes)
                for item in summary.adjacent_capacity_comparisons
            ],
            [
                (mode, policy, capacity)
                for mode in CACHE_LIFECYCLE_MODES
                for policy in ("lru", "lfu")
                for capacity in (1, 2, 3)
            ],
        )
        lru_cold = [
            item
            for item in summary.adjacent_capacity_comparisons
            if item.lifecycle_mode == "cold_per_workload" and item.policy == "lru"
        ]
        self.assertEqual(
            (
                lru_cold[0].hit_delta,
                lru_cold[0].miss_delta,
                lru_cold[0].simulated_demand_load_byte_delta,
                lru_cold[0].hit_rate_delta,
                lru_cold[0].exact_flat,
            ),
            (2, -2, -2, Fraction(1, 4), False),
        )
        self.assertTrue(lru_cold[-1].exact_flat)
        self.assertEqual(
            (lru_cold[-1].hit_delta, lru_cold[-1].miss_delta), (0, 0)
        )

    def test_same_capacity_policy_delta_is_lfu_minus_lru(self) -> None:
        summary = summarize_lifecycle_sensitivity(_grid(capacities=(2,)))
        cold = next(
            item
            for item in summary.same_capacity_policy_comparisons
            if item.lifecycle_mode == "cold_per_workload"
        )
        self.assertEqual((cold.baseline_policy, cold.comparison_policy), ("lru", "lfu"))
        self.assertEqual(
            (
                cold.hit_delta,
                cold.miss_delta,
                cold.simulated_demand_load_byte_delta,
                cold.hit_rate_delta,
            ),
            (-1, 1, 1, Fraction(-1, 8)),
        )

    def test_workload_minimum_maximum_range_preserve_identity_and_order(self) -> None:
        summary = summarize_lifecycle_sensitivity(_grid(capacities=(2,), policies=("lru",)))
        row = next(
            item
            for item in summary.workload_sensitivity
            if item.lifecycle_mode == "cold_per_workload"
        )
        self.assertEqual(
            (row.hit_rate.minimum, row.hit_rate.maximum, row.hit_rate.range),
            (Fraction(1, 3), Fraction(1, 2), Fraction(1, 6)),
        )
        self.assertEqual(
            [(item.order, item.workload_id) for item in row.hit_rate.minimum_workloads],
            [(3, "long")],
        )
        self.assertEqual(
            [(item.order, item.workload_id) for item in row.hit_rate.maximum_workloads],
            [(7, "short")],
        )
        self.assertEqual((row.misses.minimum, row.misses.maximum, row.misses.range), (1, 4, 3))
        self.assertEqual(
            (
                row.simulated_demand_load_bytes.minimum,
                row.simulated_demand_load_bytes.maximum,
                row.simulated_demand_load_bytes.range,
            ),
            (1, 4, 3),
        )

    def test_zero_request_workloads_have_undefined_hit_rate_and_exact_zero_ranges(self) -> None:
        workloads = (
            ByteCacheWorkload("empty-a", 0, ()),
            ByteCacheWorkload("empty-b", 1, ()),
        )
        summary = summarize_lifecycle_sensitivity(
            _grid(capacities=(1,), policies=("lru",), workloads=workloads)
        )
        for row in summary.workload_sensitivity:
            self.assertEqual(row.defined_hit_rate_workload_count, 0)
            self.assertEqual(
                (row.hit_rate.minimum, row.hit_rate.maximum, row.hit_rate.range),
                (None, None, None),
            )
            self.assertEqual(row.hit_rate.minimum_workloads, ())
            self.assertEqual((row.misses.minimum, row.misses.maximum, row.misses.range), (0, 0, 0))
        for comparison in summary.same_capacity_policy_comparisons:
            self.assertIsNone(comparison.hit_rate_delta)

    def test_input_order_does_not_change_output_or_source_rows(self) -> None:
        rows = _grid()
        before = tuple(rows)
        self.assertEqual(
            summarize_lifecycle_sensitivity(rows),
            summarize_lifecycle_sensitivity(reversed(rows)),
        )
        self.assertEqual(rows, before)

    def test_incomplete_duplicate_and_incompatible_inputs_are_rejected(self) -> None:
        rows = _grid(capacities=(1, 2), policies=("lru",))
        with self.assertRaisesRegex(ValueError, "requires simulation rows"):
            summarize_lifecycle_sensitivity(())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_lifecycle_sensitivity(rows + (rows[0],))
        with self.assertRaisesRegex(ValueError, "incomplete rectangular grid"):
            summarize_lifecycle_sensitivity(rows[:-1])

        altered = replace(rows[-1], workloads=tuple(reversed(rows[-1].workloads)))
        with self.assertRaisesRegex(ValueError, "not in declared order"):
            summarize_lifecycle_sensitivity(rows[:-1] + (altered,))

        wrong_child = replace(
            rows[-1].workloads[0], lifecycle_mode="cold_per_workload"
        )
        altered = replace(
            rows[-1], workloads=(wrong_child,) + rows[-1].workloads[1:]
        )
        with self.assertRaisesRegex(ValueError, "incompatible with its parent"):
            summarize_lifecycle_sensitivity(rows[:-1] + (altered,))

    def test_unsupported_mode_and_policy_do_not_cross_comparison_boundaries(self) -> None:
        rows = _grid(capacities=(1,), policies=("lru",))
        with self.assertRaisesRegex(ValueError, "unsupported lifecycle modes"):
            summarize_lifecycle_sensitivity(
                rows + (replace(rows[0], lifecycle_mode="invented"),)
            )
        with self.assertRaisesRegex(ValueError, "unsupported policies"):
            summarize_lifecycle_sensitivity(
                tuple(replace(row, policy="oracle") for row in rows)
            )


if __name__ == "__main__":
    unittest.main()
