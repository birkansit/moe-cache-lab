from dataclasses import replace
from fractions import Fraction
import unittest
from unittest.mock import patch

from moe_cache_lab.byte_cache import (
    CACHE_LIFECYCLE_MODES,
    ByteCacheWorkload,
    simulate_byte_cache_workloads,
)
from moe_cache_lab.hardware_cost import (
    HardwareTransferProfile,
    TransferOperationPlan,
    estimate_lifecycle_transfer_costs,
)
from moe_cache_lab.sensitivity_summary import summarize_lifecycle_sensitivity
from moe_cache_lab.trace import RoutingEvent


def _events(*expert_ids: int) -> tuple[RoutingEvent, ...]:
    return tuple(
        RoutingEvent("generated", position, 0, (expert_id,))
        for position, expert_id in enumerate(expert_ids)
    )


def _simulations(
    *,
    capacities: tuple[int, ...] = (100, 200),
    policies: tuple[str, ...] = ("lru", "lfu"),
    workloads: tuple[ByteCacheWorkload, ...] | None = None,
):
    workloads = workloads or (
        ByteCacheWorkload("first", 2, _events(0, 1, 0)),
        ByteCacheWorkload("second", 8, _events(0)),
    )
    sizes = {(0, 0): 100, (0, 1): 100}
    return tuple(
        simulate_byte_cache_workloads(workloads, capacity, sizes, policy, mode)
        for capacity in capacities
        for policy in policies
        for mode in CACHE_LIFECYCLE_MODES
    )


class LifecycleTransferCostTests(unittest.TestCase):
    def test_hand_computed_exact_aggregate_and_workload_reconciliation(self) -> None:
        matrix = estimate_lifecycle_transfer_costs(
            _simulations(capacities=(100,), policies=("lru",)),
            (HardwareTransferProfile("hand", 600, 100_000_000),),
            (TransferOperationPlan("two", 2),),
        )
        cold, persistent = matrix.rows

        self.assertEqual(cold.lifecycle_mode, "cold_per_workload")
        self.assertEqual(
            (
                cold.estimate.simulated_demand_load_count,
                cold.estimate.simulated_demand_load_bytes,
                cold.estimate.modeled_transfer_operation_count,
                cold.estimate.estimated_payload_service_seconds,
                cold.estimate.estimated_setup_service_seconds,
                cold.estimate.estimated_serialized_transfer_service_seconds,
            ),
            (4, 400, 8, Fraction(2, 3), Fraction(4, 5), Fraction(22, 15)),
        )
        self.assertEqual(persistent.lifecycle_mode, "persistent_sequence")
        self.assertEqual(
            (
                persistent.estimate.simulated_demand_load_count,
                persistent.estimate.simulated_demand_load_bytes,
                persistent.estimate.modeled_transfer_operation_count,
                persistent.estimate.estimated_payload_service_seconds,
                persistent.estimate.estimated_setup_service_seconds,
                persistent.estimate.estimated_serialized_transfer_service_seconds,
            ),
            (3, 300, 6, Fraction(1, 2), Fraction(3, 5), Fraction(11, 10)),
        )
        for row in matrix.rows:
            for field in (
                "simulated_demand_load_count",
                "simulated_demand_load_bytes",
                "modeled_transfer_operation_count",
                "estimated_payload_service_seconds",
                "estimated_setup_service_seconds",
                "estimated_serialized_transfer_service_seconds",
            ):
                self.assertEqual(
                    getattr(row.estimate, field),
                    sum(getattr(item.estimate, field) for item in row.workloads),
                )

    def test_multi_operation_plan_changes_operation_count_and_setup_only(self) -> None:
        simulation_rows = _simulations(capacities=(100,), policies=("lru",))
        matrix = estimate_lifecycle_transfer_costs(
            simulation_rows,
            (HardwareTransferProfile("profile", 600, 100_000_000),),
            (TransferOperationPlan("two", 2), TransferOperationPlan("one", 1)),
        )
        one, two = matrix.rows[:2]
        self.assertEqual(one.transfer_operation_plan.name, "one")
        self.assertEqual(two.transfer_operation_plan.name, "two")
        self.assertEqual(one.estimate.simulated_demand_load_count, two.estimate.simulated_demand_load_count)
        self.assertEqual(one.estimate.simulated_demand_load_bytes, two.estimate.simulated_demand_load_bytes)
        self.assertEqual(one.estimate.estimated_payload_service_seconds, two.estimate.estimated_payload_service_seconds)
        self.assertEqual(two.estimate.modeled_transfer_operation_count, 2 * one.estimate.modeled_transfer_operation_count)
        self.assertEqual(two.estimate.estimated_setup_service_seconds, 2 * one.estimate.estimated_setup_service_seconds)

    def test_assumptions_change_estimates_not_source_cache_or_sensitivity(self) -> None:
        simulation_rows = _simulations(capacities=(100,), policies=("lru",))
        before_rows = tuple(simulation_rows)
        before_summary = summarize_lifecycle_sensitivity(simulation_rows)
        matrix = estimate_lifecycle_transfer_costs(
            simulation_rows,
            (
                HardwareTransferProfile("slow", 100, 100),
                HardwareTransferProfile("fast", 1_000, 10),
            ),
        )

        fast, slow = matrix.rows[:2]
        self.assertEqual(fast.hardware_profile.name, "fast")
        self.assertEqual(slow.hardware_profile.name, "slow")
        self.assertEqual(fast.estimate.simulated_demand_load_count, slow.estimate.simulated_demand_load_count)
        self.assertEqual(fast.estimate.simulated_demand_load_bytes, slow.estimate.simulated_demand_load_bytes)
        self.assertLess(
            fast.estimate.estimated_serialized_transfer_service_seconds,
            slow.estimate.estimated_serialized_transfer_service_seconds,
        )
        self.assertEqual(simulation_rows, before_rows)
        self.assertEqual(summarize_lifecycle_sensitivity(simulation_rows), before_summary)

    def test_canonical_order_is_capacity_policy_mode_profile_plan(self) -> None:
        matrix = estimate_lifecycle_transfer_costs(
            reversed(_simulations()),
            (
                HardwareTransferProfile("zeta", 100, 0),
                HardwareTransferProfile("alpha", 100, 0),
            ),
            (TransferOperationPlan("zeta", 2), TransferOperationPlan("alpha", 1)),
        )
        actual = [
            (
                row.estimate.cache_capacity_bytes,
                row.estimate.policy,
                row.lifecycle_mode,
                row.hardware_profile.name,
                row.transfer_operation_plan.name,
            )
            for row in matrix.rows
        ]
        expected = [
            (capacity, policy, mode, profile, plan)
            for capacity in (100, 200)
            for policy in ("lru", "lfu")
            for mode in CACHE_LIFECYCLE_MODES
            for profile in ("alpha", "zeta")
            for plan in ("alpha", "zeta")
        ]
        self.assertEqual(actual, expected)
        self.assertTrue(
            all(
                [(item.order, item.workload_id) for item in row.workloads]
                == [(2, "first"), (8, "second")]
                for row in matrix.rows
            )
        )

    def test_zero_request_and_zero_miss_rows_have_exact_zero_estimates(self) -> None:
        workloads = (
            ByteCacheWorkload("empty-a", 0, ()),
            ByteCacheWorkload("empty-b", 1, ()),
        )
        matrix = estimate_lifecycle_transfer_costs(
            _simulations(capacities=(100,), policies=("lru",), workloads=workloads),
            (HardwareTransferProfile("zero", 123, 999),),
        )
        for row in matrix.rows:
            self.assertEqual(row.estimate.simulated_demand_load_count, 0)
            self.assertEqual(row.estimate.simulated_demand_load_bytes, 0)
            self.assertEqual(row.estimate.modeled_transfer_operation_count, 0)
            self.assertEqual(row.estimate.estimated_payload_service_seconds, Fraction(0, 1))
            self.assertEqual(row.estimate.estimated_setup_service_seconds, Fraction(0, 1))
            self.assertEqual(row.estimate.estimated_serialized_transfer_service_seconds, Fraction(0, 1))

    def test_nonempty_zero_miss_accounting_has_zero_service(self) -> None:
        source = _simulations(capacities=(100,), policies=("lru",))
        rows = tuple(
            replace(
                row,
                hits=row.expert_request_count,
                misses=0,
                simulated_demand_load_bytes=0,
                workloads=tuple(
                    replace(
                        workload,
                        hits=workload.expert_request_count,
                        misses=0,
                        simulated_demand_load_bytes=0,
                    )
                    for workload in row.workloads
                ),
            )
            for row in source
        )
        matrix = estimate_lifecycle_transfer_costs(
            rows, (HardwareTransferProfile("zero-miss", 100, 1_000),)
        )
        self.assertTrue(all(row.estimate.simulated_demand_load_count == 0 for row in matrix.rows))
        self.assertTrue(
            all(
                row.estimate.estimated_serialized_transfer_service_seconds
                == Fraction(0, 1)
                for row in matrix.rows
            )
        )

    def test_estimator_does_not_invoke_cache_simulation(self) -> None:
        simulation_rows = _simulations(capacities=(100,), policies=("lru",))
        with patch(
            "moe_cache_lab.hardware_cost.simulate_byte_cache",
            side_effect=AssertionError("lifecycle estimates must not replay cache"),
        ):
            matrix = estimate_lifecycle_transfer_costs(
                simulation_rows,
                (HardwareTransferProfile("profile", 100, 0),),
            )
        self.assertEqual(len(matrix.rows), 2)

    def test_invalid_incomplete_and_incompatible_inputs_reject(self) -> None:
        rows = _simulations(capacities=(100,), policies=("lru",))
        larger_grid = _simulations(capacities=(100, 200), policies=("lru",))
        profile = (HardwareTransferProfile("profile", 100, 0),)
        with self.assertRaisesRegex(ValueError, "requires simulation rows"):
            estimate_lifecycle_transfer_costs((), profile)
        with self.assertRaisesRegex(TypeError, "ByteCacheLifecycleSimulation"):
            estimate_lifecycle_transfer_costs((object(),), profile)
        with self.assertRaisesRegex(ValueError, "incomplete rectangular grid"):
            estimate_lifecycle_transfer_costs(larger_grid[:-1], profile)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            estimate_lifecycle_transfer_costs(rows + (rows[0],), profile)
        altered_workload = replace(rows[-1].workloads[0], policy="lfu")
        altered = replace(
            rows[-1], workloads=(altered_workload,) + rows[-1].workloads[1:]
        )
        with self.assertRaisesRegex(ValueError, "incompatible with its parent"):
            estimate_lifecycle_transfer_costs(rows[:-1] + (altered,), profile)
        changed_identity = replace(
            rows[-1].workloads[0], workload_id="changed"
        )
        altered = replace(
            rows[-1], workloads=(changed_identity,) + rows[-1].workloads[1:]
        )
        with self.assertRaisesRegex(ValueError, "incompatible workload identities"):
            estimate_lifecycle_transfer_costs(rows[:-1] + (altered,), profile)
        with self.assertRaisesRegex(ValueError, "at least one hardware profile"):
            estimate_lifecycle_transfer_costs(rows, ())
        with self.assertRaisesRegex(ValueError, "at least one transfer operation plan"):
            estimate_lifecycle_transfer_costs(rows, profile, ())


if __name__ == "__main__":
    unittest.main()
