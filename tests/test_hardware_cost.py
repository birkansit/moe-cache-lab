import inspect
import unittest
from fractions import Fraction

from moe_cache_lab import hardware_cost
from moe_cache_lab.byte_cache import ByteCacheSimulation
from moe_cache_lab.hardware_cost import (
    HardwareTransferProfile,
    estimate_transfer_cost,
    run_transfer_sensitivity_sweep,
)
from moe_cache_lab.trace import RoutingEvent


def event(
    layer: int,
    experts: tuple[int, ...],
    position: int,
    phase: str = "prompt",
) -> RoutingEvent:
    return RoutingEvent(phase, position, layer, experts)


def simulation(
    *,
    policy: str = "lru",
    capacity_bytes: int = 512,
    misses: int = 3,
    demand_bytes: int = 300,
    evicted_bytes: int = 0,
) -> ByteCacheSimulation:
    return ByteCacheSimulation(
        policy=policy,
        capacity_bytes=capacity_bytes,
        event_count=3,
        expert_request_count=3,
        hits=3 - misses,
        misses=misses,
        simulated_demand_load_bytes=demand_bytes,
        eviction_count=1 if evicted_bytes else 0,
        simulated_evicted_bytes=evicted_bytes,
        peak_resident_bytes=min(capacity_bytes, 300),
        final_resident_bytes=min(capacity_bytes, 300),
        final_resident_keys=(),
    )


class HardwareTransferCostTests(unittest.TestCase):
    def test_exact_payload_service_time_uses_fraction(self) -> None:
        profile = HardwareTransferProfile("hand", 600, 0)
        estimate = estimate_transfer_cost(
            simulation(demand_bytes=300, misses=3),
            profile,
        )

        self.assertEqual(estimate.hardware_profile_name, "hand")
        self.assertEqual(estimate.estimated_payload_service_seconds, Fraction(1, 2))
        self.assertIsInstance(estimate.estimated_payload_service_seconds, Fraction)

    def test_exact_setup_service_time_uses_miss_count(self) -> None:
        profile = HardwareTransferProfile("setup", 600, 100_000_000)
        estimate = estimate_transfer_cost(
            simulation(demand_bytes=0, misses=3),
            profile,
        )

        self.assertEqual(estimate.simulated_demand_load_count, 3)
        self.assertEqual(estimate.estimated_setup_service_seconds, Fraction(3, 10))

    def test_exact_serialized_total_is_payload_plus_setup(self) -> None:
        profile = HardwareTransferProfile("combined", 600, 100_000_000)
        estimate = estimate_transfer_cost(
            simulation(demand_bytes=300, misses=3),
            profile,
        )

        self.assertEqual(estimate.estimated_payload_service_seconds, Fraction(1, 2))
        self.assertEqual(estimate.estimated_setup_service_seconds, Fraction(3, 10))
        self.assertEqual(
            estimate.estimated_serialized_transfer_service_seconds,
            Fraction(4, 5),
        )
        self.assertEqual(
            estimate.estimated_serialized_transfer_service_seconds,
            estimate.estimated_payload_service_seconds
            + estimate.estimated_setup_service_seconds,
        )

    def test_zero_miss_simulation_has_zero_estimated_service_time(self) -> None:
        estimate = estimate_transfer_cost(
            simulation(misses=0, demand_bytes=0),
            HardwareTransferProfile("zero", 123, 987_654_321),
        )

        self.assertEqual(estimate.estimated_payload_service_seconds, Fraction(0, 1))
        self.assertEqual(estimate.estimated_setup_service_seconds, Fraction(0, 1))
        self.assertEqual(
            estimate.estimated_serialized_transfer_service_seconds,
            Fraction(0, 1),
        )

    def test_profiles_change_estimate_not_simulated_cache_outcome(self) -> None:
        events = [
            event(0, (0,), 0),
            event(0, (1,), 1),
            event(0, (0,), 2),
        ]
        sizes = {(0, 0): 100, (0, 1): 100}
        fast = HardwareTransferProfile("fast", 1_000, 0)
        slow = HardwareTransferProfile("slow", 100, 0)

        sweep = run_transfer_sensitivity_sweep(
            events,
            sizes,
            capacities_bytes=[100],
            policies=["lru"],
            hardware_profiles=[slow, fast],
        )

        self.assertEqual(len(sweep.rows), 2)
        fast_row, slow_row = sweep.rows
        self.assertEqual(fast_row.hardware_profile.name, "fast")
        self.assertEqual(slow_row.hardware_profile.name, "slow")
        self.assertEqual(fast_row.simulation, slow_row.simulation)
        self.assertLess(
            fast_row.estimate.estimated_serialized_transfer_service_seconds,
            slow_row.estimate.estimated_serialized_transfer_service_seconds,
        )

    def test_sweep_rows_have_documented_canonical_order(self) -> None:
        profiles = [
            HardwareTransferProfile("zeta", 1_000, 1),
            HardwareTransferProfile("alpha", 2_000, 2),
        ]
        sweep = run_transfer_sensitivity_sweep(
            [event(0, (0,), 0), event(0, (1,), 1)],
            {(0, 0): 2, (0, 1): 2},
            capacities_bytes=[6, 2],
            policies=["lfu", "lru"],
            hardware_profiles=profiles,
        )

        keys = [
            (
                row.simulation.capacity_bytes,
                row.simulation.policy,
                row.hardware_profile.name,
            )
            for row in sweep.rows
        ]
        self.assertEqual(
            keys,
            [
                (2, "lru", "alpha"),
                (2, "lru", "zeta"),
                (2, "lfu", "alpha"),
                (2, "lfu", "zeta"),
                (6, "lru", "alpha"),
                (6, "lru", "zeta"),
                (6, "lfu", "alpha"),
                (6, "lfu", "zeta"),
            ],
        )

    def test_workload_context_uses_layer_qualified_identity_and_supplied_sizes(self) -> None:
        sweep = run_transfer_sensitivity_sweep(
            [
                event(0, (1, 2), 0),
                event(1, (1,), 1),
            ],
            {(0, 1): 2, (0, 2): 4, (1, 1): 3},
            capacities_bytes=[6],
            policies=["lru"],
            hardware_profiles=[HardwareTransferProfile("p", 100, 0)],
        )

        context = sweep.workload_context
        self.assertEqual(context.unique_referenced_expert_count, 3)
        self.assertEqual(context.unique_referenced_expert_bytes, 9)
        self.assertEqual(context.maximum_atomic_event_working_set_bytes, 6)
        self.assertEqual(
            context.referenced_expert_keys,
            ((0, 1), (0, 2), (1, 1)),
        )

    def test_input_order_and_size_map_insertion_order_do_not_change_sweep(self) -> None:
        events = [
            event(0, (0, 1), 0),
            event(0, (2,), 1),
            event(0, (0,), 2),
        ]
        left_sizes = {(0, 0): 2, (0, 1): 3, (0, 2): 4, (9, 9): 999}
        right_sizes = {(9, 9): 999, (0, 2): 4, (0, 1): 3, (0, 0): 2}
        alpha = HardwareTransferProfile("alpha", 1_000, 10)
        zeta = HardwareTransferProfile("zeta", 2_000, 20)

        left = run_transfer_sensitivity_sweep(
            events,
            left_sizes,
            capacities_bytes=[9, 5, 9],
            policies=["lfu", "lru", "lfu"],
            hardware_profiles=[zeta, alpha],
        )
        right = run_transfer_sensitivity_sweep(
            events,
            right_sizes,
            capacities_bytes=[5, 9],
            policies=["lru", "lfu"],
            hardware_profiles=[alpha, zeta],
        )

        self.assertEqual(left, right)

    def test_duplicate_capacities_and_policies_normalize_but_profile_names_reject(self) -> None:
        profile = HardwareTransferProfile("only", 100, 0)
        sweep = run_transfer_sensitivity_sweep(
            [event(0, (0,), 0)],
            {(0, 0): 1},
            capacities_bytes=[2, 2],
            policies=["lfu", "lru", "lfu"],
            hardware_profiles=[profile],
        )

        self.assertEqual(sweep.capacities_bytes, (2,))
        self.assertEqual(sweep.policies, ("lru", "lfu"))
        self.assertEqual(len(sweep.rows), 2)

        with self.assertRaisesRegex(ValueError, "profile names must be unique"):
            run_transfer_sensitivity_sweep(
                [event(0, (0,), 0)],
                {(0, 0): 1},
                capacities_bytes=[2],
                policies=["lru"],
                hardware_profiles=[
                    HardwareTransferProfile("dup", 100, 0),
                    HardwareTransferProfile("dup", 200, 1),
                ],
            )

    def test_hardware_profile_validation_rejects_invalid_inputs(self) -> None:
        for name in ("", " leading", "trailing ", "line\nbreak", None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "profile name"):
                HardwareTransferProfile(name, 100, 0)

        for bandwidth in (0, -1, 1.5, True):
            with self.subTest(bandwidth=bandwidth), self.assertRaisesRegex(
                ValueError, "payload bandwidth"
            ):
                HardwareTransferProfile("p", bandwidth, 0)

        for latency in (-1, 1.5, True):
            with self.subTest(latency=latency), self.assertRaisesRegex(
                ValueError, "setup latency"
            ):
                HardwareTransferProfile("p", 100, latency)

        self.assertEqual(
            HardwareTransferProfile("zero-setup", 100, 0).setup_latency_ns_per_loaded_expert,
            0,
        )

    def test_sweep_rejects_any_capacity_below_maximum_atomic_working_set(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "sweep capacity 4 bytes is smaller than maximum atomic event working set 5 bytes",
        ):
            run_transfer_sensitivity_sweep(
                [event(0, (0, 1), 0)],
                {(0, 0): 2, (0, 1): 3},
                capacities_bytes=[5, 4],
                policies=["lru"],
                hardware_profiles=[HardwareTransferProfile("p", 100, 0)],
            )

    def test_eviction_bytes_are_not_added_to_transfer_service_cost(self) -> None:
        estimate = estimate_transfer_cost(
            simulation(
                misses=1,
                demand_bytes=100,
                evicted_bytes=10_000,
            ),
            HardwareTransferProfile("immutable", 100, 0),
        )

        self.assertEqual(estimate.simulated_demand_load_bytes, 100)
        self.assertEqual(estimate.estimated_payload_service_seconds, Fraction(1, 1))
        self.assertEqual(
            estimate.estimated_serialized_transfer_service_seconds,
            Fraction(1, 1),
        )

    def test_claim_boundaries_are_explicit_in_public_docs(self) -> None:
        module_docs = " ".join((inspect.getdoc(hardware_cost) or "").lower().split())
        estimate_docs = " ".join(
            (inspect.getdoc(hardware_cost.TransferCostEstimate) or "").lower().split()
        )
        context_docs = " ".join(
            (inspect.getdoc(hardware_cost.WorkloadByteContext) or "").lower().split()
        )

        self.assertIn("simulated cache-model", module_docs)
        self.assertIn("estimated, not measured", module_docs)
        self.assertIn("serialized/no-overlap", module_docs)
        self.assertIn("not establish physical gpu residency, latency, throughput, or speedup", module_docs)
        self.assertIn("estimated serialized transfer-service", estimate_docs)
        self.assertIn("not measured physical gpu residency", context_docs)

    def test_estimator_and_sweep_reject_wrong_public_input_types(self) -> None:
        with self.assertRaisesRegex(TypeError, "ByteCacheSimulation"):
            estimate_transfer_cost(
                object(),
                HardwareTransferProfile("p", 100, 0),
            )
        with self.assertRaisesRegex(TypeError, "HardwareTransferProfile"):
            estimate_transfer_cost(simulation(), object())

        with self.assertRaisesRegex(ValueError, "at least one byte capacity"):
            run_transfer_sensitivity_sweep(
                [],
                {},
                capacities_bytes=[],
                policies=["lru"],
                hardware_profiles=[HardwareTransferProfile("p", 100, 0)],
            )
        with self.assertRaisesRegex(ValueError, "at least one policy"):
            run_transfer_sensitivity_sweep(
                [],
                {},
                capacities_bytes=[1],
                policies=[],
                hardware_profiles=[HardwareTransferProfile("p", 100, 0)],
            )
        with self.assertRaisesRegex(ValueError, "at least one hardware profile"):
            run_transfer_sensitivity_sweep(
                [],
                {},
                capacities_bytes=[1],
                policies=["lru"],
                hardware_profiles=[],
            )


if __name__ == "__main__":
    unittest.main()
