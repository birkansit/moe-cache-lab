import importlib.util
from fractions import Fraction
from pathlib import Path
import sys
import unittest

from moe_cache_lab.byte_cache import simulate_versioned_byte_cache
from moe_cache_lab.cache import simulate
from moe_cache_lab.capacity_frontier import (
    CapacityFrontierInputError,
    analyze_capacity_frontier,
)
from moe_cache_lab.capacity_frontier_output import (
    capacity_frontier_data,
    render_capacity_frontier_json,
    render_capacity_frontier_report,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_oracle(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B1 = _load_oracle(
    "capacity_frontier_count_oracle",
    "audit_event_atomic_lru_reuse_distance.py",
)
B2 = _load_oracle(
    "capacity_frontier_byte_oracle",
    "audit_event_atomic_lru_byte_reuse_distance.py",
)


def _v1_event(
    position: int,
    experts: tuple[int, ...],
    *,
    layer: int = 0,
) -> RoutingEvent:
    return RoutingEvent(
        phase="generated",
        token_position=position,
        layer=layer,
        selected_experts=experts,
    )


def _v1_trace(events: tuple[RoutingEvent, ...]) -> RoutingTrace:
    return RoutingTrace(
        model_id="synthetic/capacity-frontier",
        num_experts=32,
        experts_per_token=None,
        events=events,
        capture_method="synthetic-test",
        created_at="synthetic",
        model_revision="test-revision",
    )


def _v2_event(
    stage: str,
    phase: str,
    position: int,
    experts: tuple[int, ...] = (),
    *,
    layer: int = 0,
    unassigned: bool = False,
) -> RoutingEventV2:
    return RoutingEventV2(
        routing_stage=stage,
        phase=phase,
        token_position=position,
        layer=layer,
        assignment_state="unassigned" if unassigned else "assigned",
        selected_experts=() if unassigned else experts,
        selected_probabilities=(),
        unassigned_reason="capacity" if unassigned else None,
    )


def _v2_trace(
    events: tuple[RoutingEventV2, ...],
    *,
    stages: tuple[str, ...] = ("decoder",),
    width: int = 1,
) -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="synthetic/capacity-frontier-v2",
        routing_stages=tuple(
            RoutingStageProfile(
                routing_stage=stage,
                num_experts=32,
                assigned_experts_per_token=width,
                allows_unassigned=True,
            )
            for stage in stages
        ),
        events=events,
        capture_method="synthetic-test",
        created_at="synthetic",
        model_revision="test-revision",
    )


class CapacityFrontierTests(unittest.TestCase):
    def test_hand_computed_count_frontier_has_exact_breakpoints(self) -> None:
        trace = _v1_trace(
            tuple(
                _v1_event(index, experts)
                for index, experts in enumerate(((0,), (0,), (1,), (0,), (2,), (1,)))
            )
        )
        result = analyze_capacity_frontier(trace)
        frontier = result.count_frontier

        self.assertEqual(result.expert_request_count, 6)
        self.assertEqual(result.first_use_count, 3)
        self.assertEqual(result.reuse_count, 3)
        self.assertEqual(result.compulsory_first_use_miss_fraction, Fraction(1, 2))
        self.assertEqual(
            result.maximum_reachable_cold_start_hit_fraction,
            Fraction(1, 2),
        )
        self.assertEqual(frontier.referenced_working_set, 3)
        self.assertEqual(frontier.feasibility_floor, 1)
        self.assertEqual(frontier.required_capacity_histogram, ((1, 1), (2, 1), (3, 1)))
        self.assertEqual(
            tuple((point.capacity, point.hits, point.misses) for point in frontier.breakpoints),
            ((1, 1, 5), (2, 2, 4), (3, 3, 3)),
        )
        self.assertEqual(frontier.first_feasible_capacity_with_observed_hit, 1)

    def test_thresholds_below_floor_collapse_into_one_floor_point(self) -> None:
        trace = _v1_trace((_v1_event(0, (0, 1)), _v1_event(1, (1,))))
        frontier = analyze_capacity_frontier(trace).count_frontier

        self.assertEqual(frontier.feasibility_floor, 2)
        self.assertEqual(frontier.required_capacity_histogram, ((1, 1),))
        self.assertEqual(
            tuple((point.capacity, point.hits) for point in frontier.breakpoints),
            ((2, 1),),
        )
        self.assertEqual(frontier.first_feasible_capacity_with_observed_hit, 2)

    def test_no_reuse_and_zero_request_boundaries_are_explicit(self) -> None:
        no_reuse = analyze_capacity_frontier(
            _v1_trace((_v1_event(0, (0,)), _v1_event(1, (1,))))
        )
        self.assertEqual(no_reuse.reuse_count, 0)
        self.assertIsNone(
            no_reuse.count_frontier.first_feasible_capacity_with_observed_hit
        )
        self.assertEqual(len(no_reuse.count_frontier.breakpoints), 1)
        self.assertTrue(all(target.unattainable for target in no_reuse.count_frontier.targets))

        zero_trace = _v2_trace(
            (_v2_event("decoder", "decoder_prompt", 0, unassigned=True),)
        )
        zero = analyze_capacity_frontier(zero_trace, {})
        self.assertEqual(zero.expert_request_count, 0)
        self.assertIsNone(zero.compulsory_first_use_miss_fraction)
        self.assertIsNone(zero.maximum_reachable_cold_start_hit_fraction)
        self.assertEqual(zero.count_frontier.feasibility_floor, 0)
        point = zero.count_frontier.breakpoints[0]
        self.assertEqual((point.capacity, point.hits, point.misses), (0, 0, 0))
        self.assertIsNone(point.hit_fraction)
        self.assertIsNone(point.miss_fraction)
        self.assertTrue(
            all(target.status == "unavailable" for target in zero.count_frontier.targets)
        )
        self.assertTrue(
            all(not target.unattainable for target in zero.count_frontier.targets)
        )
        assert zero.byte_frontier is not None
        self.assertEqual(zero.byte_frontier.referenced_working_set, 0)
        self.assertEqual(zero.byte_frontier.feasibility_floor, 0)
        self.assertTrue(
            all(target.status == "unavailable" for target in zero.byte_frontier.targets)
        )

    def test_stable_ties_tuple_order_and_identity_namespaces(self) -> None:
        left = _v1_trace(
            (
                _v1_event(0, (1, 0)),
                _v1_event(1, (2,)),
                _v1_event(2, (0, 1)),
            )
        )
        right = _v1_trace(
            (
                _v1_event(0, (0, 1)),
                _v1_event(1, (2,)),
                _v1_event(2, (1, 0)),
            )
        )
        self.assertEqual(
            analyze_capacity_frontier(left).count_frontier.required_capacity_histogram,
            ((2, 1), (3, 1)),
        )
        self.assertEqual(
            analyze_capacity_frontier(left), analyze_capacity_frontier(right)
        )

        layers = _v1_trace(
            (
                _v1_event(0, (0,), layer=0),
                _v1_event(1, (0,), layer=1),
            )
        )
        self.assertEqual(analyze_capacity_frontier(layers).first_use_count, 2)

        stages = _v2_trace(
            (
                _v2_event("encoder", "source", 0, (0,)),
                _v2_event("decoder", "decoder_prompt", 0, (0,)),
            ),
            stages=("encoder", "decoder"),
        )
        self.assertEqual(analyze_capacity_frontier(stages).first_use_count, 2)

    def test_count_matches_b1_and_simulator_across_feasible_capacities(self) -> None:
        traces = (
            _v1_trace(tuple(_v1_event(i, (key,)) for i, key in enumerate((0, 1, 0, 2, 1, 0)))),
            _v1_trace(
                (
                    _v1_event(0, (0, 1)),
                    _v1_event(1, (2,)),
                    _v1_event(2, (0, 1)),
                )
            ),
        )
        for trace in traces:
            with self.subTest(events=len(trace.events)):
                result = analyze_capacity_frontier(trace)
                oracle = B1.analyze_versioned_trace(trace)
                frontier = result.count_frontier
                self.assertEqual(
                    frontier.required_capacity_histogram,
                    oracle.required_capacity_histogram,
                )
                self.assertEqual(frontier.feasibility_floor, oracle.max_atomic_event_entries)
                for capacity in range(
                    frontier.feasibility_floor,
                    frontier.referenced_working_set + 1,
                ):
                    self.assertEqual(
                        frontier.exact_hits_at(capacity),
                        oracle.predicted_lru_hits(capacity),
                    )
                    self.assertEqual(
                        frontier.exact_hits_at(capacity),
                        simulate(trace.events, capacity, "lru").combined.hits,
                    )

        v2 = _v2_trace(
            tuple(
                _v2_event("decoder", "decoder_generated", i, (key,))
                for i, key in enumerate((0, 1, 0, 2, 1, 0))
            )
        )
        self.assertEqual(
            analyze_capacity_frontier(v2).count_frontier.required_capacity_histogram,
            B1.analyze_versioned_trace(v2).required_capacity_histogram,
        )

    def test_heterogeneous_bytes_match_b2_and_byte_simulator(self) -> None:
        traces_and_sizes = (
            (
                _v1_trace(
                    (
                        _v1_event(0, (1, 0)),
                        _v1_event(1, (2,)),
                        _v1_event(2, (0, 1)),
                    )
                ),
                {(0, 0): 2, (0, 1): 5, (0, 2): 3},
            ),
            (
                _v1_trace(tuple(_v1_event(i, (key,)) for i, key in enumerate((0, 1, 0, 2, 1, 0)))),
                {(0, 0): 1, (0, 1): 3, (0, 2): 2},
            ),
            (
                _v2_trace(
                    tuple(
                        _v2_event("decoder", "decoder_generated", i, (key,))
                        for i, key in enumerate((0, 1, 0, 2, 1, 0))
                    )
                ),
                {
                    ("decoder", 0, 0): 2,
                    ("decoder", 0, 1): 5,
                    ("decoder", 0, 2): 3,
                },
            ),
        )
        for trace, sizes in traces_and_sizes:
            with self.subTest(sizes=sizes):
                result = analyze_capacity_frontier(trace, sizes)
                frontier = result.byte_frontier
                assert frontier is not None
                oracle = B2.analyze_versioned_trace(trace, sizes)
                self.assertEqual(
                    frontier.required_capacity_histogram,
                    oracle.required_capacity_histogram,
                )
                self.assertEqual(frontier.feasibility_floor, oracle.max_atomic_event_bytes)
                self.assertEqual(
                    frontier.referenced_working_set,
                    oracle.referenced_working_set_bytes,
                )
                for capacity in range(
                    frontier.feasibility_floor,
                    frontier.referenced_working_set + 1,
                ):
                    self.assertEqual(
                        frontier.exact_hits_at(capacity),
                        oracle.predicted_lru_hits(capacity),
                    )
                    self.assertEqual(
                        frontier.exact_hits_at(capacity),
                        simulate_versioned_byte_cache(
                            trace, capacity, sizes, "lru"
                        ).hits,
                    )

    def test_uniform_sizes_scale_count_frontier_and_extra_sizes_are_inert(self) -> None:
        trace = _v1_trace(
            tuple(_v1_event(i, (key,)) for i, key in enumerate((0, 1, 0, 2, 1, 0)))
        )
        scale = 7
        sizes = {(0, key): scale for key in (0, 1, 2)}
        result = analyze_capacity_frontier(trace, sizes)
        byte = result.byte_frontier
        assert byte is not None
        count = result.count_frontier
        self.assertEqual(byte.referenced_working_set, count.referenced_working_set * scale)
        self.assertEqual(byte.feasibility_floor, count.feasibility_floor * scale)
        self.assertEqual(
            byte.required_capacity_histogram,
            tuple((capacity * scale, hits) for capacity, hits in count.required_capacity_histogram),
        )
        self.assertEqual(
            byte.breakpoints,
            tuple(
                type(point)(
                    capacity=point.capacity * scale,
                    hits=point.hits,
                    misses=point.misses,
                    hit_fraction=point.hit_fraction,
                    miss_fraction=point.miss_fraction,
                )
                for point in count.breakpoints
            ),
        )
        with_extra = analyze_capacity_frontier(trace, {**sizes, (9, 9): 999})
        self.assertEqual(result, with_extra)

    def test_missing_or_invalid_size_inputs_fail_without_inference(self) -> None:
        trace = _v1_trace((_v1_event(0, (0,)),))
        with self.assertRaisesRegex(CapacityFrontierInputError, "missing referenced"):
            analyze_capacity_frontier(trace, {})
        with self.assertRaisesRegex(CapacityFrontierInputError, "positive integer"):
            analyze_capacity_frontier(trace, {(0, 0): 0})
        with self.assertRaisesRegex(CapacityFrontierInputError, "identity shape"):
            analyze_capacity_frontier(trace, {("decoder", 0, 0): 1})

    def test_fixed_targets_are_minimal_or_explicitly_unattainable(self) -> None:
        trace = _v1_trace(
            tuple(
                _v1_event(index, (key,))
                for index, key in enumerate((0, 0, 1, 0, 2, 1))
            )
        )
        result = analyze_capacity_frontier(
            trace,
            {(0, 0): 2, (0, 1): 5, (0, 2): 3},
        )
        frontier = result.count_frontier
        assert result.byte_frontier is not None
        for mode in (frontier, result.byte_frontier):
            for target in mode.targets:
                if target.unattainable:
                    self.assertIsNone(target.capacity)
                    self.assertGreater(target.required_hits, 3)
                    continue
                assert target.capacity is not None and target.required_hits is not None
                self.assertGreaterEqual(mode.exact_hits_at(target.capacity), target.required_hits)
                if target.capacity > mode.feasibility_floor:
                    self.assertLess(
                        mode.exact_hits_at(target.capacity - 1),
                        target.required_hits,
                    )
        self.assertEqual(
            tuple((target.label, target.status, target.capacity) for target in frontier.targets),
            (
                ("25%", "reached", 2),
                ("50%", "reached", 3),
                ("75%", "unattainable", None),
                ("90%", "unattainable", None),
            ),
        )

    def test_output_is_deterministic_exact_and_bounded(self) -> None:
        trace = _v1_trace((_v1_event(0, (0,)), _v1_event(1, (0,))))
        result = analyze_capacity_frontier(trace, {(0, 0): 11})
        first = render_capacity_frontier_json(result)
        self.assertEqual(first, render_capacity_frontier_json(result))
        data = capacity_frontier_data(result)
        self.assertEqual(data["format"], "moe-cache-lab.capacity-frontier")
        self.assertEqual(data["format_version"], 1)
        self.assertEqual(
            data["maximum_reachable_cold_start_hit_fraction"],
            {"numerator": 1, "denominator": 2},
        )
        self.assertEqual(len(data["count_frontier"]["breakpoints"]), 1)
        report = render_capacity_frontier_report(result)
        self.assertIn("First feasible capacity with any observed LRU reuse hit", report)
        self.assertIn("SIMULATED", report)
        self.assertIn("caller-supplied cache-model inputs", report)
        self.assertNotIn("recommended capacity", report.lower())


if __name__ == "__main__":
    unittest.main()
