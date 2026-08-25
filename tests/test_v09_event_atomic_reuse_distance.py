import importlib.util
from fractions import Fraction
from pathlib import Path
import sys
import unittest

from moe_cache_lab.byte_cache import simulate_versioned_byte_cache
from moe_cache_lab.cache import simulate
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_event_atomic_lru_reuse_distance.py"
DESIGN = ROOT / "V09_EVENT_ATOMIC_REUSE_DISTANCE_DESIGN.md"
SPEC = importlib.util.spec_from_file_location("event_atomic_reuse_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


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
        model_id="synthetic/v09-event-atomic-reuse",
        num_experts=16,
        experts_per_token=None,
        events=events,
        created_at="synthetic",
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


def _decoder_v2_trace(events: tuple[RoutingEventV2, ...], *, width: int) -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="synthetic/v09-decoder-reuse",
        routing_stages=(
            RoutingStageProfile(
                routing_stage="decoder",
                num_experts=16,
                assigned_experts_per_token=width,
                allows_unassigned=True,
            ),
        ),
        events=events,
        created_at="synthetic",
    )


def _uniform_one_byte_sizes(trace: RoutingTraceV2) -> dict[tuple[str, int, int], int]:
    return {key: 1 for key in trace.expert_requests}


class EventAtomicLRUReuseDistanceTests(unittest.TestCase):
    def test_first_use_immediate_reuse_and_exact_fraction(self) -> None:
        trace = _v1_trace(
            (
                _v1_event(0, (0, 1)),
                _v1_event(1, (1,)),
            )
        )
        summary = AUDIT.analyze_versioned_trace(trace)

        self.assertEqual(summary.event_count, 2)
        self.assertEqual(summary.expert_request_count, 3)
        self.assertEqual(summary.distinct_referenced_key_count, 2)
        self.assertEqual(summary.first_use_count, 2)
        self.assertEqual(summary.reuse_count, 1)
        self.assertEqual(summary.max_atomic_event_entries, 2)
        self.assertEqual(summary.required_capacity_histogram, ((1, 1),))
        self.assertEqual(summary.min_required_capacity_entries, 1)
        self.assertEqual(summary.max_required_capacity_entries, 1)
        self.assertEqual(summary.first_capacity_with_observed_lru_hit_entries, 2)
        self.assertEqual(summary.predicted_lru_hits(2), 1)
        self.assertEqual(summary.reuse_fraction_within_capacity(2), Fraction(1, 1))
        with self.assertRaisesRegex(ValueError, "maximum atomic event"):
            summary.predicted_lru_hits(1)

    def test_same_event_tie_uses_stable_key_not_flattened_order(self) -> None:
        trace = _v1_trace(
            (
                _v1_event(0, (1, 0)),
                _v1_event(1, (2,)),
                _v1_event(2, (0, 1)),
            )
        )
        summary = AUDIT.analyze_versioned_trace(trace)
        samples = {sample.key: sample for sample in summary.samples if sample.event_index == 2}

        self.assertEqual(samples[(0, 0)].stack_distance_entries, 2)
        self.assertEqual(samples[(0, 0)].required_capacity_entries, 3)
        self.assertEqual(samples[(0, 1)].stack_distance_entries, 1)
        self.assertEqual(samples[(0, 1)].required_capacity_entries, 2)
        self.assertEqual(summary.required_capacity_histogram, ((2, 1), (3, 1)))

        flattened_last_access: dict[tuple[int, int], int] = {}
        serial_clock = 0
        for event in trace.events[:2]:
            for expert_id in event.selected_experts:
                flattened_last_access[(event.layer, expert_id)] = serial_clock
                serial_clock += 1
        flattened_rank = {
            key: 1
            + sum(
                other_clock > flattened_last_access[key]
                for other_key, other_clock in flattened_last_access.items()
                if other_key != key
            )
            for key in ((0, 0), (0, 1))
        }
        self.assertEqual(flattened_rank, {(0, 0): 2, (0, 1): 3})
        self.assertNotEqual(
            flattened_rank[(0, 0)], samples[(0, 0)].required_capacity_entries
        )
        self.assertNotEqual(
            flattened_rank[(0, 1)], samples[(0, 1)].required_capacity_entries
        )

    def test_selected_expert_tuple_order_is_invariant(self) -> None:
        left = _v1_trace(
            (
                _v1_event(0, (0, 1)),
                _v1_event(1, (2,)),
                _v1_event(2, (0, 1)),
            )
        )
        right = _v1_trace(
            (
                _v1_event(0, (1, 0)),
                _v1_event(1, (2,)),
                _v1_event(2, (1, 0)),
            )
        )
        self.assertEqual(
            AUDIT.analyze_versioned_trace(left),
            AUDIT.analyze_versioned_trace(right),
        )
        self.assertEqual(
            AUDIT.summary_payload(AUDIT.analyze_versioned_trace(left)),
            AUDIT.summary_payload(AUDIT.analyze_versioned_trace(right)),
        )

    def test_v1_layer_qualified_identity_is_distinct(self) -> None:
        trace = _v1_trace(
            (
                _v1_event(0, (1,), layer=0),
                _v1_event(1, (1,), layer=1),
                _v1_event(2, (1,), layer=0),
            )
        )
        summary = AUDIT.analyze_versioned_trace(trace)

        self.assertEqual(summary.distinct_referenced_key_count, 2)
        self.assertEqual(summary.first_use_count, 2)
        self.assertEqual(summary.reuse_count, 1)
        self.assertEqual(summary.samples[0].key, (0, 1))
        self.assertEqual(summary.samples[0].required_capacity_entries, 2)
        self.assertEqual(summary.predicted_lru_hits(1), 0)
        self.assertEqual(summary.predicted_lru_hits(2), 1)

    def test_v2_stage_qualified_identity_is_distinct(self) -> None:
        trace = RoutingTraceV2(
            model_id="synthetic/v09-stage-reuse",
            routing_stages=(
                RoutingStageProfile("encoder", 8, 1, False),
                RoutingStageProfile("decoder", 8, 1, False),
            ),
            events=(
                _v2_event("encoder", "source", 0, (1,)),
                _v2_event("decoder", "decoder_generated", 1, (1,)),
                _v2_event("decoder", "decoder_generated", 2, (1,)),
            ),
            created_at="synthetic",
        )
        summary = AUDIT.analyze_versioned_trace(trace)

        self.assertEqual(summary.distinct_referenced_key_count, 2)
        self.assertEqual(summary.first_use_count, 2)
        self.assertEqual(summary.reuse_count, 1)
        self.assertEqual(summary.samples[0].key, ("decoder", 0, 1))
        self.assertEqual(summary.samples[0].required_capacity_entries, 1)

        sizes = _uniform_one_byte_sizes(trace)
        simulation = simulate_versioned_byte_cache(trace, 1, sizes, "lru")
        self.assertEqual(summary.predicted_lru_hits(1), simulation.hits)
        self.assertEqual(simulation.hits, 1)

    def test_v2_unassigned_event_is_recency_inert(self) -> None:
        base = _decoder_v2_trace(
            (
                _v2_event("decoder", "decoder_generated", 0, (0,)),
                _v2_event("decoder", "decoder_generated", 2, (1,)),
                _v2_event("decoder", "decoder_generated", 3, (0,)),
            ),
            width=1,
        )
        with_unassigned = _decoder_v2_trace(
            (
                _v2_event("decoder", "decoder_generated", 0, (0,)),
                _v2_event(
                    "decoder",
                    "decoder_generated",
                    1,
                    unassigned=True,
                ),
                _v2_event("decoder", "decoder_generated", 2, (1,)),
                _v2_event("decoder", "decoder_generated", 3, (0,)),
            ),
            width=1,
        )
        base_summary = AUDIT.analyze_versioned_trace(base)
        unassigned_summary = AUDIT.analyze_versioned_trace(with_unassigned)

        self.assertEqual(unassigned_summary.event_count, base_summary.event_count + 1)
        for attribute in (
            "expert_request_count",
            "distinct_referenced_key_count",
            "first_use_count",
            "reuse_count",
            "max_atomic_event_entries",
            "required_capacity_histogram",
            "min_required_capacity_entries",
            "max_required_capacity_entries",
            "first_capacity_with_observed_lru_hit_entries",
        ):
            with self.subTest(attribute=attribute):
                self.assertEqual(
                    getattr(unassigned_summary, attribute),
                    getattr(base_summary, attribute),
                )
        for capacity in range(1, 3):
            self.assertEqual(
                unassigned_summary.predicted_lru_hits(capacity),
                base_summary.predicted_lru_hits(capacity),
            )

    def test_v1_oracle_matches_event_atomic_lru_at_every_feasible_capacity(self) -> None:
        traces = (
            _v1_trace(
                (
                    _v1_event(0, (1, 0)),
                    _v1_event(1, (2,)),
                    _v1_event(2, (0, 1)),
                    _v1_event(3, (3,)),
                    _v1_event(4, (1, 2)),
                )
            ),
            _v1_trace(
                tuple(
                    _v1_event(position, experts)
                    for position, experts in enumerate(
                        ((0,), (1,), (0,), (2,), (0,), (1,))
                    )
                )
            ),
        )
        for trace in traces:
            summary = AUDIT.analyze_versioned_trace(trace)
            for capacity in range(
                summary.max_atomic_event_entries,
                summary.distinct_referenced_key_count + 1,
            ):
                with self.subTest(trace=trace.events, capacity=capacity):
                    simulation = simulate(trace.events, capacity, "lru").combined
                    self.assertEqual(
                        summary.predicted_lru_hits(capacity), simulation.hits
                    )

    def test_v2_oracle_matches_uniform_byte_lru_at_every_feasible_capacity(self) -> None:
        trace = _decoder_v2_trace(
            (
                _v2_event("decoder", "decoder_generated", 0, (1, 0)),
                _v2_event(
                    "decoder",
                    "decoder_generated",
                    1,
                    unassigned=True,
                ),
                _v2_event("decoder", "decoder_generated", 2, (2, 3)),
                _v2_event("decoder", "decoder_generated", 3, (1, 3)),
                _v2_event("decoder", "decoder_generated", 4, (0, 2)),
            ),
            width=2,
        )
        summary = AUDIT.analyze_versioned_trace(trace)
        sizes = _uniform_one_byte_sizes(trace)

        for capacity in range(
            summary.max_atomic_event_entries,
            summary.distinct_referenced_key_count + 1,
        ):
            with self.subTest(capacity=capacity):
                simulation = simulate_versioned_byte_cache(
                    trace, capacity, sizes, "lru"
                )
                self.assertEqual(summary.predicted_lru_hits(capacity), simulation.hits)

    def test_empty_no_reuse_and_invalid_boundaries_are_explicit(self) -> None:
        empty = AUDIT.analyze_required_sets(())
        self.assertEqual(empty.event_count, 0)
        self.assertEqual(empty.expert_request_count, 0)
        self.assertEqual(empty.first_use_count, 0)
        self.assertEqual(empty.reuse_count, 0)
        self.assertEqual(empty.max_atomic_event_entries, 0)
        self.assertEqual(empty.required_capacity_histogram, ())
        self.assertIsNone(empty.first_capacity_with_observed_lru_hit_entries)
        self.assertIsNone(empty.reuse_fraction_within_capacity(1))

        one_event = AUDIT.analyze_required_sets((frozenset({(0, 0), (0, 1)}),))
        self.assertEqual(one_event.first_use_count, 2)
        self.assertEqual(one_event.reuse_count, 0)
        self.assertIsNone(one_event.min_required_capacity_entries)
        self.assertIsNone(one_event.reuse_fraction_within_capacity(2))

        all_first = AUDIT.analyze_required_sets(
            (
                frozenset({(0, 0)}),
                frozenset({(0, 1)}),
                frozenset({(0, 2)}),
            )
        )
        self.assertEqual(all_first.first_use_count, 3)
        self.assertEqual(all_first.reuse_count, 0)
        self.assertEqual(all_first.expert_request_count, 3)

        with self.assertRaisesRegex(TypeError, "one canonical key shape"):
            AUDIT.analyze_required_sets(
                (
                    frozenset({(0, 0)}),
                    frozenset({("decoder", 0, 0)}),
                )
            )

        with self.assertRaisesRegex(TypeError, "RoutingTrace"):
            AUDIT.analyze_versioned_trace([_v1_event(0, (0,))])

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _v1_trace(
                (
                    _v1_event(1, (0,)),
                    _v1_event(0, (1,)),
                )
            )

    def test_design_keeps_reuse_gap_and_claim_boundaries_distinct(self) -> None:
        text = DESIGN.read_text(encoding="utf-8")
        for required in (
            "`ReuseGapSummary` continues to answer event-spacing questions",
            "A later cache replay remains **SIMULATED**",
            "A later transfer-service result remains **ESTIMATED**",
            "not a recommendation",
            "does not define",
            "heterogeneous byte-aware reuse distance",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
