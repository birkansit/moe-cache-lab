import importlib.util
from fractions import Fraction
from pathlib import Path
import sys
import unittest

from moe_cache_lab.byte_cache import simulate_versioned_byte_cache
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_event_atomic_lru_byte_reuse_distance.py"
B1_SCRIPT = ROOT / "scripts" / "audit_event_atomic_lru_reuse_distance.py"
DESIGN = ROOT / "V09_EVENT_ATOMIC_BYTE_REUSE_DISTANCE_DESIGN.md"

SPEC = importlib.util.spec_from_file_location("event_atomic_byte_reuse_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)

B1_SPEC = importlib.util.spec_from_file_location("event_atomic_count_reuse_oracle", B1_SCRIPT)
assert B1_SPEC is not None and B1_SPEC.loader is not None
B1 = importlib.util.module_from_spec(B1_SPEC)
sys.modules[B1_SPEC.name] = B1
B1_SPEC.loader.exec_module(B1)


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
        model_id="synthetic/v09-event-atomic-byte-reuse",
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
        model_id="synthetic/v09-decoder-byte-reuse",
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


class EventAtomicLRUByteReuseDistanceTests(unittest.TestCase):
    def test_heterogeneous_hand_computed_thresholds_and_fraction(self) -> None:
        trace = _v1_trace(
            (
                _v1_event(0, (1, 0)),
                _v1_event(1, (2,)),
                _v1_event(2, (0, 1)),
            )
        )
        sizes = {(0, 0): 2, (0, 1): 5, (0, 2): 3}
        summary = AUDIT.analyze_versioned_trace(trace, sizes)
        samples = {sample.key: sample for sample in summary.samples}

        self.assertEqual(summary.event_count, 3)
        self.assertEqual(summary.expert_request_count, 5)
        self.assertEqual(summary.distinct_referenced_key_count, 3)
        self.assertEqual(summary.first_use_count, 3)
        self.assertEqual(summary.reuse_count, 2)
        self.assertEqual(summary.referenced_working_set_bytes, 10)
        self.assertEqual(summary.max_atomic_event_bytes, 7)
        self.assertEqual(samples[(0, 0)].stack_distance_bytes, 8)
        self.assertEqual(samples[(0, 0)].key_size_bytes, 2)
        self.assertEqual(samples[(0, 0)].required_capacity_bytes, 10)
        self.assertEqual(samples[(0, 1)].stack_distance_bytes, 3)
        self.assertEqual(samples[(0, 1)].key_size_bytes, 5)
        self.assertEqual(samples[(0, 1)].required_capacity_bytes, 8)
        self.assertEqual(summary.required_capacity_histogram, ((8, 1), (10, 1)))
        self.assertEqual(summary.min_required_capacity_bytes, 8)
        self.assertEqual(summary.max_required_capacity_bytes, 10)
        self.assertEqual(summary.first_capacity_with_observed_lru_hit_bytes, 8)
        self.assertEqual(summary.predicted_lru_hits(7), 0)
        self.assertEqual(summary.predicted_lru_hits(8), 1)
        self.assertEqual(summary.predicted_lru_hits(10), 2)
        self.assertEqual(summary.reuse_fraction_within_capacity(8), Fraction(1, 2))
        with self.assertRaisesRegex(ValueError, "maximum atomic event byte"):
            summary.predicted_lru_hits(6)

    def test_same_event_tie_uses_stable_key_and_unequal_weights(self) -> None:
        trace = _v1_trace(
            (
                _v1_event(0, (1, 0)),
                _v1_event(1, (2,)),
                _v1_event(2, (0, 1)),
            )
        )
        sizes = {(0, 0): 2, (0, 1): 5, (0, 2): 3}
        summary = AUDIT.analyze_versioned_trace(trace, sizes)
        samples = {sample.key: sample for sample in summary.samples}

        # Event-0 peers share one timestamp. Stable key (0, 1) outranks
        # (0, 0), so its five bytes are included above (0, 0), not vice versa.
        self.assertEqual(samples[(0, 0)].stack_distance_bytes, 5 + 3)
        self.assertEqual(samples[(0, 1)].stack_distance_bytes, 3)
        self.assertNotEqual(
            samples[(0, 0)].stack_distance_bytes,
            samples[(0, 1)].stack_distance_bytes,
        )

    def test_selected_expert_order_and_extra_valid_sizes_are_inert(self) -> None:
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
        sizes = {(0, 0): 2, (0, 1): 5, (0, 2): 3}
        with_extra = {**sizes, (9, 9): 999}

        left_summary = AUDIT.analyze_versioned_trace(left, sizes)
        self.assertEqual(left_summary, AUDIT.analyze_versioned_trace(right, sizes))
        self.assertEqual(left_summary, AUDIT.analyze_versioned_trace(left, with_extra))
        self.assertEqual(
            AUDIT.summary_payload(left_summary, capacity_bytes=8),
            AUDIT.summary_payload(AUDIT.analyze_versioned_trace(right, sizes), capacity_bytes=8),
        )

    def test_v1_layer_qualified_identity_has_independent_sizes(self) -> None:
        trace = _v1_trace(
            (
                _v1_event(0, (1,), layer=0),
                _v1_event(1, (1,), layer=1),
                _v1_event(2, (1,), layer=0),
            )
        )
        sizes = {(0, 1): 2, (1, 1): 7}
        summary = AUDIT.analyze_versioned_trace(trace, sizes)

        self.assertEqual(summary.distinct_referenced_key_count, 2)
        self.assertEqual(summary.reuse_count, 1)
        self.assertEqual(summary.samples[0].key, (0, 1))
        self.assertEqual(summary.samples[0].stack_distance_bytes, 7)
        self.assertEqual(summary.samples[0].required_capacity_bytes, 9)
        self.assertEqual(summary.referenced_working_set_bytes, 9)

    def test_v2_stage_qualified_identity_has_independent_sizes(self) -> None:
        trace = RoutingTraceV2(
            model_id="synthetic/v09-stage-byte-reuse",
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
        sizes = {("encoder", 0, 1): 2, ("decoder", 0, 1): 5}
        summary = AUDIT.analyze_versioned_trace(trace, sizes)

        self.assertEqual(summary.distinct_referenced_key_count, 2)
        self.assertEqual(summary.reuse_count, 1)
        self.assertEqual(summary.samples[0].key, ("decoder", 0, 1))
        self.assertEqual(summary.samples[0].stack_distance_bytes, 0)
        self.assertEqual(summary.samples[0].required_capacity_bytes, 5)
        simulation = simulate_versioned_byte_cache(trace, 5, sizes, "lru")
        self.assertEqual(summary.predicted_lru_hits(5), simulation.hits)
        self.assertEqual(simulation.hits, 1)

    def test_v2_unassigned_event_is_byte_recency_inert(self) -> None:
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
                _v2_event("decoder", "decoder_generated", 1, unassigned=True),
                _v2_event("decoder", "decoder_generated", 2, (1,)),
                _v2_event("decoder", "decoder_generated", 3, (0,)),
            ),
            width=1,
        )
        sizes = {("decoder", 0, 0): 2, ("decoder", 0, 1): 5}
        base_summary = AUDIT.analyze_versioned_trace(base, sizes)
        unassigned_summary = AUDIT.analyze_versioned_trace(with_unassigned, sizes)

        self.assertEqual(unassigned_summary.event_count, base_summary.event_count + 1)
        for attribute in (
            "expert_request_count",
            "distinct_referenced_key_count",
            "first_use_count",
            "reuse_count",
            "referenced_working_set_bytes",
            "max_atomic_event_bytes",
            "required_capacity_histogram",
            "min_required_capacity_bytes",
            "max_required_capacity_bytes",
            "first_capacity_with_observed_lru_hit_bytes",
        ):
            with self.subTest(attribute=attribute):
                self.assertEqual(
                    getattr(unassigned_summary, attribute),
                    getattr(base_summary, attribute),
                )
        self.assertEqual(
            [sample.required_capacity_bytes for sample in unassigned_summary.samples],
            [sample.required_capacity_bytes for sample in base_summary.samples],
        )
        for capacity in range(
            base_summary.max_atomic_event_bytes,
            base_summary.referenced_working_set_bytes + 1,
        ):
            self.assertEqual(
                unassigned_summary.predicted_lru_hits(capacity),
                base_summary.predicted_lru_hits(capacity),
            )

    def test_heterogeneous_v1_parity_at_every_feasible_integer_capacity(self) -> None:
        cases = (
            (
                _v1_trace(
                    (
                        _v1_event(0, (1, 0)),
                        _v1_event(1, (2,)),
                        _v1_event(2, (0, 1)),
                        _v1_event(3, (3,)),
                        _v1_event(4, (1, 2)),
                    )
                ),
                {(0, 0): 2, (0, 1): 5, (0, 2): 3, (0, 3): 4},
            ),
            (
                _v1_trace(
                    tuple(
                        _v1_event(position, experts)
                        for position, experts in enumerate(
                            ((0,), (1,), (0,), (2,), (0,), (1,))
                        )
                    )
                ),
                {(0, 0): 4, (0, 1): 1, (0, 2): 6},
            ),
        )

        for trace, sizes in cases:
            summary = AUDIT.analyze_versioned_trace(trace, sizes)
            for capacity in range(
                summary.max_atomic_event_bytes,
                summary.referenced_working_set_bytes + 1,
            ):
                with self.subTest(trace=trace.events, capacity=capacity):
                    simulation = simulate_versioned_byte_cache(trace, capacity, sizes, "lru")
                    self.assertEqual(summary.predicted_lru_hits(capacity), simulation.hits)

    def test_heterogeneous_v2_parity_at_every_feasible_integer_capacity(self) -> None:
        trace = _decoder_v2_trace(
            (
                _v2_event("decoder", "decoder_generated", 0, (1, 0)),
                _v2_event("decoder", "decoder_generated", 1, unassigned=True),
                _v2_event("decoder", "decoder_generated", 2, (2, 3)),
                _v2_event("decoder", "decoder_generated", 3, (1, 3)),
                _v2_event("decoder", "decoder_generated", 4, (0, 2)),
            ),
            width=2,
        )
        sizes = {
            ("decoder", 0, 0): 2,
            ("decoder", 0, 1): 5,
            ("decoder", 0, 2): 3,
            ("decoder", 0, 3): 4,
        }
        summary = AUDIT.analyze_versioned_trace(trace, sizes)

        for capacity in range(
            summary.max_atomic_event_bytes,
            summary.referenced_working_set_bytes + 1,
        ):
            with self.subTest(capacity=capacity):
                simulation = simulate_versioned_byte_cache(trace, capacity, sizes, "lru")
                self.assertEqual(summary.predicted_lru_hits(capacity), simulation.hits)

    def test_uniform_sizes_reduce_exactly_to_b1_count_contract(self) -> None:
        trace = _v1_trace(
            (
                _v1_event(0, (1, 0)),
                _v1_event(1, (2,)),
                _v1_event(2, (0, 1)),
                _v1_event(3, (3,)),
                _v1_event(4, (1, 2)),
            )
        )
        scale = 4
        sizes = {key: scale for key in {(event.layer, expert) for event in trace.events for expert in event.selected_experts}}
        count_summary = B1.analyze_versioned_trace(trace)
        byte_summary = AUDIT.analyze_versioned_trace(trace, sizes)

        self.assertEqual(
            byte_summary.max_atomic_event_bytes,
            count_summary.max_atomic_event_entries * scale,
        )
        self.assertEqual(
            byte_summary.referenced_working_set_bytes,
            count_summary.distinct_referenced_key_count * scale,
        )
        count_samples = {(sample.event_index, sample.key): sample for sample in count_summary.samples}
        for sample in byte_summary.samples:
            count_sample = count_samples[(sample.event_index, sample.key)]
            self.assertEqual(
                sample.stack_distance_bytes,
                count_sample.stack_distance_entries * scale,
            )
            self.assertEqual(
                sample.required_capacity_bytes,
                count_sample.required_capacity_entries * scale,
            )

        for capacity_entries in range(
            count_summary.max_atomic_event_entries,
            count_summary.distinct_referenced_key_count + 1,
        ):
            self.assertEqual(
                byte_summary.predicted_lru_hits(capacity_entries * scale),
                count_summary.predicted_lru_hits(capacity_entries),
            )

    def test_empty_no_reuse_and_invalid_size_boundaries_are_explicit(self) -> None:
        empty = AUDIT.analyze_required_sets((), {})
        self.assertEqual(empty.event_count, 0)
        self.assertEqual(empty.expert_request_count, 0)
        self.assertEqual(empty.referenced_working_set_bytes, 0)
        self.assertEqual(empty.max_atomic_event_bytes, 0)
        self.assertEqual(empty.required_capacity_histogram, ())
        self.assertIsNone(empty.first_capacity_with_observed_lru_hit_bytes)
        self.assertIsNone(empty.reuse_fraction_within_capacity(1))

        one_event = AUDIT.analyze_required_sets(
            (frozenset({(0, 0), (0, 1)}),),
            {(0, 0): 2, (0, 1): 5},
        )
        self.assertEqual(one_event.first_use_count, 2)
        self.assertEqual(one_event.reuse_count, 0)
        self.assertEqual(one_event.max_atomic_event_bytes, 7)
        self.assertIsNone(one_event.min_required_capacity_bytes)
        self.assertIsNone(one_event.reuse_fraction_within_capacity(7))

        with self.assertRaisesRegex(TypeError, "mapping"):
            AUDIT.analyze_required_sets((frozenset({(0, 0)}),), [((0, 0), 1)])
        with self.assertRaisesRegex(ValueError, "missing referenced"):
            AUDIT.analyze_required_sets((frozenset({(0, 0)}),), {})
        for bad_size in (True, 0, -1, 1.5, "1"):
            with self.subTest(bad_size=bad_size):
                with self.assertRaisesRegex(ValueError, "positive integer byte"):
                    AUDIT.analyze_required_sets(
                        (frozenset({(0, 0)}),),
                        {(0, 0): bad_size},
                    )
        with self.assertRaisesRegex(TypeError, "one canonical key shape"):
            AUDIT.analyze_required_sets(
                (frozenset({(0, 0)}),),
                {(0, 0): 1, ("decoder", 0, 0): 1},
            )
        with self.assertRaisesRegex(ValueError, "encoder/decoder"):
            AUDIT.analyze_required_sets(
                (frozenset({("middle", 0, 0)}),),
                {("middle", 0, 0): 1},
            )
        with self.assertRaisesRegex(TypeError, "v1 pairs or v2 triples"):
            AUDIT.analyze_required_sets(
                (frozenset({(0, 0, 0, 0)}),),
                {(0, 0, 0, 0): 1},
            )
        for bad_capacity in (True, 0, -1, 1.5, "1"):
            with self.subTest(bad_capacity=bad_capacity):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    one_event.predicted_lru_hits(bad_capacity)

    def test_payload_and_design_preserve_size_provenance_and_claim_boundaries(self) -> None:
        trace = _v1_trace((_v1_event(0, (0,)), _v1_event(1, (0,))))
        summary = AUDIT.analyze_versioned_trace(trace, {(0, 0): 7})
        payload = AUDIT.summary_payload(summary, capacity_bytes=7)
        self.assertIn("caller-supplied cache-model byte sizes", payload["size_input_boundary"])
        self.assertEqual(
            payload["tested_capacity"]["reuse_fraction_within_capacity"],
            {"numerator": 1, "denominator": 1},
        )

        text = DESIGN.read_text(encoding="utf-8")
        for required in (
            "caller-supplied",
            "not automatically MEASURED",
            "Cache replay remains **SIMULATED**",
            "Transfer-service quantities remain **ESTIMATED**",
            "not a recommendation",
            "physical residency",
            "uniform-size reduction",
            "exact simulator parity",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
