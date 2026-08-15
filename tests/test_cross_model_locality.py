from dataclasses import fields
from fractions import Fraction
import unittest

from moe_cache_lab.cross_model_locality import (
    CrossModelLocalityDiagnostics,
    analyze_cross_model_locality,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
)


def _v1(
    events: tuple[RoutingEvent, ...], *, num_experts: int | None = 4
) -> RoutingTrace:
    return RoutingTrace(
        model_id="synthetic/v1",
        num_experts=num_experts,
        experts_per_token=None,
        events=events,
        created_at="fixed",
    )


def _assigned(stage: str, phase: str, position: int, layer: int, expert: int):
    return RoutingEventV2(
        stage, phase, position, layer, "assigned", (expert,), ()
    )


def _unassigned(stage: str, phase: str, position: int, layer: int):
    return RoutingEventV2(
        stage,
        phase,
        position,
        layer,
        "unassigned",
        (),
        (),
        unassigned_reason="capacity",
    )


class CrossModelLocalityTests(unittest.TestCase):
    def test_v1_uses_configured_universe_and_exact_ordered_top_k(self) -> None:
        trace = _v1(
            (
                RoutingEvent("prompt", 0, 0, (2,)),
                RoutingEvent("prompt", 1, 0, (1,)),
                RoutingEvent("prompt", 2, 0, (2,)),
                RoutingEvent("prompt", 3, 0, (1,)),
            )
        )

        result = analyze_cross_model_locality(trace, (2, 1, 4))
        layer = result.layers[0]

        self.assertEqual(result.requested_top_k, (2, 1, 4))
        self.assertEqual(layer.layer_identity, 0)
        self.assertEqual(layer.configured_expert_count, 4)
        self.assertEqual(layer.normalized_shannon_entropy, 0.5)
        self.assertEqual(
            [
                (item.k, item.expert_ids, item.selection_count, item.share)
                for item in layer.cumulative_top_k_selection_shares
            ],
            [
                (2, (1, 2), 4, Fraction(1, 1)),
                (1, (1,), 2, Fraction(1, 2)),
                (4, (1, 2, 0, 3), 4, Fraction(1, 1)),
            ],
        )
        self.assertEqual(
            result.to_dict()["layers"][0]["cumulative_top_k_selection_shares"][1][
                "share"
            ],
            {"numerator": 1, "denominator": 2},
        )

    def test_v1_layer_order_is_canonical_not_first_observation_order(self) -> None:
        trace = _v1(
            (
                RoutingEvent("prompt", 0, 2, (0,)),
                RoutingEvent("prompt", 0, 3, (0,)),
            )
        )
        self.assertEqual(
            tuple(
                layer.layer_identity
                for layer in analyze_cross_model_locality(trace, (1,)).layers
            ),
            (2, 3),
        )

    def test_same_uneven_observed_distribution_changes_with_configured_universe(self) -> None:
        events = (
            RoutingEvent("prompt", 0, 0, (1,)),
            RoutingEvent("prompt", 1, 0, (1,)),
            RoutingEvent("prompt", 2, 0, (1,)),
            RoutingEvent("prompt", 3, 0, (2,)),
        )

        four = analyze_cross_model_locality(_v1(events, num_experts=4), (1,)).layers[0]
        eight = analyze_cross_model_locality(_v1(events, num_experts=8), (1,)).layers[0]

        self.assertAlmostEqual(four.normalized_shannon_entropy, 0.4056390622295664)
        self.assertAlmostEqual(eight.normalized_shannon_entropy, 0.2704260414863776)
        self.assertEqual(
            four.cumulative_top_k_selection_shares[0].share,
            Fraction(3, 4),
        )
        self.assertEqual(
            eight.cumulative_top_k_selection_shares[0].share,
            Fraction(3, 4),
        )

    def test_v2_stage_namespaces_and_configured_counts_remain_distinct(self) -> None:
        trace = RoutingTraceV2(
            model_id="synthetic/v2",
            routing_stages=(
                RoutingStageProfile("encoder", 4, 1, True),
                RoutingStageProfile("decoder", 8, 1, True),
            ),
            events=(
                _assigned("encoder", "source", 0, 1, 3),
                _assigned("decoder", "decoder_prompt", 0, 1, 3),
                _assigned("decoder", "decoder_generated", 1, 1, 4),
            ),
            created_at="fixed",
        )

        result = analyze_cross_model_locality(trace, (1, 4))

        self.assertEqual(
            [
                (layer.layer_identity, layer.configured_expert_count)
                for layer in result.layers
            ],
            [(('encoder', 1), 4), (('decoder', 1), 8)],
        )
        self.assertEqual(result.layers[0].normalized_shannon_entropy, 0.0)
        self.assertAlmostEqual(result.layers[1].normalized_shannon_entropy, 1 / 3)

    def test_v2_unassigned_events_have_zero_mass_and_undefined_metrics(self) -> None:
        trace = RoutingTraceV2(
            model_id="synthetic/v2",
            routing_stages=(RoutingStageProfile("decoder", 4, 1, True),),
            events=(
                _unassigned("decoder", "decoder_prompt", 0, 2),
                _unassigned("decoder", "decoder_generated", 1, 2),
            ),
            created_at="fixed",
        )

        layer = analyze_cross_model_locality(trace, (1, 4)).layers[0]

        self.assertEqual(layer.layer_identity, ("decoder", 2))
        self.assertEqual(layer.total_selection_count, 0)
        self.assertIsNone(layer.normalized_shannon_entropy)
        self.assertEqual(
            [item.expert_ids for item in layer.cumulative_top_k_selection_shares],
            [(0,), (0, 1, 2, 3)],
        )
        self.assertTrue(
            all(
                item.selection_count == 0 and item.share is None
                for item in layer.cumulative_top_k_selection_shares
            )
        )

    def test_single_configured_expert_has_undefined_normalized_entropy(self) -> None:
        layer = analyze_cross_model_locality(
            _v1((RoutingEvent("prompt", 0, 0, (0,)),), num_experts=1),
            (1,),
        ).layers[0]

        self.assertIsNone(layer.normalized_shannon_entropy)
        self.assertEqual(layer.cumulative_top_k_selection_shares[0].share, Fraction(1))

    def test_top_k_validation_is_strict_and_checks_every_layer_universe(self) -> None:
        trace = _v1((RoutingEvent("prompt", 0, 0, (0,)),))
        invalid = ([1], (), (True,), (0,), (-1,), (1, 1), (1.0,))

        for top_k in invalid:
            with self.subTest(top_k=top_k):
                with self.assertRaises((TypeError, ValueError)):
                    analyze_cross_model_locality(trace, top_k)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "routed layer 0"):
            analyze_cross_model_locality(trace, (5,))

        v2 = RoutingTraceV2(
            model_id="synthetic/v2",
            routing_stages=(
                RoutingStageProfile("encoder", 2, 1, True),
                RoutingStageProfile("decoder", 4, 1, True),
            ),
            events=(
                _assigned("encoder", "source", 0, 0, 0),
                _assigned("decoder", "decoder_prompt", 0, 0, 0),
            ),
            created_at="fixed",
        )
        with self.assertRaisesRegex(ValueError, "encoder"):
            analyze_cross_model_locality(v2, (3,))

    def test_missing_v1_universe_and_wrong_trace_type_reject(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured num_experts"):
            analyze_cross_model_locality(
                _v1((RoutingEvent("prompt", 0, 0, (0,)),), num_experts=None),
                (1,),
            )
        with self.assertRaisesRegex(ValueError, "positive integer num_experts"):
            analyze_cross_model_locality(
                _v1((RoutingEvent("prompt", 0, 0, (0,)),), num_experts=True),
                (1,),
            )
        with self.assertRaisesRegex(TypeError, "RoutingTrace or RoutingTraceV2"):
            analyze_cross_model_locality(object(), (1,))  # type: ignore[arg-type]

    def test_result_surface_contains_no_score_or_recommendation(self) -> None:
        names = {item.name for item in fields(CrossModelLocalityDiagnostics)}
        self.assertEqual(names, {"trace_format_version", "requested_top_k", "layers"})
        serialized = str(
            analyze_cross_model_locality(
                _v1((RoutingEvent("prompt", 0, 0, (0,)),)), (1,)
            ).to_dict()
        ).lower()
        for forbidden in ("score", "quality", "recommendation", "speedup"):
            self.assertNotIn(forbidden, serialized)
