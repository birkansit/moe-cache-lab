from fractions import Fraction
import unittest

from moe_cache_lab.analysis import analyze_routing
from moe_cache_lab.trace import RoutingEvent, RoutingTrace


class RoutingAnalysisTests(unittest.TestCase):
    def test_layer_qualified_identity_and_phase_counts(self) -> None:
        trace = RoutingTrace(
            "example/moe",
            8,
            2,
            (
                RoutingEvent("prompt", 0, 0, (1, 2)),
                RoutingEvent("prompt", 0, 1, (1, 3)),
                RoutingEvent("generated", 1, 0, (2, 3)),
                RoutingEvent("generated", 1, 1, (1, 2)),
            ),
        )
        summary = analyze_routing(trace)

        self.assertEqual(summary.total_event_count, 4)
        self.assertEqual(summary.total_assignment_count, 8)
        self.assertEqual(summary.unique_layer_expert_count, 6)
        self.assertEqual((summary.prompt.event_count, summary.prompt.assignment_count), (2, 4))
        self.assertEqual((summary.generated.event_count, summary.generated.assignment_count), (2, 4))
        self.assertEqual(
            [(layer.layer_id, layer.unique_expert_count) for layer in summary.layers],
            [(0, 3), (1, 3)],
        )

    def test_per_layer_frequencies_are_deterministic_for_expert_tuple_order(self) -> None:
        left = (
            RoutingEvent("prompt", 0, 0, (2, 1)),
            RoutingEvent("prompt", 1, 0, (2, 3)),
        )
        right = (
            RoutingEvent("prompt", 0, 0, (1, 2)),
            RoutingEvent("prompt", 1, 0, (3, 2)),
        )

        left_layer = analyze_routing(left).layers[0]
        right_layer = analyze_routing(right).layers[0]
        self.assertEqual(left_layer, right_layer)
        self.assertEqual(
            [(item.expert_id, item.selection_count) for item in left_layer.expert_frequencies],
            [(1, 1), (2, 2), (3, 1)],
        )

    def test_consecutive_overlap_edge_cases(self) -> None:
        summary = analyze_routing((
            RoutingEvent("prompt", 0, 0, (0, 1)),
            RoutingEvent("prompt", 1, 0, (2, 3)),
            RoutingEvent("prompt", 2, 0, (2, 3)),
            RoutingEvent("prompt", 3, 0, (3, 4)),
            RoutingEvent("prompt", 0, 1, (0, 1)),
            RoutingEvent("generated", 4, 0, (3, 4)),
        )).consecutive_overlap

        self.assertEqual(summary.comparison_count, 3)
        self.assertEqual(summary.disjoint_pair_count, 1)
        self.assertEqual(summary.identical_pair_count, 1)
        self.assertEqual(summary.partial_pair_count, 1)
        self.assertEqual(summary.mean_jaccard, Fraction(4, 9))

    def test_reuse_gap_first_immediate_separated_and_cross_layer_identity(self) -> None:
        summary = analyze_routing((
            RoutingEvent("prompt", 0, 0, (1,)),
            RoutingEvent("prompt", 1, 0, (1,)),
            RoutingEvent("prompt", 2, 0, (2,)),
            RoutingEvent("prompt", 3, 0, (1,)),
            RoutingEvent("prompt", 0, 1, (1,)),
        )).reuse_gap

        self.assertEqual(summary.first_use_count, 3)
        self.assertEqual(summary.reuse_count, 2)
        self.assertEqual(summary.min_gap_events, 0)
        self.assertEqual(summary.max_gap_events, 1)
        self.assertEqual(summary.mean_gap_events, Fraction(1, 2))

    def test_empty_input(self) -> None:
        summary = analyze_routing(())
        self.assertEqual(summary.total_event_count, 0)
        self.assertEqual(summary.total_assignment_count, 0)
        self.assertEqual(summary.unique_layer_expert_count, 0)
        self.assertEqual((summary.prompt.event_count, summary.prompt.assignment_count), (0, 0))
        self.assertEqual((summary.generated.event_count, summary.generated.assignment_count), (0, 0))
        self.assertEqual(summary.layers, ())
        self.assertEqual(summary.consecutive_overlap.comparison_count, 0)
        self.assertIsNone(summary.consecutive_overlap.mean_jaccard)
        self.assertEqual(summary.reuse_gap.first_use_count, 0)
        self.assertEqual(summary.reuse_gap.reuse_count, 0)
        self.assertIsNone(summary.reuse_gap.min_gap_events)
        self.assertIsNone(summary.reuse_gap.max_gap_events)
        self.assertIsNone(summary.reuse_gap.mean_gap_events)

    def test_invalid_event_and_sequence_validation_is_not_normalized(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            RoutingEvent("prompt", 0, 0, (1, 1))

        invalid_order = (
            RoutingEvent("prompt", 0, 0, (1,)),
            RoutingEvent("prompt", 0, 1, (2,)),
            RoutingEvent("prompt", 1, 0, (3,)),
        )
        with self.assertRaisesRegex(ValueError, "layer-major collector order"):
            analyze_routing(invalid_order)

    def test_trace_and_event_iterable_have_same_summary(self) -> None:
        events = (
            RoutingEvent("prompt", 0, 0, (0, 1)),
            RoutingEvent("prompt", 1, 0, (1, 2)),
        )
        trace = RoutingTrace("example/moe", 4, 2, events)
        self.assertEqual(analyze_routing(trace), analyze_routing(events))


if __name__ == "__main__":
    unittest.main()
