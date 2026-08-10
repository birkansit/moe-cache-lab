from fractions import Fraction
import unittest

from moe_cache_lab.analysis import analyze_routing
from moe_cache_lab.trace import RoutingEvent


class PhaseLayerRoutingAnalysisTests(unittest.TestCase):
    def test_streams_keep_phase_layer_identity_and_deterministic_order(self) -> None:
        summary = analyze_routing((
            RoutingEvent("prompt", 0, 0, (2, 1)),
            RoutingEvent("prompt", 1, 0, (1, 2)),
            RoutingEvent("prompt", 0, 2, (1,)),
            RoutingEvent("generated", 2, 0, (2, 1)),
            RoutingEvent("generated", 2, 2, (1,)),
        ))

        self.assertEqual(
            [
                (
                    item.phase,
                    item.layer_id,
                    item.event_count,
                    item.assignment_count,
                    item.unique_expert_count,
                )
                for item in summary.phase_layers
            ],
            [
                ("prompt", 0, 2, 4, 2),
                ("prompt", 2, 1, 1, 1),
                ("generated", 0, 1, 2, 2),
                ("generated", 2, 1, 1, 1),
            ],
        )
        self.assertEqual(
            [
                (item.expert_id, item.selection_count)
                for item in summary.phase_layers[0].expert_frequencies
            ],
            [(1, 2), (2, 2)],
        )

    def test_within_stream_reuse_matches_across_collector_nesting(self) -> None:
        pattern = ((0, 1), (1, 2), (0, 2))
        events = (
            RoutingEvent("prompt", 0, 0, pattern[0]),
            RoutingEvent("prompt", 1, 0, pattern[1]),
            RoutingEvent("prompt", 2, 0, pattern[2]),
            RoutingEvent("prompt", 0, 1, (3, 4)),
            RoutingEvent("prompt", 1, 1, (3, 4)),
            RoutingEvent("prompt", 2, 1, (3, 4)),
            RoutingEvent("generated", 3, 0, pattern[0]),
            RoutingEvent("generated", 3, 1, (3, 4)),
            RoutingEvent("generated", 4, 0, pattern[1]),
            RoutingEvent("generated", 4, 1, (3, 4)),
            RoutingEvent("generated", 5, 0, pattern[2]),
            RoutingEvent("generated", 5, 1, (3, 4)),
        )
        streams = {
            (item.phase, item.layer_id): item for item in analyze_routing(events).phase_layers
        }

        prompt_gap = streams[("prompt", 0)].reuse_gap
        generated_gap = streams[("generated", 0)].reuse_gap
        self.assertEqual(prompt_gap, generated_gap)
        self.assertEqual(
            (
                prompt_gap.first_use_count,
                prompt_gap.reuse_count,
                prompt_gap.min_gap_events,
                prompt_gap.max_gap_events,
                prompt_gap.mean_gap_events,
            ),
            (3, 3, 0, 1, Fraction(1, 3)),
        )

    def test_jaccard_edges_match_global_definition(self) -> None:
        summary = analyze_routing((
            RoutingEvent("prompt", 0, 0, (0, 1)),
            RoutingEvent("prompt", 1, 0, (2, 3)),
            RoutingEvent("prompt", 2, 0, (2, 3)),
            RoutingEvent("prompt", 3, 0, (3, 4)),
            RoutingEvent("prompt", 0, 1, (5, 6)),
        ))

        overlap = summary.phase_layers[0].consecutive_overlap
        self.assertEqual(
            (
                overlap.comparison_count,
                overlap.disjoint_pair_count,
                overlap.identical_pair_count,
                overlap.partial_pair_count,
                overlap.mean_jaccard,
            ),
            (3, 1, 1, 1, Fraction(4, 9)),
        )
        no_predecessor = summary.phase_layers[1].consecutive_overlap
        self.assertEqual(no_predecessor.comparison_count, 0)
        self.assertIsNone(no_predecessor.mean_jaccard)

    def test_empty_input_has_no_phase_layer_streams(self) -> None:
        self.assertEqual(analyze_routing(()).phase_layers, ())

    def test_invalid_raw_chronology_is_not_normalized(self) -> None:
        invalid_order = (
            RoutingEvent("prompt", 0, 0, (1,)),
            RoutingEvent("prompt", 0, 1, (2,)),
            RoutingEvent("prompt", 1, 0, (3,)),
        )
        with self.assertRaisesRegex(ValueError, "layer-major collector order"):
            analyze_routing(invalid_order)


if __name__ == "__main__":
    unittest.main()
