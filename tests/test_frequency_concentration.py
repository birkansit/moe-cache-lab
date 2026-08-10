from fractions import Fraction
import json
import unittest

from moe_cache_lab.analysis import (
    _summarize_frequency_concentration,
    analyze_routing,
)
from moe_cache_lab.analysis_output import render_analysis_json, render_analysis_report
from moe_cache_lab.trace import RoutingEvent


class FrequencyConcentrationTests(unittest.TestCase):
    def test_uniform_observed_frequencies_have_zero_gini_and_expected_entropy(self) -> None:
        summary = analyze_routing((
            RoutingEvent("prompt", 0, 0, (0,)),
            RoutingEvent("prompt", 1, 0, (1,)),
            RoutingEvent("prompt", 2, 0, (0,)),
            RoutingEvent("prompt", 3, 0, (1,)),
        )).layers[0].concentration

        self.assertEqual(summary.support_size, 2)
        self.assertEqual(summary.total_selections, 4)
        self.assertEqual(summary.max_selection_share, Fraction(1, 2))
        self.assertEqual(summary.observed_support_gini, Fraction(0, 1))
        self.assertEqual(summary.shannon_entropy_bits, 1.0)

    def test_skewed_distribution_has_hand_computed_exact_gini_and_max_share(self) -> None:
        summary = analyze_routing((
            RoutingEvent("prompt", 0, 0, (0,)),
            RoutingEvent("prompt", 1, 0, (0,)),
            RoutingEvent("prompt", 2, 0, (0,)),
            RoutingEvent("prompt", 3, 0, (1,)),
        )).layers[0].concentration

        self.assertEqual(summary.max_selection_share, Fraction(3, 4))
        self.assertEqual(summary.observed_support_gini, Fraction(1, 4))
        self.assertAlmostEqual(summary.shannon_entropy_bits, 0.8112781244591328)

    def test_single_expert_support_has_defined_degenerate_values(self) -> None:
        summary = analyze_routing((
            RoutingEvent("prompt", 0, 0, (3,)),
            RoutingEvent("prompt", 1, 0, (3,)),
            RoutingEvent("prompt", 2, 0, (3,)),
        )).layers[0].concentration

        self.assertEqual(summary.support_size, 1)
        self.assertEqual(summary.total_selections, 3)
        self.assertEqual(summary.max_selection_share, Fraction(1, 1))
        self.assertEqual(summary.observed_support_gini, Fraction(0, 1))
        self.assertEqual(summary.shannon_entropy_bits, 0.0)

    def test_prompt_and_generated_same_layer_are_independent(self) -> None:
        streams = {
            (item.phase, item.layer_id): item
            for item in analyze_routing((
                RoutingEvent("prompt", 0, 0, (0,)),
                RoutingEvent("prompt", 1, 0, (0,)),
                RoutingEvent("prompt", 2, 0, (0,)),
                RoutingEvent("prompt", 3, 0, (1,)),
                RoutingEvent("generated", 4, 0, (0,)),
                RoutingEvent("generated", 5, 0, (1,)),
                RoutingEvent("generated", 6, 0, (0,)),
                RoutingEvent("generated", 7, 0, (1,)),
            )).phase_layers
        }

        prompt = streams[("prompt", 0)].concentration
        generated = streams[("generated", 0)].concentration
        self.assertEqual(prompt.observed_support_gini, Fraction(1, 4))
        self.assertEqual(prompt.max_selection_share, Fraction(3, 4))
        self.assertEqual(generated.observed_support_gini, Fraction(0, 1))
        self.assertEqual(generated.max_selection_share, Fraction(1, 2))
        self.assertEqual(generated.shannon_entropy_bits, 1.0)

    def test_concentration_matches_established_deterministic_frequency_counts(self) -> None:
        summary = analyze_routing((
            RoutingEvent("prompt", 0, 0, (2, 1)),
            RoutingEvent("prompt", 1, 0, (2, 3)),
            RoutingEvent("prompt", 0, 1, (5,)),
        ))

        layer_zero = summary.layers[0]
        prompt_layer_zero = summary.phase_layers[0]
        self.assertEqual(layer_zero.expert_frequencies, prompt_layer_zero.expert_frequencies)
        self.assertEqual(layer_zero.concentration, prompt_layer_zero.concentration)
        counts = tuple(item.selection_count for item in layer_zero.expert_frequencies)
        self.assertEqual(layer_zero.concentration.support_size, len(counts))
        self.assertEqual(layer_zero.concentration.total_selections, sum(counts))
        self.assertEqual(
            layer_zero.concentration.max_selection_share,
            Fraction(max(counts), sum(counts)),
        )

    def test_human_output_documents_observed_support_and_entropy_units(self) -> None:
        report = render_analysis_report(analyze_routing((
            RoutingEvent("prompt", 0, 0, (0,)),
            RoutingEvent("prompt", 1, 0, (1,)),
        )))

        self.assertIn("observed-support Gini", report)
        self.assertIn("unselected experts are excluded", report)
        self.assertIn("Shannon entropy", report)
        self.assertIn("bits", report)
        self.assertIn("not an exact rational result", report)
        self.assertNotIn("GOOD", report)
        self.assertNotIn("BAD", report)
        self.assertNotIn("recommendation", report.lower())
        self.assertIn("does not establish cache/offload speedup", report)

    def test_json_preserves_exact_rationals_and_is_deterministic(self) -> None:
        summary = analyze_routing((
            RoutingEvent("prompt", 0, 0, (0,)),
            RoutingEvent("prompt", 1, 0, (0,)),
            RoutingEvent("prompt", 2, 0, (0,)),
            RoutingEvent("prompt", 3, 0, (1,)),
        ))
        first = render_analysis_json(summary)
        second = render_analysis_json(summary)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))

        payload = json.loads(first)
        layer = payload["layers"][0]["frequency_concentration"]
        stream = payload["phase_layers"][0]["frequency_concentration"]
        for concentration in (layer, stream):
            self.assertEqual(
                concentration["max_selection_share"],
                {"numerator": 3, "denominator": 4},
            )
            self.assertEqual(
                concentration["observed_support_gini"],
                {"numerator": 1, "denominator": 4},
            )
            self.assertIsInstance(concentration["shannon_entropy_bits"], float)

    def test_empty_helper_behavior_is_defined(self) -> None:
        summary = _summarize_frequency_concentration(())
        self.assertEqual((summary.support_size, summary.total_selections), (0, 0))
        self.assertIsNone(summary.max_selection_share)
        self.assertIsNone(summary.observed_support_gini)
        self.assertIsNone(summary.shannon_entropy_bits)


if __name__ == "__main__":
    unittest.main()
