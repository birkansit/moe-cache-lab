import unittest

from moe_cache_lab.cache import simulate
from moe_cache_lab.report import render_report
from moe_cache_lab.trace import RoutingEvent, RoutingTrace


class ReportTests(unittest.TestCase):
    def test_report_labels_boundaries_phases_identity_and_metrics(self) -> None:
        trace = RoutingTrace(
            "example/moe",
            4,
            1,
            (
                RoutingEvent("prompt", 0, 0, (2,)),
                RoutingEvent("generated", 1, 0, (2,)),
            ),
        )
        simulations = [
            simulate(trace.events, 1, "lru"),
            simulate(trace.events, 1, "offline_oracle_frequency"),
        ]
        report = render_report(trace, simulations)
        for expected in (
            "**measured**",
            "**simulated**",
            "layer-qualified",
            "prompt/prefill",
            "generated/decode",
            "prewarm loads",
            "demand loads",
            "estimated transfers",
            "explicitly non-causal",
            "restored after every event",
            "two demand loads and two evictions",
            "not measured data movement",
        ):
            self.assertIn(expected, report)
