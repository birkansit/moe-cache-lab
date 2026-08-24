import unittest
from unittest.mock import patch

from moe_cache_lab.producer import ProducerResult
from moe_cache_lab.switch_collector import (
    SwitchCollectionOutcome,
    SwitchDispatchCheck,
    SwitchTraceCollector,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.trace_v2 import RoutingEventV2, RoutingStageProfile, RoutingTraceV2


def _granite_trace() -> RoutingTrace:
    return RoutingTrace(
        model_id="synthetic/granite",
        num_experts=8,
        experts_per_token=1,
        events=(RoutingEvent("prompt", 0, 3, (5,), selected_probabilities=(0.75,)),),
        created_at="fixed",
    )


def _switch_outcome() -> SwitchCollectionOutcome:
    trace = RoutingTraceV2(
        model_id="synthetic/switch",
        routing_stages=(
            RoutingStageProfile("encoder", 8, 1, True),
            RoutingStageProfile("decoder", 8, 1, True),
        ),
        events=(
            RoutingEventV2("encoder", "source", 0, 3, "assigned", (5,), (0.75,)),
            RoutingEventV2(
                "encoder",
                "source",
                1,
                3,
                "unassigned",
                (),
                (),
                unassigned_reason="capacity",
            ),
            RoutingEventV2(
                "decoder", "decoder_prompt", 0, 3, "assigned", (5,), (0.8,)
            ),
        ),
        created_at="fixed",
    )
    return SwitchCollectionOutcome(
        trace=trace,
        source_token_ids=(10, 11),
        decoder_prompt_token_ids=(0,),
        fed_decoder_generated_token_ids=(),
        candidate_token_ids=(1,),
        terminal_eos_token_id=1,
        horizon_exhausted=False,
        max_fed_back_non_eos_steps=4,
        logits_exact=True,
        max_logit_abs_diff=0.0,
        registered_sparse_layers=(("encoder", (3,)), ("decoder", (3,))),
        router_output_shapes=(((2, 1), (2, 1, 8), (2, 1)),),
        dispatch_checks=(SwitchDispatchCheck("encoder", 3, 2, 2),),
    )


class BuiltInProducerNormalizationTests(unittest.TestCase):
    def test_granite_trace_normalizes_without_reconstruction(self) -> None:
        trace = _granite_trace()
        original_events = trace.events

        result = ProducerResult(trace)

        self.assertIs(result.trace, trace)
        self.assertIs(result.trace.events, original_events)
        self.assertIs(result.trace.events[0], original_events[0])

    def test_switch_outcome_normalizes_trace_and_retains_evidence_separately(self) -> None:
        outcome = _switch_outcome()
        evidence = outcome.dispatch_checks
        original_events = outcome.trace.events

        result = ProducerResult(outcome.trace)

        self.assertIs(result.trace, outcome.trace)
        self.assertIs(result.trace.events, original_events)
        for actual, expected in zip(result.trace.events, original_events, strict=True):
            self.assertIs(actual, expected)
        self.assertIs(outcome.dispatch_checks, evidence)
        self.assertNotIn("dispatch_checks", vars(result))
        self.assertNotIn("logits_exact", vars(result))

    def test_switch_stage_identity_and_unassigned_semantics_are_unchanged(self) -> None:
        outcome = _switch_outcome()

        result = ProducerResult(outcome.trace)

        self.assertEqual(
            result.trace.expert_requests,
            (("encoder", 3, 5), ("decoder", 3, 5)),
        )
        unassigned = result.trace.events[1]
        self.assertEqual(unassigned.assignment_state, "unassigned")
        self.assertEqual(unassigned.selected_experts, ())
        self.assertEqual(unassigned.selected_probabilities, ())
        self.assertEqual(unassigned.expert_requests, ())

    def test_normalization_does_not_invoke_collectors_or_serializers(self) -> None:
        trace = _granite_trace()
        outcome = _switch_outcome()
        with (
            patch("moe_cache_lab.collector.GraniteTraceCollector.collect") as granite,
            patch.object(SwitchTraceCollector, "collect_and_verify") as switch,
            patch("moe_cache_lab.trace.write_trace") as write_v1,
            patch("moe_cache_lab.trace_v2.write_trace_v2") as write_v2,
            patch("moe_cache_lab.trace.read_trace") as read_v1,
            patch("moe_cache_lab.trace_v2.read_trace_v2") as read_v2,
        ):
            granite_result = ProducerResult(trace)
            switch_result = ProducerResult(outcome.trace)

        self.assertIs(granite_result.trace, trace)
        self.assertIs(switch_result.trace, outcome.trace)
        for mocked in (granite, switch, write_v1, write_v2, read_v1, read_v2):
            mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
