from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import moe_cache_lab.producer as producer
from moe_cache_lab.producer import ProducerResult
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.trace_v2 import RoutingEventV2, RoutingStageProfile, RoutingTraceV2


ROOT = Path(__file__).resolve().parents[1]


def _v1_trace() -> RoutingTrace:
    return RoutingTrace(
        model_id="synthetic/v1",
        num_experts=8,
        experts_per_token=1,
        events=(RoutingEvent("prompt", 0, 3, (5,)),),
        created_at="2026-01-01T00:00:00+00:00",
    )


def _v2_trace() -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="synthetic/v2",
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
        created_at="2026-01-01T00:00:00+00:00",
    )


class ProducerResultTests(unittest.TestCase):
    def test_v1_trace_is_retained_by_object_identity(self) -> None:
        trace = _v1_trace()

        result = ProducerResult(trace)

        self.assertIs(result.trace, trace)
        self.assertIs(result.trace.events, trace.events)
        self.assertIs(result.trace.events[0], trace.events[0])
        self.assertEqual(result.trace.expert_requests, ((3, 5),))

    def test_v2_stage_identity_and_unassigned_event_remain_untouched(self) -> None:
        trace = _v2_trace()

        result = ProducerResult(trace)

        self.assertIs(result.trace, trace)
        self.assertIs(result.trace.events, trace.events)
        for actual, expected in zip(result.trace.events, trace.events, strict=True):
            self.assertIs(actual, expected)
        self.assertEqual(
            result.trace.expert_requests,
            (("encoder", 3, 5), ("decoder", 3, 5)),
        )
        unassigned = result.trace.events[1]
        self.assertEqual(unassigned.assignment_state, "unassigned")
        self.assertEqual(unassigned.selected_experts, ())
        self.assertEqual(unassigned.selected_probabilities, ())
        self.assertEqual(unassigned.expert_requests, ())

    def test_result_is_immutable(self) -> None:
        result = ProducerResult(_v1_trace())

        with self.assertRaises(FrozenInstanceError):
            result.trace = _v1_trace()  # type: ignore[misc]

    def test_unsupported_payload_types_reject_deterministically(self) -> None:
        unsupported = (None, object(), {}, _v1_trace().events[0])

        for payload in unsupported:
            with self.subTest(payload_type=type(payload).__name__), self.assertRaisesRegex(
                TypeError,
                "trace must be a canonical RoutingTrace or RoutingTraceV2",
            ):
                ProducerResult(payload)  # type: ignore[arg-type]

    def test_construction_does_not_serialize_or_invoke_collectors(self) -> None:
        trace = _v2_trace()
        with (
            patch("moe_cache_lab.trace.write_trace") as write_v1,
            patch("moe_cache_lab.trace_v2.write_trace_v2") as write_v2,
        ):
            result = ProducerResult(trace)
            self.assertIs(result.trace, trace)

        write_v1.assert_not_called()
        write_v2.assert_not_called()

    def test_public_surface_has_no_framework_or_claim_flags(self) -> None:
        self.assertEqual(
            producer.__all__,
            ["CanonicalRoutingTrace", "ProducerResult"],
        )
        forbidden = {
            "collect",
            "produce",
            "registry",
            "discover",
            "plugin",
            "validated",
            "conforming",
            "non_interfering",
        }
        self.assertTrue(forbidden.isdisjoint(vars(producer)))
        self.assertTrue(forbidden.isdisjoint(vars(ProducerResult)))

    def test_module_imports_with_heavy_dependencies_blocked(self) -> None:
        script = r'''
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'transformers'}:
        raise AssertionError('producer import attempted heavy dependency: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from moe_cache_lab.producer import ProducerResult
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
trace = RoutingTrace('synthetic/v1', 2, 1, (RoutingEvent('prompt', 0, 0, (1,)),))
assert ProducerResult(trace).trace is trace
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
