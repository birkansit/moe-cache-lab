from fractions import Fraction
import json
import unittest

from moe_cache_lab.evidence import (
    DeclaredCaptureMetadata,
    EvidenceWorkload,
    WARNING_CODES,
    analyze_evidence,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
)


def _v1_trace(
    events: tuple[RoutingEvent, ...],
    *,
    model_id: str = "model/a",
    revision: str | None = "rev-a",
    capture: str = "observer-a",
    transformers: str | None = "5.12.0",
    num_experts: int | None = 4,
) -> RoutingTrace:
    return RoutingTrace(
        model_id=model_id,
        num_experts=num_experts,
        experts_per_token=None,
        events=events,
        capture_method=capture,
        created_at="synthetic",
        transformers_version=transformers,
        model_revision=revision,
    )


def _v2_trace(
    events: tuple[RoutingEventV2, ...],
    *,
    model_id: str = "model/switch",
    revision: str | None = "rev-switch",
    capture: str = "switch-observer",
    transformers: str | None = "5.12.0",
    encoder_experts: int = 8,
    decoder_experts: int = 8,
) -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id=model_id,
        routing_stages=(
            RoutingStageProfile("encoder", encoder_experts, 1, True),
            RoutingStageProfile("decoder", decoder_experts, 1, True),
        ),
        events=events,
        capture_method=capture,
        created_at="synthetic",
        transformers_version=transformers,
        model_revision=revision,
    )


def _assigned_v2(
    stage: str, phase: str, position: int, layer: int, expert: int
) -> RoutingEventV2:
    return RoutingEventV2(
        stage,
        phase,
        position,
        layer,
        "assigned",
        (expert,),
        (),
    )


def _unassigned_v2(
    stage: str, phase: str, position: int, layer: int
) -> RoutingEventV2:
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


def _phase(summary, stage, phase):
    return next(
        item
        for item in summary.phases
        if item.routing_stage == stage and item.phase == phase
    )


def _provenance(result, field):
    return next(item for item in result.provenance if item.field == field)


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


class EvidenceCoverageTests(unittest.TestCase):
    def test_single_v1_counts_shares_phases_decode_and_support(self) -> None:
        trace = _v1_trace(
            (
                RoutingEvent("prompt", 0, 0, (0,)),
                RoutingEvent("prompt", 1, 0, (1,)),
                RoutingEvent("generated", 2, 0, (1,)),
            )
        )
        result = analyze_evidence((EvidenceWorkload("only", trace),))
        workload = result.workloads[0]

        self.assertEqual(result.to_dict()["workload_count"], 1)
        self.assertEqual((workload.event_count, workload.expert_request_count), (3, 3))
        self.assertEqual((workload.event_share, workload.expert_request_share), (Fraction(1), Fraction(1)))
        self.assertEqual(
            [
                (phase.routing_stage, phase.phase, phase.event_count, phase.expert_request_count)
                for phase in workload.phases
            ],
            [(None, "prompt", 2, 2), (None, "generated", 1, 1)],
        )
        self.assertEqual(workload.observed_decode_input_positions, (2,))
        support = workload.expert_support[0]
        self.assertEqual(support.observed_expert_ids, (0, 1))
        self.assertEqual(support.configured_expert_count, 4)
        self.assertEqual(support.support_fraction, Fraction(1, 2))
        self.assertIsNone(workload.layer_coverage.configured_coverage)
        capture = workload.to_dict()["declared_capture"]
        self.assertEqual(capture, {"available": False, "requested_decode_horizon": None, "termination": None})

    def test_event_and_request_dominance_are_independent_exact_shares(self) -> None:
        event_heavy = _v1_trace(
            (
                RoutingEvent("prompt", 0, 0, (0,)),
                RoutingEvent("prompt", 1, 0, (0,)),
                RoutingEvent("prompt", 2, 0, (0,)),
            )
        )
        request_heavy = _v1_trace(
            (RoutingEvent("prompt", 0, 0, (0, 1, 2, 3)),),
        )
        result = analyze_evidence(
            (
                EvidenceWorkload("event-heavy", event_heavy),
                EvidenceWorkload("request-heavy", request_heavy),
            )
        )

        self.assertEqual(result.dominance.largest_event_share, Fraction(3, 4))
        self.assertEqual(result.dominance.largest_event_share_workload_ids, ("event-heavy",))
        self.assertEqual(result.dominance.largest_expert_request_share, Fraction(4, 7))
        self.assertEqual(result.dominance.largest_expert_request_share_workload_ids, ("request-heavy",))
        self.assertNotIn("dominance", [warning.code for warning in result.warnings])

    def test_v2_phases_unassigned_counts_and_stage_namespaces_remain_distinct(self) -> None:
        trace = _v2_trace(
            (
                _assigned_v2("encoder", "source", 0, 1, 3),
                _unassigned_v2("decoder", "decoder_prompt", 0, 1),
                _assigned_v2("decoder", "decoder_generated", 1, 1, 3),
            ),
            encoder_experts=8,
            decoder_experts=4,
        )
        workload = analyze_evidence((EvidenceWorkload("switch", trace),)).workloads[0]

        self.assertEqual((workload.event_count, workload.expert_request_count), (3, 2))
        self.assertEqual(
            (
                _phase(workload, "encoder", "source").event_count,
                _phase(workload, "decoder", "decoder_prompt").unassigned_event_count,
                _phase(workload, "decoder", "decoder_generated").expert_request_count,
            ),
            (1, 1, 1),
        )
        self.assertEqual(workload.observed_decode_input_positions, (1,))
        supports = {item.layer_identity: item for item in workload.expert_support}
        self.assertEqual(set(supports), {("encoder", 1), ("decoder", 1)})
        self.assertEqual(supports[("encoder", 1)].support_fraction, Fraction(1, 8))
        self.assertEqual(supports[("decoder", 1)].support_fraction, Fraction(1, 4))
        self.assertEqual(supports[("decoder", 1)].observed_expert_ids, (3,))

    def test_all_unassigned_v2_events_have_events_but_no_requests_or_support(self) -> None:
        trace = _v2_trace(
            (_unassigned_v2("decoder", "decoder_prompt", 0, 1),)
        )
        result = analyze_evidence((EvidenceWorkload("dropped", trace),))
        workload = result.workloads[0]

        self.assertEqual((workload.event_count, workload.expert_request_count), (1, 0))
        self.assertIsNone(workload.expert_request_share)
        self.assertEqual(workload.expert_support[0].observed_expert_ids, ())
        self.assertEqual(workload.expert_support[0].support_fraction, Fraction(0, 1))
        self.assertIn("zero_expert_requests", [warning.code for warning in result.warnings])
        self.assertIsNone(result.dominance.largest_expert_request_share)
        self.assertEqual(result.dominance.largest_expert_request_share_workload_ids, ())

    def test_v2_layer_union_and_intersection_preserve_stage_identity(self) -> None:
        first = _v2_trace(
            (
                _assigned_v2("encoder", "source", 0, 1, 0),
                _assigned_v2("decoder", "decoder_prompt", 0, 1, 0),
            )
        )
        second = _v2_trace(
            (
                _assigned_v2("encoder", "source", 0, 1, 1),
                _assigned_v2("decoder", "decoder_prompt", 0, 2, 1),
            )
        )
        comparison = analyze_evidence(
            (EvidenceWorkload("first", first), EvidenceWorkload("second", second))
        ).layer_set_comparisons[0]

        self.assertEqual(comparison.trace_format_version, 2)
        self.assertEqual(
            comparison.union,
            (("encoder", 1), ("decoder", 1), ("decoder", 2)),
        )
        self.assertEqual(comparison.intersection, (("encoder", 1),))

    def test_configured_layer_coverage_is_explicit_or_unavailable(self) -> None:
        trace = _v1_trace((RoutingEvent("prompt", 0, 0, (0,)),))
        configured = analyze_evidence(
            (EvidenceWorkload("configured", trace, configured_routed_layers=(0, 1)),)
        ).workloads[0]
        unknown = analyze_evidence((EvidenceWorkload("unknown", trace),)).workloads[0]

        self.assertEqual(configured.layer_coverage.configured_coverage, Fraction(1, 2))
        self.assertEqual(configured.layer_coverage.missing_configured, (1,))
        self.assertEqual(configured.layer_coverage.unexpected_observed, ())
        self.assertIsNone(unknown.layer_coverage.configured)
        self.assertIsNone(unknown.layer_coverage.configured_coverage)
        self.assertFalse(unknown.layer_coverage.to_dict()["configured_available"])

    def test_declared_capture_is_separate_from_observed_decode_positions(self) -> None:
        trace = _v1_trace(
            (
                RoutingEvent("prompt", 0, 0, (0,)),
                RoutingEvent("generated", 1, 0, (1,)),
            )
        )
        explicit = EvidenceWorkload(
            "explicit",
            trace,
            declared_capture=DeclaredCaptureMetadata(8, "eos"),
        )
        explicit_data = analyze_evidence((explicit,)).workloads[0].to_dict()
        absent_data = analyze_evidence((EvidenceWorkload("absent", trace),)).workloads[0].to_dict()

        self.assertEqual(explicit_data["observed_decode_input"]["position_count"], 1)
        self.assertEqual(explicit_data["declared_capture"]["requested_decode_horizon"], 8)
        self.assertEqual(explicit_data["declared_capture"]["termination"], "eos")
        self.assertEqual(absent_data["declared_capture"]["available"], False)
        self.assertIsNone(absent_data["declared_capture"]["requested_decode_horizon"])
        self.assertIsNone(absent_data["declared_capture"]["termination"])

    def test_mixed_and_missing_provenance_are_distinct_and_deterministic(self) -> None:
        event = (RoutingEvent("prompt", 0, 0, (0,)),)
        workloads = (
            EvidenceWorkload("a", _v1_trace(event, revision="a", capture="hook-a", transformers="5.12")),
            EvidenceWorkload("b", _v1_trace(event, revision="b", capture="hook-b", transformers="5.13")),
            EvidenceWorkload("missing", _v1_trace(event, revision=None, capture="unknown", transformers=None)),
        )
        result = analyze_evidence(workloads)

        self.assertEqual(_provenance(result, "model_revision").concrete_values, ("a", "b"))
        self.assertEqual(
            _provenance(result, "trace_format").concrete_values,
            ("moe-cache-lab.routing-jsonl",),
        )
        self.assertEqual(_provenance(result, "model_revision").missing_workload_ids, ("missing",))
        self.assertEqual(_provenance(result, "model_revision").status, "mixed_and_inconsistently_missing")
        self.assertEqual(_provenance(result, "capture_method").status, "mixed_and_inconsistently_missing")
        self.assertEqual(_provenance(result, "transformers_version").status, "mixed_and_inconsistently_missing")
        codes = [warning.code for warning in result.warnings]
        self.assertEqual(
            codes[-6:],
            [
                "mixed_model_revision",
                "missing_model_revision",
                "mixed_capture_method",
                "missing_capture_method",
                "mixed_transformers_version",
                "missing_transformers_version",
            ],
        )

    def test_warning_codes_order_and_result_have_no_quality_score(self) -> None:
        trace = _v1_trace(
            (RoutingEvent("prompt", 0, 0, (0,)),),
            revision=None,
            capture="unknown",
            transformers=None,
        )
        result = analyze_evidence(
            (EvidenceWorkload("thin", trace, configured_routed_layers=(0, 1)),)
        )
        codes = [warning.code for warning in result.warnings]

        self.assertEqual(
            codes,
            [
                "single_workload",
                "no_decode_routing",
                "configured_layers_not_fully_observed",
                "missing_model_revision",
                "missing_capture_method",
                "missing_transformers_version",
            ],
        )
        self.assertTrue(set(codes).issubset(WARNING_CODES))
        forbidden = {"representativeness_score", "confidence_score", "quality_score", "quality_grade"}
        self.assertTrue(forbidden.isdisjoint(_all_keys(result.to_dict())))

    def test_input_order_and_serialization_are_deterministic(self) -> None:
        trace = _v1_trace((RoutingEvent("prompt", 0, 0, (0,)),))
        workloads = (EvidenceWorkload("z", trace), EvidenceWorkload("a", trace))
        first = analyze_evidence(workloads)
        second = analyze_evidence(workloads)

        self.assertEqual([item.workload_id for item in first.workloads], ["z", "a"])
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            json.dumps(first.to_dict(), sort_keys=True, separators=(",", ":")),
            json.dumps(second.to_dict(), sort_keys=True, separators=(",", ":")),
        )

    def test_invalid_workload_boundaries_fail_without_trace_repair(self) -> None:
        trace = _v1_trace((RoutingEvent("prompt", 0, 0, (0,)),))
        with self.assertRaisesRegex(ValueError, "at least one workload"):
            analyze_evidence(())
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze_evidence((EvidenceWorkload("same", trace), EvidenceWorkload("same", trace)))
        with self.assertRaisesRegex(ValueError, "canonical sorted order"):
            EvidenceWorkload("bad-order", trace, configured_routed_layers=(1, 0))
        with self.assertRaisesRegex(ValueError, "does not match trace version"):
            EvidenceWorkload("bad-identity", trace, configured_routed_layers=(("encoder", 0),))
        with self.assertRaisesRegex(ValueError, "explicit fact"):
            DeclaredCaptureMetadata()


if __name__ == "__main__":
    unittest.main()
