"""Version-aware, deterministic evidence coverage diagnostics.

This module describes validated routing traces.  It does not collect routing,
simulate cache state, estimate transfers, or grade evidence quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from .trace import TRACE_FORMAT, RoutingTrace
from .trace_v2 import RoutingTraceV2, VersionedRoutingTrace


LayerIdentity = int | tuple[str, int]

_PROVENANCE_FIELDS = (
    "trace_format",
    "trace_format_version",
    "model_id",
    "model_revision",
    "capture_method",
    "transformers_version",
)
_MIXED_WARNING_CODES = {
    "trace_format": "mixed_trace_format",
    "trace_format_version": "mixed_trace_format_version",
    "model_id": "mixed_model_id",
    "model_revision": "mixed_model_revision",
    "capture_method": "mixed_capture_method",
    "transformers_version": "mixed_transformers_version",
}
_MISSING_WARNING_CODES = {
    "model_revision": "missing_model_revision",
    "capture_method": "missing_capture_method",
    "transformers_version": "missing_transformers_version",
}

WARNING_CODES = (
    "single_workload",
    "no_decode_routing",
    "configured_layers_not_fully_observed",
    "zero_expert_requests",
    "mixed_trace_format",
    "mixed_trace_format_version",
    "mixed_model_id",
    "mixed_model_revision",
    "mixed_capture_method",
    "mixed_transformers_version",
    "missing_model_revision",
    "missing_capture_method",
    "missing_transformers_version",
)


@dataclass(frozen=True)
class DeclaredCaptureMetadata:
    """Optional caller-supplied capture facts absent from canonical traces."""

    requested_decode_horizon: int | None = None
    termination: str | None = None

    def __post_init__(self) -> None:
        horizon = self.requested_decode_horizon
        if horizon is not None and (
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0
        ):
            raise ValueError("requested_decode_horizon must be a non-negative integer")
        if self.termination not in {None, "eos", "horizon"}:
            raise ValueError("termination must be 'eos', 'horizon', or None")
        if horizon is None and self.termination is None:
            raise ValueError("declared capture metadata must contain an explicit fact")

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": True,
            "requested_decode_horizon": self.requested_decode_horizon,
            "termination": self.termination,
        }


@dataclass(frozen=True)
class EvidenceWorkload:
    """One stable workload identity and one validated canonical trace."""

    workload_id: str
    trace: VersionedRoutingTrace
    configured_routed_layers: tuple[LayerIdentity, ...] | None = None
    declared_capture: DeclaredCaptureMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workload_id, str) or not self.workload_id.strip():
            raise ValueError("workload_id must be a non-empty string")
        if not isinstance(self.trace, (RoutingTrace, RoutingTraceV2)):
            raise TypeError("trace must be a validated RoutingTrace or RoutingTraceV2")
        if self.declared_capture is not None and not isinstance(
            self.declared_capture, DeclaredCaptureMetadata
        ):
            raise TypeError("declared_capture must be DeclaredCaptureMetadata or None")
        if self.configured_routed_layers is not None:
            if not isinstance(self.configured_routed_layers, tuple):
                raise TypeError("configured_routed_layers must be a tuple or None")
            _validate_configured_layers(self.trace, self.configured_routed_layers)


@dataclass(frozen=True)
class EvidenceWarning:
    code: str
    text: str
    workload_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "text": self.text,
            "workload_id": self.workload_id,
        }


@dataclass(frozen=True)
class PhaseCoverage:
    trace_format_version: int
    routing_stage: str | None
    phase: str
    event_count: int
    expert_request_count: int
    distinct_token_positions: tuple[int, ...]
    assigned_event_count: int | None
    unassigned_event_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_format_version": self.trace_format_version,
            "routing_stage": self.routing_stage,
            "phase": self.phase,
            "event_count": self.event_count,
            "expert_request_count": self.expert_request_count,
            "distinct_token_positions": list(self.distinct_token_positions),
            "distinct_token_position_count": len(self.distinct_token_positions),
            "assigned_event_count": self.assigned_event_count,
            "unassigned_event_count": self.unassigned_event_count,
        }


@dataclass(frozen=True)
class AggregatePhaseCoverage:
    trace_format_version: int
    routing_stage: str | None
    phase: str
    workload_count: int
    event_count: int
    expert_request_count: int
    assigned_event_count: int | None
    unassigned_event_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_format_version": self.trace_format_version,
            "routing_stage": self.routing_stage,
            "phase": self.phase,
            "workload_count": self.workload_count,
            "event_count": self.event_count,
            "expert_request_count": self.expert_request_count,
            "assigned_event_count": self.assigned_event_count,
            "unassigned_event_count": self.unassigned_event_count,
        }


@dataclass(frozen=True)
class LayerCoverage:
    trace_format_version: int
    observed: tuple[LayerIdentity, ...]
    configured: tuple[LayerIdentity, ...] | None
    missing_configured: tuple[LayerIdentity, ...]
    unexpected_observed: tuple[LayerIdentity, ...]
    configured_coverage: Fraction | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_format_version": self.trace_format_version,
            "observed": [_layer_identity_data(self.trace_format_version, item) for item in self.observed],
            "configured_available": self.configured is not None,
            "configured": (
                None
                if self.configured is None
                else [_layer_identity_data(self.trace_format_version, item) for item in self.configured]
            ),
            "missing_configured": [
                _layer_identity_data(self.trace_format_version, item)
                for item in self.missing_configured
            ],
            "unexpected_observed": [
                _layer_identity_data(self.trace_format_version, item)
                for item in self.unexpected_observed
            ],
            "configured_coverage": _fraction_data(self.configured_coverage),
        }


@dataclass(frozen=True)
class ExpertSupport:
    trace_format_version: int
    layer_identity: LayerIdentity
    observed_expert_ids: tuple[int, ...]
    configured_expert_count: int | None
    support_fraction: Fraction | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_identity": _layer_identity_data(
                self.trace_format_version, self.layer_identity
            ),
            "observed_expert_ids": list(self.observed_expert_ids),
            "observed_expert_count": len(self.observed_expert_ids),
            "configured_expert_count": self.configured_expert_count,
            "support_fraction": _fraction_data(self.support_fraction),
        }


@dataclass(frozen=True)
class WorkloadEvidence:
    workload_id: str
    trace_format_version: int
    event_count: int
    expert_request_count: int
    event_share: Fraction
    expert_request_share: Fraction | None
    phases: tuple[PhaseCoverage, ...]
    observed_decode_input_positions: tuple[int, ...]
    layer_coverage: LayerCoverage
    expert_support: tuple[ExpertSupport, ...]
    declared_capture: DeclaredCaptureMetadata | None

    def to_dict(self) -> dict[str, Any]:
        positions = self.observed_decode_input_positions
        return {
            "workload_id": self.workload_id,
            "trace_format_version": self.trace_format_version,
            "event_count": self.event_count,
            "expert_request_count": self.expert_request_count,
            "event_share": _fraction_data(self.event_share),
            "expert_request_share": _fraction_data(self.expert_request_share),
            "phase_coverage": [phase.to_dict() for phase in self.phases],
            "observed_decode_input": {
                "positions": list(positions),
                "position_count": len(positions),
                "position_span": (
                    None if not positions else {"first": positions[0], "last": positions[-1]}
                ),
            },
            "routed_layer_coverage": self.layer_coverage.to_dict(),
            "expert_support": [support.to_dict() for support in self.expert_support],
            "declared_capture": (
                {"available": False, "requested_decode_horizon": None, "termination": None}
                if self.declared_capture is None
                else self.declared_capture.to_dict()
            ),
        }


@dataclass(frozen=True)
class DominanceSummary:
    total_event_count: int
    total_expert_request_count: int
    largest_event_share: Fraction
    largest_event_share_workload_ids: tuple[str, ...]
    largest_expert_request_share: Fraction | None
    largest_expert_request_share_workload_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_event_count": self.total_event_count,
            "total_expert_request_count": self.total_expert_request_count,
            "largest_event_share": _fraction_data(self.largest_event_share),
            "largest_event_share_workload_ids": list(
                self.largest_event_share_workload_ids
            ),
            "largest_expert_request_share": _fraction_data(
                self.largest_expert_request_share
            ),
            "largest_expert_request_share_workload_ids": list(
                self.largest_expert_request_share_workload_ids
            ),
        }


@dataclass(frozen=True)
class LayerSetComparison:
    trace_format_version: int
    workload_ids: tuple[str, ...]
    union: tuple[LayerIdentity, ...]
    intersection: tuple[LayerIdentity, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_format_version": self.trace_format_version,
            "workload_ids": list(self.workload_ids),
            "union": [
                _layer_identity_data(self.trace_format_version, item)
                for item in self.union
            ],
            "intersection": [
                _layer_identity_data(self.trace_format_version, item)
                for item in self.intersection
            ],
        }


@dataclass(frozen=True)
class ProvenanceFieldSummary:
    field: str
    concrete_values: tuple[str | int, ...]
    missing_workload_ids: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "concrete_values": list(self.concrete_values),
            "missing_workload_ids": list(self.missing_workload_ids),
            "status": self.status,
        }


@dataclass(frozen=True)
class EvidenceDiagnostics:
    workloads: tuple[WorkloadEvidence, ...]
    dominance: DominanceSummary
    aggregate_phases: tuple[AggregatePhaseCoverage, ...]
    layer_set_comparisons: tuple[LayerSetComparison, ...]
    provenance: tuple[ProvenanceFieldSummary, ...]
    warnings: tuple[EvidenceWarning, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic structured boundary for later renderers."""
        return {
            "format": "moe-cache-lab.evidence-coverage",
            "format_version": 1,
            "workload_count": len(self.workloads),
            "dominance": self.dominance.to_dict(),
            "workloads": [workload.to_dict() for workload in self.workloads],
            "aggregate_phase_coverage": [
                phase.to_dict() for phase in self.aggregate_phases
            ],
            "routed_layer_comparisons": [
                comparison.to_dict() for comparison in self.layer_set_comparisons
            ],
            "provenance": [field.to_dict() for field in self.provenance],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


def analyze_evidence(workloads: Iterable[EvidenceWorkload]) -> EvidenceDiagnostics:
    """Describe an explicitly ordered, non-empty collection of canonical traces."""
    try:
        ordered = tuple(workloads)
    except TypeError as exc:
        raise TypeError("workloads must be an iterable of EvidenceWorkload objects") from exc
    if not ordered:
        raise ValueError("evidence diagnostics require at least one workload")
    if any(not isinstance(item, EvidenceWorkload) for item in ordered):
        raise TypeError("workloads must contain EvidenceWorkload objects")
    ids = tuple(item.workload_id for item in ordered)
    if len(set(ids)) != len(ids):
        raise ValueError("workload IDs must be unique")

    total_events = sum(len(item.trace.events) for item in ordered)
    total_requests = sum(len(item.trace.expert_requests) for item in ordered)
    summaries = tuple(
        _summarize_workload(item, total_events, total_requests) for item in ordered
    )
    largest_event_share = max(item.event_share for item in summaries)
    if total_requests:
        largest_request_share = max(
            item.expert_request_share
            for item in summaries
            if item.expert_request_share is not None
        )
        largest_request_ids = tuple(
            item.workload_id
            for item in summaries
            if item.expert_request_share == largest_request_share
        )
    else:
        largest_request_share = None
        largest_request_ids = ()

    provenance = _summarize_provenance(ordered)
    warnings = _warnings(ordered, summaries, provenance)
    return EvidenceDiagnostics(
        workloads=summaries,
        dominance=DominanceSummary(
            total_event_count=total_events,
            total_expert_request_count=total_requests,
            largest_event_share=largest_event_share,
            largest_event_share_workload_ids=tuple(
                item.workload_id
                for item in summaries
                if item.event_share == largest_event_share
            ),
            largest_expert_request_share=largest_request_share,
            largest_expert_request_share_workload_ids=largest_request_ids,
        ),
        aggregate_phases=_aggregate_phases(summaries),
        layer_set_comparisons=_layer_comparisons(summaries),
        provenance=provenance,
        warnings=warnings,
    )


def _summarize_workload(
    workload: EvidenceWorkload, total_events: int, total_requests: int
) -> WorkloadEvidence:
    trace = workload.trace
    version = 1 if isinstance(trace, RoutingTrace) else 2
    event_count = len(trace.events)
    request_count = len(trace.expert_requests)
    phases = _phase_coverage(trace)
    observed_layers = _observed_layers(trace)
    configured = workload.configured_routed_layers
    if configured is None:
        missing = ()
        unexpected = ()
        coverage = None
    else:
        observed_set = set(observed_layers)
        configured_set = set(configured)
        missing = _sort_layers(version, configured_set.difference(observed_set))
        unexpected = _sort_layers(version, observed_set.difference(configured_set))
        coverage = Fraction(
            len(configured_set.intersection(observed_set)), len(configured_set)
        )

    decode_phase = "generated" if version == 1 else "decoder_generated"
    decode_positions = tuple(
        sorted(
            {
                event.token_position
                for event in trace.events
                if event.phase == decode_phase
            }
        )
    )
    return WorkloadEvidence(
        workload_id=workload.workload_id,
        trace_format_version=version,
        event_count=event_count,
        expert_request_count=request_count,
        event_share=Fraction(event_count, total_events),
        expert_request_share=(
            None if not total_requests else Fraction(request_count, total_requests)
        ),
        phases=phases,
        observed_decode_input_positions=decode_positions,
        layer_coverage=LayerCoverage(
            trace_format_version=version,
            observed=observed_layers,
            configured=configured,
            missing_configured=missing,
            unexpected_observed=unexpected,
            configured_coverage=coverage,
        ),
        expert_support=_expert_support(trace, observed_layers),
        declared_capture=workload.declared_capture,
    )


def _phase_coverage(trace: VersionedRoutingTrace) -> tuple[PhaseCoverage, ...]:
    if isinstance(trace, RoutingTrace):
        keys: tuple[tuple[str | None, str], ...] = (
            (None, "prompt"),
            (None, "generated"),
        )
        version = 1
    else:
        profile_stages = {profile.routing_stage for profile in trace.routing_stages}
        keys = tuple(
            key
            for key in (
                ("encoder", "source"),
                ("decoder", "decoder_prompt"),
                ("decoder", "decoder_generated"),
            )
            if key[0] in profile_stages
        )
        version = 2
    result = []
    for stage, phase in keys:
        events = tuple(
            event
            for event in trace.events
            if event.phase == phase
            and (stage is None or getattr(event, "routing_stage", None) == stage)
        )
        assigned = (
            None
            if version == 1
            else sum(event.assignment_state == "assigned" for event in events)
        )
        unassigned = (
            None
            if version == 1
            else sum(event.assignment_state == "unassigned" for event in events)
        )
        result.append(
            PhaseCoverage(
                trace_format_version=version,
                routing_stage=stage,
                phase=phase,
                event_count=len(events),
                expert_request_count=sum(len(event.selected_experts) for event in events),
                distinct_token_positions=tuple(
                    sorted({event.token_position for event in events})
                ),
                assigned_event_count=assigned,
                unassigned_event_count=unassigned,
            )
        )
    return tuple(result)


def _observed_layers(trace: VersionedRoutingTrace) -> tuple[LayerIdentity, ...]:
    if isinstance(trace, RoutingTrace):
        return tuple(sorted({event.layer for event in trace.events}))
    return tuple(
        sorted(
            {(event.routing_stage, event.layer) for event in trace.events},
            key=_v2_layer_sort_key,
        )
    )


def _expert_support(
    trace: VersionedRoutingTrace, observed_layers: tuple[LayerIdentity, ...]
) -> tuple[ExpertSupport, ...]:
    supports = []
    if isinstance(trace, RoutingTrace):
        for layer in observed_layers:
            assert isinstance(layer, int)
            observed = tuple(
                sorted(
                    {
                        expert
                        for event in trace.events
                        if event.layer == layer
                        for expert in event.selected_experts
                    }
                )
            )
            configured = trace.num_experts
            supports.append(
                ExpertSupport(
                    1,
                    layer,
                    observed,
                    configured,
                    None if configured is None else Fraction(len(observed), configured),
                )
            )
    else:
        configured_by_stage = {
            profile.routing_stage: profile.num_experts
            for profile in trace.routing_stages
        }
        for identity in observed_layers:
            assert isinstance(identity, tuple)
            stage, layer = identity
            observed = tuple(
                sorted(
                    {
                        expert
                        for event in trace.events
                        if event.routing_stage == stage and event.layer == layer
                        for expert in event.selected_experts
                    }
                )
            )
            configured = configured_by_stage[stage]
            supports.append(
                ExpertSupport(
                    2,
                    identity,
                    observed,
                    configured,
                    Fraction(len(observed), configured),
                )
            )
    return tuple(supports)


def _aggregate_phases(
    summaries: tuple[WorkloadEvidence, ...]
) -> tuple[AggregatePhaseCoverage, ...]:
    keys = []
    for summary in summaries:
        for phase in summary.phases:
            key = phase.trace_format_version, phase.routing_stage, phase.phase
            if key not in keys:
                keys.append(key)
    result = []
    for version, stage, phase_name in keys:
        phases = tuple(
            phase
            for summary in summaries
            for phase in summary.phases
            if (phase.trace_format_version, phase.routing_stage, phase.phase)
            == (version, stage, phase_name)
        )
        result.append(
            AggregatePhaseCoverage(
                trace_format_version=version,
                routing_stage=stage,
                phase=phase_name,
                workload_count=len(phases),
                event_count=sum(item.event_count for item in phases),
                expert_request_count=sum(item.expert_request_count for item in phases),
                assigned_event_count=(
                    None
                    if version == 1
                    else sum(item.assigned_event_count or 0 for item in phases)
                ),
                unassigned_event_count=(
                    None
                    if version == 1
                    else sum(item.unassigned_event_count or 0 for item in phases)
                ),
            )
        )
    return tuple(result)


def _layer_comparisons(
    summaries: tuple[WorkloadEvidence, ...]
) -> tuple[LayerSetComparison, ...]:
    result = []
    for version in (1, 2):
        comparable = tuple(
            item for item in summaries if item.trace_format_version == version
        )
        if len(comparable) < 2:
            continue
        sets = tuple(set(item.layer_coverage.observed) for item in comparable)
        result.append(
            LayerSetComparison(
                trace_format_version=version,
                workload_ids=tuple(item.workload_id for item in comparable),
                union=_sort_layers(version, set().union(*sets)),
                intersection=_sort_layers(version, set.intersection(*sets)),
            )
        )
    return tuple(result)


def _summarize_provenance(
    workloads: tuple[EvidenceWorkload, ...]
) -> tuple[ProvenanceFieldSummary, ...]:
    summaries = []
    for field in _PROVENANCE_FIELDS:
        concrete = []
        missing = []
        for workload in workloads:
            value = _provenance_value(workload.trace, field)
            if value is None or (field == "capture_method" and value == "unknown"):
                missing.append(workload.workload_id)
            elif value not in concrete:
                concrete.append(value)
        concrete_values = tuple(sorted(concrete, key=lambda value: (str(type(value)), str(value))))
        if not concrete_values:
            status = "unavailable"
        elif len(concrete_values) > 1 and missing:
            status = "mixed_and_inconsistently_missing"
        elif len(concrete_values) > 1:
            status = "mixed"
        elif missing:
            status = "inconsistently_missing"
        else:
            status = "consistent"
        summaries.append(
            ProvenanceFieldSummary(field, concrete_values, tuple(missing), status)
        )
    return tuple(summaries)


def _warnings(
    workloads: tuple[EvidenceWorkload, ...],
    summaries: tuple[WorkloadEvidence, ...],
    provenance: tuple[ProvenanceFieldSummary, ...],
) -> tuple[EvidenceWarning, ...]:
    result = []
    if len(workloads) == 1:
        result.append(
            EvidenceWarning(
                "single_workload",
                "Only one workload is present; dominance is descriptive and cross-workload coverage is unavailable.",
            )
        )
    for workload, summary in zip(workloads, summaries):
        if not summary.observed_decode_input_positions:
            result.append(
                EvidenceWarning(
                    "no_decode_routing",
                    "No generated/decode-input routing positions were observed for this workload.",
                    workload.workload_id,
                )
            )
        if summary.layer_coverage.missing_configured:
            result.append(
                EvidenceWarning(
                    "configured_layers_not_fully_observed",
                    "One or more explicitly configured routed layers were not observed.",
                    workload.workload_id,
                )
            )
        if summary.expert_request_count == 0:
            result.append(
                EvidenceWarning(
                    "zero_expert_requests",
                    "The workload contains routing events but no actual expert requests.",
                    workload.workload_id,
                )
            )
    for field in provenance:
        if len(field.concrete_values) > 1:
            result.append(
                EvidenceWarning(
                    _MIXED_WARNING_CODES[field.field],
                    f"Concrete {field.field} values are mixed; direct cross-workload comparison must preserve that distinction.",
                )
            )
        missing_code = _MISSING_WARNING_CODES.get(field.field)
        if field.missing_workload_ids and missing_code is not None:
            result.append(
                EvidenceWarning(
                    missing_code,
                    f"{field.field} is unavailable for one or more workloads; missing provenance was not normalized.",
                )
            )
    return tuple(result)


def _provenance_value(trace: VersionedRoutingTrace, field: str) -> str | int | None:
    if field == "trace_format":
        return TRACE_FORMAT
    if field == "trace_format_version":
        return 1 if isinstance(trace, RoutingTrace) else 2
    return getattr(trace, field)


def _validate_configured_layers(
    trace: VersionedRoutingTrace, configured: tuple[LayerIdentity, ...]
) -> None:
    if not configured:
        raise ValueError("configured_routed_layers cannot be empty when supplied")
    version = 1 if isinstance(trace, RoutingTrace) else 2
    if version == 1:
        valid = all(
            not isinstance(item, bool) and isinstance(item, int) and item >= 0
            for item in configured
        )
    else:
        stages = {profile.routing_stage for profile in trace.routing_stages}
        valid = all(
            isinstance(item, tuple)
            and len(item) == 2
            and item[0] in stages
            and not isinstance(item[1], bool)
            and isinstance(item[1], int)
            and item[1] >= 0
            for item in configured
        )
    if not valid:
        raise ValueError("configured routed-layer identity does not match trace version")
    if len(set(configured)) != len(configured):
        raise ValueError("configured_routed_layers must be unique")
    if configured != _sort_layers(version, set(configured)):
        raise ValueError("configured_routed_layers must use canonical sorted order")


def _sort_layers(
    version: int, values: set[LayerIdentity]
) -> tuple[LayerIdentity, ...]:
    if version == 1:
        return tuple(sorted(values))
    return tuple(sorted(values, key=_v2_layer_sort_key))


def _v2_layer_sort_key(identity: LayerIdentity) -> tuple[int, int]:
    assert isinstance(identity, tuple)
    stage, layer = identity
    return ((0 if stage == "encoder" else 1), layer)


def _layer_identity_data(version: int, identity: LayerIdentity) -> dict[str, Any]:
    if version == 1:
        assert isinstance(identity, int)
        return {"layer": identity}
    assert isinstance(identity, tuple)
    return {"routing_stage": identity[0], "layer": identity[1]}


def _fraction_data(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


__all__ = [
    "DeclaredCaptureMetadata",
    "EvidenceDiagnostics",
    "EvidenceWarning",
    "EvidenceWorkload",
    "WARNING_CODES",
    "analyze_evidence",
]
