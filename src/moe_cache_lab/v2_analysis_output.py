"""Deterministic descriptive output for one canonical routing trace v2.

This module renders the reviewed evidence-coverage and cross-model-locality
results.  It does not flatten routing stages, simulate cache state, estimate
transfers, execute a model, or produce a policy/capacity recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .cross_model_locality import (
    CrossModelLocalityDiagnostics,
    analyze_cross_model_locality,
)
from .evidence import EvidenceDiagnostics, EvidenceWorkload, analyze_evidence
from .trace import TRACE_FORMAT
from .trace_v2 import TRACE_VERSION_V2, RoutingTraceV2


V2_ANALYSIS_FORMAT = "moe-cache-lab.routing-evidence-analysis-v2"
V2_ANALYSIS_VERSION = 1


@dataclass(frozen=True)
class V2DescriptiveAnalysis:
    """Reviewed offline diagnostics plus immutable trace provenance."""

    trace: RoutingTraceV2
    workload_id: str
    evidence: EvidenceDiagnostics
    locality: CrossModelLocalityDiagnostics | None


def analyze_v2_trace(
    trace: RoutingTraceV2,
    *,
    workload_id: str = "trace",
    top_k: tuple[int, ...] | None = None,
) -> V2DescriptiveAnalysis:
    """Build the v2 CLI result without inventing locality parameters."""

    if not isinstance(trace, RoutingTraceV2):
        raise TypeError("v2 descriptive analysis requires RoutingTraceV2")
    evidence = analyze_evidence((EvidenceWorkload(workload_id, trace),))
    locality = (
        None if top_k is None else analyze_cross_model_locality(trace, top_k)
    )
    return V2DescriptiveAnalysis(trace, workload_id, evidence, locality)


def v2_analysis_data(result: V2DescriptiveAnalysis) -> dict[str, Any]:
    """Return the closed deterministic machine-readable output boundary."""

    if not isinstance(result, V2DescriptiveAnalysis):
        raise TypeError("result must be V2DescriptiveAnalysis")
    trace = result.trace
    return {
        "format": V2_ANALYSIS_FORMAT,
        "format_version": V2_ANALYSIS_VERSION,
        "trace": {
            "format": TRACE_FORMAT,
            "format_version": TRACE_VERSION_V2,
            "model_id": trace.model_id,
            "model_revision": trace.model_revision,
            "capture_method": trace.capture_method,
            "transformers_version": trace.transformers_version,
            "routing_stages": [
                {
                    "routing_stage": profile.routing_stage,
                    "num_experts": profile.num_experts,
                    "assigned_experts_per_token": (
                        profile.assigned_experts_per_token
                    ),
                    "allows_unassigned": profile.allows_unassigned,
                }
                for profile in trace.routing_stages
            ],
        },
        "workload_id": result.workload_id,
        "evidence": result.evidence.to_dict(),
        "locality": None if result.locality is None else result.locality.to_dict(),
        "claim_boundary": {
            "routing": (
                "Trace-derived descriptive evidence; MEASURED only when trace "
                "provenance establishes measurement."
            ),
            "cache": "No cache simulation is run by this v2 report.",
            "transfer_service": "No transfer-service estimate is produced.",
            "runtime": (
                "This report does not establish native cache behavior, physical "
                "residency, speedup, latency, throughput, or optimal policy/capacity."
            ),
        },
    }


def render_v2_analysis_json(result: V2DescriptiveAnalysis) -> str:
    """Serialize deterministic JSON while retaining exact Fraction objects."""

    return json.dumps(
        v2_analysis_data(result),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def render_v2_analysis_report(result: V2DescriptiveAnalysis) -> str:
    """Render one concise stage-qualified Markdown report."""

    if not isinstance(result, V2DescriptiveAnalysis):
        raise TypeError("result must be V2DescriptiveAnalysis")
    trace = result.trace
    workload = result.evidence.workloads[0]
    assigned = sum(phase.assigned_event_count or 0 for phase in workload.phases)
    unassigned = sum(phase.unassigned_event_count or 0 for phase in workload.phases)
    lines = [
        "# MoE routing evidence report (trace v2)",
        "",
        "## Scope and claim boundary",
        "",
        "- Routing values are descriptive and trace-derived. Treat them as **MEASURED** only when the trace provenance establishes measurement.",
        "- This report runs no cache simulation and presents no **SIMULATED** cache outcome.",
        "- This report produces no **ESTIMATED** transfer-service result.",
        "- It does not establish native cache behavior, physical residency, speedup, latency, throughput, or optimal policy/capacity.",
        "",
        "## Trace provenance",
        "",
        f"- Trace contract: `{TRACE_FORMAT}` version {TRACE_VERSION_V2}",
        f"- Workload ID: `{result.workload_id}`",
        f"- Model: `{trace.model_id}`",
        f"- Model revision: `{_available(trace.model_revision)}`",
        f"- Capture method: `{trace.capture_method}`",
        f"- Transformers version: `{_available(trace.transformers_version)}`",
        "",
        "## Routing evidence",
        "",
        f"- Routing events: {workload.event_count}",
        f"- Stage-qualified expert requests: {workload.expert_request_count}",
        f"- Assigned events: {assigned}",
        f"- Unassigned events: {unassigned}",
        "",
        "### Phase coverage",
        "",
        "| routing stage | phase | events | expert requests | assigned | unassigned | token positions |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for phase in workload.phases:
        positions = ", ".join(str(value) for value in phase.distinct_token_positions)
        lines.append(
            f"| {phase.routing_stage} | {phase.phase} | {phase.event_count} | "
            f"{phase.expert_request_count} | {phase.assigned_event_count} | "
            f"{phase.unassigned_event_count} | {positions or 'N/A'} |"
        )

    lines.extend([
        "",
        "### Routed-layer expert support",
        "",
        "Encoder and decoder layers remain separate `(routing_stage, layer)` identities.",
        "",
        "| routing stage | layer | observed expert IDs | observed/configured support | support fraction |",
        "| --- | ---: | --- | ---: | --- |",
    ])
    for support in workload.expert_support:
        stage, layer = support.layer_identity
        expert_ids = ", ".join(str(value) for value in support.observed_expert_ids)
        lines.append(
            f"| {stage} | {layer} | {expert_ids or 'none'} | "
            f"{len(support.observed_expert_ids)}/{support.configured_expert_count} | "
            f"{_fraction(support.support_fraction)} |"
        )

    lines.extend(["", "### Evidence warnings", ""])
    if result.evidence.warnings:
        lines.extend(
            f"- `{warning.code}`: {warning.text}"
            for warning in result.evidence.warnings
        )
    else:
        lines.append("- None")

    if result.locality is not None:
        lines.extend([
            "",
            "## Caller-requested locality diagnostics",
            "",
            "Requested top-k ranks in caller order: "
            + ", ".join(str(value) for value in result.locality.requested_top_k),
            "",
            "Normalized entropy uses the configured expert universe. Cumulative top-k shares are exact descriptive selection fractions.",
            "",
        ])
        for layer in result.locality.layers:
            stage, layer_id = layer.layer_identity
            lines.extend([
                f"### {stage} / layer {layer_id}",
                "",
                f"- Total selections: {layer.total_selection_count}",
                f"- Configured experts: {layer.configured_expert_count}",
                f"- Normalized Shannon entropy: {_number(layer.normalized_shannon_entropy)}",
                "- Cumulative shares:",
            ])
            for share in layer.cumulative_top_k_selection_shares:
                experts = ", ".join(str(value) for value in share.expert_ids)
                lines.append(
                    f"  - k={share.k}: experts [{experts}], selections "
                    f"{share.selection_count}/{share.total_selection_count}, "
                    f"share {_fraction(share.share)}"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _available(value: str | None) -> str:
    return "unavailable" if value is None or value == "" else value


def _fraction(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{value.numerator}/{value.denominator}"


def _number(value: float | None) -> str:
    return "N/A" if value is None else format(value, ".12g")


__all__ = [
    "V2_ANALYSIS_FORMAT",
    "V2_ANALYSIS_VERSION",
    "V2DescriptiveAnalysis",
    "analyze_v2_trace",
    "v2_analysis_data",
    "render_v2_analysis_json",
    "render_v2_analysis_report",
]
