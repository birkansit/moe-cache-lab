"""Deterministic rendering and JSON serialization for routing analysis summaries."""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from .analysis import (
    ConsecutiveOverlapSummary,
    ExpertFrequency,
    FrequencyConcentrationSummary,
    ReuseGapSummary,
    RoutingLocalitySummary,
)

_JSON_FORMAT = "moe-cache-lab.routing-analysis"
_JSON_FORMAT_VERSION = 1


def render_analysis_report(summary: RoutingLocalitySummary) -> str:
    """Render a deterministic human-readable report from an analysis summary."""
    lines = [
        "# MoE routing analysis report",
        "",
        "## Scope and claim boundary",
        "",
        "- **MEASURED:** routing selections may originate from a measured trace; this command does not collect new measurements.",
        "- **SIMULATED:** this report does not run a cache simulation or present simulated cache outcomes.",
        "- **ESTIMATED:** this report does not produce hardware or runtime estimates.",
        "- This is descriptive routing-trace analysis. It does not establish cache/offload speedup, latency, throughput, physical GPU residency, or transfer performance.",
        "- Frequency concentration is reported per layer and per phase-layer stream only; no global concentration or cacheability score is produced.",
        "- observed-support Gini uses only expert IDs selected in that summary; unselected experts are excluded rather than treated as zero-count members of the configured expert universe.",
        "- Shannon entropy is reported in bits using floating-point log2 arithmetic; unlike Fraction-valued metrics, it is not an exact rational result.",
        "",
        "## Global summary",
        "",
        f"- Routing events: {summary.total_event_count}",
        f"- Layer-qualified expert assignments: {summary.total_assignment_count}",
        f"- Unique layer-qualified experts: {summary.unique_layer_expert_count}",
        "",
        "## Phase counts",
        "",
        "| phase | events | assignments |",
        "| --- | ---: | ---: |",
        f"| prompt/prefill | {summary.prompt.event_count} | {summary.prompt.assignment_count} |",
        f"| generated/decode | {summary.generated.event_count} | {summary.generated.assignment_count} |",
        "",
        "## Per-layer expert frequencies",
        "",
    ]
    if summary.layers:
        for layer in summary.layers:
            lines.extend([
                f"### Layer {layer.layer_id}",
                "",
                f"- Events: {layer.event_count}",
                f"- Assignments: {layer.assignment_count}",
                f"- Unique experts: {layer.unique_expert_count}",
                f"- Expert frequencies: {_format_frequencies(layer.expert_frequencies)}",
                "- Frequency concentration:",
                *[f"  {line}" for line in _concentration_report_lines(layer.concentration)],
                "",
            ])
    else:
        lines.extend(["N/A", ""])

    lines.extend([
        "## Global locality",
        "",
        "### Consecutive expert-set overlap",
        "",
        *_overlap_report_lines(summary.consecutive_overlap),
        "",
        "### Reuse gaps",
        "",
        *_reuse_report_lines(summary.reuse_gap),
        "",
        "## Phase-layer summaries",
        "",
    ])
    if summary.phase_layers:
        for stream in summary.phase_layers:
            lines.extend([
                f"### {_phase_label(stream.phase)} / layer {stream.layer_id}",
                "",
                f"- Events: {stream.event_count}",
                f"- Assignments: {stream.assignment_count}",
                f"- Unique experts: {stream.unique_expert_count}",
                f"- Expert frequencies: {_format_frequencies(stream.expert_frequencies)}",
                "- Frequency concentration:",
                *[f"  {line}" for line in _concentration_report_lines(stream.concentration)],
                "- Consecutive overlap:",
                *[f"  {line}" for line in _overlap_report_lines(stream.consecutive_overlap)],
                "- Reuse gaps:",
                *[f"  {line}" for line in _reuse_report_lines(stream.reuse_gap)],
                "",
            ])
    else:
        lines.extend(["N/A", ""])
    return "\n".join(lines)


def render_analysis_json(summary: RoutingLocalitySummary) -> str:
    """Serialize a summary deterministically while preserving exact fractions.

    Every :class:`fractions.Fraction` is represented as an object containing
    integer ``numerator`` and ``denominator`` fields. Undefined values are JSON
    ``null``. Shannon entropy is the one distribution metric emitted as a JSON
    floating-point number because its base-2 logarithm is not generally exact.
    """
    return json.dumps(
        analysis_summary_data(summary),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def analysis_summary_data(summary: RoutingLocalitySummary) -> dict[str, Any]:
    """Return the stable machine-readable representation used by JSON output."""
    return {
        "format": _JSON_FORMAT,
        "format_version": _JSON_FORMAT_VERSION,
        "total_event_count": summary.total_event_count,
        "total_assignment_count": summary.total_assignment_count,
        "unique_layer_expert_count": summary.unique_layer_expert_count,
        "phases": [
            {
                "phase": summary.prompt.phase,
                "event_count": summary.prompt.event_count,
                "assignment_count": summary.prompt.assignment_count,
            },
            {
                "phase": summary.generated.phase,
                "event_count": summary.generated.event_count,
                "assignment_count": summary.generated.assignment_count,
            },
        ],
        "layers": [
            {
                "layer_id": layer.layer_id,
                "event_count": layer.event_count,
                "assignment_count": layer.assignment_count,
                "unique_expert_count": layer.unique_expert_count,
                "expert_frequencies": _frequency_data(layer.expert_frequencies),
                "frequency_concentration": _concentration_data(layer.concentration),
            }
            for layer in summary.layers
        ],
        "consecutive_overlap": _overlap_data(summary.consecutive_overlap),
        "reuse_gap": _reuse_data(summary.reuse_gap),
        "phase_layers": [
            {
                "phase": stream.phase,
                "layer_id": stream.layer_id,
                "event_count": stream.event_count,
                "assignment_count": stream.assignment_count,
                "unique_expert_count": stream.unique_expert_count,
                "expert_frequencies": _frequency_data(stream.expert_frequencies),
                "frequency_concentration": _concentration_data(stream.concentration),
                "consecutive_overlap": _overlap_data(stream.consecutive_overlap),
                "reuse_gap": _reuse_data(stream.reuse_gap),
            }
            for stream in summary.phase_layers
        ],
    }


def _phase_label(phase: str) -> str:
    return "prompt/prefill" if phase == "prompt" else "generated/decode"


def _format_frequencies(frequencies: tuple[ExpertFrequency, ...]) -> str:
    if not frequencies:
        return "N/A"
    return ", ".join(
        f"expert {item.expert_id}: {item.selection_count}" for item in frequencies
    )


def _format_fraction(value: Fraction | None) -> str:
    return "N/A" if value is None else f"{value.numerator}/{value.denominator}"


def _format_entropy_bits(value: float | None) -> str:
    if value is None:
        return "N/A"
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    return f"{rendered} bits"


def _concentration_report_lines(summary: FrequencyConcentrationSummary) -> list[str]:
    return [
        f"- Support size: {summary.support_size}",
        f"- Total selections: {summary.total_selections}",
        f"- Maximum expert selection share: {_format_fraction(summary.max_selection_share)}",
        f"- observed-support Gini: {_format_fraction(summary.observed_support_gini)}",
        f"- Shannon entropy: {_format_entropy_bits(summary.shannon_entropy_bits)}",
    ]


def _overlap_report_lines(summary: ConsecutiveOverlapSummary) -> list[str]:
    return [
        f"- Comparable pairs: {summary.comparison_count}",
        f"- Disjoint / identical / partial pairs: {summary.disjoint_pair_count} / {summary.identical_pair_count} / {summary.partial_pair_count}",
        f"- Mean Jaccard: {_format_fraction(summary.mean_jaccard)}",
    ]


def _reuse_report_lines(summary: ReuseGapSummary) -> list[str]:
    return [
        f"- First uses: {summary.first_use_count}",
        f"- Reuses: {summary.reuse_count}",
        f"- Min / max gap events: {_format_optional_int(summary.min_gap_events)} / {_format_optional_int(summary.max_gap_events)}",
        f"- Mean gap events: {_format_fraction(summary.mean_gap_events)}",
    ]


def _format_optional_int(value: int | None) -> str:
    return "N/A" if value is None else str(value)


def _fraction_data(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _frequency_data(
    frequencies: tuple[ExpertFrequency, ...],
) -> list[dict[str, int]]:
    return [
        {"expert_id": item.expert_id, "selection_count": item.selection_count}
        for item in frequencies
    ]


def _concentration_data(summary: FrequencyConcentrationSummary) -> dict[str, Any]:
    return {
        "support_size": summary.support_size,
        "total_selections": summary.total_selections,
        "max_selection_share": _fraction_data(summary.max_selection_share),
        "observed_support_gini": _fraction_data(summary.observed_support_gini),
        "shannon_entropy_bits": summary.shannon_entropy_bits,
    }


def _overlap_data(summary: ConsecutiveOverlapSummary) -> dict[str, Any]:
    return {
        "comparison_count": summary.comparison_count,
        "disjoint_pair_count": summary.disjoint_pair_count,
        "identical_pair_count": summary.identical_pair_count,
        "partial_pair_count": summary.partial_pair_count,
        "mean_jaccard": _fraction_data(summary.mean_jaccard),
    }


def _reuse_data(summary: ReuseGapSummary) -> dict[str, Any]:
    return {
        "first_use_count": summary.first_use_count,
        "reuse_count": summary.reuse_count,
        "min_gap_events": summary.min_gap_events,
        "max_gap_events": summary.max_gap_events,
        "mean_gap_events": _fraction_data(summary.mean_gap_events),
    }
