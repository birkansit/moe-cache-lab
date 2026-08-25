"""Deterministic human and machine output for exact LRU capacity frontiers."""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from .capacity_frontier import (
    CapacityFrontierAnalysis,
    CapacityFrontierMode,
    CapacityFrontierTarget,
)
from .trace import TRACE_FORMAT


CAPACITY_FRONTIER_FORMAT = "moe-cache-lab.capacity-frontier"
CAPACITY_FRONTIER_VERSION = 1


def capacity_frontier_data(result: CapacityFrontierAnalysis) -> dict[str, Any]:
    """Return the deterministic exact machine boundary."""

    if not isinstance(result, CapacityFrontierAnalysis):
        raise TypeError("result must be CapacityFrontierAnalysis")
    return {
        "format": CAPACITY_FRONTIER_FORMAT,
        "format_version": CAPACITY_FRONTIER_VERSION,
        "trace": {
            "format": TRACE_FORMAT,
            "format_version": result.trace_format_version,
            "model_id": result.model_id,
            "model_revision": result.model_revision,
            "capture_method": result.capture_method,
        },
        "event_count": result.event_count,
        "expert_request_count": result.expert_request_count,
        "first_use_count": result.first_use_count,
        "reuse_count": result.reuse_count,
        "compulsory_first_use_miss_fraction": _optional_fraction_data(
            result.compulsory_first_use_miss_fraction
        ),
        "maximum_reachable_cold_start_hit_fraction": _optional_fraction_data(
            result.maximum_reachable_cold_start_hit_fraction
        ),
        "count_frontier": _mode_data(result.count_frontier),
        "byte_frontier": (
            None if result.byte_frontier is None else _mode_data(result.byte_frontier)
        ),
        "claim_boundary": {
            "routing": (
                "Trace-derived routing is MEASURED only when provenance establishes "
                "measurement."
            ),
            "cache": (
                "Exact deterministic derivation under the SIMULATED event-atomic "
                "LRU cache abstraction."
            ),
            "byte_sizes": (
                "Caller-supplied cache-model inputs unless separate provenance "
                "establishes measured/model-matched physical extents."
            ),
            "runtime": (
                "Does not establish physical residency or transfer, latency, "
                "throughput, tokens/sec, speedup, an optimal capacity, a "
                "recommendation, LFU behavior, or workload representativeness."
            ),
        },
    }


def render_capacity_frontier_json(result: CapacityFrontierAnalysis) -> str:
    """Serialize deterministic JSON with exact rational fields."""

    return json.dumps(
        capacity_frontier_data(result),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def render_capacity_frontier_report(result: CapacityFrontierAnalysis) -> str:
    """Render a bounded Markdown summary without a full breakpoint table."""

    if not isinstance(result, CapacityFrontierAnalysis):
        raise TypeError("result must be CapacityFrontierAnalysis")
    lines = [
        "# Exact event-atomic LRU capacity frontier",
        "",
        "## Scope and claim boundary",
        "",
        "- This is an exact deterministic derivation under the **SIMULATED** event-atomic LRU cache abstraction.",
        "- Routing observations are **MEASURED** only when trace provenance establishes measurement.",
        "- It is not runtime cache behavior, physical residency/transfer evidence, a latency/throughput/tokens/sec/speedup prediction, an LFU result, an optimum, or a recommendation.",
        "- Fixed 25%/50%/75%/90% rows are descriptive reporting checkpoints only; they are not SLAs, knees, or deployment goals.",
        "",
        "## Trace context",
        "",
        f"- Trace contract: `{TRACE_FORMAT}` version {result.trace_format_version}",
        f"- Model ID: `{result.model_id}`",
        f"- Model revision: `{_available(result.model_revision)}`",
        f"- Capture method: `{result.capture_method}`",
        f"- Routing events: {result.event_count}",
        f"- Expert requests: {result.expert_request_count}",
        f"- First uses / compulsory misses: {result.first_use_count}",
        f"- Reuses: {result.reuse_count}",
        f"- Compulsory first-use miss fraction: {_format_fraction(result.compulsory_first_use_miss_fraction)}",
        f"- Maximum reachable cold-start hit fraction: {_format_fraction(result.maximum_reachable_cold_start_hit_fraction)}",
        "",
    ]
    lines.extend(_mode_report(result.count_frontier, title="Count-capacity frontier"))
    if result.byte_frontier is not None:
        lines.extend([
            "",
            "## Byte-size assumption boundary",
            "",
            "The supplied `size_bytes` values are caller-supplied cache-model inputs unless separate provenance establishes measured/model-matched physical extents. This frontier does not establish physical expert sizes, residency, or transfers.",
            "",
        ])
        lines.extend(_mode_report(result.byte_frontier, title="Byte-capacity frontier"))
    return "\n".join(lines).rstrip() + "\n"


def _mode_data(mode: CapacityFrontierMode) -> dict[str, Any]:
    suffix = mode.unit
    return {
        "capacity_unit": mode.unit,
        f"referenced_working_set_{suffix}": mode.referenced_working_set,
        f"max_atomic_event_{suffix}": mode.feasibility_floor,
        f"required_capacity_{suffix}_histogram": [
            {f"required_capacity_{suffix}": capacity, "reuse_count": count}
            for capacity, count in mode.required_capacity_histogram
        ],
        f"first_feasible_capacity_with_any_observed_lru_reuse_hit_{suffix}": (
            mode.first_feasible_capacity_with_observed_hit
        ),
        "breakpoints": [
            {
                f"capacity_{suffix}": point.capacity,
                "hits": point.hits,
                "misses": point.misses,
                "hit_fraction": _optional_fraction_data(point.hit_fraction),
                "miss_fraction": _optional_fraction_data(point.miss_fraction),
            }
            for point in mode.breakpoints
        ],
        "fixed_descriptive_targets": [
            _target_data(target, suffix=suffix) for target in mode.targets
        ],
    }


def _target_data(target: CapacityFrontierTarget, *, suffix: str) -> dict[str, Any]:
    return {
        "label": target.label,
        "target_hit_fraction": _fraction_data(target.target_fraction),
        "required_hits": target.required_hits,
        "status": target.status,
        "unattainable": target.unattainable,
        f"capacity_{suffix}": target.capacity,
        "capacity_fraction_of_referenced_working_set": _optional_fraction_data(
            target.capacity_fraction_of_referenced_working_set
        ),
    }


def _mode_report(mode: CapacityFrontierMode, *, title: str) -> list[str]:
    unit = mode.unit
    first_hit = mode.first_feasible_capacity_with_observed_hit
    lines = [
        f"## {title}",
        "",
        "The complete exact machine frontier contains only the feasibility floor and hit-changing thresholds; it is not generated by sweeping every integer capacity.",
        "",
        f"- Referenced working set ({unit}): {mode.referenced_working_set}",
        f"- Atomic-event feasibility floor ({unit}): {mode.feasibility_floor}",
        "- First feasible capacity with any observed LRU reuse hit: "
        + ("unavailable (no observed reuse)" if first_hit is None else f"{first_hit} {unit}"),
        f"- Exact hit-changing machine breakpoints: {len(mode.breakpoints)}",
        "",
        "### Fixed descriptive total-request hit-rate checkpoints",
        "",
        f"| target | required hits | smallest feasible capacity ({unit}) | capacity / referenced working set | status |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for target in mode.targets:
        lines.append(
            f"| {target.label} | {_optional_number(target.required_hits)} | "
            f"{_optional_number(target.capacity)} | "
            f"{_format_fraction(target.capacity_fraction_of_referenced_working_set)} | "
            f"{target.status} |"
        )
    return lines


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _optional_fraction_data(value: Fraction | None) -> dict[str, int] | None:
    return None if value is None else _fraction_data(value)


def _format_fraction(value: Fraction | None) -> str:
    return "unavailable" if value is None else f"{value.numerator}/{value.denominator}"


def _optional_number(value: int | None) -> str:
    return "unavailable" if value is None else str(value)


def _available(value: str | None) -> str:
    return "unavailable" if value is None or value == "" else value


__all__ = [
    "CAPACITY_FRONTIER_FORMAT",
    "CAPACITY_FRONTIER_VERSION",
    "capacity_frontier_data",
    "render_capacity_frontier_json",
    "render_capacity_frontier_report",
]
