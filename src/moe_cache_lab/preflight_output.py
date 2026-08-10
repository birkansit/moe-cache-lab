"""Deterministic Markdown and JSON output for combined pre-flight analysis."""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from .analysis_output import analysis_summary_data
from .byte_cache import ByteCacheSimulation
from .hardware_cost import (
    TransferCostEstimate,
    TransferSensitivitySweep,
    WorkloadByteContext,
)
from .preflight import PreflightAnalysisResult
from .preflight_config import preflight_config_data

PREFLIGHT_ANALYSIS_FORMAT = "moe-cache-lab.preflight-analysis"
PREFLIGHT_ANALYSIS_VERSION = 1


def render_preflight_report(result: PreflightAnalysisResult) -> str:
    """Render one deterministic report with explicit evidence-class boundaries."""
    if not isinstance(result, PreflightAnalysisResult):
        raise TypeError("render_preflight_report requires a PreflightAnalysisResult")

    routing = result.routing
    sweep = result.transfer_sensitivity
    context = sweep.workload_context
    lines = [
        "# MoE pre-flight analysis report",
        "",
        "## Scope / claim boundary",
        "",
        "- **ROUTING OBSERVATIONS:** treat the supplied trace as **MEASURED only if trace provenance establishes that**; this command does not collect new measurements.",
        "- **SIMULATED:** cache hits, misses, demand-load bytes, eviction bytes, and cache residency are offline cache-model outcomes.",
        "- **ESTIMATED:** serialized transfer-service times are derived from caller-supplied bandwidth and setup-latency assumptions.",
        "- This pre-flight feasibility/sensitivity analysis does not establish actual GPU residency, physical transfer timing, end-to-end latency, throughput, tokens/sec, or runtime speedup.",
        "",
        "## Routing observations",
        "",
        f"- Trace model ID: {result.trace_provenance.model_id}",
        f"- Capture method: {result.trace_provenance.capture_method}",
        f"- Model revision: {_optional_text(result.trace_provenance.model_revision)}",
        f"- Trace created_at: {_optional_text(result.trace_provenance.created_at)}",
        f"- Routing events: {routing.total_event_count}",
        f"- Layer-qualified expert assignments: {routing.total_assignment_count}",
        f"- Unique layer-qualified experts: {routing.unique_layer_expert_count}",
        f"- Prompt/prefill events / assignments: {routing.prompt.event_count} / {routing.prompt.assignment_count}",
        f"- Generated/decode events / assignments: {routing.generated.event_count} / {routing.generated.assignment_count}",
        f"- Global mean consecutive Jaccard: {_format_fraction(routing.consecutive_overlap.mean_jaccard)}",
        f"- Global mean reuse gap events: {_format_fraction(routing.reuse_gap.mean_gap_events)}",
        "",
        "### Per-layer routing locality",
        "",
        "| layer | events | assignments | unique experts | expert frequencies | max share | observed-support Gini | entropy bits |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    if routing.layers:
        for layer in routing.layers:
            lines.append(
                f"| {layer.layer_id} | {layer.event_count} | {layer.assignment_count} | "
                f"{layer.unique_expert_count} | {_format_frequencies(layer.expert_frequencies)} | "
                f"{_format_fraction(layer.concentration.max_selection_share)} | "
                f"{_format_fraction(layer.concentration.observed_support_gini)} | "
                f"{_format_float(layer.concentration.shannon_entropy_bits)} |"
            )
    else:
        lines.append("| N/A | 0 | 0 | 0 | N/A | N/A | N/A | N/A |")

    lines.extend([
        "",
        "### Phase-layer routing locality",
        "",
        "| phase | layer | events | assignments | unique experts | expert frequencies | mean Jaccard | mean reuse gap |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ])
    if routing.phase_layers:
        for stream in routing.phase_layers:
            lines.append(
                f"| {_phase_label(stream.phase)} | {stream.layer_id} | {stream.event_count} | "
                f"{stream.assignment_count} | {stream.unique_expert_count} | "
                f"{_format_frequencies(stream.expert_frequencies)} | "
                f"{_format_fraction(stream.consecutive_overlap.mean_jaccard)} | "
                f"{_format_fraction(stream.reuse_gap.mean_gap_events)} |"
            )
    else:
        lines.append("| N/A | N/A | 0 | 0 | 0 | N/A | N/A | N/A |")

    lines.extend([
        "",
        "## Supplied workload-size / cache-budget context",
        "",
        f"- Unique referenced expert objects: {context.unique_referenced_expert_count}",
        f"- Total supplied bytes for unique referenced experts: {context.unique_referenced_expert_bytes}",
        f"- Maximum atomic event working-set bytes: {context.maximum_atomic_event_working_set_bytes}",
        f"- Cache capacities analyzed (bytes): {', '.join(str(value) for value in sweep.capacities_bytes)}",
        f"- Policies: {', '.join(sweep.policies)}",
        "",
        "### Caller-supplied expert sizes",
        "",
        "| layer | expert | size bytes |",
        "| ---: | ---: | ---: |",
    ])
    for item in result.config.expert_sizes:
        lines.append(f"| {item.layer_id} | {item.expert_id} | {item.size_bytes} |")

    lines.extend([
        "",
        "## SIMULATED byte-cache sensitivity",
        "",
        "| capacity bytes | policy | requests | hits | misses | hit rate | demand-load bytes | evictions | evicted bytes | peak resident bytes | final resident bytes |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for simulation in _unique_simulations(sweep):
        lines.append(
            f"| {simulation.capacity_bytes} | {simulation.policy} | "
            f"{simulation.expert_request_count} | {simulation.hits} | {simulation.misses} | "
            f"{_format_float(simulation.hit_rate)} | {simulation.simulated_demand_load_bytes} | "
            f"{simulation.eviction_count} | {simulation.simulated_evicted_bytes} | "
            f"{simulation.peak_resident_bytes} | {simulation.final_resident_bytes} |"
        )

    lines.extend([
        "",
        "## ESTIMATED serialized transfer-service sensitivity",
        "",
        "| capacity bytes | policy | hardware profile | bandwidth B/s | setup ns/load | payload service s | setup service s | serialized service s |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in sweep.rows:
        estimate = row.estimate
        lines.append(
            f"| {estimate.cache_capacity_bytes} | {estimate.policy} | "
            f"{estimate.hardware_profile_name} | "
            f"{estimate.assumed_h2d_payload_bandwidth_bytes_per_second} | "
            f"{estimate.assumed_setup_latency_ns_per_loaded_expert} | "
            f"{_format_fraction(estimate.estimated_payload_service_seconds)} | "
            f"{_format_fraction(estimate.estimated_setup_service_seconds)} | "
            f"{_format_fraction(estimate.estimated_serialized_transfer_service_seconds)} |"
        )

    lines.extend([
        "",
        "## Assumptions / limitations",
        "",
        "- Expert sizes, H2D payload bandwidth, and per-loaded-expert setup latency are caller-supplied assumptions unless established separately by measurement or calibration.",
        "- Transfer-service estimation is serialized/no-overlap: simulated demand-load bytes / assumed bandwidth + simulated misses × assumed setup latency.",
        "- Simulated evictions drop immutable/read-only expert residency and are not charged as D2H writeback.",
        "- The model excludes GPU compute, transfer/compute overlap, concurrency, kernel scheduling, allocator effects, CPU overhead, runtime synchronization, protocol details, and D2H writeback.",
        "- Lower simulated byte traffic or lower estimated serialized service time is a neutral sensitivity result, not a runtime-benefit recommendation.",
        "",
    ])
    return "\n".join(lines)


def render_preflight_json(result: PreflightAnalysisResult) -> str:
    """Serialize the combined result deterministically without losing exact Fractions."""
    return json.dumps(
        preflight_analysis_data(result),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def preflight_analysis_data(result: PreflightAnalysisResult) -> dict[str, Any]:
    """Return the stable machine-readable pre-flight analysis representation."""
    if not isinstance(result, PreflightAnalysisResult):
        raise TypeError("preflight_analysis_data requires a PreflightAnalysisResult")
    sweep = result.transfer_sensitivity
    return {
        "format": PREFLIGHT_ANALYSIS_FORMAT,
        "format_version": PREFLIGHT_ANALYSIS_VERSION,
        "trace_provenance": {
            "model_id": result.trace_provenance.model_id,
            "capture_method": result.trace_provenance.capture_method,
            "model_revision": result.trace_provenance.model_revision,
            "created_at": result.trace_provenance.created_at,
        },
        "config": preflight_config_data(result.config),
        "routing_analysis": analysis_summary_data(result.routing),
        "workload_byte_context": _workload_context_data(sweep.workload_context),
        "simulated_byte_cache_sensitivity": [
            _simulation_data(simulation) for simulation in _unique_simulations(sweep)
        ],
        "estimated_transfer_service_sensitivity": [
            _estimate_data(row.estimate) for row in sweep.rows
        ],
    }


def _unique_simulations(
    sweep: TransferSensitivitySweep,
) -> tuple[ByteCacheSimulation, ...]:
    seen: set[tuple[int, str]] = set()
    simulations: list[ByteCacheSimulation] = []
    for row in sweep.rows:
        key = row.simulation.capacity_bytes, row.simulation.policy
        if key not in seen:
            seen.add(key)
            simulations.append(row.simulation)
    return tuple(simulations)


def _simulation_data(simulation: ByteCacheSimulation) -> dict[str, Any]:
    return {
        "policy": simulation.policy,
        "capacity_bytes": simulation.capacity_bytes,
        "event_count": simulation.event_count,
        "expert_request_count": simulation.expert_request_count,
        "hits": simulation.hits,
        "misses": simulation.misses,
        "hit_rate": simulation.hit_rate,
        "simulated_demand_load_bytes": simulation.simulated_demand_load_bytes,
        "eviction_count": simulation.eviction_count,
        "simulated_evicted_bytes": simulation.simulated_evicted_bytes,
        "peak_resident_bytes": simulation.peak_resident_bytes,
        "final_resident_bytes": simulation.final_resident_bytes,
        "final_resident_keys": [
            {"layer_id": layer_id, "expert_id": expert_id}
            for layer_id, expert_id in simulation.final_resident_keys
        ],
    }


def _estimate_data(estimate: TransferCostEstimate) -> dict[str, Any]:
    return {
        "hardware_profile_name": estimate.hardware_profile_name,
        "policy": estimate.policy,
        "cache_capacity_bytes": estimate.cache_capacity_bytes,
        "simulated_demand_load_count": estimate.simulated_demand_load_count,
        "simulated_demand_load_bytes": estimate.simulated_demand_load_bytes,
        "assumed_h2d_payload_bandwidth_bytes_per_second": (
            estimate.assumed_h2d_payload_bandwidth_bytes_per_second
        ),
        "assumed_setup_latency_ns_per_loaded_expert": (
            estimate.assumed_setup_latency_ns_per_loaded_expert
        ),
        "estimated_payload_service_seconds": _fraction_data(
            estimate.estimated_payload_service_seconds
        ),
        "estimated_setup_service_seconds": _fraction_data(
            estimate.estimated_setup_service_seconds
        ),
        "estimated_serialized_transfer_service_seconds": _fraction_data(
            estimate.estimated_serialized_transfer_service_seconds
        ),
    }


def _workload_context_data(context: WorkloadByteContext) -> dict[str, Any]:
    return {
        "unique_referenced_expert_count": context.unique_referenced_expert_count,
        "unique_referenced_expert_bytes": context.unique_referenced_expert_bytes,
        "maximum_atomic_event_working_set_bytes": (
            context.maximum_atomic_event_working_set_bytes
        ),
        "referenced_expert_keys": [
            {"layer_id": layer_id, "expert_id": expert_id}
            for layer_id, expert_id in context.referenced_expert_keys
        ],
    }


def _format_frequencies(frequencies: tuple[Any, ...]) -> str:
    if not frequencies:
        return "N/A"
    return ", ".join(
        f"expert {item.expert_id}: {item.selection_count}" for item in frequencies
    )


def _format_fraction(value: Fraction | None) -> str:
    if value is None:
        return "N/A"
    return f"{value.numerator}/{value.denominator}"


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _format_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _optional_text(value: str | None) -> str:
    return "N/A" if value is None else value


def _phase_label(phase: str) -> str:
    return "prompt/prefill" if phase == "prompt" else "generated/decode"
