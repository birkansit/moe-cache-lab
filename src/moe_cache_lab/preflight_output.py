"""Deterministic Markdown and JSON output for combined pre-flight analysis."""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from .analysis_output import analysis_summary_data
from .byte_cache import ByteCacheLifecycleSimulation, ByteCacheSimulation
from .hardware_cost import (
    LifecycleTransferEstimateRow,
    TransferCostEstimate,
    TransferSensitivitySweep,
    WorkloadByteContext,
)
from .preflight import PreflightAnalysisResult, PreflightLifecycleAnalysisResult
from .preflight_config import (
    PREFLIGHT_CONFIG_LEGACY_VERSION,
    PREFLIGHT_CONFIG_VERSION,
    preflight_config_data,
)
from .sensitivity_summary import (
    LifecycleSensitivitySummary,
    WorkloadIdentity,
    WorkloadMetricRange,
)

PREFLIGHT_ANALYSIS_FORMAT = "moe-cache-lab.preflight-analysis"
PREFLIGHT_ANALYSIS_VERSION = 2
PREFLIGHT_LIFECYCLE_FORMAT = "moe-cache-lab.preflight-cache-lifecycle-analysis"
PREFLIGHT_LIFECYCLE_VERSION = 3


def render_preflight_report(result: PreflightAnalysisResult) -> str:
    """Render one deterministic report with explicit evidence-class boundaries."""
    if not isinstance(result, PreflightAnalysisResult):
        raise TypeError("render_preflight_report requires a PreflightAnalysisResult")

    routing = result.routing
    sweep = result.transfer_sensitivity
    context = sweep.workload_context
    legacy_output = result.config.format_version == PREFLIGHT_CONFIG_LEGACY_VERSION
    estimated_scope = (
        "- **ESTIMATED:** serialized transfer-service times are derived from caller-supplied bandwidth and setup-latency assumptions."
        if legacy_output
        else "- **ESTIMATED:** serialized transfer-service times are derived from caller-supplied bandwidth, per-operation setup latency, and transfer-operation-plan assumptions."
    )
    lines = [
        "# MoE pre-flight analysis report",
        "",
        "## Scope / claim boundary",
        "",
        "- **ROUTING OBSERVATIONS:** treat the supplied trace as **MEASURED only if trace provenance establishes that**; this command does not collect new measurements.",
        "- **SIMULATED:** cache hits, misses, demand-load bytes, eviction bytes, and cache residency are offline cache-model outcomes.",
        estimated_scope,
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

    lines.extend(["", "## ESTIMATED serialized transfer-service sensitivity", ""])
    if legacy_output:
        lines.extend([
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
        assumptions = [
            "- Expert sizes, H2D payload bandwidth, and per-loaded-expert setup latency are caller-supplied assumptions unless established separately by measurement or calibration.",
            "- Transfer-service estimation is serialized/no-overlap: simulated demand-load bytes / assumed bandwidth + simulated misses × assumed setup latency.",
        ]
    else:
        lines.extend([
            "| capacity bytes | policy | hardware profile | transfer plan | logical loads | operations/load | modeled operations | bandwidth B/s | setup ns/operation | payload service s | setup service s | serialized service s |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in sweep.rows:
            estimate = row.estimate
            lines.append(
                f"| {estimate.cache_capacity_bytes} | {estimate.policy} | "
                f"{estimate.hardware_profile_name} | "
                f"{estimate.transfer_operation_plan_name} | "
                f"{estimate.simulated_demand_load_count} | "
                f"{estimate.assumed_transfer_operations_per_logical_load} | "
                f"{estimate.modeled_transfer_operation_count} | "
                f"{estimate.assumed_h2d_payload_bandwidth_bytes_per_second} | "
                f"{estimate.assumed_setup_latency_ns_per_transfer_operation} | "
                f"{_format_fraction(estimate.estimated_payload_service_seconds)} | "
                f"{_format_fraction(estimate.estimated_setup_service_seconds)} | "
                f"{_format_fraction(estimate.estimated_serialized_transfer_service_seconds)} |"
            )
        assumptions = [
            "- Expert sizes, H2D payload bandwidth, per-operation setup latency, and transfer-operation plans are caller-supplied assumptions unless established separately by measurement or calibration.",
            "- A transfer-operation plan maps each simulated logical expert load to a fixed number of modeled setup-bearing operations; it does not establish how packed model parameters move in a real runtime.",
            "- Transfer-service estimation is serialized/no-overlap: simulated demand-load bytes / assumed bandwidth + modeled transfer-operation count × assumed setup latency per operation.",
        ]

    lines.extend([
        "",
        "## Assumptions / limitations",
        "",
        *assumptions,
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


def render_preflight_lifecycle_report(
    result: PreflightLifecycleAnalysisResult,
) -> str:
    """Render deterministic SIMULATED cold/persistent lifecycle comparisons."""
    if not isinstance(result, PreflightLifecycleAnalysisResult):
        raise TypeError(
            "render_preflight_lifecycle_report requires a "
            "PreflightLifecycleAnalysisResult"
        )
    lines = [
        "# MoE cache lifecycle pre-flight analysis",
        "",
        "## Scope / claim boundary",
        "",
        "- **SIMULATED:** every cache result below is an offline byte-cache lifecycle scenario over recorded routing events.",
        "- **ESTIMATED:** serialized H2D transfer-service rows apply caller-supplied bandwidth, setup-latency, and transfer-operation-granularity assumptions to those existing simulated rows.",
        "- `cold_per_workload` starts every explicitly identified workload with an empty cache.",
        "- `persistent_sequence` preserves one cache across the exact manifest workload order shown below.",
        "- Lifecycle mode changes reset boundaries only; layer-qualified identity, atomic routing-event working sets, and LRU/LFU admission and eviction semantics are unchanged.",
        "- Demand-load and eviction bytes are simulated accounting, not measured transfers or physical GPU residency.",
        "- This comparison does not establish runtime latency, throughput, tokens/sec, speedup, or a deployment recommendation.",
        "",
        "## Reproducible input order",
        "",
        f"- Manifest: `{result.manifest_filename}`; SHA-256 `{result.manifest_sha256}`",
        f"- Corpus version: `{result.corpus_version}`; canonical SHA-256 `{result.corpus_sha256}`",
        "- Workload chronology is read from the validated manifest/corpus order; it is not inferred from IDs, names, categories, or prompt semantics.",
        "",
        "| order | workload | trace path | trace SHA-256 | model | capture method |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for workload in result.workloads:
        lines.append(
            f"| {workload.order} | {workload.workload_id} | "
            f"{workload.trace_relative_path} | {workload.trace_sha256} | "
            f"{workload.model_id} | {workload.capture_method} |"
        )

    lines.extend([
        "",
        "## SIMULATED lifecycle aggregates",
        "",
        "| capacity bytes | policy | lifecycle mode | workloads | events | requests | hits | misses | hit rate | demand-load bytes | evictions | evicted bytes | peak resident bytes | shared final resident bytes | shared final resident keys |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for simulation in result.simulations:
        lines.append(
            f"| {simulation.capacity_bytes} | {simulation.policy} | "
            f"{simulation.lifecycle_mode} | {simulation.workload_count} | "
            f"{simulation.event_count} | {simulation.expert_request_count} | "
            f"{simulation.hits} | {simulation.misses} | "
            f"{_format_float(simulation.hit_rate)} | "
            f"{simulation.simulated_demand_load_bytes} | "
            f"{simulation.eviction_count} | "
            f"{simulation.simulated_evicted_bytes} | "
            f"{simulation.peak_resident_bytes} | "
            f"{_optional_number(simulation.shared_sequence_final_resident_bytes)} | "
            f"{_format_resident_keys(simulation.shared_sequence_final_resident_keys)} |"
        )

    summary = result.sensitivity_summary
    lines.extend([
        "",
        "## DESCRIPTIVE adjacent tested-capacity summaries",
        "",
        "Each delta is upper tested capacity minus the immediately lower tested capacity within the same lifecycle mode and policy. `exact flat` means exact equality of integer hits, misses, and simulated demand-load bytes; it does not interpolate between capacities.",
        "",
        "| lifecycle mode | policy | lower capacity bytes | upper capacity bytes | hit delta | miss delta | demand-load byte delta | hit-rate delta | exact flat |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for comparison in summary.adjacent_capacity_comparisons:
        lines.append(
            f"| {comparison.lifecycle_mode} | {comparison.policy} | "
            f"{comparison.lower_capacity_bytes} | "
            f"{comparison.upper_capacity_bytes} | {comparison.hit_delta} | "
            f"{comparison.miss_delta} | "
            f"{comparison.simulated_demand_load_byte_delta} | "
            f"{_format_fraction(comparison.hit_rate_delta)} | "
            f"{str(comparison.exact_flat).lower()} |"
        )

    lines.extend([
        "",
        "## DESCRIPTIVE same-capacity policy summaries",
        "",
        "Policy deltas are LFU minus LRU at the same tested capacity and lifecycle mode. They are descriptive comparisons, not recommendations.",
        "",
        "| lifecycle mode | capacity bytes | baseline policy | comparison policy | hit delta | miss delta | demand-load byte delta | hit-rate delta |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for comparison in summary.same_capacity_policy_comparisons:
        lines.append(
            f"| {comparison.lifecycle_mode} | {comparison.capacity_bytes} | "
            f"{comparison.baseline_policy} | {comparison.comparison_policy} | "
            f"{comparison.hit_delta} | {comparison.miss_delta} | "
            f"{comparison.simulated_demand_load_byte_delta} | "
            f"{_format_fraction(comparison.hit_rate_delta)} |"
        )

    lines.extend([
        "",
        "## DESCRIPTIVE workload spread",
        "",
        "Minimum, maximum, and range describe only the observed workload rows. Tied extrema retain every manifest workload identity in declared order; no semantic meaning is inferred from workload IDs.",
        "",
        "| capacity bytes | policy | lifecycle mode | metric | defined workloads | minimum | minimum workloads | maximum | maximum workloads | range |",
        "| ---: | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: |",
    ])
    for row in summary.workload_sensitivity:
        for metric in (row.hit_rate, row.misses, row.simulated_demand_load_bytes):
            defined_count = (
                row.defined_hit_rate_workload_count
                if metric.metric == "hit_rate"
                else row.workload_count
            )
            lines.append(
                f"| {row.capacity_bytes} | {row.policy} | "
                f"{row.lifecycle_mode} | {metric.metric} | {defined_count} | "
                f"{_format_exact(metric.minimum)} | "
                f"{_format_workload_identities(metric.minimum_workloads)} | "
                f"{_format_exact(metric.maximum)} | "
                f"{_format_workload_identities(metric.maximum_workloads)} | "
                f"{_format_exact(metric.range)} |"
            )

    lines.extend([
        "",
        "## SIMULATED per-workload attribution",
        "",
        "| capacity bytes | policy | lifecycle mode | order | workload | events | requests | hits | misses | hit rate | demand-load bytes | evictions | evicted bytes | start resident bytes | start keys | end resident bytes | end keys |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ])
    for simulation in result.simulations:
        for workload in simulation.workloads:
            lines.append(
                f"| {workload.capacity_bytes} | {workload.policy} | "
                f"{workload.lifecycle_mode} | {workload.order} | "
                f"{workload.workload_id} | {workload.event_count} | "
                f"{workload.expert_request_count} | {workload.hits} | "
                f"{workload.misses} | {_format_float(workload.hit_rate)} | "
                f"{workload.simulated_demand_load_bytes} | "
                f"{workload.eviction_count} | {workload.simulated_evicted_bytes} | "
                f"{workload.starting_resident_bytes} | "
                f"{_format_resident_keys(workload.starting_resident_keys)} | "
                f"{workload.ending_resident_bytes} | "
                f"{_format_resident_keys(workload.ending_resident_keys)} |"
            )

    lines.extend([
        "",
        "## ESTIMATED serialized H2D transfer service",
        "",
        "These rows apply the serialized/no-overlap equation to existing SIMULATED demand loads. They use caller-supplied hardware and transfer-operation-granularity assumptions and do not rerun cache simulation.",
        "",
        "| capacity bytes | policy | lifecycle mode | hardware profile | transfer plan | logical loads | demand-load bytes | operations/load | modeled operations | bandwidth bytes/s | setup ns/operation | payload seconds | setup seconds | serialized total seconds |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in result.transfer_estimates.rows:
        estimate = row.estimate
        lines.append(
            f"| {estimate.cache_capacity_bytes} | {estimate.policy} | "
            f"{row.lifecycle_mode} | {estimate.hardware_profile_name} | "
            f"{estimate.transfer_operation_plan_name} | "
            f"{estimate.simulated_demand_load_count} | "
            f"{estimate.simulated_demand_load_bytes} | "
            f"{estimate.assumed_transfer_operations_per_logical_load} | "
            f"{estimate.modeled_transfer_operation_count} | "
            f"{estimate.assumed_h2d_payload_bandwidth_bytes_per_second} | "
            f"{estimate.assumed_setup_latency_ns_per_transfer_operation} | "
            f"{_format_fraction(estimate.estimated_payload_service_seconds)} | "
            f"{_format_fraction(estimate.estimated_setup_service_seconds)} | "
            f"{_format_fraction(estimate.estimated_serialized_transfer_service_seconds)} |"
        )

    lines.extend([
        "",
        "### ESTIMATED per-workload attribution",
        "",
        "Cold workload rows are independent empty-cache simulations. Persistent workload rows are attributed slices of one chronological persistent-cache replay; cold and persistent totals are never combined.",
        "",
        "| capacity bytes | policy | lifecycle mode | hardware profile | transfer plan | order | workload | logical loads | demand-load bytes | modeled operations | payload seconds | setup seconds | serialized total seconds |",
        "| ---: | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in result.transfer_estimates.rows:
        for workload in row.workloads:
            estimate = workload.estimate
            lines.append(
                f"| {estimate.cache_capacity_bytes} | {estimate.policy} | "
                f"{row.lifecycle_mode} | {estimate.hardware_profile_name} | "
                f"{estimate.transfer_operation_plan_name} | {workload.order} | "
                f"{workload.workload_id} | "
                f"{estimate.simulated_demand_load_count} | "
                f"{estimate.simulated_demand_load_bytes} | "
                f"{estimate.modeled_transfer_operation_count} | "
                f"{_format_fraction(estimate.estimated_payload_service_seconds)} | "
                f"{_format_fraction(estimate.estimated_setup_service_seconds)} | "
                f"{_format_fraction(estimate.estimated_serialized_transfer_service_seconds)} |"
            )

    lines.extend([
        "",
        "## Assumptions / limitations",
        "",
        "- Aggregate integer counters are exact sums of the per-workload rows for each capacity, policy, and lifecycle mode.",
        "- Sensitivity summaries are derived from those existing rows without rerunning simulation; exact integer deltas remain integers and hit-rate deltas remain exact fractions.",
        "- Adjacent comparisons cover tested neighboring capacities only. No untested capacity, optimum, ranking, or recommendation is inferred.",
        "- Cold rows have no shared sequence-final residency; each workload row records its own empty start and ending state.",
        "- Persistent rows expose stable sorted layer-qualified start/end resident keys only, not internal timestamps or frequency counters.",
        "- Hardware profiles and transfer-operation plans are caller-supplied assumptions used only by the separate ESTIMATED service rows; they do not change SIMULATED cache outcomes.",
        "- Transfer estimates model serialized/no-overlap H2D service only. They exclude compute, overlap/concurrency, kernel/runtime scheduling, allocator effects, synchronization/protocol effects, D2H writeback, and end-to-end runtime behavior.",
        "- A smaller ESTIMATED service value is not proof of measured latency reduction, throughput, tokens/sec, speedup, physical transfer granularity, or physical expert residency.",
        "- No expert semantics are inferred from expert IDs, routing frequency, workload IDs, or corpus categories.",
        "",
    ])
    return "\n".join(lines)


def render_preflight_lifecycle_json(
    result: PreflightLifecycleAnalysisResult,
) -> str:
    """Serialize lifecycle analysis deterministically."""
    return json.dumps(
        preflight_lifecycle_analysis_data(result),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def preflight_lifecycle_analysis_data(
    result: PreflightLifecycleAnalysisResult,
) -> dict[str, Any]:
    """Return the stable machine-readable lifecycle analysis representation."""
    if not isinstance(result, PreflightLifecycleAnalysisResult):
        raise TypeError(
            "preflight_lifecycle_analysis_data requires a "
            "PreflightLifecycleAnalysisResult"
        )
    return {
        "format": PREFLIGHT_LIFECYCLE_FORMAT,
        "format_version": PREFLIGHT_LIFECYCLE_VERSION,
        "evidence_classes": {
            "cache_lifecycle_scenarios": "SIMULATED",
            "descriptive_sensitivity_summaries": "SIMULATED",
            "serialized_h2d_transfer_service": "ESTIMATED",
        },
        "source": {
            "manifest_filename": result.manifest_filename,
            "manifest_sha256": result.manifest_sha256,
            "corpus_version": result.corpus_version,
            "corpus_sha256": result.corpus_sha256,
        },
        "config": preflight_config_data(result.config),
        "workload_order": [
            {
                "workload_id": workload.workload_id,
                "order": workload.order,
                "trace_relative_path": workload.trace_relative_path,
                "trace_sha256": workload.trace_sha256,
                "model_id": workload.model_id,
                "capture_method": workload.capture_method,
                "model_revision": workload.model_revision,
                "created_at": workload.created_at,
            }
            for workload in result.workloads
        ],
        "simulated_cache_lifecycle_scenarios": [
            _lifecycle_simulation_data(simulation)
            for simulation in result.simulations
        ],
        "descriptive_sensitivity_summaries": _lifecycle_sensitivity_data(
            result.sensitivity_summary
        ),
        "estimated_serialized_h2d_transfer_service": {
            "model": {
                "direction": "H2D",
                "serialization": "serialized_no_overlap",
                "hardware_assumptions_are_caller_supplied": True,
                "transfer_operation_granularity_is_caller_supplied": True,
                "excludes": [
                    "compute",
                    "overlap_or_concurrency",
                    "kernel_or_runtime_scheduling",
                    "allocator_effects",
                    "synchronization_or_protocol_effects",
                    "D2H_writeback",
                    "end_to_end_runtime_behavior",
                ],
            },
            "rows": [
                _lifecycle_transfer_estimate_data(row)
                for row in result.transfer_estimates.rows
            ],
        },
        "claim_boundary": {
            "routing_is_measured_only_if_trace_provenance_establishes_measurement": True,
            "cache_outcomes_are_simulated": True,
            "demand_and_eviction_bytes_are_simulated": True,
            "transfer_service_is_estimated": True,
            "hardware_and_operation_assumptions_are_caller_supplied": True,
            "runtime_performance_is_not_measured": True,
            "physical_residency_is_not_established": True,
        },
    }


def preflight_analysis_data(result: PreflightAnalysisResult) -> dict[str, Any]:
    """Return the stable machine-readable pre-flight analysis representation."""
    if not isinstance(result, PreflightAnalysisResult):
        raise TypeError("preflight_analysis_data requires a PreflightAnalysisResult")
    sweep = result.transfer_sensitivity
    output_version = result.config.format_version
    if output_version not in (
        PREFLIGHT_CONFIG_LEGACY_VERSION,
        PREFLIGHT_CONFIG_VERSION,
    ):
        raise ValueError("unsupported preflight analysis output version")
    return {
        "format": PREFLIGHT_ANALYSIS_FORMAT,
        "format_version": output_version,
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
            _estimate_data(row.estimate, output_version) for row in sweep.rows
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


def _estimate_data(
    estimate: TransferCostEstimate,
    output_version: int,
) -> dict[str, Any]:
    payload = {
        "hardware_profile_name": estimate.hardware_profile_name,
        "policy": estimate.policy,
        "cache_capacity_bytes": estimate.cache_capacity_bytes,
        "simulated_demand_load_bytes": estimate.simulated_demand_load_bytes,
        "assumed_h2d_payload_bandwidth_bytes_per_second": (
            estimate.assumed_h2d_payload_bandwidth_bytes_per_second
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
    if output_version == PREFLIGHT_CONFIG_LEGACY_VERSION:
        payload.update({
            "simulated_demand_load_count": estimate.simulated_demand_load_count,
            "assumed_setup_latency_ns_per_loaded_expert": (
                estimate.assumed_setup_latency_ns_per_loaded_expert
            ),
        })
    else:
        payload.update({
            "transfer_operation_plan_name": (
                estimate.transfer_operation_plan_name
            ),
            "simulated_logical_demand_load_count": (
                estimate.simulated_demand_load_count
            ),
            "assumed_transfer_operations_per_logical_load": (
                estimate.assumed_transfer_operations_per_logical_load
            ),
            "modeled_transfer_operation_count": (
                estimate.modeled_transfer_operation_count
            ),
            "assumed_setup_latency_ns_per_transfer_operation": (
                estimate.assumed_setup_latency_ns_per_transfer_operation
            ),
        })
    return payload


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


def _lifecycle_simulation_data(
    simulation: ByteCacheLifecycleSimulation,
) -> dict[str, Any]:
    return {
        "lifecycle_mode": simulation.lifecycle_mode,
        "policy": simulation.policy,
        "capacity_bytes": simulation.capacity_bytes,
        "workload_count": simulation.workload_count,
        "event_count": simulation.event_count,
        "expert_request_count": simulation.expert_request_count,
        "hits": simulation.hits,
        "misses": simulation.misses,
        "hit_rate": simulation.hit_rate,
        "simulated_demand_load_bytes": simulation.simulated_demand_load_bytes,
        "eviction_count": simulation.eviction_count,
        "simulated_evicted_bytes": simulation.simulated_evicted_bytes,
        "peak_resident_bytes": simulation.peak_resident_bytes,
        "shared_sequence_final_resident_bytes": (
            simulation.shared_sequence_final_resident_bytes
        ),
        "shared_sequence_final_resident_keys": _resident_key_data(
            simulation.shared_sequence_final_resident_keys
        ),
        "workloads": [
            {
                "workload_id": workload.workload_id,
                "order": workload.order,
                "lifecycle_mode": workload.lifecycle_mode,
                "policy": workload.policy,
                "capacity_bytes": workload.capacity_bytes,
                "event_count": workload.event_count,
                "expert_request_count": workload.expert_request_count,
                "hits": workload.hits,
                "misses": workload.misses,
                "hit_rate": workload.hit_rate,
                "simulated_demand_load_bytes": (
                    workload.simulated_demand_load_bytes
                ),
                "eviction_count": workload.eviction_count,
                "simulated_evicted_bytes": workload.simulated_evicted_bytes,
                "peak_resident_bytes": workload.peak_resident_bytes,
                "starting_resident_bytes": workload.starting_resident_bytes,
                "starting_resident_keys": _resident_key_data(
                    workload.starting_resident_keys
                ),
                "ending_resident_bytes": workload.ending_resident_bytes,
                "ending_resident_keys": _resident_key_data(
                    workload.ending_resident_keys
                ),
            }
            for workload in simulation.workloads
        ],
    }


def _lifecycle_sensitivity_data(
    summary: LifecycleSensitivitySummary,
) -> dict[str, Any]:
    return {
        "capacities_bytes": list(summary.capacities_bytes),
        "policies": list(summary.policies),
        "lifecycle_modes": list(summary.lifecycle_modes),
        "adjacent_capacity_comparisons": [
            {
                "lifecycle_mode": item.lifecycle_mode,
                "policy": item.policy,
                "lower_capacity_bytes": item.lower_capacity_bytes,
                "upper_capacity_bytes": item.upper_capacity_bytes,
                "hit_delta": item.hit_delta,
                "miss_delta": item.miss_delta,
                "simulated_demand_load_byte_delta": (
                    item.simulated_demand_load_byte_delta
                ),
                "hit_rate_delta": _optional_fraction_data(item.hit_rate_delta),
                "exact_flat": item.exact_flat,
            }
            for item in summary.adjacent_capacity_comparisons
        ],
        "same_capacity_policy_comparisons": [
            {
                "lifecycle_mode": item.lifecycle_mode,
                "capacity_bytes": item.capacity_bytes,
                "baseline_policy": item.baseline_policy,
                "comparison_policy": item.comparison_policy,
                "hit_delta": item.hit_delta,
                "miss_delta": item.miss_delta,
                "simulated_demand_load_byte_delta": (
                    item.simulated_demand_load_byte_delta
                ),
                "hit_rate_delta": _optional_fraction_data(item.hit_rate_delta),
            }
            for item in summary.same_capacity_policy_comparisons
        ],
        "workload_sensitivity": [
            {
                "lifecycle_mode": row.lifecycle_mode,
                "policy": row.policy,
                "capacity_bytes": row.capacity_bytes,
                "workload_count": row.workload_count,
                "defined_hit_rate_workload_count": (
                    row.defined_hit_rate_workload_count
                ),
                "metrics": {
                    metric.metric: _workload_metric_range_data(metric)
                    for metric in (
                        row.hit_rate,
                        row.misses,
                        row.simulated_demand_load_bytes,
                    )
                },
            }
            for row in summary.workload_sensitivity
        ],
    }


def _lifecycle_transfer_estimate_data(
    row: LifecycleTransferEstimateRow,
) -> dict[str, Any]:
    return {
        "lifecycle_mode": row.lifecycle_mode,
        **_lifecycle_transfer_accounting_data(row.estimate),
        "workloads": [
            {
                "workload_id": workload.workload_id,
                "order": workload.order,
                **_lifecycle_transfer_accounting_data(workload.estimate),
            }
            for workload in row.workloads
        ],
    }


def _lifecycle_transfer_accounting_data(
    estimate: TransferCostEstimate,
) -> dict[str, Any]:
    return {
        "capacity_bytes": estimate.cache_capacity_bytes,
        "policy": estimate.policy,
        "hardware_profile_name": estimate.hardware_profile_name,
        "transfer_operation_plan_name": estimate.transfer_operation_plan_name,
        "simulated_logical_demand_load_count": (
            estimate.simulated_demand_load_count
        ),
        "simulated_demand_load_bytes": estimate.simulated_demand_load_bytes,
        "modeled_transfer_operation_count": (
            estimate.modeled_transfer_operation_count
        ),
        "assumed_transfer_operations_per_logical_load": (
            estimate.assumed_transfer_operations_per_logical_load
        ),
        "assumed_h2d_payload_bandwidth_bytes_per_second": (
            estimate.assumed_h2d_payload_bandwidth_bytes_per_second
        ),
        "assumed_setup_latency_ns_per_transfer_operation": (
            estimate.assumed_setup_latency_ns_per_transfer_operation
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


def _workload_metric_range_data(metric: WorkloadMetricRange) -> dict[str, Any]:
    return {
        "minimum": _exact_value_data(metric.minimum),
        "maximum": _exact_value_data(metric.maximum),
        "range": _exact_value_data(metric.range),
        "minimum_workloads": _workload_identity_data(metric.minimum_workloads),
        "maximum_workloads": _workload_identity_data(metric.maximum_workloads),
    }


def _workload_identity_data(
    identities: tuple[WorkloadIdentity, ...],
) -> list[dict[str, int | str]]:
    return [
        {"order": identity.order, "workload_id": identity.workload_id}
        for identity in identities
    ]


def _exact_value_data(value: int | Fraction | None) -> int | dict[str, int] | None:
    if isinstance(value, Fraction):
        return _fraction_data(value)
    return value


def _optional_fraction_data(value: Fraction | None) -> dict[str, int] | None:
    return None if value is None else _fraction_data(value)


def _resident_key_data(
    keys: tuple[tuple[int, int], ...] | None,
) -> list[dict[str, int]] | None:
    if keys is None:
        return None
    return [
        {"layer_id": layer_id, "expert_id": expert_id}
        for layer_id, expert_id in keys
    ]


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


def _format_exact(value: int | Fraction | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, Fraction):
        return _format_fraction(value)
    return str(value)


def _format_workload_identities(
    identities: tuple[WorkloadIdentity, ...],
) -> str:
    if not identities:
        return "N/A"
    return ", ".join(
        f"{identity.order}:{identity.workload_id}" for identity in identities
    )


def _optional_text(value: str | None) -> str:
    return "N/A" if value is None else value


def _optional_number(value: int | None) -> str:
    return "N/A" if value is None else str(value)


def _format_resident_keys(keys: tuple[tuple[int, int], ...] | None) -> str:
    if keys is None:
        return "N/A"
    if not keys:
        return "empty"
    return ", ".join(f"({layer_id}, {expert_id})" for layer_id, expert_id in keys)


def _phase_label(phase: str) -> str:
    return "prompt/prefill" if phase == "prompt" else "generated/decode"
