"""Combined offline pre-flight analysis over validated routing traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .analysis import RoutingLocalitySummary, analyze_routing
from .byte_cache import (
    CACHE_LIFECYCLE_MODES,
    ByteCacheLifecycleSimulation,
    ByteCacheWorkload,
    simulate_byte_cache_workloads,
)
from .hardware_cost import (
    LifecycleTransferEstimateMatrix,
    TransferSensitivitySweep,
    estimate_lifecycle_transfer_costs,
    run_transfer_sensitivity_sweep,
)
from .preflight_config import PREFLIGHT_CONFIG_LEGACY_VERSION, PreflightConfig
from .sensitivity_summary import (
    LifecycleSensitivitySummary,
    summarize_lifecycle_sensitivity,
)
from .trace import RoutingTrace
from .workflow import load_suite_inputs


@dataclass(frozen=True)
class TraceProvenanceSummary:
    """Trace metadata relevant to interpreting routing observations."""

    model_id: str
    capture_method: str
    model_revision: str | None
    created_at: str | None


@dataclass(frozen=True)
class PreflightAnalysisResult:
    """Immutable combination of routing analysis and pre-flight sensitivity results."""

    trace_provenance: TraceProvenanceSummary
    config: PreflightConfig
    routing: RoutingLocalitySummary
    transfer_sensitivity: TransferSensitivitySweep


@dataclass(frozen=True)
class LifecycleWorkloadProvenance:
    """Manifest-bound identity and trace provenance for one lifecycle workload."""

    workload_id: str
    order: int
    trace_relative_path: str
    trace_sha256: str
    model_id: str
    capture_method: str
    model_revision: str | None
    created_at: str | None


@dataclass(frozen=True)
class PreflightLifecycleAnalysisResult:
    """Deterministic simulated lifecycle comparison for manifest evaluation traces."""

    manifest_filename: str
    manifest_sha256: str
    corpus_version: str
    corpus_sha256: str
    config: PreflightConfig
    workloads: tuple[LifecycleWorkloadProvenance, ...]
    simulations: tuple[ByteCacheLifecycleSimulation, ...]
    sensitivity_summary: LifecycleSensitivitySummary
    transfer_estimates: LifecycleTransferEstimateMatrix


def run_preflight_analysis(
    trace: RoutingTrace,
    config: PreflightConfig,
) -> PreflightAnalysisResult:
    """Reuse existing routing and transfer-sensitivity cores for one trace/config."""
    if not isinstance(trace, RoutingTrace):
        raise TypeError("preflight analysis requires a validated RoutingTrace")
    if not isinstance(config, PreflightConfig):
        raise TypeError("preflight analysis requires a PreflightConfig")

    sweep_arguments = (
        (
            trace.events,
            config.expert_size_map(),
            config.capacities_bytes,
            config.policies,
            config.hardware_profiles,
        )
        if config.format_version == PREFLIGHT_CONFIG_LEGACY_VERSION
        else (
            trace.events,
            config.expert_size_map(),
            config.capacities_bytes,
            config.policies,
            config.hardware_profiles,
            config.transfer_operation_plans,
        )
    )
    return PreflightAnalysisResult(
        trace_provenance=TraceProvenanceSummary(
            model_id=trace.model_id,
            capture_method=trace.capture_method,
            model_revision=trace.model_revision,
            created_at=trace.created_at,
        ),
        config=config,
        routing=analyze_routing(trace),
        transfer_sensitivity=run_transfer_sensitivity_sweep(*sweep_arguments),
    )


def run_preflight_lifecycle_analysis(
    manifest_path: str | Path,
    config: PreflightConfig,
) -> PreflightLifecycleAnalysisResult:
    """Compare explicit cold and persistent lifecycle modes in manifest order."""
    if not isinstance(config, PreflightConfig):
        raise TypeError("preflight lifecycle analysis requires a PreflightConfig")
    inputs = load_suite_inputs(manifest_path)
    workload_inputs = tuple(
        ByteCacheWorkload(
            workload_id=item.prompt.id,
            order=item.prompt.order,
            events=item.trace.events,
        )
        for item in inputs.evaluation
    )
    simulations = tuple(
        simulate_byte_cache_workloads(
            workload_inputs,
            capacity,
            config.expert_size_map(),
            policy,
            lifecycle_mode,
        )
        for capacity in config.capacities_bytes
        for policy in config.policies
        for lifecycle_mode in CACHE_LIFECYCLE_MODES
    )
    return PreflightLifecycleAnalysisResult(
        manifest_filename=inputs.manifest_path.name,
        manifest_sha256=inputs.manifest_sha256,
        corpus_version=inputs.corpus.version,
        corpus_sha256=inputs.corpus.sha256,
        config=config,
        workloads=tuple(
            LifecycleWorkloadProvenance(
                workload_id=item.prompt.id,
                order=item.prompt.order,
                trace_relative_path=item.relative_path,
                trace_sha256=item.sha256,
                model_id=item.trace.model_id,
                capture_method=item.trace.capture_method,
                model_revision=item.trace.model_revision,
                created_at=item.trace.created_at,
            )
            for item in inputs.evaluation
        ),
        simulations=simulations,
        sensitivity_summary=summarize_lifecycle_sensitivity(simulations),
        transfer_estimates=estimate_lifecycle_transfer_costs(
            simulations,
            config.hardware_profiles,
            config.transfer_operation_plans,
        ),
    )
