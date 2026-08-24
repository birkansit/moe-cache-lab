"""Combined offline pre-flight analysis over validated routing traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .analysis import RoutingLocalitySummary, analyze_routing
from .byte_cache import (
    CACHE_LIFECYCLE_MODES,
    ByteCacheLifecycleSimulation,
    ByteCacheSimulation,
    ByteCacheWorkload,
    simulate_byte_cache_workloads,
    simulate_versioned_byte_cache,
)
from .hardware_cost import (
    LifecycleTransferEstimateMatrix,
    StageQualifiedTransferSensitivitySweep,
    TransferSensitivitySweep,
    derive_stage_qualified_transfer_sensitivity,
    estimate_lifecycle_transfer_costs,
    run_transfer_sensitivity_sweep,
)
from .preflight_config import (
    PREFLIGHT_CONFIG_LEGACY_VERSION,
    PreflightConfig,
    PreflightConfigV3,
    VersionedPreflightConfig,
)
from .sensitivity_summary import (
    LifecycleSensitivitySummary,
    summarize_lifecycle_sensitivity,
)
from .trace import RoutingTrace
from .trace_v2 import RoutingTraceV2, VersionedRoutingTrace
from .v2_analysis_output import V2DescriptiveAnalysis, analyze_v2_trace
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
class VersionedPreflightEngineResult:
    """Internal cache-only pre-flight result for one validated trace/config pair.

    The simulations are **SIMULATED** cache-model accounting. This B3 result
    deliberately has no hardware-cost estimate or serialized report contract.
    """

    trace: VersionedRoutingTrace
    config: VersionedPreflightConfig
    simulations: tuple[ByteCacheSimulation, ...]


@dataclass(frozen=True)
class StageQualifiedPreflightAnalysisResult:
    """Internal B4 result with stage-safe cache and transfer evidence."""

    engine: VersionedPreflightEngineResult
    routing: V2DescriptiveAnalysis
    transfer_sensitivity: StageQualifiedTransferSensitivitySweep


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


def run_versioned_preflight_engine(
    trace: VersionedRoutingTrace,
    config: VersionedPreflightConfig,
) -> VersionedPreflightEngineResult:
    """Run the compatible single-trace byte-cache sweep without public output."""
    if not isinstance(trace, (RoutingTrace, RoutingTraceV2)):
        raise TypeError(
            "version-aware preflight engine requires RoutingTrace or RoutingTraceV2"
        )
    if not isinstance(config, (PreflightConfig, PreflightConfigV3)):
        raise TypeError(
            "version-aware preflight engine requires PreflightConfig or "
            "PreflightConfigV3"
        )
    if isinstance(trace, RoutingTraceV2):
        if not isinstance(config, PreflightConfigV3):
            raise ValueError(
                "routing trace v2 requires stage-qualified preflight config "
                "format_version 3"
            )
    elif isinstance(config, PreflightConfigV3):
        raise ValueError(
            "routing trace v1 requires layer-qualified preflight config "
            "format_version 1 or 2"
        )

    expert_sizes = config.expert_size_map()
    simulations = tuple(
        simulate_versioned_byte_cache(trace, capacity, expert_sizes, policy)
        for capacity in config.capacities_bytes
        for policy in config.policies
    )
    return VersionedPreflightEngineResult(
        trace=trace,
        config=config,
        simulations=simulations,
    )


def run_stage_qualified_preflight_analysis(
    trace: RoutingTraceV2,
    config: PreflightConfigV3,
    *,
    workload_id: str = "trace",
) -> StageQualifiedPreflightAnalysisResult:
    """Build one stage-qualified single-trace pre-flight result."""

    if not isinstance(trace, RoutingTraceV2):
        raise TypeError(
            "stage-qualified preflight analysis requires a RoutingTraceV2"
        )
    if not isinstance(config, PreflightConfigV3):
        raise TypeError(
            "stage-qualified preflight analysis requires a PreflightConfigV3"
        )

    engine = run_versioned_preflight_engine(trace, config)
    return StageQualifiedPreflightAnalysisResult(
        engine=engine,
        routing=analyze_v2_trace(trace, workload_id=workload_id),
        transfer_sensitivity=derive_stage_qualified_transfer_sensitivity(
            trace,
            config.expert_size_map(),
            engine.simulations,
            config.capacities_bytes,
            config.policies,
            config.hardware_profiles,
            config.transfer_operation_plans,
        ),
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
