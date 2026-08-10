"""Combined offline pre-flight analysis over validated routing traces."""

from __future__ import annotations

from dataclasses import dataclass

from .analysis import RoutingLocalitySummary, analyze_routing
from .hardware_cost import TransferSensitivitySweep, run_transfer_sensitivity_sweep
from .preflight_config import PreflightConfig
from .trace import RoutingTrace


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


def run_preflight_analysis(
    trace: RoutingTrace,
    config: PreflightConfig,
) -> PreflightAnalysisResult:
    """Reuse existing routing and transfer-sensitivity cores for one trace/config."""
    if not isinstance(trace, RoutingTrace):
        raise TypeError("preflight analysis requires a validated RoutingTrace")
    if not isinstance(config, PreflightConfig):
        raise TypeError("preflight analysis requires a PreflightConfig")

    return PreflightAnalysisResult(
        trace_provenance=TraceProvenanceSummary(
            model_id=trace.model_id,
            capture_method=trace.capture_method,
            model_revision=trace.model_revision,
            created_at=trace.created_at,
        ),
        config=config,
        routing=analyze_routing(trace),
        transfer_sensitivity=run_transfer_sensitivity_sweep(
            trace.events,
            config.expert_size_map(),
            config.capacities_bytes,
            config.policies,
            config.hardware_profiles,
        ),
    )
