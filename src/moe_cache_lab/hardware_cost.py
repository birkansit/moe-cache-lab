"""Pure offline transfer-service estimates over simulated byte-cache outcomes.

Cache hits, misses, demand-load bytes, and eviction bytes supplied by this
module's inputs are simulated cache-model quantities. Transfer-service times
computed from caller-supplied hardware assumptions are estimated, not measured.

The transfer model is deliberately serialized/no-overlap:

    serialized_service_time =
        simulated_demand_load_bytes / bandwidth_bytes_per_second
        + modeled_transfer_operation_count * setup_latency_ns / 1_000_000_000

Transfer-operation plans are caller-supplied assumptions. They convert each
simulated logical expert demand load into a fixed positive number of modeled
setup-bearing transfer operations. The default plan preserves the v0.5 model:
one logical load is one modeled transfer operation. These plans do not describe
or establish actual physical movement of packed model parameters.

Simulated evictions are not modeled as D2H transfers. Expert weights are treated
as immutable/read-only cache objects, so eviction only drops simulated device
residency. The estimate excludes GPU compute, overlap, concurrency, kernel
scheduling, allocator effects, CPU overhead, runtime synchronization, protocol
details, D2H writeback, and end-to-end runtime behavior. A lower estimate does
not establish physical GPU residency, latency, throughput, or speedup.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from .byte_cache import (
    CACHE_LIFECYCLE_MODES,
    ByteCacheLifecycleSimulation,
    ByteCacheSimulation,
    ByteCacheWorkloadResult,
    simulate_byte_cache,
)
from .trace import ExpertKey, RoutingEvent
from .trace_v2 import RoutingExpertKeyV2, RoutingTraceV2

_POLICY_ORDER = ("lru", "lfu")
_NS_PER_SECOND = 1_000_000_000


def _validate_name(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(not character.isprintable() for character in value)
    ):
        raise ValueError(
            f"{label} name must be a non-empty printable string without leading "
            "or trailing whitespace"
        )


@dataclass(frozen=True)
class TransferOperationPlan:
    """Assumed setup-bearing transfer operations per simulated logical load."""

    name: str
    operations_per_logical_load: int

    def __post_init__(self) -> None:
        _validate_name(self.name, "transfer operation plan")
        if (
            isinstance(self.operations_per_logical_load, bool)
            or not isinstance(self.operations_per_logical_load, int)
            or self.operations_per_logical_load <= 0
        ):
            raise ValueError(
                "operations_per_logical_load must be a positive integer"
            )


DEFAULT_TRANSFER_OPERATION_PLAN = TransferOperationPlan(
    name="one-operation-per-logical-load",
    operations_per_logical_load=1,
)


@dataclass(frozen=True)
class HardwareTransferProfile:
    """Caller-supplied assumptions for an estimated serialized H2D service model."""

    name: str
    h2d_payload_bandwidth_bytes_per_second: int
    setup_latency_ns_per_loaded_expert: int

    def __post_init__(self) -> None:
        _validate_name(self.name, "hardware transfer profile")
        if (
            isinstance(self.h2d_payload_bandwidth_bytes_per_second, bool)
            or not isinstance(self.h2d_payload_bandwidth_bytes_per_second, int)
            or self.h2d_payload_bandwidth_bytes_per_second <= 0
        ):
            raise ValueError(
                "H2D payload bandwidth must be a positive integer number of bytes/second"
            )
        if (
            isinstance(self.setup_latency_ns_per_loaded_expert, bool)
            or not isinstance(self.setup_latency_ns_per_loaded_expert, int)
            or self.setup_latency_ns_per_loaded_expert < 0
        ):
            raise ValueError(
                "setup latency must be a non-negative integer number of nanoseconds"
            )

    @property
    def setup_latency_ns_per_transfer_operation(self) -> int:
        """Return setup latency applied to each modeled transfer operation.

        The stored field name is retained for v0.5 API/config compatibility.
        Explicit v0.6 configuration serializes this assumption with the
        transfer-operation terminology.
        """

        return self.setup_latency_ns_per_loaded_expert


@dataclass(frozen=True)
class TransferCostEstimate:
    """Exact estimated serialized transfer-service cost for one simulated cache result."""

    hardware_profile_name: str
    transfer_operation_plan_name: str
    policy: str
    cache_capacity_bytes: int
    simulated_demand_load_count: int
    simulated_demand_load_bytes: int
    modeled_transfer_operation_count: int
    assumed_transfer_operations_per_logical_load: int
    assumed_h2d_payload_bandwidth_bytes_per_second: int
    assumed_setup_latency_ns_per_loaded_expert: int
    assumed_setup_latency_ns_per_transfer_operation: int
    estimated_payload_service_seconds: Fraction
    estimated_setup_service_seconds: Fraction
    estimated_serialized_transfer_service_seconds: Fraction


@dataclass(frozen=True)
class WorkloadByteContext:
    """Supplied-size context for layer-qualified experts referenced by a workload.

    These byte totals are descriptive quantities derived from caller-supplied
    expert sizes. They are not measured physical GPU residency.
    """

    unique_referenced_expert_count: int
    unique_referenced_expert_bytes: int
    maximum_atomic_event_working_set_bytes: int
    referenced_expert_keys: tuple[ExpertKey, ...]


@dataclass(frozen=True)
class StageQualifiedWorkloadByteContext:
    """Supplied-size context for assigned stage-qualified expert requests.

    Unassigned trace-v2 events contribute no expert keys or bytes. Extra valid
    caller-supplied size records are not referenced and do not inflate this
    context. These values are descriptive caller-assumption context, not
    measured physical residency.
    """

    unique_referenced_expert_count: int
    unique_referenced_expert_bytes: int
    maximum_atomic_event_working_set_bytes: int
    referenced_expert_keys: tuple[RoutingExpertKeyV2, ...]


@dataclass(frozen=True)
class TransferSensitivityRow:
    """One simulated byte-cache outcome paired with one estimated transfer cost."""

    hardware_profile: HardwareTransferProfile
    transfer_operation_plan: TransferOperationPlan
    simulation: ByteCacheSimulation
    estimate: TransferCostEstimate


@dataclass(frozen=True)
class TransferSensitivitySweep:
    """Canonical pure/offline sensitivity sweep.

    Rows are ordered by ascending byte capacity, then policy order ``lru`` /
    ``lfu``, ascending hardware-profile name, then ascending transfer-operation
    plan name. Duplicate capacities and policies are normalized; duplicate
    profile or plan names are rejected.
    """

    capacities_bytes: tuple[int, ...]
    policies: tuple[str, ...]
    hardware_profiles: tuple[HardwareTransferProfile, ...]
    transfer_operation_plans: tuple[TransferOperationPlan, ...]
    workload_context: WorkloadByteContext
    rows: tuple[TransferSensitivityRow, ...]


@dataclass(frozen=True)
class StageQualifiedTransferSensitivitySweep:
    """ESTIMATED transfer rows derived from existing trace-v2 simulations.

    This type never runs cache simulation. Rows retain the exact
    :class:`ByteCacheSimulation` objects supplied by the B3 engine and are
    ordered by capacity, policy, profile name, then operation-plan name.
    """

    capacities_bytes: tuple[int, ...]
    policies: tuple[str, ...]
    hardware_profiles: tuple[HardwareTransferProfile, ...]
    transfer_operation_plans: tuple[TransferOperationPlan, ...]
    workload_context: StageQualifiedWorkloadByteContext
    rows: tuple[TransferSensitivityRow, ...]


@dataclass(frozen=True)
class LifecycleWorkloadTransferEstimate:
    """One ESTIMATED service slice attributed to an existing workload row."""

    workload_id: str
    order: int
    estimate: TransferCostEstimate


@dataclass(frozen=True)
class LifecycleTransferEstimateRow:
    """One aggregate lifecycle estimate and its exact workload reconciliation."""

    lifecycle_mode: str
    hardware_profile: HardwareTransferProfile
    transfer_operation_plan: TransferOperationPlan
    estimate: TransferCostEstimate
    workloads: tuple[LifecycleWorkloadTransferEstimate, ...]


@dataclass(frozen=True)
class LifecycleTransferEstimateMatrix:
    """Canonical ESTIMATED matrix derived from existing lifecycle simulations.

    Rows are ordered by capacity, LRU/LFU policy order, explicit lifecycle-mode
    order, hardware-profile name, and transfer-operation-plan name.
    """

    capacities_bytes: tuple[int, ...]
    policies: tuple[str, ...]
    lifecycle_modes: tuple[str, ...]
    hardware_profiles: tuple[HardwareTransferProfile, ...]
    transfer_operation_plans: tuple[TransferOperationPlan, ...]
    rows: tuple[LifecycleTransferEstimateRow, ...]


def estimate_transfer_cost(
    simulation: ByteCacheSimulation,
    profile: HardwareTransferProfile,
    transfer_operation_plan: TransferOperationPlan = DEFAULT_TRANSFER_OPERATION_PLAN,
) -> TransferCostEstimate:
    """Estimate exact serialized/no-overlap H2D service time in seconds.

    Only simulated demand loads are charged. ``simulation.simulated_evicted_bytes``
    is intentionally excluded because immutable expert eviction is modeled as
    dropping simulated residency, not as a D2H writeback.
    """

    if not isinstance(simulation, ByteCacheSimulation):
        raise TypeError("transfer-cost estimation requires a ByteCacheSimulation")
    if not isinstance(profile, HardwareTransferProfile):
        raise TypeError("transfer-cost estimation requires a HardwareTransferProfile")
    if not isinstance(transfer_operation_plan, TransferOperationPlan):
        raise TypeError(
            "transfer-cost estimation requires a TransferOperationPlan"
        )

    return _estimate_transfer_accounting(
        policy=simulation.policy,
        cache_capacity_bytes=simulation.capacity_bytes,
        simulated_demand_load_count=simulation.misses,
        simulated_demand_load_bytes=simulation.simulated_demand_load_bytes,
        profile=profile,
        transfer_operation_plan=transfer_operation_plan,
    )


def estimate_lifecycle_transfer_costs(
    simulations: Iterable[ByteCacheLifecycleSimulation],
    hardware_profiles: Iterable[HardwareTransferProfile],
    transfer_operation_plans: Iterable[TransferOperationPlan] = (
        DEFAULT_TRANSFER_OPERATION_PLAN,
    ),
) -> LifecycleTransferEstimateMatrix:
    """Derive exact ESTIMATED service from existing lifecycle rows only.

    This function does not invoke byte-cache simulation. Cold and persistent
    modes remain separate cells; workload estimates are exact attributed slices
    of their existing parent lifecycle row.
    """

    rows = tuple(simulations)
    if not rows:
        raise ValueError("lifecycle transfer estimation requires simulation rows")
    if any(not isinstance(row, ByteCacheLifecycleSimulation) for row in rows):
        raise TypeError(
            "lifecycle transfer estimation requires ByteCacheLifecycleSimulation rows"
        )
    profiles = _normalize_profiles(hardware_profiles)
    plans = _normalize_transfer_operation_plans(transfer_operation_plans)
    by_key, capacities, policies, lifecycle_modes = _validate_lifecycle_rows(rows)

    estimate_rows = tuple(
        _lifecycle_transfer_estimate_row(
            by_key[(mode, policy, capacity)], profile, plan
        )
        for capacity in capacities
        for policy in policies
        for mode in lifecycle_modes
        for profile in profiles
        for plan in plans
    )
    return LifecycleTransferEstimateMatrix(
        capacities_bytes=capacities,
        policies=policies,
        lifecycle_modes=lifecycle_modes,
        hardware_profiles=profiles,
        transfer_operation_plans=plans,
        rows=estimate_rows,
    )


def _estimate_transfer_accounting(
    *,
    policy: str,
    cache_capacity_bytes: int,
    simulated_demand_load_count: int,
    simulated_demand_load_bytes: int,
    profile: HardwareTransferProfile,
    transfer_operation_plan: TransferOperationPlan,
) -> TransferCostEstimate:
    """Apply the one authoritative exact serialized transfer-cost equation."""

    payload_seconds = Fraction(
        simulated_demand_load_bytes,
        profile.h2d_payload_bandwidth_bytes_per_second,
    )
    transfer_operation_count = (
        simulated_demand_load_count
        * transfer_operation_plan.operations_per_logical_load
    )
    setup_seconds = Fraction(
        transfer_operation_count * profile.setup_latency_ns_per_transfer_operation,
        _NS_PER_SECOND,
    )
    return TransferCostEstimate(
        hardware_profile_name=profile.name,
        transfer_operation_plan_name=transfer_operation_plan.name,
        policy=policy,
        cache_capacity_bytes=cache_capacity_bytes,
        simulated_demand_load_count=simulated_demand_load_count,
        simulated_demand_load_bytes=simulated_demand_load_bytes,
        modeled_transfer_operation_count=transfer_operation_count,
        assumed_transfer_operations_per_logical_load=(
            transfer_operation_plan.operations_per_logical_load
        ),
        assumed_h2d_payload_bandwidth_bytes_per_second=(
            profile.h2d_payload_bandwidth_bytes_per_second
        ),
        assumed_setup_latency_ns_per_loaded_expert=(
            profile.setup_latency_ns_per_loaded_expert
        ),
        assumed_setup_latency_ns_per_transfer_operation=(
            profile.setup_latency_ns_per_transfer_operation
        ),
        estimated_payload_service_seconds=payload_seconds,
        estimated_setup_service_seconds=setup_seconds,
        estimated_serialized_transfer_service_seconds=(
            payload_seconds + setup_seconds
        ),
    )


def _lifecycle_transfer_estimate_row(
    simulation: ByteCacheLifecycleSimulation,
    profile: HardwareTransferProfile,
    plan: TransferOperationPlan,
) -> LifecycleTransferEstimateRow:
    aggregate = _estimate_transfer_accounting(
        policy=simulation.policy,
        cache_capacity_bytes=simulation.capacity_bytes,
        simulated_demand_load_count=simulation.misses,
        simulated_demand_load_bytes=simulation.simulated_demand_load_bytes,
        profile=profile,
        transfer_operation_plan=plan,
    )
    workloads = tuple(
        LifecycleWorkloadTransferEstimate(
            workload_id=workload.workload_id,
            order=workload.order,
            estimate=_estimate_workload_transfer_accounting(workload, profile, plan),
        )
        for workload in simulation.workloads
    )
    for field in (
        "simulated_demand_load_count",
        "simulated_demand_load_bytes",
        "modeled_transfer_operation_count",
        "estimated_payload_service_seconds",
        "estimated_setup_service_seconds",
        "estimated_serialized_transfer_service_seconds",
    ):
        if getattr(aggregate, field) != sum(
            (getattr(workload.estimate, field) for workload in workloads),
            Fraction(0, 1) if field.startswith("estimated_") else 0,
        ):
            raise ValueError(
                f"lifecycle transfer aggregate {field} does not reconcile with workloads"
            )
    return LifecycleTransferEstimateRow(
        lifecycle_mode=simulation.lifecycle_mode,
        hardware_profile=profile,
        transfer_operation_plan=plan,
        estimate=aggregate,
        workloads=workloads,
    )


def _estimate_workload_transfer_accounting(
    workload: ByteCacheWorkloadResult,
    profile: HardwareTransferProfile,
    plan: TransferOperationPlan,
) -> TransferCostEstimate:
    return _estimate_transfer_accounting(
        policy=workload.policy,
        cache_capacity_bytes=workload.capacity_bytes,
        simulated_demand_load_count=workload.misses,
        simulated_demand_load_bytes=workload.simulated_demand_load_bytes,
        profile=profile,
        transfer_operation_plan=plan,
    )


def _validate_lifecycle_rows(
    rows: tuple[ByteCacheLifecycleSimulation, ...],
) -> tuple[
    dict[tuple[str, str, int], ByteCacheLifecycleSimulation],
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    by_key: dict[tuple[str, str, int], ByteCacheLifecycleSimulation] = {}
    workload_signature: tuple[tuple[int, str, int, int], ...] | None = None
    for row in rows:
        if row.lifecycle_mode not in CACHE_LIFECYCLE_MODES:
            raise ValueError(
                f"unsupported lifecycle transfer mode: {row.lifecycle_mode}"
            )
        if row.policy not in _POLICY_ORDER:
            raise ValueError(
                f"unsupported lifecycle transfer policy: {row.policy}"
            )
        if (
            isinstance(row.capacity_bytes, bool)
            or not isinstance(row.capacity_bytes, int)
            or row.capacity_bytes <= 0
        ):
            raise ValueError("lifecycle transfer capacities must be positive integers")
        key = row.lifecycle_mode, row.policy, row.capacity_bytes
        if key in by_key:
            raise ValueError("duplicate lifecycle transfer simulation cell")
        current_signature = _validate_lifecycle_workload_reconciliation(row)
        if workload_signature is None:
            workload_signature = current_signature
        elif current_signature != workload_signature:
            raise ValueError(
                "lifecycle transfer rows use incompatible workload identities, "
                "orders, events, or request counts"
            )
        by_key[key] = row

    capacities = tuple(sorted({row.capacity_bytes for row in rows}))
    policies = tuple(
        policy for policy in _POLICY_ORDER if any(row.policy == policy for row in rows)
    )
    modes = tuple(
        mode
        for mode in CACHE_LIFECYCLE_MODES
        if any(row.lifecycle_mode == mode for row in rows)
    )
    if modes != CACHE_LIFECYCLE_MODES:
        raise ValueError(
            "lifecycle transfer input requires both explicit lifecycle modes"
        )
    expected = {
        (mode, policy, capacity)
        for capacity in capacities
        for policy in policies
        for mode in modes
    }
    if set(by_key) != expected:
        raise ValueError("lifecycle transfer input is an incomplete rectangular grid")
    return by_key, capacities, policies, modes


def _validate_lifecycle_workload_reconciliation(
    row: ByteCacheLifecycleSimulation,
) -> tuple[tuple[int, str, int, int], ...]:
    if row.workload_count != len(row.workloads):
        raise ValueError("lifecycle transfer workload count does not reconcile")
    signature = tuple(
        (
            workload.order,
            workload.workload_id,
            workload.event_count,
            workload.expert_request_count,
        )
        for workload in row.workloads
    )
    identities = tuple(item[:2] for item in signature)
    if identities != tuple(sorted(identities)):
        raise ValueError("lifecycle transfer workloads are not in declared order")
    if len(set(identities)) != len(identities):
        raise ValueError("lifecycle transfer workload identities must be unique")
    for workload in row.workloads:
        if (
            workload.lifecycle_mode != row.lifecycle_mode
            or workload.policy != row.policy
            or workload.capacity_bytes != row.capacity_bytes
        ):
            raise ValueError(
                "lifecycle transfer workload is incompatible with its parent"
            )
        if workload.expert_request_count != workload.hits + workload.misses:
            raise ValueError("lifecycle transfer workload requests do not reconcile")
    for field in ("misses", "simulated_demand_load_bytes"):
        if getattr(row, field) != sum(
            getattr(workload, field) for workload in row.workloads
        ):
            raise ValueError(
                f"lifecycle transfer aggregate {field} does not reconcile with workloads"
            )
    return signature


def run_transfer_sensitivity_sweep(
    events: Iterable[RoutingEvent],
    expert_sizes_bytes: Mapping[ExpertKey, int],
    capacities_bytes: Iterable[int],
    policies: Iterable[str],
    hardware_profiles: Iterable[HardwareTransferProfile],
    transfer_operation_plans: Iterable[TransferOperationPlan] = (
        DEFAULT_TRANSFER_OPERATION_PLAN,
    ),
) -> TransferSensitivitySweep:
    """Run deterministic byte-cache simulations and exact transfer-cost estimates.

    The existing :func:`simulate_byte_cache` implementation remains authoritative
    for LRU/LFU behavior and validation. Hardware profiles are caller-supplied
    assumptions only; this function does not detect, calibrate, or benchmark
    hardware.
    """

    sequence = tuple(events)
    normalized_capacities = _normalize_capacities(capacities_bytes)
    normalized_policies = _normalize_policies(policies)
    normalized_profiles = _normalize_profiles(hardware_profiles)
    normalized_plans = _normalize_transfer_operation_plans(
        transfer_operation_plans
    )

    # Use the existing simulator as the validation authority for RoutingEvent
    # inputs and the complete size map, including extra entries.
    baseline_key = (normalized_capacities[-1], normalized_policies[0])
    simulations: dict[tuple[int, str], ByteCacheSimulation] = {
        baseline_key: simulate_byte_cache(
            sequence,
            baseline_key[0],
            expert_sizes_bytes,
            baseline_key[1],
        )
    }

    workload_context = _workload_byte_context(sequence, expert_sizes_bytes)
    infeasible = tuple(
        capacity
        for capacity in normalized_capacities
        if capacity < workload_context.maximum_atomic_event_working_set_bytes
    )
    if infeasible:
        raise ValueError(
            "sweep capacity "
            f"{infeasible[0]} bytes is smaller than maximum atomic event working "
            f"set {workload_context.maximum_atomic_event_working_set_bytes} bytes"
        )

    for capacity in normalized_capacities:
        for policy in normalized_policies:
            key = (capacity, policy)
            if key not in simulations:
                simulations[key] = simulate_byte_cache(
                    sequence,
                    capacity,
                    expert_sizes_bytes,
                    policy,
                )

    rows = tuple(
        TransferSensitivityRow(
            hardware_profile=profile,
            transfer_operation_plan=plan,
            simulation=simulations[(capacity, policy)],
            estimate=estimate_transfer_cost(
                simulations[(capacity, policy)], profile, plan
            ),
        )
        for capacity in normalized_capacities
        for policy in normalized_policies
        for profile in normalized_profiles
        for plan in normalized_plans
    )
    return TransferSensitivitySweep(
        capacities_bytes=normalized_capacities,
        policies=normalized_policies,
        hardware_profiles=normalized_profiles,
        transfer_operation_plans=normalized_plans,
        workload_context=workload_context,
        rows=rows,
    )


def derive_stage_qualified_transfer_sensitivity(
    trace: RoutingTraceV2,
    expert_sizes_bytes: Mapping[RoutingExpertKeyV2, int],
    simulations: Iterable[ByteCacheSimulation],
    capacities_bytes: Iterable[int],
    policies: Iterable[str],
    hardware_profiles: Iterable[HardwareTransferProfile],
    transfer_operation_plans: Iterable[TransferOperationPlan],
) -> StageQualifiedTransferSensitivitySweep:
    """Apply the existing transfer estimator to exact B3 simulation cells.

    The supplied simulation grid is validated and reused by identity; cache
    state is never replayed here. All cache counters remain **SIMULATED** and
    all returned transfer-service values remain **ESTIMATED**.
    """

    if not isinstance(trace, RoutingTraceV2):
        raise TypeError(
            "stage-qualified transfer sensitivity requires a RoutingTraceV2"
        )
    normalized_capacities = _normalize_capacities(capacities_bytes)
    normalized_policies = _normalize_policies(policies)
    normalized_profiles = _normalize_profiles(hardware_profiles)
    normalized_plans = _normalize_transfer_operation_plans(
        transfer_operation_plans
    )
    simulation_rows = tuple(simulations)
    if not simulation_rows:
        raise ValueError(
            "stage-qualified transfer sensitivity requires B3 simulation rows"
        )
    if any(not isinstance(row, ByteCacheSimulation) for row in simulation_rows):
        raise TypeError(
            "stage-qualified transfer sensitivity requires ByteCacheSimulation rows"
        )

    expected_cells = tuple(
        (capacity, policy)
        for capacity in normalized_capacities
        for policy in normalized_policies
    )
    actual_cells = tuple(
        (row.capacity_bytes, row.policy) for row in simulation_rows
    )
    if actual_cells != expected_cells:
        raise ValueError(
            "B3 simulation rows must form the canonical capacity/policy grid"
        )

    event_count = len(trace.events)
    expert_request_count = len(trace.expert_requests)
    for row in simulation_rows:
        if row.event_count != event_count:
            raise ValueError("B3 simulation event count does not match trace v2")
        if row.expert_request_count != expert_request_count:
            raise ValueError(
                "B3 simulation expert-request count does not match trace v2"
            )
        if row.expert_request_count != row.hits + row.misses:
            raise ValueError("B3 simulation hit/miss accounting does not reconcile")

    context = _stage_qualified_workload_byte_context(
        trace, expert_sizes_bytes
    )
    rows = tuple(
        TransferSensitivityRow(
            hardware_profile=profile,
            transfer_operation_plan=plan,
            simulation=simulation,
            estimate=estimate_transfer_cost(simulation, profile, plan),
        )
        for simulation in simulation_rows
        for profile in normalized_profiles
        for plan in normalized_plans
    )
    return StageQualifiedTransferSensitivitySweep(
        capacities_bytes=normalized_capacities,
        policies=normalized_policies,
        hardware_profiles=normalized_profiles,
        transfer_operation_plans=normalized_plans,
        workload_context=context,
        rows=rows,
    )


def _normalize_capacities(capacities_bytes: Iterable[int]) -> tuple[int, ...]:
    values = tuple(capacities_bytes)
    if not values:
        raise ValueError("transfer sensitivity sweep requires at least one byte capacity")
    for capacity in values:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("sweep byte capacities must be positive integers")
    return tuple(sorted(set(values)))


def _normalize_policies(policies: Iterable[str]) -> tuple[str, ...]:
    values = tuple(policies)
    if not values:
        raise ValueError("transfer sensitivity sweep requires at least one policy")
    unsupported = sorted(
        {str(policy) for policy in values if policy not in _POLICY_ORDER}
    )
    if unsupported:
        raise ValueError(
            "unsupported transfer sensitivity policy: " + ", ".join(unsupported)
        )
    selected = set(values)
    return tuple(policy for policy in _POLICY_ORDER if policy in selected)


def _normalize_profiles(
    hardware_profiles: Iterable[HardwareTransferProfile],
) -> tuple[HardwareTransferProfile, ...]:
    values = tuple(hardware_profiles)
    if not values:
        raise ValueError("transfer sensitivity sweep requires at least one hardware profile")
    if any(not isinstance(profile, HardwareTransferProfile) for profile in values):
        raise TypeError("hardware profiles must be HardwareTransferProfile objects")
    names = [profile.name for profile in values]
    duplicate_names = sorted(
        name for name in set(names) if names.count(name) > 1
    )
    if duplicate_names:
        raise ValueError(
            "hardware profile names must be unique within a sweep: "
            + ", ".join(duplicate_names)
        )
    return tuple(sorted(values, key=lambda profile: profile.name))


def _normalize_transfer_operation_plans(
    transfer_operation_plans: Iterable[TransferOperationPlan],
) -> tuple[TransferOperationPlan, ...]:
    values = tuple(transfer_operation_plans)
    if not values:
        raise ValueError(
            "transfer sensitivity sweep requires at least one transfer operation plan"
        )
    if any(not isinstance(plan, TransferOperationPlan) for plan in values):
        raise TypeError(
            "transfer operation plans must be TransferOperationPlan objects"
        )
    names = [plan.name for plan in values]
    duplicate_names = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicate_names:
        raise ValueError(
            "transfer operation plan names must be unique within a sweep: "
            + ", ".join(duplicate_names)
        )
    return tuple(sorted(values, key=lambda plan: plan.name))


def _workload_byte_context(
    events: tuple[RoutingEvent, ...],
    expert_sizes_bytes: Mapping[ExpertKey, int],
) -> WorkloadByteContext:
    required_by_event = tuple(
        tuple(sorted((event.layer, expert_id) for expert_id in event.selected_experts))
        for event in events
    )
    referenced_keys = tuple(
        sorted({key for required in required_by_event for key in required})
    )
    return WorkloadByteContext(
        unique_referenced_expert_count=len(referenced_keys),
        unique_referenced_expert_bytes=sum(
            expert_sizes_bytes[key] for key in referenced_keys
        ),
        maximum_atomic_event_working_set_bytes=max(
            (
                sum(expert_sizes_bytes[key] for key in required)
                for required in required_by_event
            ),
            default=0,
        ),
        referenced_expert_keys=referenced_keys,
    )


def _stage_qualified_workload_byte_context(
    trace: RoutingTraceV2,
    expert_sizes_bytes: Mapping[RoutingExpertKeyV2, int],
) -> StageQualifiedWorkloadByteContext:
    required_by_event = tuple(event.expert_requests for event in trace.events)
    referenced_keys = tuple(
        sorted(
            {key for required in required_by_event for key in required},
            key=_stage_qualified_key_sort_key,
        )
    )
    missing_sizes = tuple(
        key for key in referenced_keys if key not in expert_sizes_bytes
    )
    if missing_sizes:
        raise ValueError(
            "expert size map is missing referenced stage-qualified experts: "
            + ", ".join(str(key) for key in missing_sizes)
        )
    return StageQualifiedWorkloadByteContext(
        unique_referenced_expert_count=len(referenced_keys),
        unique_referenced_expert_bytes=sum(
            expert_sizes_bytes[key] for key in referenced_keys
        ),
        maximum_atomic_event_working_set_bytes=max(
            (
                sum(expert_sizes_bytes[key] for key in required)
                for required in required_by_event
            ),
            default=0,
        ),
        referenced_expert_keys=referenced_keys,
    )


def _stage_qualified_key_sort_key(
    key: RoutingExpertKeyV2,
) -> tuple[int, int, int]:
    stage, layer_id, expert_id = key
    return (0 if stage == "encoder" else 1), layer_id, expert_id
