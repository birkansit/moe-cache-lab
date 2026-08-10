"""Pure offline transfer-service estimates over simulated byte-cache outcomes.

Cache hits, misses, demand-load bytes, and eviction bytes supplied by this
module's inputs are simulated cache-model quantities. Transfer-service times
computed from caller-supplied hardware assumptions are estimated, not measured.

The transfer model is deliberately serialized/no-overlap:

    serialized_service_time =
        simulated_demand_load_bytes / bandwidth_bytes_per_second
        + simulated_demand_load_count * setup_latency_ns / 1_000_000_000

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

from .byte_cache import ByteCacheSimulation, simulate_byte_cache
from .trace import ExpertKey, RoutingEvent

_POLICY_ORDER = ("lru", "lfu")
_NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class HardwareTransferProfile:
    """Caller-supplied assumptions for an estimated serialized H2D service model."""

    name: str
    h2d_payload_bandwidth_bytes_per_second: int
    setup_latency_ns_per_loaded_expert: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
            or any(not character.isprintable() for character in self.name)
        ):
            raise ValueError(
                "hardware transfer profile name must be a non-empty printable "
                "string without leading or trailing whitespace"
            )
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


@dataclass(frozen=True)
class TransferCostEstimate:
    """Exact estimated serialized transfer-service cost for one simulated cache result."""

    hardware_profile_name: str
    policy: str
    cache_capacity_bytes: int
    simulated_demand_load_count: int
    simulated_demand_load_bytes: int
    assumed_h2d_payload_bandwidth_bytes_per_second: int
    assumed_setup_latency_ns_per_loaded_expert: int
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
class TransferSensitivityRow:
    """One simulated byte-cache outcome paired with one estimated transfer cost."""

    hardware_profile: HardwareTransferProfile
    simulation: ByteCacheSimulation
    estimate: TransferCostEstimate


@dataclass(frozen=True)
class TransferSensitivitySweep:
    """Canonical pure/offline sensitivity sweep.

    Rows are ordered by ascending byte capacity, then policy order ``lru`` /
    ``lfu``, then ascending hardware-profile name. Duplicate capacities and
    policies are normalized; duplicate profile names are rejected.
    """

    capacities_bytes: tuple[int, ...]
    policies: tuple[str, ...]
    hardware_profiles: tuple[HardwareTransferProfile, ...]
    workload_context: WorkloadByteContext
    rows: tuple[TransferSensitivityRow, ...]


def estimate_transfer_cost(
    simulation: ByteCacheSimulation,
    profile: HardwareTransferProfile,
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

    payload_seconds = Fraction(
        simulation.simulated_demand_load_bytes,
        profile.h2d_payload_bandwidth_bytes_per_second,
    )
    setup_seconds = Fraction(
        simulation.misses * profile.setup_latency_ns_per_loaded_expert,
        _NS_PER_SECOND,
    )
    total_seconds = payload_seconds + setup_seconds
    return TransferCostEstimate(
        hardware_profile_name=profile.name,
        policy=simulation.policy,
        cache_capacity_bytes=simulation.capacity_bytes,
        simulated_demand_load_count=simulation.misses,
        simulated_demand_load_bytes=simulation.simulated_demand_load_bytes,
        assumed_h2d_payload_bandwidth_bytes_per_second=(
            profile.h2d_payload_bandwidth_bytes_per_second
        ),
        assumed_setup_latency_ns_per_loaded_expert=(
            profile.setup_latency_ns_per_loaded_expert
        ),
        estimated_payload_service_seconds=payload_seconds,
        estimated_setup_service_seconds=setup_seconds,
        estimated_serialized_transfer_service_seconds=total_seconds,
    )


def run_transfer_sensitivity_sweep(
    events: Iterable[RoutingEvent],
    expert_sizes_bytes: Mapping[ExpertKey, int],
    capacities_bytes: Iterable[int],
    policies: Iterable[str],
    hardware_profiles: Iterable[HardwareTransferProfile],
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
            simulation=simulations[(capacity, policy)],
            estimate=estimate_transfer_cost(simulations[(capacity, policy)], profile),
        )
        for capacity in normalized_capacities
        for policy in normalized_policies
        for profile in normalized_profiles
    )
    return TransferSensitivitySweep(
        capacities_bytes=normalized_capacities,
        policies=normalized_policies,
        hardware_profiles=normalized_profiles,
        workload_context=workload_context,
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
