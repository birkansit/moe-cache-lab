"""Deterministic offline byte-capacity cache simulation.

All load and eviction byte counts in this module are simulated cache-model
accounting based on caller-supplied expert sizes. They are not measured
host-device/GPU transfers and do not establish latency, throughput, or speedup.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .trace import ExpertKey, RoutingEvent, RoutingTrace
from .trace_v2 import RoutingExpertKeyV2, RoutingTraceV2, VersionedRoutingTrace

CACHE_LIFECYCLE_MODES = ("cold_per_workload", "persistent_sequence")
VersionedExpertKey = ExpertKey | RoutingExpertKeyV2


@dataclass(frozen=True)
class ByteCacheSimulation:
    """Immutable result of simulated byte-capacity cache accounting.

    ``simulated_demand_load_bytes`` and ``simulated_evicted_bytes`` are offline
    cache-model accounting quantities. They do not represent measured physical
    transfers, GPU residency, latency, throughput, or speedup.
    """

    policy: str
    capacity_bytes: int
    event_count: int
    expert_request_count: int
    hits: int
    misses: int
    simulated_demand_load_bytes: int
    eviction_count: int
    simulated_evicted_bytes: int
    peak_resident_bytes: int
    final_resident_bytes: int
    final_resident_keys: tuple[VersionedExpertKey, ...]

    @property
    def hit_rate(self) -> float:
        """Return the established zero-for-empty request hit-rate behavior."""
        return self.hits / self.expert_request_count if self.expert_request_count else 0.0


@dataclass(frozen=True)
class ByteCacheWorkload:
    """One explicitly identified workload in a caller-declared sequence."""

    workload_id: str
    order: int
    events: tuple[RoutingEvent, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.workload_id, str)
            or not self.workload_id
            or self.workload_id != self.workload_id.strip()
            or any(not character.isprintable() for character in self.workload_id)
        ):
            raise ValueError(
                "workload_id must be a non-empty printable string without leading "
                "or trailing whitespace"
            )
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("workload order must be a non-negative integer")
        events = tuple(self.events)
        if any(not isinstance(event, RoutingEvent) for event in events):
            raise TypeError("byte-cache workloads require RoutingEvent objects")
        object.__setattr__(self, "events", events)


@dataclass(frozen=True)
class ByteCacheWorkloadResult:
    """Attributed counters and residency boundaries for one workload replay."""

    workload_id: str
    order: int
    lifecycle_mode: str
    policy: str
    capacity_bytes: int
    event_count: int
    expert_request_count: int
    hits: int
    misses: int
    simulated_demand_load_bytes: int
    eviction_count: int
    simulated_evicted_bytes: int
    peak_resident_bytes: int
    starting_resident_bytes: int
    starting_resident_keys: tuple[ExpertKey, ...]
    ending_resident_bytes: int
    ending_resident_keys: tuple[ExpertKey, ...]

    @property
    def hit_rate(self) -> float | None:
        """Return the hit rate, or ``None`` when this workload has no requests."""
        if not self.expert_request_count:
            return None
        return self.hits / self.expert_request_count


@dataclass(frozen=True)
class ByteCacheLifecycleSimulation:
    """One lifecycle scenario over an explicitly ordered workload collection."""

    lifecycle_mode: str
    policy: str
    capacity_bytes: int
    workload_count: int
    event_count: int
    expert_request_count: int
    hits: int
    misses: int
    simulated_demand_load_bytes: int
    eviction_count: int
    simulated_evicted_bytes: int
    peak_resident_bytes: int
    shared_sequence_final_resident_bytes: int | None
    shared_sequence_final_resident_keys: tuple[ExpertKey, ...] | None
    workloads: tuple[ByteCacheWorkloadResult, ...]

    @property
    def hit_rate(self) -> float | None:
        """Return the aggregate hit rate, or ``None`` for zero total requests."""
        if not self.expert_request_count:
            return None
        return self.hits / self.expert_request_count


class _ByteDynamicCache:
    name: str

    def __init__(
        self,
        capacity_bytes: int,
        expert_sizes_bytes: Mapping[VersionedExpertKey, int],
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.expert_sizes_bytes = expert_sizes_bytes
        self.clock = 0

    @property
    def resident_keys(self) -> frozenset[VersionedExpertKey]:
        raise NotImplementedError

    @property
    def resident_bytes(self) -> int:
        return sum(self.expert_sizes_bytes[key] for key in self.resident_keys)

    def _evict_one(
        self, candidates: frozenset[VersionedExpertKey]
    ) -> VersionedExpertKey:
        raise NotImplementedError

    def _touch_required(self, required: frozenset[VersionedExpertKey]) -> None:
        raise NotImplementedError

    def access_bundle(
        self,
        required: frozenset[VersionedExpertKey],
    ) -> tuple[int, int, int, int, int, int]:
        """Access one atomic pinned working set and return simulated accounting."""

        self.clock += 1
        resident_at_start = self.resident_keys
        hits = len(required.intersection(resident_at_start))
        missing = required.difference(resident_at_start)
        misses = len(missing)
        demand_load_bytes = sum(self.expert_sizes_bytes[key] for key in missing)

        resident_bytes = self.resident_bytes
        eviction_count = 0
        evicted_bytes = 0
        while resident_bytes + demand_load_bytes > self.capacity_bytes:
            candidates = self.resident_keys.difference(required)
            if not candidates:
                raise AssertionError(
                    "validated byte-cache event cannot fit without evicting a pinned requirement"
                )
            victim = self._evict_one(candidates)
            victim_bytes = self.expert_sizes_bytes[victim]
            resident_bytes -= victim_bytes
            eviction_count += 1
            evicted_bytes += victim_bytes

        self._touch_required(required)
        resident_bytes += demand_load_bytes
        if resident_bytes != self.resident_bytes:
            raise AssertionError("byte-cache resident-byte accounting diverged from residency")
        if resident_bytes > self.capacity_bytes:
            raise AssertionError("byte-cache residency exceeded configured capacity")

        return (
            hits,
            misses,
            demand_load_bytes,
            eviction_count,
            evicted_bytes,
            resident_bytes,
        )


class _ByteLRUCache(_ByteDynamicCache):
    """Event-atomic byte-capacity LRU with stable layer/expert tie-breaking."""

    name = "lru"

    def __init__(
        self,
        capacity_bytes: int,
        expert_sizes_bytes: Mapping[VersionedExpertKey, int],
    ) -> None:
        super().__init__(capacity_bytes, expert_sizes_bytes)
        self.entries: dict[VersionedExpertKey, int] = {}

    @property
    def resident_keys(self) -> frozenset[VersionedExpertKey]:
        return frozenset(self.entries)

    def _evict_one(
        self, candidates: frozenset[VersionedExpertKey]
    ) -> VersionedExpertKey:
        victim = min(candidates, key=lambda key: (self.entries[key], key))
        del self.entries[victim]
        return victim

    def _touch_required(self, required: frozenset[VersionedExpertKey]) -> None:
        # One routing event is atomic: every required key receives the same
        # logical timestamp, so selected-expert tuple ordering cannot affect LRU.
        for key in required:
            self.entries[key] = self.clock


class _ByteLFUCache(_ByteDynamicCache):
    """Resident-lifetime LFU with stable oldest/key tie-breaking."""

    name = "lfu"

    def __init__(
        self,
        capacity_bytes: int,
        expert_sizes_bytes: Mapping[VersionedExpertKey, int],
    ) -> None:
        super().__init__(capacity_bytes, expert_sizes_bytes)
        self.entries: dict[VersionedExpertKey, tuple[int, int]] = {}

    @property
    def resident_keys(self) -> frozenset[VersionedExpertKey]:
        return frozenset(self.entries)

    def _evict_one(
        self, candidates: frozenset[VersionedExpertKey]
    ) -> VersionedExpertKey:
        victim = min(
            candidates,
            key=lambda key: (self.entries[key][0], self.entries[key][1], key),
        )
        del self.entries[victim]
        return victim

    def _touch_required(self, required: frozenset[VersionedExpertKey]) -> None:
        for key in required:
            if key in self.entries:
                frequency, _ = self.entries[key]
                self.entries[key] = (frequency + 1, self.clock)
            else:
                # Match the count-capacity LFU: evicted frequency history does
                # not survive a later simulated demand load.
                self.entries[key] = (1, self.clock)


def simulate_byte_cache(
    events: Iterable[RoutingEvent],
    capacity_bytes: int,
    expert_sizes_bytes: Mapping[ExpertKey, int],
    policy_name: str,
) -> ByteCacheSimulation:
    """Replay routing events through a byte-capacity LRU or LFU simulation.

    Hits/misses are determined from residency at event start. Each routing
    event's complete layer-qualified required set is pinned atomically while
    deterministic non-required victims are evicted until all missing required
    bytes fit. Extra valid size-map entries are permitted and do not affect the
    result.

    The byte totals returned here are simulated cache-model accounting only;
    they are not measured physical transfer volumes or runtime performance.
    """

    sequence = tuple(events)
    capacity_bytes = _validate_capacity_bytes(capacity_bytes)
    _validate_policy_name(policy_name)
    validated_sizes = _validate_expert_sizes_bytes(expert_sizes_bytes)
    required_by_event = _validate_event_sequence(
        sequence, capacity_bytes, validated_sizes
    )
    policy = _new_policy(policy_name, capacity_bytes, validated_sizes)
    return _replay_byte_cache(sequence, required_by_event, policy)


def simulate_versioned_byte_cache(
    trace: VersionedRoutingTrace,
    capacity_bytes: int,
    expert_sizes_bytes: Mapping[VersionedExpertKey, int],
    policy_name: str,
) -> ByteCacheSimulation:
    """Simulate one validated canonical v1 or v2 trace by byte capacity.

    Version 1 delegates to the established :func:`simulate_byte_cache` path.
    Version 2 preserves stage-qualified ``(routing_stage, layer, expert_id)``
    identity and keeps capacity-unassigned events in ``event_count`` while
    projecting them to an empty required set. Returned byte and cache counters
    remain **SIMULATED** cache-model accounting.
    """

    if isinstance(trace, RoutingTraceV2):
        capacity_bytes = _validate_capacity_bytes(capacity_bytes)
        _validate_policy_name(policy_name)
        validated_sizes = _validate_v2_expert_sizes_bytes(expert_sizes_bytes)
        required_by_event = tuple(
            frozenset(event.expert_requests) for event in trace.events
        )
        _validate_required_by_event(
            required_by_event,
            capacity_bytes,
            validated_sizes,
            "stage-qualified experts",
        )
        policy = _new_policy(policy_name, capacity_bytes, validated_sizes)
        return _replay_byte_cache(trace.events, required_by_event, policy)
    if isinstance(trace, RoutingTrace):
        return simulate_byte_cache(
            trace.events,
            capacity_bytes,
            expert_sizes_bytes,
            policy_name,
        )
    raise TypeError(
        "version-aware byte-cache simulation requires RoutingTrace or RoutingTraceV2"
    )


def simulate_byte_cache_workloads(
    workloads: Iterable[ByteCacheWorkload],
    capacity_bytes: int,
    expert_sizes_bytes: Mapping[ExpertKey, int],
    policy_name: str,
    lifecycle_mode: str,
) -> ByteCacheLifecycleSimulation:
    """Simulate cold or persistent cache state across declared workloads.

    ``cold_per_workload`` creates an empty cache for every workload.
    ``persistent_sequence`` reuses one cache in the exact supplied workload
    order. The lifecycle changes only reset boundaries; event-atomic admission,
    eviction, and layer-qualified identity semantics are shared with
    :func:`simulate_byte_cache`.
    """

    normalized_workloads = _validate_workloads(workloads)
    capacity_bytes = _validate_capacity_bytes(capacity_bytes)
    _validate_policy_name(policy_name)
    if lifecycle_mode not in CACHE_LIFECYCLE_MODES:
        raise ValueError(f"unknown byte-cache lifecycle mode: {lifecycle_mode}")
    validated_sizes = _validate_expert_sizes_bytes(expert_sizes_bytes)
    prepared = tuple(
        (
            workload,
            _validate_event_sequence(
                workload.events, capacity_bytes, validated_sizes
            ),
        )
        for workload in normalized_workloads
    )

    shared_policy = (
        _new_policy(policy_name, capacity_bytes, validated_sizes)
        if lifecycle_mode == "persistent_sequence"
        else None
    )
    results: list[ByteCacheWorkloadResult] = []
    for workload, required_by_event in prepared:
        policy = shared_policy or _new_policy(
            policy_name, capacity_bytes, validated_sizes
        )
        starting_keys = tuple(sorted(policy.resident_keys))
        starting_bytes = policy.resident_bytes
        simulation = _replay_byte_cache(workload.events, required_by_event, policy)
        results.append(
            ByteCacheWorkloadResult(
                workload_id=workload.workload_id,
                order=workload.order,
                lifecycle_mode=lifecycle_mode,
                policy=simulation.policy,
                capacity_bytes=simulation.capacity_bytes,
                event_count=simulation.event_count,
                expert_request_count=simulation.expert_request_count,
                hits=simulation.hits,
                misses=simulation.misses,
                simulated_demand_load_bytes=(
                    simulation.simulated_demand_load_bytes
                ),
                eviction_count=simulation.eviction_count,
                simulated_evicted_bytes=simulation.simulated_evicted_bytes,
                peak_resident_bytes=simulation.peak_resident_bytes,
                starting_resident_bytes=starting_bytes,
                starting_resident_keys=starting_keys,
                ending_resident_bytes=simulation.final_resident_bytes,
                ending_resident_keys=simulation.final_resident_keys,
            )
        )

    frozen_results = tuple(results)
    persistent_final_keys = (
        tuple(sorted(shared_policy.resident_keys))
        if shared_policy is not None
        else None
    )
    persistent_final_bytes = (
        shared_policy.resident_bytes if shared_policy is not None else None
    )
    return ByteCacheLifecycleSimulation(
        lifecycle_mode=lifecycle_mode,
        policy=policy_name,
        capacity_bytes=capacity_bytes,
        workload_count=len(frozen_results),
        event_count=sum(result.event_count for result in frozen_results),
        expert_request_count=sum(
            result.expert_request_count for result in frozen_results
        ),
        hits=sum(result.hits for result in frozen_results),
        misses=sum(result.misses for result in frozen_results),
        simulated_demand_load_bytes=sum(
            result.simulated_demand_load_bytes for result in frozen_results
        ),
        eviction_count=sum(result.eviction_count for result in frozen_results),
        simulated_evicted_bytes=sum(
            result.simulated_evicted_bytes for result in frozen_results
        ),
        peak_resident_bytes=max(
            (result.peak_resident_bytes for result in frozen_results),
            default=0,
        ),
        shared_sequence_final_resident_bytes=persistent_final_bytes,
        shared_sequence_final_resident_keys=persistent_final_keys,
        workloads=frozen_results,
    )


def _replay_byte_cache(
    sequence: tuple[object, ...],
    required_by_event: tuple[frozenset[VersionedExpertKey], ...],
    policy: _ByteDynamicCache,
) -> ByteCacheSimulation:
    """Replay one prepared event sequence through an existing policy state."""

    hits = 0
    misses = 0
    demand_load_bytes = 0
    eviction_count = 0
    evicted_bytes = 0
    peak_resident_bytes = policy.resident_bytes

    for required in required_by_event:
        if not required:
            continue
        (
            event_hits,
            event_misses,
            event_demand_load_bytes,
            event_eviction_count,
            event_evicted_bytes,
            resident_bytes,
        ) = policy.access_bundle(required)
        hits += event_hits
        misses += event_misses
        demand_load_bytes += event_demand_load_bytes
        eviction_count += event_eviction_count
        evicted_bytes += event_evicted_bytes
        peak_resident_bytes = max(peak_resident_bytes, resident_bytes)

    return ByteCacheSimulation(
        policy=policy.name,
        capacity_bytes=policy.capacity_bytes,
        event_count=len(sequence),
        expert_request_count=sum(len(required) for required in required_by_event),
        hits=hits,
        misses=misses,
        simulated_demand_load_bytes=demand_load_bytes,
        eviction_count=eviction_count,
        simulated_evicted_bytes=evicted_bytes,
        peak_resident_bytes=peak_resident_bytes,
        final_resident_bytes=policy.resident_bytes,
        final_resident_keys=tuple(sorted(policy.resident_keys)),
    )


def _validate_event_sequence(
    sequence: tuple[RoutingEvent, ...],
    capacity_bytes: int,
    validated_sizes: Mapping[ExpertKey, int],
) -> tuple[frozenset[ExpertKey], ...]:
    if any(not isinstance(event, RoutingEvent) for event in sequence):
        raise TypeError("byte-cache simulation requires RoutingEvent objects")
    required_by_event = tuple(
        frozenset((event.layer, expert_id) for expert_id in event.selected_experts)
        for event in sequence
    )
    _validate_required_by_event(
        required_by_event,
        capacity_bytes,
        validated_sizes,
        "layer-qualified experts",
    )
    return required_by_event


def _validate_required_by_event(
    required_by_event: tuple[frozenset[VersionedExpertKey], ...],
    capacity_bytes: int,
    validated_sizes: Mapping[VersionedExpertKey, int],
    identity_description: str,
) -> None:
    referenced_keys = frozenset(
        key for required in required_by_event for key in required
    )
    missing_sizes = sorted(referenced_keys.difference(validated_sizes))
    if missing_sizes:
        raise ValueError(
            f"expert size map is missing referenced {identity_description}: "
            + ", ".join(str(key) for key in missing_sizes)
        )
    for required in required_by_event:
        required_bytes = sum(validated_sizes[key] for key in required)
        if required_bytes > capacity_bytes:
            raise ValueError(
                f"routing-event required working set {required_bytes} bytes "
                f"exceeds byte-cache capacity {capacity_bytes} bytes"
            )


def _validate_workloads(
    workloads: Iterable[ByteCacheWorkload],
) -> tuple[ByteCacheWorkload, ...]:
    sequence = tuple(workloads)
    if not sequence:
        raise ValueError("byte-cache lifecycle simulation requires at least one workload")
    if any(not isinstance(workload, ByteCacheWorkload) for workload in sequence):
        raise TypeError("byte-cache lifecycle simulation requires ByteCacheWorkload objects")
    identities = [workload.workload_id for workload in sequence]
    if len(set(identities)) != len(identities):
        raise ValueError("byte-cache workload IDs must be unique")
    orders = [workload.order for workload in sequence]
    if len(set(orders)) != len(orders):
        raise ValueError("byte-cache workload orders must be unique")
    if orders != sorted(orders):
        raise ValueError(
            "byte-cache workloads must be supplied in ascending declared order"
        )
    return sequence


def _validate_policy_name(policy_name: str) -> None:
    if policy_name not in {"lru", "lfu"}:
        raise ValueError(f"unknown byte-cache policy: {policy_name}")


def _new_policy(
    policy_name: str,
    capacity_bytes: int,
    validated_sizes: Mapping[VersionedExpertKey, int],
) -> _ByteDynamicCache:
    if policy_name == "lru":
        return _ByteLRUCache(capacity_bytes, validated_sizes)
    return _ByteLFUCache(capacity_bytes, validated_sizes)


def _validate_capacity_bytes(capacity_bytes: int) -> int:
    if (
        isinstance(capacity_bytes, bool)
        or not isinstance(capacity_bytes, int)
        or capacity_bytes <= 0
    ):
        raise ValueError("byte-cache capacity must be a positive integer number of bytes")
    return capacity_bytes


def _validate_expert_sizes_bytes(
    expert_sizes_bytes: Mapping[ExpertKey, int],
) -> dict[ExpertKey, int]:
    if not isinstance(expert_sizes_bytes, Mapping):
        raise TypeError("expert_sizes_bytes must be a mapping")

    items = tuple(expert_sizes_bytes.items())
    for key, size_bytes in items:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, int) or part < 0
                for part in key
            )
        ):
            raise ValueError(
                "expert size map keys must be non-negative integer "
                "(layer_id, expert_id) pairs"
            )
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError("expert sizes must be positive integer byte counts")

    return {key: size_bytes for key, size_bytes in sorted(items)}


def _validate_v2_expert_sizes_bytes(
    expert_sizes_bytes: Mapping[VersionedExpertKey, int],
) -> dict[RoutingExpertKeyV2, int]:
    if not isinstance(expert_sizes_bytes, Mapping):
        raise TypeError("expert_sizes_bytes must be a mapping")

    items = tuple(expert_sizes_bytes.items())
    for key, size_bytes in items:
        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or not isinstance(key[0], str)
            or key[0] not in {"encoder", "decoder"}
            or any(
                isinstance(part, bool) or not isinstance(part, int) or part < 0
                for part in key[1:]
            )
        ):
            raise ValueError(
                "v2 expert size map keys must be "
                "(routing_stage, layer_id, expert_id) triples with an encoder "
                "or decoder stage and non-negative integer IDs"
            )
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError("expert sizes must be positive integer byte counts")

    return {key: size_bytes for key, size_bytes in sorted(items)}
