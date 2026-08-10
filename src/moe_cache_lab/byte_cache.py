"""Deterministic offline byte-capacity cache simulation.

All load and eviction byte counts in this module are simulated cache-model
accounting based on caller-supplied expert sizes. They are not measured
host-device/GPU transfers and do not establish latency, throughput, or speedup.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .trace import ExpertKey, RoutingEvent


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
    final_resident_keys: tuple[ExpertKey, ...]

    @property
    def hit_rate(self) -> float:
        """Return the established zero-for-empty request hit-rate behavior."""
        return self.hits / self.expert_request_count if self.expert_request_count else 0.0


class _ByteDynamicCache:
    name: str

    def __init__(
        self,
        capacity_bytes: int,
        expert_sizes_bytes: Mapping[ExpertKey, int],
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.expert_sizes_bytes = expert_sizes_bytes
        self.clock = 0

    @property
    def resident_keys(self) -> frozenset[ExpertKey]:
        raise NotImplementedError

    @property
    def resident_bytes(self) -> int:
        return sum(self.expert_sizes_bytes[key] for key in self.resident_keys)

    def _evict_one(self, candidates: frozenset[ExpertKey]) -> ExpertKey:
        raise NotImplementedError

    def _touch_required(self, required: frozenset[ExpertKey]) -> None:
        raise NotImplementedError

    def access_bundle(
        self,
        required: frozenset[ExpertKey],
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
        expert_sizes_bytes: Mapping[ExpertKey, int],
    ) -> None:
        super().__init__(capacity_bytes, expert_sizes_bytes)
        self.entries: dict[ExpertKey, int] = {}

    @property
    def resident_keys(self) -> frozenset[ExpertKey]:
        return frozenset(self.entries)

    def _evict_one(self, candidates: frozenset[ExpertKey]) -> ExpertKey:
        victim = min(candidates, key=lambda key: (self.entries[key], key))
        del self.entries[victim]
        return victim

    def _touch_required(self, required: frozenset[ExpertKey]) -> None:
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
        expert_sizes_bytes: Mapping[ExpertKey, int],
    ) -> None:
        super().__init__(capacity_bytes, expert_sizes_bytes)
        self.entries: dict[ExpertKey, tuple[int, int]] = {}

    @property
    def resident_keys(self) -> frozenset[ExpertKey]:
        return frozenset(self.entries)

    def _evict_one(self, candidates: frozenset[ExpertKey]) -> ExpertKey:
        victim = min(
            candidates,
            key=lambda key: (self.entries[key][0], self.entries[key][1], key),
        )
        del self.entries[victim]
        return victim

    def _touch_required(self, required: frozenset[ExpertKey]) -> None:
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
    if any(not isinstance(event, RoutingEvent) for event in sequence):
        raise TypeError("byte-cache simulation requires RoutingEvent objects")
    capacity_bytes = _validate_capacity_bytes(capacity_bytes)
    if policy_name not in {"lru", "lfu"}:
        raise ValueError(f"unknown byte-cache policy: {policy_name}")
    validated_sizes = _validate_expert_sizes_bytes(expert_sizes_bytes)

    required_by_event = tuple(
        frozenset((event.layer, expert_id) for expert_id in event.selected_experts)
        for event in sequence
    )
    referenced_keys = frozenset(
        key for required in required_by_event for key in required
    )
    missing_sizes = sorted(referenced_keys.difference(validated_sizes))
    if missing_sizes:
        raise ValueError(
            "expert size map is missing referenced layer-qualified experts: "
            + ", ".join(str(key) for key in missing_sizes)
        )

    for required in required_by_event:
        required_bytes = sum(validated_sizes[key] for key in required)
        if required_bytes > capacity_bytes:
            raise ValueError(
                f"routing-event required working set {required_bytes} bytes "
                f"exceeds byte-cache capacity {capacity_bytes} bytes"
            )

    policy: _ByteDynamicCache
    if policy_name == "lru":
        policy = _ByteLRUCache(capacity_bytes, validated_sizes)
    else:
        policy = _ByteLFUCache(capacity_bytes, validated_sizes)

    hits = 0
    misses = 0
    demand_load_bytes = 0
    eviction_count = 0
    evicted_bytes = 0
    peak_resident_bytes = 0

    for required in required_by_event:
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
        capacity_bytes=capacity_bytes,
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
