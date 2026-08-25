"""Exact trace-specific event-atomic LRU capacity frontiers.

The derivation is independent of cache simulator state.  It consumes one
already-validated canonical trace in supplied chronology and applies the
project's frozen event-atomic LRU/stable-key contract.  Returned cache outcomes
are exact under that **SIMULATED** abstraction; they are not runtime or physical
residency measurements and do not recommend a capacity.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

from .trace import ExpertKey, RoutingTrace
from .trace_v2 import RoutingExpertKeyV2, RoutingTraceV2, VersionedRoutingTrace


CanonicalKey: TypeAlias = ExpertKey | RoutingExpertKeyV2
TARGET_HIT_FRACTIONS = (
    ("25%", Fraction(1, 4)),
    ("50%", Fraction(1, 2)),
    ("75%", Fraction(3, 4)),
    ("90%", Fraction(9, 10)),
)


class CapacityFrontierInputError(ValueError):
    """Expected caller-supplied byte-size input failure."""


@dataclass(frozen=True)
class CapacityFrontierBreakpoint:
    """One exact hit-changing capacity point in the feasible domain."""

    capacity: int
    hits: int
    misses: int
    hit_fraction: Fraction | None
    miss_fraction: Fraction | None


@dataclass(frozen=True)
class CapacityFrontierTarget:
    """Smallest feasible capacity for one fixed descriptive checkpoint."""

    label: str
    target_fraction: Fraction
    required_hits: int | None
    status: str
    unattainable: bool
    capacity: int | None
    capacity_fraction_of_referenced_working_set: Fraction | None


@dataclass(frozen=True)
class CapacityFrontierMode:
    """Exact count- or byte-capacity frontier derived from one histogram."""

    unit: str
    referenced_working_set: int
    feasibility_floor: int
    required_capacity_histogram: tuple[tuple[int, int], ...]
    breakpoints: tuple[CapacityFrontierBreakpoint, ...]
    first_feasible_capacity_with_observed_hit: int | None
    targets: tuple[CapacityFrontierTarget, ...]

    def exact_hits_at(self, capacity: int) -> int:
        """Return exact LRU hits at one feasible capacity without simulation."""

        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("capacity must be an integer")
        if capacity < self.feasibility_floor:
            raise ValueError("capacity is below the atomic-event feasibility floor")
        return sum(
            count
            for required_capacity, count in self.required_capacity_histogram
            if required_capacity <= capacity
        )


@dataclass(frozen=True)
class CapacityFrontierAnalysis:
    """Complete exact count frontier and optional heterogeneous-byte frontier."""

    trace_format_version: int
    model_id: str
    model_revision: str | None
    capture_method: str
    event_count: int
    expert_request_count: int
    first_use_count: int
    reuse_count: int
    compulsory_first_use_miss_fraction: Fraction | None
    maximum_reachable_cold_start_hit_fraction: Fraction | None
    count_frontier: CapacityFrontierMode
    byte_frontier: CapacityFrontierMode | None


@dataclass(frozen=True)
class _ModeDerivation:
    mode: CapacityFrontierMode
    event_count: int
    expert_request_count: int
    first_use_count: int
    reuse_count: int


def analyze_capacity_frontier(
    trace: VersionedRoutingTrace,
    expert_sizes_bytes: Mapping[CanonicalKey, int] | None = None,
) -> CapacityFrontierAnalysis:
    """Derive exact count and optional byte LRU frontiers for one trace."""

    required_by_event, trace_version = _required_sets(trace)
    count = _derive_mode(required_by_event, unit="entries")
    byte = None
    if expert_sizes_bytes is not None:
        sizes = _validate_expert_sizes(
            expert_sizes_bytes,
            expected_key_shape=2 if trace_version == 1 else 3,
        )
        referenced = frozenset(key for event in required_by_event for key in event)
        missing = tuple(sorted(referenced.difference(sizes)))
        if missing:
            raise CapacityFrontierInputError(
                "expert size map is missing referenced canonical keys: "
                + ", ".join(str(key) for key in missing)
            )
        byte = _derive_mode(required_by_event, unit="bytes", weights=sizes)
        for attribute in (
            "event_count",
            "expert_request_count",
            "first_use_count",
            "reuse_count",
        ):
            if getattr(byte, attribute) != getattr(count, attribute):
                raise AssertionError(
                    "count/byte frontier request accounting does not reconcile"
                )

    requests = count.expert_request_count
    compulsory = None if requests == 0 else Fraction(count.first_use_count, requests)
    maximum_hit = None if requests == 0 else Fraction(count.reuse_count, requests)
    return CapacityFrontierAnalysis(
        trace_format_version=trace_version,
        model_id=trace.model_id,
        model_revision=trace.model_revision,
        capture_method=trace.capture_method,
        event_count=count.event_count,
        expert_request_count=requests,
        first_use_count=count.first_use_count,
        reuse_count=count.reuse_count,
        compulsory_first_use_miss_fraction=compulsory,
        maximum_reachable_cold_start_hit_fraction=maximum_hit,
        count_frontier=count.mode,
        byte_frontier=None if byte is None else byte.mode,
    )


def _required_sets(
    trace: VersionedRoutingTrace,
) -> tuple[tuple[frozenset[CanonicalKey], ...], int]:
    if isinstance(trace, RoutingTrace):
        return (
            tuple(
                frozenset(
                    (event.layer, expert_id)
                    for expert_id in event.selected_experts
                )
                for event in trace.events
            ),
            1,
        )
    if isinstance(trace, RoutingTraceV2):
        return (
            tuple(frozenset(event.expert_requests) for event in trace.events),
            2,
        )
    raise TypeError(
        "capacity-frontier analysis requires RoutingTrace or RoutingTraceV2"
    )


def _derive_mode(
    required_by_event: tuple[frozenset[CanonicalKey], ...],
    *,
    unit: str,
    weights: Mapping[CanonicalKey, int] | None = None,
) -> _ModeDerivation:
    last_touch_event: dict[CanonicalKey, int] = {}
    threshold_counts: Counter[int] = Counter()
    expert_request_count = 0
    first_use_count = 0
    feasibility_floor = 0

    def weight(key: CanonicalKey) -> int:
        return 1 if weights is None else weights[key]

    for event_index, required in enumerate(required_by_event):
        expert_request_count += len(required)
        feasibility_floor = max(
            feasibility_floor,
            sum(weight(key) for key in required),
        )

        # Each request observes the same pre-event recency state.  Sorting is
        # solely the frozen stable-key tie-break, never serial access order.
        for key in sorted(required):
            previous_event_index = last_touch_event.get(key)
            if previous_event_index is None:
                first_use_count += 1
                continue
            priority = (previous_event_index, key)
            stack_distance = sum(
                weight(other_key)
                for other_key, other_event_index in last_touch_event.items()
                if other_key != key and (other_event_index, other_key) > priority
            )
            threshold_counts[stack_distance + weight(key)] += 1

        for key in required:
            last_touch_event[key] = event_index

    histogram = tuple(sorted(threshold_counts.items()))
    reuse_count = sum(threshold_counts.values())
    referenced_working_set = sum(weight(key) for key in last_touch_event)
    if first_use_count + reuse_count != expert_request_count:
        raise AssertionError("first-use/reuse accounting does not reconcile")
    if first_use_count != len(last_touch_event):
        raise AssertionError("first-use count does not equal referenced key count")

    breakpoints = _derive_breakpoints(
        histogram,
        feasibility_floor=feasibility_floor,
        expert_request_count=expert_request_count,
    )
    targets = _derive_targets(
        breakpoints,
        expert_request_count=expert_request_count,
        reuse_count=reuse_count,
        referenced_working_set=referenced_working_set,
    )
    first_hit = (
        None
        if reuse_count == 0
        else max(feasibility_floor, histogram[0][0])
    )
    return _ModeDerivation(
        mode=CapacityFrontierMode(
            unit=unit,
            referenced_working_set=referenced_working_set,
            feasibility_floor=feasibility_floor,
            required_capacity_histogram=histogram,
            breakpoints=breakpoints,
            first_feasible_capacity_with_observed_hit=first_hit,
            targets=targets,
        ),
        event_count=len(required_by_event),
        expert_request_count=expert_request_count,
        first_use_count=first_use_count,
        reuse_count=reuse_count,
    )


def _derive_breakpoints(
    histogram: tuple[tuple[int, int], ...],
    *,
    feasibility_floor: int,
    expert_request_count: int,
) -> tuple[CapacityFrontierBreakpoint, ...]:
    hits = sum(count for capacity, count in histogram if capacity <= feasibility_floor)
    points = [
        _breakpoint(feasibility_floor, hits, expert_request_count)
    ]
    for capacity, count in histogram:
        if capacity <= feasibility_floor:
            continue
        hits += count
        points.append(_breakpoint(capacity, hits, expert_request_count))
    return tuple(points)


def _breakpoint(
    capacity: int,
    hits: int,
    expert_request_count: int,
) -> CapacityFrontierBreakpoint:
    misses = expert_request_count - hits
    return CapacityFrontierBreakpoint(
        capacity=capacity,
        hits=hits,
        misses=misses,
        hit_fraction=(
            None
            if expert_request_count == 0
            else Fraction(hits, expert_request_count)
        ),
        miss_fraction=(
            None
            if expert_request_count == 0
            else Fraction(misses, expert_request_count)
        ),
    )


def _derive_targets(
    breakpoints: tuple[CapacityFrontierBreakpoint, ...],
    *,
    expert_request_count: int,
    reuse_count: int,
    referenced_working_set: int,
) -> tuple[CapacityFrontierTarget, ...]:
    targets = []
    for label, target in TARGET_HIT_FRACTIONS:
        if expert_request_count == 0:
            targets.append(
                CapacityFrontierTarget(
                    label=label,
                    target_fraction=target,
                    required_hits=None,
                    status="unavailable",
                    unattainable=False,
                    capacity=None,
                    capacity_fraction_of_referenced_working_set=None,
                )
            )
            continue

        required_hits = (
            target.numerator * expert_request_count + target.denominator - 1
        ) // target.denominator
        if required_hits > reuse_count:
            targets.append(
                CapacityFrontierTarget(
                    label=label,
                    target_fraction=target,
                    required_hits=required_hits,
                    status="unattainable",
                    unattainable=True,
                    capacity=None,
                    capacity_fraction_of_referenced_working_set=None,
                )
            )
            continue

        capacity = next(
            point.capacity for point in breakpoints if point.hits >= required_hits
        )
        targets.append(
            CapacityFrontierTarget(
                label=label,
                target_fraction=target,
                required_hits=required_hits,
                status="reached",
                unattainable=False,
                capacity=capacity,
                capacity_fraction_of_referenced_working_set=Fraction(
                    capacity, referenced_working_set
                ),
            )
        )
    return tuple(targets)


def _validate_expert_sizes(
    expert_sizes_bytes: Mapping[CanonicalKey, int],
    *,
    expected_key_shape: int,
) -> dict[CanonicalKey, int]:
    if not isinstance(expert_sizes_bytes, Mapping):
        raise TypeError("expert_sizes_bytes must be a mapping")
    validated: dict[CanonicalKey, int] = {}
    for key, size in expert_sizes_bytes.items():
        if not isinstance(key, tuple) or len(key) != expected_key_shape:
            raise CapacityFrontierInputError(
                "expert size keys must match the trace's canonical identity shape"
            )
        if expected_key_shape == 3 and key[0] not in {"encoder", "decoder"}:
            raise CapacityFrontierInputError(
                "stage-qualified expert sizes require encoder/decoder identity"
            )
        numeric_fields = key if expected_key_shape == 2 else key[1:]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in numeric_fields
        ):
            raise CapacityFrontierInputError(
                "expert size keys require non-negative integer layer/expert fields"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CapacityFrontierInputError(
                "expert sizes must be positive integer byte values"
            )
        validated[key] = size
    return validated


__all__ = [
    "CapacityFrontierAnalysis",
    "CapacityFrontierBreakpoint",
    "CapacityFrontierInputError",
    "CapacityFrontierMode",
    "CapacityFrontierTarget",
    "TARGET_HIT_FRACTIONS",
    "analyze_capacity_frontier",
]
