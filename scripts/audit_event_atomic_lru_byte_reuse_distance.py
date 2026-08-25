#!/usr/bin/env python3
"""Reference oracle for heterogeneous byte-capacity event-atomic LRU reuse distance.

This v0.9 audit helper derives byte-weighted LRU residency thresholds from
canonical routing events plus caller-supplied positive integer expert sizes. It
does not import cache simulator state and does not establish physical extents,
physical residency or transfer, latency, throughput, or speedup.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

from moe_cache_lab.trace import ExpertKey, RoutingTrace
from moe_cache_lab.trace_v2 import RoutingExpertKeyV2, RoutingTraceV2, VersionedRoutingTrace

CanonicalKey: TypeAlias = ExpertKey | RoutingExpertKeyV2


@dataclass(frozen=True)
class ByteReuseSample:
    """One repeated canonical request evaluated against the pre-event LRU stack."""

    event_index: int
    previous_event_index: int
    key: CanonicalKey
    key_size_bytes: int
    stack_distance_bytes: int
    required_capacity_bytes: int


@dataclass(frozen=True)
class EventAtomicLRUByteReuseSummary:
    """Exact byte-weighted recency summary under the frozen event-atomic rule."""

    event_count: int
    expert_request_count: int
    distinct_referenced_key_count: int
    first_use_count: int
    reuse_count: int
    referenced_working_set_bytes: int
    max_atomic_event_bytes: int
    required_capacity_histogram: tuple[tuple[int, int], ...]
    min_required_capacity_bytes: int | None
    max_required_capacity_bytes: int | None
    first_capacity_with_observed_lru_hit_bytes: int | None
    samples: tuple[ByteReuseSample, ...]

    def predicted_lru_hits(self, capacity_bytes: int) -> int:
        """Return exact predicted LRU hits for one feasible byte capacity."""

        capacity_bytes = _validate_capacity_bytes(capacity_bytes)
        if capacity_bytes < self.max_atomic_event_bytes:
            raise ValueError(
                "capacity is smaller than the maximum atomic event byte working set"
            )
        return sum(
            count
            for required_capacity, count in self.required_capacity_histogram
            if required_capacity <= capacity_bytes
        )

    def reuse_fraction_within_capacity(self, capacity_bytes: int) -> Fraction | None:
        """Return the exact fraction of repeated requests predicted to hit."""

        hits = self.predicted_lru_hits(capacity_bytes)
        if not self.reuse_count:
            return None
        return Fraction(hits, self.reuse_count)


def analyze_versioned_trace(
    trace: VersionedRoutingTrace,
    expert_sizes_bytes: Mapping[CanonicalKey, int],
) -> EventAtomicLRUByteReuseSummary:
    """Analyze one already-valid canonical v1 or v2 trace with supplied sizes.

    Version 1 uses ``(layer, expert_id)`` identity. Version 2 uses
    ``(routing_stage, layer, expert_id)`` and projects unassigned events to an
    empty required set. Event order is consumed exactly as stored; no sorting,
    chronology repair, size inference, or probability/rank reconstruction is
    performed.
    """

    if isinstance(trace, RoutingTrace):
        required_by_event: tuple[frozenset[CanonicalKey], ...] = tuple(
            frozenset((event.layer, expert_id) for expert_id in event.selected_experts)
            for event in trace.events
        )
        expected_key_shape = 2
    elif isinstance(trace, RoutingTraceV2):
        required_by_event = tuple(frozenset(event.expert_requests) for event in trace.events)
        expected_key_shape = 3
    else:
        raise TypeError(
            "byte reuse-distance analysis requires RoutingTrace or RoutingTraceV2"
        )

    return analyze_required_sets(
        required_by_event,
        expert_sizes_bytes,
        expected_key_shape=expected_key_shape,
    )


def analyze_required_sets(
    required_by_event: Iterable[Iterable[CanonicalKey]],
    expert_sizes_bytes: Mapping[CanonicalKey, int],
    *,
    expected_key_shape: int | None = None,
) -> EventAtomicLRUByteReuseSummary:
    """Reference primitive over atomic required sets and a supplied size map.

    ``expected_key_shape`` is used by the canonical v1/v2 entry point. The
    primitive leaves it optional so empty and hand-computed contract cases can
    be tested without inventing an invalid empty canonical trace.
    """

    if expected_key_shape not in {None, 2, 3}:
        raise ValueError("expected_key_shape must be None, 2, or 3")

    events = tuple(frozenset(required) for required in required_by_event)
    sequence_key_shape: int | None = expected_key_shape
    for required in events:
        event_key_shape = _validate_required_set(required)
        if event_key_shape is None:
            continue
        if sequence_key_shape is None:
            sequence_key_shape = event_key_shape
        elif event_key_shape != sequence_key_shape:
            raise TypeError("one required-set sequence must use one canonical key shape")

    validated_sizes = _validate_expert_sizes_bytes(
        expert_sizes_bytes,
        expected_key_shape=sequence_key_shape,
    )
    if sequence_key_shape is None and validated_sizes:
        sequence_key_shape = len(next(iter(validated_sizes)))

    referenced_keys = frozenset(key for required in events for key in required)
    missing_sizes = tuple(sorted(referenced_keys.difference(validated_sizes)))
    if missing_sizes:
        raise ValueError(
            "expert size map is missing referenced canonical keys: "
            + ", ".join(str(key) for key in missing_sizes)
        )

    last_touch_event: dict[CanonicalKey, int] = {}
    samples: list[ByteReuseSample] = []
    first_use_count = 0
    expert_request_count = 0
    max_atomic_event_bytes = 0

    for event_index, required in enumerate(events):
        event_bytes = sum(validated_sizes[key] for key in required)
        max_atomic_event_bytes = max(max_atomic_event_bytes, event_bytes)
        expert_request_count += len(required)

        # Every request in one atomic routing event sees the same pre-event
        # recency state. Stable canonical-key ordering is used only for the
        # already-frozen LRU tie-break, never as invented serial access order.
        for key in sorted(required):
            previous_event_index = last_touch_event.get(key)
            if previous_event_index is None:
                first_use_count += 1
                continue

            priority = (previous_event_index, key)
            stack_distance_bytes = sum(
                validated_sizes[other_key]
                for other_key, other_event_index in last_touch_event.items()
                if other_key != key and (other_event_index, other_key) > priority
            )
            key_size_bytes = validated_sizes[key]
            samples.append(
                ByteReuseSample(
                    event_index=event_index,
                    previous_event_index=previous_event_index,
                    key=key,
                    key_size_bytes=key_size_bytes,
                    stack_distance_bytes=stack_distance_bytes,
                    required_capacity_bytes=stack_distance_bytes + key_size_bytes,
                )
            )

        for key in required:
            last_touch_event[key] = event_index

    histogram_counter = Counter(sample.required_capacity_bytes for sample in samples)
    histogram = tuple(sorted(histogram_counter.items()))
    required_capacities = tuple(sample.required_capacity_bytes for sample in samples)
    min_required = min(required_capacities) if required_capacities else None
    max_required = max(required_capacities) if required_capacities else None
    first_hit_capacity = (
        max(max_atomic_event_bytes, min_required)
        if min_required is not None
        else None
    )
    referenced_working_set_bytes = sum(
        validated_sizes[key] for key in last_touch_event
    )

    if first_use_count + len(samples) != expert_request_count:
        raise AssertionError("first-use/reuse accounting does not reconcile")
    if first_use_count != len(last_touch_event):
        raise AssertionError("first-use count does not equal distinct referenced keys")

    return EventAtomicLRUByteReuseSummary(
        event_count=len(events),
        expert_request_count=expert_request_count,
        distinct_referenced_key_count=len(last_touch_event),
        first_use_count=first_use_count,
        reuse_count=len(samples),
        referenced_working_set_bytes=referenced_working_set_bytes,
        max_atomic_event_bytes=max_atomic_event_bytes,
        required_capacity_histogram=histogram,
        min_required_capacity_bytes=min_required,
        max_required_capacity_bytes=max_required,
        first_capacity_with_observed_lru_hit_bytes=first_hit_capacity,
        samples=tuple(samples),
    )


def _validate_expert_sizes_bytes(
    expert_sizes_bytes: Mapping[CanonicalKey, int],
    *,
    expected_key_shape: int | None,
) -> dict[CanonicalKey, int]:
    if not isinstance(expert_sizes_bytes, Mapping):
        raise TypeError("expert_sizes_bytes must be a mapping")

    items = tuple(expert_sizes_bytes.items())
    observed_shape: int | None = expected_key_shape
    for key, size_bytes in items:
        key_shape = _validate_key(key)
        if observed_shape is None:
            observed_shape = key_shape
        elif key_shape != observed_shape:
            raise TypeError("expert size map must use one canonical key shape")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError("expert sizes must be positive integer byte counts")

    return {key: size_bytes for key, size_bytes in sorted(items)}


def _validate_required_set(required: frozenset[CanonicalKey]) -> int | None:
    key_shapes = set()
    for key in required:
        key_shapes.add(_validate_key(key))
    if len(key_shapes) > 1:
        raise TypeError("one required-set sequence must use one canonical key shape")
    return next(iter(key_shapes), None)


def _validate_key(key: object) -> int:
    if not isinstance(key, tuple):
        raise TypeError("canonical cache identities must be tuples")
    if len(key) == 2:
        layer, expert = key
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (layer, expert)
        ):
            raise ValueError("v1 canonical keys require non-negative integer fields")
        return 2
    if len(key) == 3:
        stage, layer, expert = key
        if stage not in {"encoder", "decoder"}:
            raise ValueError("v2 canonical keys require encoder/decoder stage")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (layer, expert)
        ):
            raise ValueError("v2 canonical keys require non-negative numeric fields")
        return 3
    raise TypeError("canonical cache identities must be v1 pairs or v2 triples")


def _validate_capacity_bytes(capacity_bytes: int) -> int:
    if (
        isinstance(capacity_bytes, bool)
        or not isinstance(capacity_bytes, int)
        or capacity_bytes <= 0
    ):
        raise ValueError("capacity_bytes must be a positive integer")
    return capacity_bytes


def _key_json(key: CanonicalKey) -> list[str | int]:
    return list(key)


def summary_payload(
    summary: EventAtomicLRUByteReuseSummary,
    *,
    capacity_bytes: int | None = None,
) -> dict[str, object]:
    """Return deterministic JSON-compatible audit output."""

    payload: dict[str, object] = {
        "format": "moe-cache-lab.event-atomic-lru-byte-reuse-distance-audit",
        "format_version": 1,
        "size_input_boundary": (
            "caller-supplied cache-model byte sizes; physical extent provenance is not inferred"
        ),
        "event_count": summary.event_count,
        "expert_request_count": summary.expert_request_count,
        "distinct_referenced_key_count": summary.distinct_referenced_key_count,
        "first_use_count": summary.first_use_count,
        "reuse_count": summary.reuse_count,
        "referenced_working_set_bytes": summary.referenced_working_set_bytes,
        "max_atomic_event_bytes": summary.max_atomic_event_bytes,
        "required_capacity_histogram": [
            {"required_capacity_bytes": capacity, "reuse_count": count}
            for capacity, count in summary.required_capacity_histogram
        ],
        "min_required_capacity_bytes": summary.min_required_capacity_bytes,
        "max_required_capacity_bytes": summary.max_required_capacity_bytes,
        "first_capacity_with_observed_lru_hit_bytes": (
            summary.first_capacity_with_observed_lru_hit_bytes
        ),
        "samples": [
            {
                "event_index": sample.event_index,
                "previous_event_index": sample.previous_event_index,
                "key": _key_json(sample.key),
                "key_size_bytes": sample.key_size_bytes,
                "stack_distance_bytes": sample.stack_distance_bytes,
                "required_capacity_bytes": sample.required_capacity_bytes,
            }
            for sample in summary.samples
        ],
    }
    if capacity_bytes is not None:
        hits = summary.predicted_lru_hits(capacity_bytes)
        fraction = summary.reuse_fraction_within_capacity(capacity_bytes)
        payload["tested_capacity"] = {
            "capacity_bytes": capacity_bytes,
            "predicted_lru_hits": hits,
            "reuse_fraction_within_capacity": (
                None
                if fraction is None
                else {"numerator": fraction.numerator, "denominator": fraction.denominator}
            ),
        }
    return payload
