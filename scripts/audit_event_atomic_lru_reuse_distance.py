#!/usr/bin/env python3
"""Reference oracle for count-capacity event-atomic LRU reuse distance.

This script is a v0.9 contract/audit helper, not a public package API.  It derives
recency ranks from canonical routing events without importing cache simulator
state.  Simulator parity is tested separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Iterable, TypeAlias

from moe_cache_lab.trace import ExpertKey, RoutingTrace
from moe_cache_lab.trace_v2 import (
    RoutingExpertKeyV2,
    RoutingTraceV2,
    VersionedRoutingTrace,
    read_versioned_trace,
)

CanonicalKey: TypeAlias = ExpertKey | RoutingExpertKeyV2


@dataclass(frozen=True)
class ReuseSample:
    """One repeated canonical expert request evaluated before its event touch."""

    event_index: int
    previous_event_index: int
    key: CanonicalKey
    stack_distance_entries: int
    required_capacity_entries: int


@dataclass(frozen=True)
class EventAtomicLRUReuseSummary:
    """Exact count-capacity recency summary under the frozen event-atomic rule."""

    event_count: int
    expert_request_count: int
    distinct_referenced_key_count: int
    first_use_count: int
    reuse_count: int
    max_atomic_event_entries: int
    required_capacity_histogram: tuple[tuple[int, int], ...]
    min_required_capacity_entries: int | None
    max_required_capacity_entries: int | None
    first_capacity_with_observed_lru_hit_entries: int | None
    samples: tuple[ReuseSample, ...]

    def predicted_lru_hits(self, capacity_entries: int) -> int:
        """Return exact predicted hits for one feasible count capacity."""

        capacity_entries = _validate_capacity(capacity_entries)
        if capacity_entries < self.max_atomic_event_entries:
            raise ValueError(
                "capacity is smaller than the maximum atomic event entries"
            )
        return sum(
            count
            for required_capacity, count in self.required_capacity_histogram
            if required_capacity <= capacity_entries
        )

    def reuse_fraction_within_capacity(
        self, capacity_entries: int
    ) -> Fraction | None:
        """Return the exact fraction of reuse requests predicted to hit."""

        hits = self.predicted_lru_hits(capacity_entries)
        if not self.reuse_count:
            return None
        return Fraction(hits, self.reuse_count)


def analyze_versioned_trace(
    trace: VersionedRoutingTrace,
) -> EventAtomicLRUReuseSummary:
    """Analyze one already-validated canonical v1 or v2 trace.

    Version 1 uses ``(layer, expert_id)`` identity.  Version 2 uses
    ``(routing_stage, layer, expert_id)`` and projects unassigned events to an
    empty required set.  Event order is consumed exactly as stored; this
    function performs no sorting or chronology repair.
    """

    if isinstance(trace, RoutingTrace):
        required_by_event: tuple[frozenset[CanonicalKey], ...] = tuple(
            frozenset((event.layer, expert_id) for expert_id in event.selected_experts)
            for event in trace.events
        )
    elif isinstance(trace, RoutingTraceV2):
        required_by_event = tuple(
            frozenset(event.expert_requests) for event in trace.events
        )
    else:
        raise TypeError(
            "event-atomic reuse-distance analysis requires RoutingTrace or RoutingTraceV2"
        )
    return analyze_required_sets(required_by_event)


def analyze_required_sets(
    required_by_event: Iterable[Iterable[CanonicalKey]],
) -> EventAtomicLRUReuseSummary:
    """Reference primitive over an already-projected atomic required-set sequence.

    The canonical entry point is :func:`analyze_versioned_trace`.  This helper
    exists so the mathematical empty-sequence edge and hand-computed oracle
    cases can be tested without inventing an invalid empty canonical trace.
    """

    events = tuple(frozenset(required) for required in required_by_event)
    last_touch_event: dict[CanonicalKey, int] = {}
    samples: list[ReuseSample] = []
    first_use_count = 0
    expert_request_count = 0
    max_atomic_event_entries = 0
    sequence_key_shape: int | None = None

    for event_index, required in enumerate(events):
        event_key_shape = _validate_required_set(required)
        if event_key_shape is not None:
            if sequence_key_shape is None:
                sequence_key_shape = event_key_shape
            elif event_key_shape != sequence_key_shape:
                raise TypeError(
                    "one required-set sequence must use one canonical key shape"
                )
        expert_request_count += len(required)
        max_atomic_event_entries = max(max_atomic_event_entries, len(required))

        # Every request in one routing event is evaluated against this same
        # pre-event recency state.  No selected-expert tuple ordering exists.
        for key in sorted(required):
            previous_event_index = last_touch_event.get(key)
            if previous_event_index is None:
                first_use_count += 1
                continue

            priority = (previous_event_index, key)
            stack_distance = sum(
                (other_event_index, other_key) > priority
                for other_key, other_event_index in last_touch_event.items()
                if other_key != key
            )
            samples.append(
                ReuseSample(
                    event_index=event_index,
                    previous_event_index=previous_event_index,
                    key=key,
                    stack_distance_entries=stack_distance,
                    required_capacity_entries=stack_distance + 1,
                )
            )

        for key in required:
            last_touch_event[key] = event_index

    histogram_counter = Counter(
        sample.required_capacity_entries for sample in samples
    )
    histogram = tuple(sorted(histogram_counter.items()))
    required_capacities = tuple(
        sample.required_capacity_entries for sample in samples
    )
    min_required = min(required_capacities) if required_capacities else None
    max_required = max(required_capacities) if required_capacities else None
    first_hit_capacity = (
        max(max_atomic_event_entries, min_required)
        if min_required is not None
        else None
    )

    if first_use_count + len(samples) != expert_request_count:
        raise AssertionError("first-use/reuse accounting does not reconcile")
    if first_use_count != len(last_touch_event):
        raise AssertionError("first-use count does not equal distinct referenced keys")

    return EventAtomicLRUReuseSummary(
        event_count=len(events),
        expert_request_count=expert_request_count,
        distinct_referenced_key_count=len(last_touch_event),
        first_use_count=first_use_count,
        reuse_count=len(samples),
        max_atomic_event_entries=max_atomic_event_entries,
        required_capacity_histogram=histogram,
        min_required_capacity_entries=min_required,
        max_required_capacity_entries=max_required,
        first_capacity_with_observed_lru_hit_entries=first_hit_capacity,
        samples=tuple(samples),
    )


def _validate_required_set(required: frozenset[CanonicalKey]) -> int | None:
    key_shapes = {len(key) for key in required if isinstance(key, tuple)}
    if any(not isinstance(key, tuple) for key in required):
        raise TypeError("canonical cache identities must be tuples")
    if key_shapes.difference({2, 3}) or len(key_shapes) > 1:
        raise TypeError("one required-set sequence must use one canonical key shape")
    for key in required:
        if len(key) == 2:
            layer, expert = key
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (layer, expert)
            ):
                raise ValueError("v1 canonical keys require non-negative integer fields")
        else:
            stage, layer, expert = key
            if stage not in {"encoder", "decoder"}:
                raise ValueError("v2 canonical keys require encoder/decoder stage")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (layer, expert)
            ):
                raise ValueError("v2 canonical keys require non-negative numeric fields")
    return next(iter(key_shapes), None)


def _validate_capacity(capacity_entries: int) -> int:
    if (
        isinstance(capacity_entries, bool)
        or not isinstance(capacity_entries, int)
        or capacity_entries <= 0
    ):
        raise ValueError("capacity_entries must be a positive integer")
    return capacity_entries


def _key_json(key: CanonicalKey) -> list[str | int]:
    return list(key)


def summary_payload(
    summary: EventAtomicLRUReuseSummary,
    *,
    capacity_entries: int | None = None,
) -> dict[str, object]:
    """Return deterministic JSON-compatible audit output."""

    payload: dict[str, object] = {
        "format": "moe-cache-lab.event-atomic-lru-reuse-distance-audit",
        "format_version": 1,
        "event_count": summary.event_count,
        "expert_request_count": summary.expert_request_count,
        "distinct_referenced_key_count": summary.distinct_referenced_key_count,
        "first_use_count": summary.first_use_count,
        "reuse_count": summary.reuse_count,
        "max_atomic_event_entries": summary.max_atomic_event_entries,
        "required_capacity_histogram": [
            {"required_capacity_entries": capacity, "reuse_count": count}
            for capacity, count in summary.required_capacity_histogram
        ],
        "min_required_capacity_entries": summary.min_required_capacity_entries,
        "max_required_capacity_entries": summary.max_required_capacity_entries,
        "first_capacity_with_observed_lru_hit_entries": (
            summary.first_capacity_with_observed_lru_hit_entries
        ),
        "samples": [
            {
                "event_index": sample.event_index,
                "previous_event_index": sample.previous_event_index,
                "key": _key_json(sample.key),
                "stack_distance_entries": sample.stack_distance_entries,
                "required_capacity_entries": sample.required_capacity_entries,
            }
            for sample in summary.samples
        ],
    }
    if capacity_entries is not None:
        hits = summary.predicted_lru_hits(capacity_entries)
        fraction = summary.reuse_fraction_within_capacity(capacity_entries)
        payload["tested_capacity"] = {
            "capacity_entries": capacity_entries,
            "predicted_lru_hits": hits,
            "reuse_fraction_within_capacity": (
                None
                if fraction is None
                else {"numerator": fraction.numerator, "denominator": fraction.denominator}
            ),
        }
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capacity-entries", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    trace = read_versioned_trace(args.trace)
    summary = analyze_versioned_trace(trace)
    print(
        json.dumps(
            summary_payload(summary, capacity_entries=args.capacity_entries),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
