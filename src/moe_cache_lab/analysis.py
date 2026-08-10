"""Deterministic descriptive locality summaries for validated routing traces.

This module is pure/offline analysis over :class:`RoutingEvent` data.  It does
not simulate cache residency, estimate transfers, or measure runtime behavior.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import log2
from typing import Hashable, Iterable

from .trace import ExpertKey, RoutingEvent, RoutingTrace

_PHASE_ORDER = ("prompt", "generated")


@dataclass(frozen=True)
class PhaseRoutingSummary:
    """Event and expert-assignment counts for one routing phase."""

    phase: str
    event_count: int
    assignment_count: int


@dataclass(frozen=True)
class ExpertFrequency:
    """Selection count for one numeric expert ID within one layer."""

    expert_id: int
    selection_count: int


@dataclass(frozen=True)
class FrequencyConcentrationSummary:
    """Descriptive concentration statistics over positive observed counts.

    ``support_size`` counts only expert IDs with at least one observed
    selection, and ``total_selections`` is the sum of those positive counts.
    ``max_selection_share`` and ``observed_support_gini`` are exact
    :class:`fractions.Fraction` values.  Observed-support Gini uses

    ``sum_i sum_j |x_i - x_j| / (2 * n * sum_i x_i)``

    over the positive observed counts only; experts with zero observed
    selections are intentionally excluded.  ``shannon_entropy_bits`` uses
    ``-sum_i p_i * log2(p_i)`` and is therefore a non-exact floating-point
    value measured in bits.

    Empty helper input is defined as support and total equal to zero with all
    ratio/distribution values ``None``.  One observed expert has maximum share
    ``1``, observed-support Gini ``0``, and entropy ``0.0``.
    """

    support_size: int
    total_selections: int
    max_selection_share: Fraction | None
    observed_support_gini: Fraction | None
    shannon_entropy_bits: float | None


@dataclass(frozen=True)
class LayerRoutingSummary:
    """Deterministic aggregate selection counts for one routed layer."""

    layer_id: int
    event_count: int
    assignment_count: int
    unique_expert_count: int
    expert_frequencies: tuple[ExpertFrequency, ...]
    concentration: FrequencyConcentrationSummary


@dataclass(frozen=True)
class ConsecutiveOverlapSummary:
    """Jaccard overlap across comparable consecutive routing events.

    Events are comparable only with the immediately preceding event in the same
    ``(phase, layer)`` stream.  For expert sets ``A`` and ``B``, pair overlap is
    ``|A ∩ B| / |A ∪ B|``.  ``mean_jaccard`` is the exact arithmetic mean of
    those pair overlaps.  It is ``None`` when no event has a comparable
    predecessor.  Disjoint, identical, and partial pair counts partition all
    comparable pairs.
    """

    comparison_count: int
    disjoint_pair_count: int
    identical_pair_count: int
    partial_pair_count: int
    mean_jaccard: Fraction | None


@dataclass(frozen=True)
class ReuseGapSummary:
    """Event-gap statistics for repeated expert references.

    The identity and event stream are chosen by the caller.  A reuse gap is the
    number of events in that stream strictly between consecutive selections of
    the same identity: ``current_event_index - previous_event_index - 1``.
    First uses produce no gap sample.  Immediate reuse therefore has gap zero.
    ``mean_gap_events`` is exact and is ``None`` when there are no reuses.

    The top-level Issue #1 summary uses layer-qualified ``(layer_id, expert_id)``
    identities over the full validated trace sequence.  Phase-layer summaries
    use numeric expert IDs over only their own ``(phase, layer)`` event stream.
    """

    first_use_count: int
    reuse_count: int
    min_gap_events: int | None
    max_gap_events: int | None
    mean_gap_events: Fraction | None


@dataclass(frozen=True)
class PhaseLayerRoutingSummary:
    """Locality summary for one validated ``(phase, layer_id)`` event stream."""

    phase: str
    layer_id: int
    event_count: int
    assignment_count: int
    unique_expert_count: int
    expert_frequencies: tuple[ExpertFrequency, ...]
    concentration: FrequencyConcentrationSummary
    consecutive_overlap: ConsecutiveOverlapSummary
    reuse_gap: ReuseGapSummary


@dataclass(frozen=True)
class RoutingLocalitySummary:
    """Backend-independent descriptive summary of measured routing selections."""

    total_event_count: int
    total_assignment_count: int
    unique_layer_expert_count: int
    prompt: PhaseRoutingSummary
    generated: PhaseRoutingSummary
    layers: tuple[LayerRoutingSummary, ...]
    consecutive_overlap: ConsecutiveOverlapSummary
    reuse_gap: ReuseGapSummary
    phase_layers: tuple[PhaseLayerRoutingSummary, ...] = ()


def analyze_routing(
    source: RoutingTrace | Iterable[RoutingEvent],
) -> RoutingLocalitySummary:
    """Analyze one validated trace or a raw chronological event iterable.

    ``RoutingTrace`` inputs have already passed the repository's trace
    validation.  Non-empty raw event iterables are passed through
    :class:`RoutingTrace` validation with neutral metadata so analysis never
    sorts, repairs, or silently normalizes invalid chronology.  An empty raw
    iterable is the one supported no-trace edge case and produces all-zero
    global counts plus an empty ``phase_layers`` collection.
    """

    events = _validated_events(source)
    phase_counts: dict[str, list[int]] = {
        "prompt": [0, 0],
        "generated": [0, 0],
    }
    per_layer_events: Counter[int] = Counter()
    per_layer_frequencies: dict[int, Counter[int]] = {}
    events_by_phase_layer: dict[tuple[str, int], list[RoutingEvent]] = {}
    unique_keys: set[ExpertKey] = set()

    for event in events:
        assignments = len(event.selected_experts)
        phase_counts[event.phase][0] += 1
        phase_counts[event.phase][1] += assignments
        per_layer_events[event.layer] += 1
        layer_frequencies = per_layer_frequencies.setdefault(event.layer, Counter())
        stream = events_by_phase_layer.setdefault((event.phase, event.layer), [])
        stream.append(event)

        for expert_id in event.selected_experts:
            key = event.layer, expert_id
            unique_keys.add(key)
            layer_frequencies[expert_id] += 1

    ordered_stream_keys = tuple(
        (phase, layer_id)
        for phase in _PHASE_ORDER
        for layer_id in sorted(
            layer
            for stream_phase, layer in events_by_phase_layer
            if stream_phase == phase
        )
    )
    ordered_streams = tuple(
        tuple(events_by_phase_layer[key]) for key in ordered_stream_keys
    )

    layer_summaries: list[LayerRoutingSummary] = []
    for layer_id in sorted(per_layer_frequencies):
        expert_frequencies = _expert_frequencies(per_layer_frequencies[layer_id])
        layer_summaries.append(
            LayerRoutingSummary(
                layer_id=layer_id,
                event_count=per_layer_events[layer_id],
                assignment_count=sum(per_layer_frequencies[layer_id].values()),
                unique_expert_count=len(per_layer_frequencies[layer_id]),
                expert_frequencies=expert_frequencies,
                concentration=_summarize_frequency_concentration(expert_frequencies),
            )
        )
    layers = tuple(layer_summaries)
    phase_layers = tuple(
        _phase_layer_summary(phase, layer_id, stream)
        for (phase, layer_id), stream in zip(ordered_stream_keys, ordered_streams)
    )

    return RoutingLocalitySummary(
        total_event_count=len(events),
        total_assignment_count=sum(len(event.selected_experts) for event in events),
        unique_layer_expert_count=len(unique_keys),
        prompt=PhaseRoutingSummary("prompt", *phase_counts["prompt"]),
        generated=PhaseRoutingSummary("generated", *phase_counts["generated"]),
        layers=layers,
        consecutive_overlap=_summarize_consecutive_overlap(ordered_streams),
        reuse_gap=_summarize_reuse_gaps(
            tuple(
                tuple((event.layer, expert_id) for expert_id in event.selected_experts)
                for event in events
            )
        ),
        phase_layers=phase_layers,
    )


def _phase_layer_summary(
    phase: str,
    layer_id: int,
    events: tuple[RoutingEvent, ...],
) -> PhaseLayerRoutingSummary:
    frequencies = Counter(
        expert_id for event in events for expert_id in event.selected_experts
    )
    expert_frequencies = _expert_frequencies(frequencies)
    return PhaseLayerRoutingSummary(
        phase=phase,
        layer_id=layer_id,
        event_count=len(events),
        assignment_count=sum(frequencies.values()),
        unique_expert_count=len(frequencies),
        expert_frequencies=expert_frequencies,
        concentration=_summarize_frequency_concentration(expert_frequencies),
        consecutive_overlap=_summarize_consecutive_overlap((events,)),
        reuse_gap=_summarize_reuse_gaps(
            tuple(event.selected_experts for event in events)
        ),
    )


def _expert_frequencies(frequencies: Counter[int]) -> tuple[ExpertFrequency, ...]:
    return tuple(
        ExpertFrequency(expert_id, count)
        for expert_id, count in sorted(frequencies.items())
    )


def _summarize_frequency_concentration(
    frequencies: Iterable[ExpertFrequency],
) -> FrequencyConcentrationSummary:
    """Summarize positive observed frequencies without a configured zero universe.

    Empty input is valid and returns zero support/total with ``None`` for
    maximum share, observed-support Gini, and Shannon entropy.  This helper does
    not add unselected experts with zero counts; doing so would change the Gini
    interpretation and is intentionally outside this analysis.
    """

    frequencies = tuple(frequencies)
    counts = tuple(item.selection_count for item in frequencies)
    if any(count <= 0 for count in counts):
        raise ValueError("frequency concentration requires positive selection counts")
    if not counts:
        return FrequencyConcentrationSummary(0, 0, None, None, None)

    support_size = len(counts)
    total_selections = sum(counts)
    max_selection_share = Fraction(max(counts), total_selections)
    if support_size == 1:
        observed_support_gini = Fraction(0, 1)
        shannon_entropy_bits = 0.0
    else:
        pairwise_absolute_difference = sum(
            abs(left - right) for left in counts for right in counts
        )
        observed_support_gini = Fraction(
            pairwise_absolute_difference,
            2 * support_size * total_selections,
        )
        shannon_entropy_bits = -sum(
            (count / total_selections) * log2(count / total_selections)
            for count in counts
        )

    return FrequencyConcentrationSummary(
        support_size=support_size,
        total_selections=total_selections,
        max_selection_share=max_selection_share,
        observed_support_gini=observed_support_gini,
        shannon_entropy_bits=shannon_entropy_bits,
    )


def _summarize_consecutive_overlap(
    streams: Iterable[Iterable[RoutingEvent]],
) -> ConsecutiveOverlapSummary:
    overlap_values: list[Fraction] = []
    disjoint_pairs = 0
    identical_pairs = 0
    partial_pairs = 0

    for stream in streams:
        previous_set: frozenset[int] | None = None
        for event in stream:
            selected_set = frozenset(event.selected_experts)
            if previous_set is not None:
                intersection_size = len(previous_set.intersection(selected_set))
                union_size = len(previous_set.union(selected_set))
                overlap_values.append(Fraction(intersection_size, union_size))
                if intersection_size == 0:
                    disjoint_pairs += 1
                elif previous_set == selected_set:
                    identical_pairs += 1
                else:
                    partial_pairs += 1
            previous_set = selected_set

    return ConsecutiveOverlapSummary(
        comparison_count=len(overlap_values),
        disjoint_pair_count=disjoint_pairs,
        identical_pair_count=identical_pairs,
        partial_pair_count=partial_pairs,
        mean_jaccard=(
            sum(overlap_values, Fraction()) / len(overlap_values)
            if overlap_values
            else None
        ),
    )


def _summarize_reuse_gaps(
    event_selections: Iterable[Iterable[Hashable]],
) -> ReuseGapSummary:
    last_event_by_key: dict[Hashable, int] = {}
    gaps: list[int] = []

    for event_index, selected_keys in enumerate(event_selections):
        for key in selected_keys:
            previous_event_index = last_event_by_key.get(key)
            if previous_event_index is not None:
                gaps.append(event_index - previous_event_index - 1)
            last_event_by_key[key] = event_index

    return ReuseGapSummary(
        first_use_count=len(last_event_by_key),
        reuse_count=len(gaps),
        min_gap_events=min(gaps) if gaps else None,
        max_gap_events=max(gaps) if gaps else None,
        mean_gap_events=Fraction(sum(gaps), len(gaps)) if gaps else None,
    )


def _validated_events(
    source: RoutingTrace | Iterable[RoutingEvent],
) -> tuple[RoutingEvent, ...]:
    if isinstance(source, RoutingTrace):
        return source.events
    try:
        events = tuple(source)
    except TypeError as exc:
        raise TypeError("routing analysis requires a RoutingTrace or RoutingEvent iterable") from exc
    if any(not isinstance(event, RoutingEvent) for event in events):
        raise TypeError("routing analysis requires RoutingEvent objects")
    if events:
        RoutingTrace(
            model_id="routing-analysis-validation",
            num_experts=None,
            experts_per_token=None,
            events=events,
            created_at="validation-only",
        )
    return events
