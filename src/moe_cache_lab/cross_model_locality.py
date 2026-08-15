"""Version-aware, offline routing-locality diagnostics.

The metrics in this module describe validated routing selections.  They do not
simulate cache state, estimate transfers, score model quality, or imply that a
trace is measured unless its provenance establishes that fact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any

from .trace import RoutingTrace
from .trace_v2 import RoutingTraceV2, VersionedRoutingTrace


LayerIdentity = int | tuple[str, int]


@dataclass(frozen=True)
class CumulativeTopKSelectionShare:
    """Exact cumulative selection share for one caller-requested rank cut."""

    k: int
    expert_ids: tuple[int, ...]
    selection_count: int
    total_selection_count: int
    share: Fraction | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "expert_ids": list(self.expert_ids),
            "selection_count": self.selection_count,
            "total_selection_count": self.total_selection_count,
            "share": _fraction_data(self.share),
        }


@dataclass(frozen=True)
class RoutedLayerLocality:
    """Configured-universe locality metrics for one routed-layer identity."""

    trace_format_version: int
    layer_identity: LayerIdentity
    configured_expert_count: int
    total_selection_count: int
    normalized_shannon_entropy: float | None
    cumulative_top_k_selection_shares: tuple[
        CumulativeTopKSelectionShare, ...
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_format_version": self.trace_format_version,
            "layer_identity": _layer_identity_data(
                self.trace_format_version, self.layer_identity
            ),
            "configured_expert_count": self.configured_expert_count,
            "total_selection_count": self.total_selection_count,
            "normalized_shannon_entropy": self.normalized_shannon_entropy,
            "cumulative_top_k_selection_shares": [
                item.to_dict() for item in self.cumulative_top_k_selection_shares
            ],
        }


@dataclass(frozen=True)
class CrossModelLocalityDiagnostics:
    """Deterministic per-layer diagnostics for one canonical routing trace."""

    trace_format_version: int
    requested_top_k: tuple[int, ...]
    layers: tuple[RoutedLayerLocality, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_format_version": self.trace_format_version,
            "requested_top_k": list(self.requested_top_k),
            "layers": [layer.to_dict() for layer in self.layers],
        }


def analyze_cross_model_locality(
    trace: VersionedRoutingTrace,
    top_k: tuple[int, ...],
) -> CrossModelLocalityDiagnostics:
    """Compute the two Issue-56 metrics without changing trace semantics.

    Version 1 identities are integer layers with one configured expert count.
    Version 2 identities are ``(routing_stage, layer)`` pairs with the matching
    stage profile's configured count.  Unassigned v2 events contribute no
    selection mass.
    """

    requested_top_k = _validate_top_k(top_k)
    if isinstance(trace, RoutingTrace):
        if trace.num_experts is None:
            raise ValueError(
                "version-1 locality diagnostics require configured num_experts"
            )
        if (
            isinstance(trace.num_experts, bool)
            or not isinstance(trace.num_experts, int)
            or trace.num_experts <= 0
        ):
            raise ValueError(
                "version-1 locality diagnostics require positive integer num_experts"
            )
        format_version = 1
        configured_counts: dict[LayerIdentity, int] = {
            layer: trace.num_experts for layer in {event.layer for event in trace.events}
        }
        selections = {
            layer: tuple(
                expert
                for event in trace.events
                if event.layer == layer
                for expert in event.selected_experts
            )
            for layer in configured_counts
        }
        identities: tuple[LayerIdentity, ...] = tuple(sorted(configured_counts))
    elif isinstance(trace, RoutingTraceV2):
        format_version = 2
        stage_counts = {
            profile.routing_stage: profile.num_experts
            for profile in trace.routing_stages
        }
        identities = tuple(
            (stage, layer)
            for stage in stage_counts
            for layer in sorted(
                {
                    event.layer
                    for event in trace.events
                    if event.routing_stage == stage
                }
            )
        )
        configured_counts = {
            identity: stage_counts[identity[0]] for identity in identities
        }
        selections = {
            identity: tuple(
                expert
                for event in trace.events
                if (event.routing_stage, event.layer) == identity
                for expert in event.selected_experts
            )
            for identity in identities
        }
    else:
        raise TypeError(
            "cross-model locality diagnostics require RoutingTrace or RoutingTraceV2"
        )

    for identity in identities:
        configured = configured_counts[identity]
        if any(k > configured for k in requested_top_k):
            raise ValueError(
                f"top_k exceeds configured expert count for routed layer {identity!r}"
            )

    layers = tuple(
        _summarize_layer(
            format_version,
            identity,
            configured_counts[identity],
            selections[identity],
            requested_top_k,
        )
        for identity in identities
    )
    return CrossModelLocalityDiagnostics(format_version, requested_top_k, layers)


def _validate_top_k(top_k: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(top_k, tuple):
        raise TypeError("top_k must be a tuple of positive integers")
    if not top_k:
        raise ValueError("top_k must contain at least one requested rank")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in top_k
    ):
        raise ValueError("top_k must contain positive integers and cannot contain bool")
    if len(set(top_k)) != len(top_k):
        raise ValueError("top_k cannot contain duplicate ranks")
    return top_k


def _summarize_layer(
    format_version: int,
    identity: LayerIdentity,
    configured_expert_count: int,
    selections: tuple[int, ...],
    requested_top_k: tuple[int, ...],
) -> RoutedLayerLocality:
    frequencies = Counter(selections)
    total = len(selections)
    ranked = tuple(
        sorted(
            range(configured_expert_count),
            key=lambda expert_id: (-frequencies[expert_id], expert_id),
        )
    )

    top_k_shares = tuple(
        CumulativeTopKSelectionShare(
            k=k,
            expert_ids=ranked[:k],
            selection_count=sum(frequencies[expert_id] for expert_id in ranked[:k]),
            total_selection_count=total,
            share=(
                Fraction(
                    sum(frequencies[expert_id] for expert_id in ranked[:k]), total
                )
                if total
                else None
            ),
        )
        for k in requested_top_k
    )

    normalized_entropy = None
    if total and configured_expert_count > 1:
        if len(frequencies) == 1:
            normalized_entropy = 0.0
        else:
            probabilities = tuple(count / total for count in frequencies.values())
            entropy_bits = -math.fsum(
                probability * math.log2(probability)
                for probability in probabilities
            )
            normalized_entropy = entropy_bits / math.log2(configured_expert_count)

    return RoutedLayerLocality(
        trace_format_version=format_version,
        layer_identity=identity,
        configured_expert_count=configured_expert_count,
        total_selection_count=total,
        normalized_shannon_entropy=normalized_entropy,
        cumulative_top_k_selection_shares=top_k_shares,
    )


def _fraction_data(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _layer_identity_data(
    trace_format_version: int, identity: LayerIdentity
) -> dict[str, int | str | None]:
    if trace_format_version == 1:
        return {"routing_stage": None, "layer": identity}
    routing_stage, layer = identity
    return {"routing_stage": routing_stage, "layer": layer}
