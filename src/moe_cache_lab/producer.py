"""Core-only normalization boundary for already-built canonical traces.

Canonical JSONL remains the external producer interoperability boundary.  This
module is internal plumbing and does not establish producer semantic validation
or non-interference.
"""

from __future__ import annotations

from dataclasses import dataclass

from .trace import RoutingTrace
from .trace_v2 import RoutingTraceV2


CanonicalRoutingTrace = RoutingTrace | RoutingTraceV2


@dataclass(frozen=True)
class ProducerResult:
    """An immutable reference to one unchanged canonical in-memory trace."""

    trace: CanonicalRoutingTrace

    def __post_init__(self) -> None:
        if not isinstance(self.trace, (RoutingTrace, RoutingTraceV2)):
            raise TypeError("trace must be a canonical RoutingTrace or RoutingTraceV2")


__all__ = ["CanonicalRoutingTrace", "ProducerResult"]
