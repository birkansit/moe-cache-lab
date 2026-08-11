"""Deterministic descriptive summaries over simulated cache lifecycle results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from .byte_cache import (
    CACHE_LIFECYCLE_MODES,
    ByteCacheLifecycleSimulation,
    ByteCacheWorkloadResult,
)

_POLICY_ORDER = ("lru", "lfu")


@dataclass(frozen=True, order=True)
class WorkloadIdentity:
    """Stable workload identity retained in descriptive extrema."""

    order: int
    workload_id: str


@dataclass(frozen=True)
class WorkloadMetricRange:
    """Exact observed minimum, maximum, and range over workload rows."""

    metric: str
    minimum: int | Fraction | None
    maximum: int | Fraction | None
    range: int | Fraction | None
    minimum_workloads: tuple[WorkloadIdentity, ...]
    maximum_workloads: tuple[WorkloadIdentity, ...]


@dataclass(frozen=True)
class WorkloadSensitivitySummary:
    """Descriptive workload spread for one capacity/policy/lifecycle cell."""

    lifecycle_mode: str
    policy: str
    capacity_bytes: int
    workload_count: int
    defined_hit_rate_workload_count: int
    hit_rate: WorkloadMetricRange
    misses: WorkloadMetricRange
    simulated_demand_load_bytes: WorkloadMetricRange


@dataclass(frozen=True)
class AdjacentCapacityComparison:
    """Upper-tested-capacity minus lower-tested-capacity deltas."""

    lifecycle_mode: str
    policy: str
    lower_capacity_bytes: int
    upper_capacity_bytes: int
    hit_delta: int
    miss_delta: int
    simulated_demand_load_byte_delta: int
    hit_rate_delta: Fraction | None
    exact_flat: bool


@dataclass(frozen=True)
class SameCapacityPolicyComparison:
    """LFU minus LRU deltas at one tested capacity and lifecycle mode."""

    lifecycle_mode: str
    capacity_bytes: int
    baseline_policy: str
    comparison_policy: str
    hit_delta: int
    miss_delta: int
    simulated_demand_load_byte_delta: int
    hit_rate_delta: Fraction | None


@dataclass(frozen=True)
class LifecycleSensitivitySummary:
    """Canonical descriptive comparisons over a complete lifecycle grid."""

    capacities_bytes: tuple[int, ...]
    policies: tuple[str, ...]
    lifecycle_modes: tuple[str, ...]
    adjacent_capacity_comparisons: tuple[AdjacentCapacityComparison, ...]
    same_capacity_policy_comparisons: tuple[SameCapacityPolicyComparison, ...]
    workload_sensitivity: tuple[WorkloadSensitivitySummary, ...]


def summarize_lifecycle_sensitivity(
    simulations: Iterable[ByteCacheLifecycleSimulation],
) -> LifecycleSensitivitySummary:
    """Build exact descriptive summaries without rerunning cache simulation."""

    rows = tuple(simulations)
    if not rows:
        raise ValueError("lifecycle sensitivity summary requires simulation rows")
    if any(not isinstance(row, ByteCacheLifecycleSimulation) for row in rows):
        raise TypeError(
            "lifecycle sensitivity summary requires ByteCacheLifecycleSimulation rows"
        )

    by_key: dict[tuple[str, str, int], ByteCacheLifecycleSimulation] = {}
    for row in rows:
        if (
            isinstance(row.capacity_bytes, bool)
            or not isinstance(row.capacity_bytes, int)
            or row.capacity_bytes <= 0
        ):
            raise ValueError(
                "lifecycle sensitivity capacities must be positive integers"
            )
        key = row.lifecycle_mode, row.policy, row.capacity_bytes
        if key in by_key:
            raise ValueError(
                "lifecycle sensitivity input contains duplicate mode/policy/capacity rows"
            )
        by_key[key] = row

    unsupported_modes = sorted(
        {row.lifecycle_mode for row in rows}.difference(CACHE_LIFECYCLE_MODES)
    )
    if unsupported_modes:
        raise ValueError(
            "lifecycle sensitivity input contains unsupported lifecycle modes: "
            + ", ".join(unsupported_modes)
        )
    lifecycle_modes = tuple(
        mode for mode in CACHE_LIFECYCLE_MODES
        if any(row.lifecycle_mode == mode for row in rows)
    )
    if lifecycle_modes != CACHE_LIFECYCLE_MODES:
        raise ValueError(
            "lifecycle sensitivity input requires both explicit lifecycle modes"
        )

    unsupported_policies = sorted(
        {row.policy for row in rows}.difference(_POLICY_ORDER)
    )
    if unsupported_policies:
        raise ValueError(
            "lifecycle sensitivity input contains unsupported policies: "
            + ", ".join(unsupported_policies)
        )
    policies = tuple(
        policy for policy in _POLICY_ORDER if any(row.policy == policy for row in rows)
    )
    capacities = tuple(sorted({row.capacity_bytes for row in rows}))
    expected_keys = {
        (mode, policy, capacity)
        for mode in lifecycle_modes
        for policy in policies
        for capacity in capacities
    }
    missing_keys = sorted(expected_keys.difference(by_key))
    if missing_keys:
        raise ValueError(
            "lifecycle sensitivity input is an incomplete rectangular grid: "
            + ", ".join(str(key) for key in missing_keys)
        )
    unexpected_keys = sorted(set(by_key).difference(expected_keys))
    if unexpected_keys:
        raise ValueError(
            "lifecycle sensitivity input contains incompatible grid rows: "
            + ", ".join(str(key) for key in unexpected_keys)
        )

    canonical_rows = tuple(
        by_key[(mode, policy, capacity)]
        for capacity in capacities
        for policy in policies
        for mode in lifecycle_modes
    )
    workload_signature = _validate_rows(canonical_rows)

    adjacent = tuple(
        _adjacent_capacity_comparison(
            by_key[(mode, policy, lower)],
            by_key[(mode, policy, upper)],
        )
        for mode in lifecycle_modes
        for policy in policies
        for lower, upper in zip(capacities, capacities[1:])
    )
    policy_comparisons = (
        tuple(
            _same_capacity_policy_comparison(
                by_key[(mode, "lru", capacity)],
                by_key[(mode, "lfu", capacity)],
            )
            for capacity in capacities
            for mode in lifecycle_modes
        )
        if policies == _POLICY_ORDER
        else ()
    )
    workload_sensitivity = tuple(
        _workload_sensitivity(row, workload_signature)
        for row in canonical_rows
    )
    return LifecycleSensitivitySummary(
        capacities_bytes=capacities,
        policies=policies,
        lifecycle_modes=lifecycle_modes,
        adjacent_capacity_comparisons=adjacent,
        same_capacity_policy_comparisons=policy_comparisons,
        workload_sensitivity=workload_sensitivity,
    )


def _validate_rows(
    rows: tuple[ByteCacheLifecycleSimulation, ...],
) -> tuple[tuple[int, str, int, int], ...]:
    signature: tuple[tuple[int, str, int, int], ...] | None = None
    for row in rows:
        if row.workload_count != len(row.workloads):
            raise ValueError("lifecycle workload_count does not match workload rows")
        if row.expert_request_count != row.hits + row.misses:
            raise ValueError("lifecycle aggregate requests do not equal hits plus misses")
        for field in (
            "event_count",
            "expert_request_count",
            "hits",
            "misses",
            "simulated_demand_load_bytes",
            "eviction_count",
            "simulated_evicted_bytes",
        ):
            if getattr(row, field) != sum(
                getattr(workload, field) for workload in row.workloads
            ):
                raise ValueError(
                    f"lifecycle aggregate {field} does not reconcile with workloads"
                )
        current_signature = tuple(
            (
                workload.order,
                workload.workload_id,
                workload.event_count,
                workload.expert_request_count,
            )
            for workload in row.workloads
        )
        if tuple(item[:2] for item in current_signature) != tuple(
            sorted(item[:2] for item in current_signature)
        ):
            raise ValueError("lifecycle workload rows are not in declared order")
        if len({item[0] for item in current_signature}) != len(current_signature):
            raise ValueError("lifecycle workload orders are not unique")
        if len({item[1] for item in current_signature}) != len(current_signature):
            raise ValueError("lifecycle workload IDs are not unique")
        for workload in row.workloads:
            _validate_workload_parent(row, workload)
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise ValueError(
                "lifecycle comparison rows use incompatible workload identities, "
                "orders, events, or request counts"
            )
    if signature is None:
        raise AssertionError("validated lifecycle grid unexpectedly has no rows")
    return signature


def _validate_workload_parent(
    row: ByteCacheLifecycleSimulation,
    workload: ByteCacheWorkloadResult,
) -> None:
    if (
        workload.lifecycle_mode != row.lifecycle_mode
        or workload.policy != row.policy
        or workload.capacity_bytes != row.capacity_bytes
    ):
        raise ValueError(
            "lifecycle workload row is incompatible with its parent scenario"
        )
    if workload.expert_request_count != workload.hits + workload.misses:
        raise ValueError("lifecycle workload requests do not equal hits plus misses")


def _adjacent_capacity_comparison(
    lower: ByteCacheLifecycleSimulation,
    upper: ByteCacheLifecycleSimulation,
) -> AdjacentCapacityComparison:
    if lower.lifecycle_mode != upper.lifecycle_mode or lower.policy != upper.policy:
        raise ValueError(
            "adjacent capacity comparison cannot mix lifecycle modes or policies"
        )
    if lower.capacity_bytes >= upper.capacity_bytes:
        raise ValueError("adjacent capacity comparison requires ascending capacities")
    hit_delta = upper.hits - lower.hits
    miss_delta = upper.misses - lower.misses
    demand_delta = (
        upper.simulated_demand_load_bytes - lower.simulated_demand_load_bytes
    )
    return AdjacentCapacityComparison(
        lifecycle_mode=lower.lifecycle_mode,
        policy=lower.policy,
        lower_capacity_bytes=lower.capacity_bytes,
        upper_capacity_bytes=upper.capacity_bytes,
        hit_delta=hit_delta,
        miss_delta=miss_delta,
        simulated_demand_load_byte_delta=demand_delta,
        hit_rate_delta=_hit_rate_delta(lower, upper),
        exact_flat=hit_delta == 0 and miss_delta == 0 and demand_delta == 0,
    )


def _same_capacity_policy_comparison(
    lru: ByteCacheLifecycleSimulation,
    lfu: ByteCacheLifecycleSimulation,
) -> SameCapacityPolicyComparison:
    if lru.lifecycle_mode != lfu.lifecycle_mode:
        raise ValueError("same-capacity policy comparison cannot mix lifecycle modes")
    if lru.capacity_bytes != lfu.capacity_bytes:
        raise ValueError("policy comparison requires the same tested capacity")
    if lru.policy != "lru" or lfu.policy != "lfu":
        raise ValueError("policy comparison requires LRU baseline and LFU comparison")
    return SameCapacityPolicyComparison(
        lifecycle_mode=lru.lifecycle_mode,
        capacity_bytes=lru.capacity_bytes,
        baseline_policy="lru",
        comparison_policy="lfu",
        hit_delta=lfu.hits - lru.hits,
        miss_delta=lfu.misses - lru.misses,
        simulated_demand_load_byte_delta=(
            lfu.simulated_demand_load_bytes - lru.simulated_demand_load_bytes
        ),
        hit_rate_delta=_hit_rate_delta(lru, lfu),
    )


def _hit_rate_delta(
    baseline: ByteCacheLifecycleSimulation,
    comparison: ByteCacheLifecycleSimulation,
) -> Fraction | None:
    if not baseline.expert_request_count or not comparison.expert_request_count:
        return None
    return Fraction(comparison.hits, comparison.expert_request_count) - Fraction(
        baseline.hits, baseline.expert_request_count
    )


def _workload_sensitivity(
    row: ByteCacheLifecycleSimulation,
    signature: tuple[tuple[int, str, int, int], ...],
) -> WorkloadSensitivitySummary:
    identities = tuple(
        WorkloadIdentity(order, workload_id)
        for order, workload_id, _, _ in signature
    )
    hit_rate_values = tuple(
        (identity, Fraction(workload.hits, workload.expert_request_count))
        for identity, workload in zip(identities, row.workloads)
        if workload.expert_request_count
    )
    return WorkloadSensitivitySummary(
        lifecycle_mode=row.lifecycle_mode,
        policy=row.policy,
        capacity_bytes=row.capacity_bytes,
        workload_count=len(row.workloads),
        defined_hit_rate_workload_count=len(hit_rate_values),
        hit_rate=_metric_range("hit_rate", hit_rate_values),
        misses=_metric_range(
            "misses",
            tuple(
                (identity, workload.misses)
                for identity, workload in zip(identities, row.workloads)
            ),
        ),
        simulated_demand_load_bytes=_metric_range(
            "simulated_demand_load_bytes",
            tuple(
                (identity, workload.simulated_demand_load_bytes)
                for identity, workload in zip(identities, row.workloads)
            ),
        ),
    )


def _metric_range(
    metric: str,
    values: tuple[tuple[WorkloadIdentity, int | Fraction], ...],
) -> WorkloadMetricRange:
    if not values:
        return WorkloadMetricRange(metric, None, None, None, (), ())
    minimum = min(value for _, value in values)
    maximum = max(value for _, value in values)
    return WorkloadMetricRange(
        metric=metric,
        minimum=minimum,
        maximum=maximum,
        range=maximum - minimum,
        minimum_workloads=tuple(
            identity for identity, value in values if value == minimum
        ),
        maximum_workloads=tuple(
            identity for identity, value in values if value == maximum
        ),
    )
