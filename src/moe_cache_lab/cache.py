"""Atomic, offline cache-policy simulation over measured routing events."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .trace import ExpertKey, RoutingEvent
from .views import SimulationBundle, adapt_events, summarize_bundles


@dataclass(frozen=True)
class CacheResult:
    """Counters for one scope of a shared chronological simulation."""

    scope: str
    samples: int
    requests: int
    hits: int
    misses: int
    prewarm_loads: int
    demand_loads: int
    evictions: int
    estimated_expert_transfers: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0


@dataclass(frozen=True)
class CacheSimulation:
    """Combined and phase-attributed views of one shared-cache replay."""

    policy: str
    capacity: int
    combined: CacheResult
    prompt: CacheResult
    generated: CacheResult
    fixed_entries: frozenset[ExpertKey] = frozenset()

    def rows(self) -> tuple[CacheResult, CacheResult, CacheResult]:
        return self.combined, self.prompt, self.generated


@dataclass(frozen=True)
class FixedTargetPlan:
    """Immutable, auditable fixed targets selected once for later replay."""

    policy: str
    capacity: int
    selection_bundles: tuple[SimulationBundle, ...]
    opposite_bundles: tuple[SimulationBundle, ...]
    view_name: str = field(init=False)
    selection_scope: str = field(init=False)
    target_entries: frozenset[ExpertKey] = field(init=False)
    selected_prompt_provenance: frozenset[tuple[int, str]] = field(init=False)
    opposite_prompt_provenance: frozenset[tuple[int, str]] = field(init=False)

    def __post_init__(self) -> None:
        selection_scope = {
            "calibrated_static_frequency": "calibration",
            "offline_oracle_frequency": "full_evaluation",
        }.get(self.policy)
        if selection_scope is None:
            raise ValueError("fixed-target plan policy/selection scope is invalid")
        _validate_capacity(self.capacity)
        if (
            not isinstance(self.selection_bundles, tuple)
            or not self.selection_bundles
            or not isinstance(self.opposite_bundles, tuple)
            or not self.opposite_bundles
        ):
            raise ValueError("fixed-target plans require nonempty immutable selected and opposite bundles")
        selected_summary = summarize_bundles(self.selection_bundles)
        opposite_summary = summarize_bundles(self.opposite_bundles)
        if selected_summary.view_name != opposite_summary.view_name:
            raise ValueError("fixed-target selected and opposite splits must use the same view")
        _reject_prompt_provenance_overlap(
            self.selection_bundles, self.opposite_bundles
        )
        object.__setattr__(self, "view_name", selected_summary.view_name)
        object.__setattr__(self, "selection_scope", selection_scope)
        object.__setattr__(
            self,
            "target_entries",
            frequency_targets(self.selection_bundles, self.capacity),
        )
        object.__setattr__(
            self,
            "selected_prompt_provenance",
            _prompt_provenance(self.selection_bundles),
        )
        object.__setattr__(
            self,
            "opposite_prompt_provenance",
            _prompt_provenance(self.opposite_bundles),
        )


@dataclass
class _MutableCounters:
    samples: int = 0
    requests: int = 0
    hits: int = 0
    misses: int = 0
    prewarm_loads: int = 0
    demand_loads: int = 0
    evictions: int = 0

    def freeze(self, scope: str) -> CacheResult:
        return CacheResult(
            scope=scope,
            samples=self.samples,
            requests=self.requests,
            hits=self.hits,
            misses=self.misses,
            prewarm_loads=self.prewarm_loads,
            demand_loads=self.demand_loads,
            evictions=self.evictions,
            estimated_expert_transfers=self.prewarm_loads + self.demand_loads,
        )


class _DynamicCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.clock = 0

    def access_bundle(self, required: frozenset[ExpertKey]) -> tuple[int, int, int, int]:
        raise NotImplementedError


class LRUCache(_DynamicCache):
    """Event-atomic LRU with stable layer/expert tie-breaking."""

    name = "lru"

    def __init__(self, capacity: int) -> None:
        super().__init__(_validate_capacity(capacity))
        self.entries: dict[ExpertKey, int] = {}

    def access_bundle(self, required: frozenset[ExpertKey]) -> tuple[int, int, int, int]:
        self.clock += 1
        hits = len(required.intersection(self.entries))
        misses = len(required) - hits
        evictions = 0
        while len(self.entries) + misses > self.capacity:
            candidates = self.entries.keys() - required
            victim = min(candidates, key=lambda key: (self.entries[key], key))
            del self.entries[victim]
            evictions += 1
        # One routing event is atomic: every required key receives the same
        # timestamp, and tuple ordering therefore cannot affect future victims.
        for key in required:
            self.entries[key] = self.clock
        return hits, misses, misses, evictions


class LFUCache(_DynamicCache):
    """Online resident-frequency LFU with stable oldest/key tie-breaking."""

    name = "lfu"

    def __init__(self, capacity: int) -> None:
        super().__init__(_validate_capacity(capacity))
        self.entries: dict[ExpertKey, tuple[int, int]] = {}

    def access_bundle(self, required: frozenset[ExpertKey]) -> tuple[int, int, int, int]:
        self.clock += 1
        hits = len(required.intersection(self.entries))
        misses = len(required) - hits
        evictions = 0
        while len(self.entries) + misses > self.capacity:
            candidates = self.entries.keys() - required
            victim = min(
                candidates,
                key=lambda key: (self.entries[key][0], self.entries[key][1], key),
            )
            del self.entries[victim]
            evictions += 1
        for key in required:
            if key in self.entries:
                frequency, _ = self.entries[key]
                self.entries[key] = (frequency + 1, self.clock)
            else:
                # Frequencies describe only the current residency lifetime.
                self.entries[key] = (1, self.clock)
        return hits, misses, misses, evictions


class FixedFrequencyCache:
    """Fixed frequency target set restored after every atomic event."""

    def __init__(
        self,
        capacity: int,
        selection_bundles: tuple[SimulationBundle, ...],
        name: str,
        target_entries: frozenset[ExpertKey] | None = None,
    ) -> None:
        self.name = name
        self.capacity = _validate_capacity(capacity)
        self.target_entries = (
            frequency_targets(selection_bundles, capacity)
            if target_entries is None else target_entries
        )
        if len(self.target_entries) > self.capacity:
            raise ValueError("fixed target entries exceed cache capacity")
        self.entries = set(self.target_entries)

    def access_bundle(self, required: frozenset[ExpertKey]) -> tuple[int, int, int, int]:
        hits = len(required.intersection(self.entries))
        missing = required.difference(self.entries)
        # Preserve the hard capacity limit during execution. When the fixed
        # target fills capacity, each miss displaces one stable non-required
        # target resident before the complete required bundle is loaded.
        displacement_count = max(0, len(self.entries) + len(missing) - self.capacity)
        displaced = sorted(self.entries.difference(required))[:displacement_count]
        for key in displaced:
            self.entries.remove(key)
        self.entries.update(missing)
        if len(self.entries) > self.capacity:
            raise AssertionError("fixed-frequency cache exceeded capacity")

        # The event has executed. Temporary requirements leave residency and
        # every displaced fixed target is loaded again, restoring the invariant.
        self.entries.difference_update(missing)
        self.entries.update(displaced)
        if self.entries != set(self.target_entries):
            raise AssertionError("fixed-frequency cache failed to restore its target set")
        misses = len(missing)
        demand_loads = misses + len(displaced)
        evictions = misses + len(displaced)
        return hits, misses, demand_loads, evictions


class OfflineOracleFrequencyCache(FixedFrequencyCache):
    """Non-causal fixed targets selected from the full evaluation trace."""

    name = "offline_oracle_frequency"

    def __init__(self, capacity: int, bundles: tuple[SimulationBundle, ...]) -> None:
        super().__init__(capacity, bundles, self.name)


def simulate(
    events: Iterable[RoutingEvent],
    capacity: int,
    policy_name: str,
    *,
    calibration_events: Iterable[RoutingEvent] | None = None,
) -> CacheSimulation:
    """Replay event bundles once, preserving one cache across both phases.

    Hits are determined from residency at event start. Required keys are pinned
    while enough non-required residents are evicted for all misses, after which
    the entire required bundle is loaded/touched with one logical timestamp.
    """

    sequence = tuple(events)
    if any(not isinstance(event, RoutingEvent) for event in sequence):
        raise TypeError("cache simulation requires RoutingEvent objects")
    capacity = _validate_capacity(capacity)
    largest_bundle = max((_bundle_size(event) for event in sequence), default=0)
    if capacity < largest_bundle:
        raise ValueError(
            f"cache capacity {capacity} is smaller than the largest routing-event bundle {largest_bundle}"
        )
    calibration_bundle_sequence: tuple[SimulationBundle, ...] | None = None
    if policy_name == "calibrated_static_frequency":
        if calibration_events is None:
            raise ValueError("calibrated_static_frequency requires disjoint calibration events")
        calibration_sequence = tuple(calibration_events)
        if any(not isinstance(event, RoutingEvent) for event in calibration_sequence):
            raise TypeError("calibration data must contain RoutingEvent objects")
        calibration_bundle_sequence = adapt_events(
            calibration_sequence,
            prompt_id="routing-event-compatibility-calibration",
            prompt_order=1,
        )
    return _replay_bundles(
        adapt_events(
            sequence,
            prompt_id="routing-event-compatibility-evaluation",
            prompt_order=0,
        ),
        capacity,
        policy_name,
        calibration_bundles=calibration_bundle_sequence,
        oracle_bundles=None,
        validate_chronology=False,
    )


def simulate_bundles(
    bundles: Iterable[SimulationBundle],
    capacity: int,
    policy_name: str,
    *,
    calibration_bundles: Iterable[SimulationBundle] | None = None,
    oracle_bundles: Iterable[SimulationBundle] | None = None,
    fixed_target_plan: FixedTargetPlan | None = None,
) -> CacheSimulation:
    """Replay immutable simulation bundles with one cache across both phases.

    Fixed-frequency target counts contain one occurrence for each unique
    layer-qualified key required by each selection bundle.  Capacity remains a
    hard bound on simultaneous resident objects, including grouped bundles. An
    empty evaluation sequence adopts a nonempty calibration sequence's view so
    all-EOS controls can still charge fixed prewarm without simulated accesses.
    """

    return _replay_bundles(
        bundles,
        capacity,
        policy_name,
        calibration_bundles=calibration_bundles,
        oracle_bundles=oracle_bundles,
        fixed_target_plan=fixed_target_plan,
        validate_chronology=True,
    )


def _replay_bundles(
    bundles: Iterable[SimulationBundle],
    capacity: int,
    policy_name: str,
    *,
    calibration_bundles: Iterable[SimulationBundle] | None,
    validate_chronology: bool,
    oracle_bundles: Iterable[SimulationBundle] | None = None,
    fixed_target_plan: FixedTargetPlan | None = None,
) -> CacheSimulation:
    sequence = tuple(bundles)
    if any(not isinstance(bundle, SimulationBundle) for bundle in sequence):
        raise TypeError("bundle simulation requires SimulationBundle objects")
    capacity = _validate_capacity(capacity)
    evaluation_view = _bundle_view(sequence)
    if validate_chronology:
        evaluation_summary = summarize_bundles(sequence)
        evaluation_summary.require_feasible(capacity)
    else:
        largest_bundle = max((len(bundle.required_keys) for bundle in sequence), default=0)
        if capacity < largest_bundle:
            raise ValueError(
                f"cache capacity {capacity} is smaller than the largest "
                f"{evaluation_view or 'unknown'} bundle {largest_bundle}"
            )
    if fixed_target_plan is not None and policy_name in {"lru", "lfu"}:
        raise ValueError("dynamic policies cannot use a fixed-target plan")
    if policy_name == "lru":
        policy: _DynamicCache | FixedFrequencyCache = LRUCache(capacity)
    elif policy_name == "lfu":
        policy = LFUCache(capacity)
    elif policy_name == "offline_oracle_frequency":
        if fixed_target_plan is not None:
            _validate_fixed_plan(fixed_target_plan, policy_name, capacity, evaluation_view)
            _require_plan_replay_subsequence(
                sequence, fixed_target_plan.selection_bundles
            )
            policy = FixedFrequencyCache(
                capacity, (), policy_name, fixed_target_plan.target_entries
            )
        else:
            oracle_sequence = sequence if oracle_bundles is None else tuple(oracle_bundles)
            if any(not isinstance(bundle, SimulationBundle) for bundle in oracle_sequence):
                raise TypeError("oracle selection data must contain SimulationBundle objects")
            oracle_view = _bundle_view(oracle_sequence)
            if validate_chronology:
                summarize_bundles(oracle_sequence)
            if oracle_bundles is not None and oracle_sequence != sequence:
                raise ValueError(
                    "raw oracle target selection must exactly equal the replay bundle sequence"
                )
            if evaluation_view is not None and oracle_view != evaluation_view:
                raise ValueError("oracle selection and evaluation bundles must use the same simulation view")
            if evaluation_view is None:
                evaluation_view = oracle_view
            policy = OfflineOracleFrequencyCache(capacity, oracle_sequence)
    elif policy_name == "calibrated_static_frequency":
        if fixed_target_plan is not None:
            _validate_fixed_plan(fixed_target_plan, policy_name, capacity, evaluation_view)
            _require_plan_replay_subsequence(
                sequence, fixed_target_plan.opposite_bundles
            )
            policy = FixedFrequencyCache(
                capacity, (), policy_name, fixed_target_plan.target_entries
            )
        elif calibration_bundles is None:
            raise ValueError("calibrated_static_frequency requires disjoint calibration bundles")
        else:
            calibration_sequence = tuple(calibration_bundles)
            if any(not isinstance(bundle, SimulationBundle) for bundle in calibration_sequence):
                raise TypeError("calibration data must contain SimulationBundle objects")
            if calibration_sequence:
                calibration_view = _bundle_view(calibration_sequence)
                if validate_chronology:
                    summarize_bundles(calibration_sequence)
                if evaluation_view is not None and calibration_view != evaluation_view:
                    raise ValueError("calibration and evaluation bundles must use the same simulation view")
                if evaluation_view is None:
                    evaluation_view = calibration_view
                _reject_prompt_provenance_overlap(sequence, calibration_sequence)
            policy = FixedFrequencyCache(capacity, calibration_sequence, policy_name)
    else:
        raise ValueError(f"unknown policy: {policy_name}")

    combined = _MutableCounters()
    phases = {"prompt": _MutableCounters(), "generated": _MutableCounters()}
    if isinstance(policy, FixedFrequencyCache):
        # Prewarm is a one-time, phase-independent setup cost. Phase rows show
        # only outcomes attributable to events; the combined row includes it.
        combined.prewarm_loads = len(policy.target_entries)

    for bundle in sequence:
        required = bundle.required_keys
        hits, misses, demand_loads, evictions = policy.access_bundle(required)
        for counters in (combined, phases[bundle.phase]):
            counters.samples += 1
            counters.requests += len(required)
            counters.hits += hits
            counters.misses += misses
            counters.demand_loads += demand_loads
            counters.evictions += evictions

    return CacheSimulation(
        policy=policy.name,
        capacity=capacity,
        combined=combined.freeze("combined"),
        prompt=phases["prompt"].freeze("prompt/prefill"),
        generated=phases["generated"].freeze("generated/decode"),
        fixed_entries=policy.target_entries if isinstance(policy, FixedFrequencyCache) else frozenset(),
    )


def _bundle_view(bundles: tuple[SimulationBundle, ...]) -> str | None:
    views = {bundle.view_name for bundle in bundles}
    if len(views) > 1:
        raise ValueError("cache replay cannot mix simulation views")
    return next(iter(views), None)


def _reject_prompt_provenance_overlap(
    evaluation: tuple[SimulationBundle, ...],
    calibration: tuple[SimulationBundle, ...],
) -> None:
    evaluation_pairs = {(bundle.prompt_order, bundle.prompt_id) for bundle in evaluation}
    calibration_pairs = {(bundle.prompt_order, bundle.prompt_id) for bundle in calibration}
    evaluation_orders = {order for order, _ in evaluation_pairs}
    calibration_orders = {order for order, _ in calibration_pairs}
    evaluation_ids = {prompt_id for _, prompt_id in evaluation_pairs}
    calibration_ids = {prompt_id for _, prompt_id in calibration_pairs}
    if (
        evaluation_pairs.intersection(calibration_pairs)
        or evaluation_orders.intersection(calibration_orders)
        or evaluation_ids.intersection(calibration_ids)
    ):
        raise ValueError("calibration and evaluation prompt provenance must be disjoint")


def _validate_fixed_plan(
    plan: FixedTargetPlan,
    policy_name: str,
    capacity: int,
    evaluation_view: str | None,
) -> None:
    if not isinstance(plan, FixedTargetPlan):
        raise TypeError("fixed_target_plan must be a FixedTargetPlan")
    if plan.policy != policy_name or plan.capacity != capacity:
        raise ValueError("fixed-target plan policy/capacity mismatch")
    if evaluation_view is not None and plan.view_name != evaluation_view:
        raise ValueError("fixed-target plan and evaluation bundles must use the same view")


def _require_plan_replay_subsequence(
    evaluation: tuple[SimulationBundle, ...],
    declared_evaluation: tuple[SimulationBundle, ...],
) -> None:
    if not evaluation:
        return
    width = len(evaluation)
    if not any(
        declared_evaluation[start:start + width] == evaluation
        for start in range(len(declared_evaluation) - width + 1)
    ):
        raise ValueError(
            "fixed-target replay must be an exact contiguous subsequence of "
            "the declared evaluation bundles"
        )


def _prompt_provenance(
    bundles: tuple[SimulationBundle, ...],
) -> frozenset[tuple[int, str]]:
    provenance = frozenset(
        (bundle.prompt_order, bundle.prompt_id) for bundle in bundles
    )
    if any(
        isinstance(order, bool)
        or not isinstance(order, int)
        or order < 0
        or not isinstance(prompt_id, str)
        or not prompt_id
        for order, prompt_id in provenance
    ):
        raise ValueError("fixed-target prompt provenance is invalid")
    return provenance


def frequency_targets(
    bundles: Iterable[SimulationBundle], capacity: int
) -> frozenset[ExpertKey]:
    """Select stable frequency-ranked fixed targets from an audited bundle set."""

    sequence = tuple(bundles)
    if any(not isinstance(bundle, SimulationBundle) for bundle in sequence):
        raise TypeError("fixed-target selection requires SimulationBundle objects")
    capacity = _validate_capacity(capacity)
    counts = Counter(
        key
        for bundle in sequence
        for key in bundle.required_keys
    )
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    return frozenset(ranked[:capacity])


def build_fixed_target_plan(
    selection_bundles: Iterable[SimulationBundle],
    opposite_bundles: Iterable[SimulationBundle],
    capacity: int,
    policy_name: str,
) -> FixedTargetPlan:
    """Select one immutable fixed-target plan from the declared full split."""

    return FixedTargetPlan(
        policy=policy_name,
        capacity=_validate_capacity(capacity),
        selection_bundles=tuple(selection_bundles),
        opposite_bundles=tuple(opposite_bundles),
    )


def _bundle_size(event: RoutingEvent) -> int:
    return len(set(event.selected_experts))


def _validate_capacity(capacity: int) -> int:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("cache capacity must be a positive integer")
    return capacity
