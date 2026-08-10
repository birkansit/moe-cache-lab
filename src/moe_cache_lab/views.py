"""Immutable simulation bundles derived from measured routing traces.

The adapters in this module never rewrite :class:`RoutingTrace` or invent
router probabilities.  They provide alternate atomic access views for the
offline cache simulator while retaining prompt and source-event provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .trace import ExpertKey, RoutingEvent, RoutingTrace

TOKEN_LAYER_ATOMIC = "token_layer_atomic"
PREFILL_LAYER_UNION_ATOMIC = "prefill_layer_union_atomic"
SIMULATION_VIEWS = frozenset({TOKEN_LAYER_ATOMIC, PREFILL_LAYER_UNION_ATOMIC})


@dataclass(frozen=True)
class PromptTraceSource:
    """One independent prompt trace plus its stable suite identity."""

    prompt_id: str
    prompt_order: int
    trace: RoutingTrace

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("prompt_id must be a non-empty string")
        if (
            isinstance(self.prompt_order, bool)
            or not isinstance(self.prompt_order, int)
            or self.prompt_order < 0
        ):
            raise ValueError("prompt_order must be a non-negative integer")
        if not isinstance(self.trace, RoutingTrace):
            raise TypeError("trace source must contain a RoutingTrace")


@dataclass(frozen=True)
class SimulationBundle:
    """One immutable atomic cache requirement with measured provenance."""

    phase: str
    required_keys: frozenset[ExpertKey]
    view_name: str
    prompt_id: str
    prompt_order: int
    layer: int
    token_position: int | None
    source_event_count: int
    source_assignment_count: int
    source_token_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.phase not in {"prompt", "generated"}:
            raise ValueError("bundle phase must be 'prompt' or 'generated'")
        if self.view_name not in SIMULATION_VIEWS:
            raise ValueError(f"unknown simulation view: {self.view_name}")
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("bundle prompt_id must be a non-empty string")
        if (
            isinstance(self.prompt_order, bool)
            or not isinstance(self.prompt_order, int)
            or self.prompt_order < 0
        ):
            raise ValueError("bundle prompt_order must be a non-negative integer")
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError("bundle layer must be a non-negative integer")
        if not isinstance(self.required_keys, frozenset) or not self.required_keys:
            raise ValueError("bundle required_keys must be a non-empty frozenset")
        for key in self.required_keys:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in key)
            ):
                raise ValueError("bundle keys must be non-negative (layer_id, expert_id) pairs")
            if key[0] != self.layer:
                raise ValueError("all required keys must be qualified by the bundle layer")
        if self.token_position is not None and (
            isinstance(self.token_position, bool)
            or not isinstance(self.token_position, int)
            or self.token_position < 0
        ):
            raise ValueError("bundle token_position must be non-negative when provided")
        if self.view_name == TOKEN_LAYER_ATOMIC and self.token_position is None:
            raise ValueError("token_layer_atomic bundles require a token_position")
        if (
            self.view_name == PREFILL_LAYER_UNION_ATOMIC
            and self.phase == "prompt"
            and self.token_position is not None
        ):
            raise ValueError("grouped prompt bundles must not have a token_position")
        if self.phase == "generated" and self.token_position is None:
            raise ValueError("generated bundles require a token_position")
        if (
            isinstance(self.source_event_count, bool)
            or not isinstance(self.source_event_count, int)
            or self.source_event_count <= 0
        ):
            raise ValueError("source_event_count must be a positive integer")
        if (
            isinstance(self.source_assignment_count, bool)
            or not isinstance(self.source_assignment_count, int)
            or self.source_assignment_count < len(self.required_keys)
            or self.source_assignment_count < self.source_event_count
        ):
            raise ValueError(
                "source_assignment_count cannot be smaller than source events or unique requirements"
            )
        if len(self.source_token_positions) != self.source_event_count:
            raise ValueError("source_token_positions must identify every measured source event")
        positions_are_valid = all(
            not isinstance(position, bool) and isinstance(position, int) and position >= 0
            for position in self.source_token_positions
        )
        if not positions_are_valid or (
            tuple(sorted(self.source_token_positions)) != self.source_token_positions
            or len(set(self.source_token_positions)) != len(self.source_token_positions)
        ):
            raise ValueError("source token positions must be ascending unique non-negative integers")
        if self.view_name == TOKEN_LAYER_ATOMIC and (
            self.source_event_count != 1
            or self.source_assignment_count != len(self.required_keys)
            or self.source_token_positions != (self.token_position,)
        ):
            raise ValueError("token_layer_atomic bundles must represent exactly one source event")
        if self.view_name == PREFILL_LAYER_UNION_ATOMIC and self.phase == "generated" and (
            self.source_event_count != 1
            or self.source_assignment_count != len(self.required_keys)
            or self.source_token_positions != (self.token_position,)
        ):
            raise ValueError("grouped generated bundles must preserve exactly one source event")
        if (
            self.view_name == PREFILL_LAYER_UNION_ATOMIC
            and self.phase == "prompt"
            and self.source_event_count == 1
            and self.source_assignment_count != len(self.required_keys)
        ):
            raise ValueError(
                "a one-event grouped prompt bundle must preserve its exact assignment set"
            )


@dataclass(frozen=True)
class BundleScopeSummary:
    """Auditable counts for one phase or the combined derived view."""

    scope: str
    bundles: int
    requests: int
    source_events: int
    source_assignments: int
    max_bundle_size: int


@dataclass(frozen=True)
class SimulationViewSummary:
    """Bundle/request/source counts and hard-capacity feasibility for a view."""

    view_name: str
    prompt_count: int
    combined: BundleScopeSummary
    prompt: BundleScopeSummary
    generated: BundleScopeSummary

    @property
    def max_bundle_size(self) -> int:
        return self.combined.max_bundle_size

    def is_feasible(self, capacity: int) -> bool:
        _validate_capacity(capacity)
        return capacity >= self.max_bundle_size

    def require_feasible(self, capacity: int) -> None:
        _validate_capacity(capacity)
        if capacity < self.max_bundle_size:
            raise ValueError(
                f"cache capacity {capacity} is smaller than the largest "
                f"{self.view_name} bundle {self.max_bundle_size}"
            )


def adapt_prompt_traces(
    prompt_traces: Iterable[PromptTraceSource],
    view_name: str,
) -> tuple[SimulationBundle, ...]:
    """Derive one of the two frozen Stage 1 views from ordered prompt traces."""

    if view_name not in SIMULATION_VIEWS:
        raise ValueError(f"unknown simulation view: {view_name}")
    sources = tuple(prompt_traces)
    if any(not isinstance(source, PromptTraceSource) for source in sources):
        raise TypeError("view adapters require PromptTraceSource objects")
    identities = [(source.prompt_order, source.prompt_id) for source in sources]
    if len({order for order, _ in identities}) != len(identities):
        raise ValueError("prompt trace orders must be unique")
    if len({prompt_id for _, prompt_id in identities}) != len(identities):
        raise ValueError("prompt trace IDs must be unique")
    if [order for order, _ in identities] != sorted(order for order, _ in identities):
        raise ValueError("prompt traces must be supplied in ascending prompt order")

    bundles: list[SimulationBundle] = []
    for source in sources:
        if view_name == TOKEN_LAYER_ATOMIC:
            bundles.extend(_event_bundle(event, source, view_name) for event in source.trace.events)
            continue

        prompt_by_layer: dict[int, list[RoutingEvent]] = {}
        for event in source.trace.events:
            if event.phase == "prompt":
                prompt_by_layer.setdefault(event.layer, []).append(event)
        for layer in sorted(prompt_by_layer):
            events = prompt_by_layer[layer]
            bundles.append(SimulationBundle(
                phase="prompt",
                required_keys=frozenset(
                    (layer, expert) for event in events for expert in event.selected_experts
                ),
                view_name=view_name,
                prompt_id=source.prompt_id,
                prompt_order=source.prompt_order,
                layer=layer,
                token_position=None,
                source_event_count=len(events),
                source_assignment_count=sum(len(event.selected_experts) for event in events),
                source_token_positions=tuple(event.token_position for event in events),
            ))
        bundles.extend(
            _event_bundle(event, source, view_name)
            for event in source.trace.events
            if event.phase == "generated"
        )
    return tuple(bundles)


def adapt_events(
    events: Iterable[RoutingEvent],
    *,
    prompt_id: str = "routing-event-compatibility",
    prompt_order: int = 0,
) -> tuple[SimulationBundle, ...]:
    """Compatibility adapter for the public V0.2 ``simulate(events, ...)`` API."""

    sequence = tuple(events)
    if any(not isinstance(event, RoutingEvent) for event in sequence):
        raise TypeError("cache simulation requires RoutingEvent objects")
    return tuple(
        SimulationBundle(
            phase=event.phase,
            required_keys=frozenset((event.layer, expert) for expert in event.selected_experts),
            view_name=TOKEN_LAYER_ATOMIC,
            prompt_id=prompt_id,
            prompt_order=prompt_order,
            layer=event.layer,
            token_position=event.token_position,
            source_event_count=1,
            source_assignment_count=len(event.selected_experts),
            source_token_positions=(event.token_position,),
        )
        for event in sequence
    )


def summarize_bundles(bundles: Iterable[SimulationBundle]) -> SimulationViewSummary:
    """Summarize derived accesses without simulating a cache policy."""

    sequence = tuple(bundles)
    if any(not isinstance(bundle, SimulationBundle) for bundle in sequence):
        raise TypeError("bundle summary requires SimulationBundle objects")
    view_names = {bundle.view_name for bundle in sequence}
    if len(view_names) > 1:
        raise ValueError("a bundle summary cannot mix simulation views")
    _validate_bundle_sequence(sequence)
    view_name = next(iter(view_names), TOKEN_LAYER_ATOMIC)
    return SimulationViewSummary(
        view_name=view_name,
        prompt_count=len({(bundle.prompt_order, bundle.prompt_id) for bundle in sequence}),
        combined=_scope_summary("combined", sequence),
        prompt=_scope_summary("prompt/prefill", tuple(
            bundle for bundle in sequence if bundle.phase == "prompt"
        )),
        generated=_scope_summary("generated/decode", tuple(
            bundle for bundle in sequence if bundle.phase == "generated"
        )),
    )


def _event_bundle(
    event: RoutingEvent,
    source: PromptTraceSource,
    view_name: str,
) -> SimulationBundle:
    return SimulationBundle(
        phase=event.phase,
        required_keys=frozenset((event.layer, expert) for expert in event.selected_experts),
        view_name=view_name,
        prompt_id=source.prompt_id,
        prompt_order=source.prompt_order,
        layer=event.layer,
        token_position=event.token_position,
        source_event_count=1,
        source_assignment_count=len(event.selected_experts),
        source_token_positions=(event.token_position,),
    )


def _scope_summary(scope: str, bundles: tuple[SimulationBundle, ...]) -> BundleScopeSummary:
    return BundleScopeSummary(
        scope=scope,
        bundles=len(bundles),
        requests=sum(len(bundle.required_keys) for bundle in bundles),
        source_events=sum(bundle.source_event_count for bundle in bundles),
        source_assignments=sum(bundle.source_assignment_count for bundle in bundles),
        max_bundle_size=max((len(bundle.required_keys) for bundle in bundles), default=0),
    )


def _validate_bundle_sequence(sequence: tuple[SimulationBundle, ...]) -> None:
    """Enforce the frozen view chronology without requiring dense positions."""

    if not sequence:
        return
    prompt_blocks: list[tuple[int, str]] = []
    current_prompt: tuple[int, str] | None = None
    bundles_by_prompt: dict[tuple[int, str], list[SimulationBundle]] = {}
    for bundle in sequence:
        prompt = bundle.prompt_order, bundle.prompt_id
        if prompt != current_prompt:
            prompt_blocks.append(prompt)
            current_prompt = prompt
        bundles_by_prompt.setdefault(prompt, []).append(bundle)

    if len(set(prompt_blocks)) != len(prompt_blocks):
        raise ValueError("bundles for each prompt must form one contiguous block")
    orders = [order for order, _ in prompt_blocks]
    prompt_ids = [prompt_id for _, prompt_id in prompt_blocks]
    if len(set(orders)) != len(orders) or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("bundle prompt IDs and prompt_order values must be unique")
    if orders != sorted(orders):
        raise ValueError("bundle prompt blocks must be ascending by prompt_order")

    for prompt, prompt_bundles in bundles_by_prompt.items():
        phases = [bundle.phase for bundle in prompt_bundles]
        if "generated" in phases:
            generated_start = phases.index("generated")
            if any(phase == "prompt" for phase in phases[generated_start:]):
                raise ValueError("prompt bundles must precede generated bundles")

        prompt_part = [bundle for bundle in prompt_bundles if bundle.phase == "prompt"]
        generated_part = [bundle for bundle in prompt_bundles if bundle.phase == "generated"]
        view_name = prompt_bundles[0].view_name
        if view_name == TOKEN_LAYER_ATOMIC:
            prompt_order = [(bundle.layer, bundle.token_position) for bundle in prompt_part]
            if prompt_order != sorted(prompt_order):
                raise ValueError("token view prompt bundles must use layer-major order")
        else:
            prompt_layers = [bundle.layer for bundle in prompt_part]
            if len(set(prompt_layers)) != len(prompt_layers):
                raise ValueError("grouped view requires one prompt bundle per prompt/layer")
            if prompt_layers != sorted(prompt_layers):
                raise ValueError("grouped prompt bundles must use ascending layer order")

        generated_order = [
            (bundle.token_position, bundle.layer) for bundle in generated_part
        ]
        if generated_order != sorted(generated_order):
            raise ValueError("generated bundles must use token-major order")

        physical_prompt = {
            (position, bundle.layer)
            for bundle in prompt_part
            for position in bundle.source_token_positions
        }
        prompt_event_count = sum(bundle.source_event_count for bundle in prompt_part)
        if len(physical_prompt) != prompt_event_count:
            raise ValueError("prompt bundles contain duplicate physical token-position/layer events")
        physical_generated = {
            (bundle.token_position, bundle.layer) for bundle in generated_part
        }
        if len(physical_generated) != len(generated_part):
            raise ValueError("generated bundles contain duplicate physical token-position/layer events")
        if physical_prompt.intersection(physical_generated):
            raise ValueError("prompt and generated bundles collide at a physical token-position/layer")
        if physical_prompt and physical_generated and (
            max(position for position, _ in physical_prompt)
            >= min(position for position, _ in physical_generated)
        ):
            raise ValueError(
                f"all prompt source positions must precede generated positions for prompt {prompt}"
            )


def _validate_capacity(capacity: int) -> None:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("cache capacity must be a positive integer")
