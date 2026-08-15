"""Strict storage boundary for canonical routing trace version 2.

The established :mod:`moe_cache_lab.trace` API remains version-1-only so its
analysis and cache consumers cannot accidentally claim version-2 support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.resources import files
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .trace import TRACE_FORMAT, RoutingTrace, validate_trace_records


TRACE_VERSION_V2 = 2
TRACE_SCHEMA_RESOURCE_V2 = "schemas/routing-trace-v2.schema.json"
RoutingExpertKeyV2 = tuple[str, int, int]

_STAGES = ("encoder", "decoder")
_PHASE_BY_STAGE = {
    "encoder": frozenset(("source",)),
    "decoder": frozenset(("decoder_prompt", "decoder_generated")),
}


@dataclass(frozen=True)
class RoutingStageProfile:
    """Configured routing shape for one encoder or decoder expert namespace."""

    routing_stage: str
    num_experts: int
    assigned_experts_per_token: int
    allows_unassigned: bool

    def __post_init__(self) -> None:
        if self.routing_stage not in _STAGES:
            raise ValueError("routing_stage must be 'encoder' or 'decoder'")
        for name in ("num_experts", "assigned_experts_per_token"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.allows_unassigned, bool):
            raise ValueError("allows_unassigned must be a boolean")


@dataclass(frozen=True)
class RoutingEventV2:
    """One post-capacity routing event in a stage-qualified namespace."""

    routing_stage: str
    phase: str
    token_position: int
    layer: int
    assignment_state: str
    selected_experts: tuple[int, ...]
    selected_probabilities: tuple[float, ...]
    token_id: int | None = None
    unassigned_reason: str | None = None

    def __post_init__(self) -> None:
        if self.routing_stage not in _STAGES:
            raise ValueError("routing_stage must be 'encoder' or 'decoder'")
        if self.phase not in _PHASE_BY_STAGE[self.routing_stage]:
            raise ValueError("phase is incompatible with routing_stage")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.token_position, self.layer)
        ):
            raise ValueError("token_position and layer must be non-negative integers")
        if self.token_id is not None and (
            isinstance(self.token_id, bool)
            or not isinstance(self.token_id, int)
            or self.token_id < 0
        ):
            raise ValueError("token_id must be None or a non-negative integer")
        if any(
            isinstance(expert, bool) or not isinstance(expert, int) or expert < 0
            for expert in self.selected_experts
        ):
            raise ValueError("selected_experts must contain non-negative integer IDs")
        if len(set(self.selected_experts)) != len(self.selected_experts):
            raise ValueError("selected_experts must be unique within a routing event")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in self.selected_probabilities
        ):
            raise ValueError(
                "selected_probabilities must be finite values between zero and one"
            )

        if self.assignment_state == "assigned":
            if not self.selected_experts:
                raise ValueError("assigned events must select at least one expert")
            if self.unassigned_reason is not None:
                raise ValueError("assigned events cannot have unassigned_reason")
            if self.selected_probabilities and (
                len(self.selected_probabilities) != len(self.selected_experts)
            ):
                raise ValueError("selected_probabilities must match selected_experts")
        elif self.assignment_state == "unassigned":
            if self.selected_experts or self.selected_probabilities:
                raise ValueError("unassigned events must have empty selection arrays")
            if self.unassigned_reason != "capacity":
                raise ValueError("unassigned events require capacity reason")
        else:
            raise ValueError("assignment_state must be 'assigned' or 'unassigned'")

    @property
    def expert_requests(self) -> tuple[RoutingExpertKeyV2, ...]:
        """Actual post-capacity expert requests for this atomic event."""
        return tuple(
            (self.routing_stage, self.layer, expert)
            for expert in self.selected_experts
        )


@dataclass(frozen=True)
class RoutingTraceV2:
    """Canonical version-2 metadata and immutable stage-qualified events."""

    model_id: str
    routing_stages: tuple[RoutingStageProfile, ...]
    events: tuple[RoutingEventV2, ...]
    source_text: str | None = None
    generated_text: str | None = None
    capture_method: str = "unknown"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    transformers_version: str | None = None
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not self.routing_stages:
            raise ValueError("routing_stages must contain at least one stage profile")
        if any(
            not isinstance(profile, RoutingStageProfile)
            for profile in self.routing_stages
        ):
            raise TypeError("routing_stages must contain RoutingStageProfile objects")
        stages = tuple(profile.routing_stage for profile in self.routing_stages)
        if len(set(stages)) != len(stages):
            raise ValueError("routing_stages must have unique stage profiles")
        if stages != tuple(stage for stage in _STAGES if stage in stages):
            raise ValueError("routing_stages must use canonical encoder/decoder order")
        if not self.events:
            raise ValueError("a routing trace must contain at least one event")
        if any(not isinstance(event, RoutingEventV2) for event in self.events):
            raise TypeError("events must contain RoutingEventV2 objects")
        _validate_trace_v2_semantics(self.routing_stages, self.events)

        for name in (
            "source_text",
            "generated_text",
            "transformers_version",
            "model_revision",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be null or a string")
        for name in ("capture_method", "created_at"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")

    @property
    def expert_requests(self) -> tuple[RoutingExpertKeyV2, ...]:
        """Stage-qualified actual requests in authoritative event order."""
        return tuple(
            request for event in self.events for request in event.expert_requests
        )

    @property
    def assigned_event_count(self) -> int:
        return sum(event.assignment_state == "assigned" for event in self.events)

    @property
    def unassigned_event_count(self) -> int:
        return sum(event.assignment_state == "unassigned" for event in self.events)

    def metadata(self) -> dict[str, Any]:
        return {
            "record_type": "metadata",
            "format": TRACE_FORMAT,
            "format_version": TRACE_VERSION_V2,
            "model_id": self.model_id,
            "routing_stages": [asdict(profile) for profile in self.routing_stages],
            "source_text": self.source_text,
            "generated_text": self.generated_text,
            "capture_method": self.capture_method,
            "created_at": self.created_at,
            "transformers_version": self.transformers_version,
            "model_revision": self.model_revision,
        }


VersionedRoutingTrace = RoutingTrace | RoutingTraceV2


def write_trace_v2(path: str | Path, trace: RoutingTraceV2) -> Path:
    """Explicitly write one canonical version-2 JSONL trace."""
    if not isinstance(trace, RoutingTraceV2):
        raise TypeError("write_trace_v2 requires a RoutingTraceV2")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(trace.metadata(), sort_keys=True) + "\n")
        for event in trace.events:
            payload = {"record_type": "routing_selection", **asdict(event)}
            if event.unassigned_reason is None:
                del payload["unassigned_reason"]
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return destination


def load_trace_v2_schema() -> dict[str, Any]:
    """Load the packaged machine-readable schema for canonical v2 records."""
    resource = files("moe_cache_lab").joinpath(TRACE_SCHEMA_RESOURCE_V2)
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_trace_v2_records(
    records: Iterable[Mapping[str, Any]],
) -> RoutingTraceV2:
    """Validate canonical v2 records without sorting, repair, or fallback."""
    materialized = tuple(records)
    metadata = _require_metadata_record(materialized)
    schema = load_trace_v2_schema()
    _validate_record_shape(metadata, schema["$defs"]["metadata_record"], 1)
    _validate_v2_header(metadata)

    profiles_raw = metadata["routing_stages"]
    if not isinstance(profiles_raw, list):
        raise ValueError("routing_stages must be a JSON array")
    profile_contract = schema["$defs"]["routing_stage_profile"]
    profiles = []
    for index, profile in enumerate(profiles_raw):
        if not isinstance(profile, Mapping):
            raise ValueError(f"routing stage profile {index} must be a JSON object")
        _validate_record_shape(profile, profile_contract, f"1 profile {index}")
        profiles.append(
            RoutingStageProfile(
                routing_stage=profile["routing_stage"],
                num_experts=profile["num_experts"],
                assigned_experts_per_token=profile["assigned_experts_per_token"],
                allows_unassigned=profile["allows_unassigned"],
            )
        )

    events = []
    event_contract = schema["$defs"]["routing_selection_record"]
    for line_number, record in enumerate(materialized[1:], start=2):
        if not isinstance(record, Mapping):
            raise ValueError(f"trace line {line_number} must be a JSON object")
        if record.get("record_type") != "routing_selection":
            raise ValueError(f"unexpected record type at trace line {line_number}")
        _validate_record_shape(record, event_contract, line_number)
        for name in ("selected_experts", "selected_probabilities"):
            if not isinstance(record[name], list):
                raise ValueError(f"{name} at trace line {line_number} must be a JSON array")
        events.append(
            RoutingEventV2(
                routing_stage=record["routing_stage"],
                phase=record["phase"],
                token_position=record["token_position"],
                layer=record["layer"],
                assignment_state=record["assignment_state"],
                selected_experts=tuple(record["selected_experts"]),
                selected_probabilities=tuple(record["selected_probabilities"]),
                token_id=record.get("token_id"),
                unassigned_reason=record.get("unassigned_reason"),
            )
        )

    return RoutingTraceV2(
        model_id=metadata["model_id"],
        routing_stages=tuple(profiles),
        events=tuple(events),
        source_text=metadata.get("source_text"),
        generated_text=metadata.get("generated_text"),
        capture_method=metadata.get("capture_method", "unknown"),
        created_at=metadata.get("created_at", ""),
        transformers_version=metadata.get("transformers_version"),
        model_revision=metadata.get("model_revision"),
    )


def read_trace_v2(path: str | Path) -> RoutingTraceV2:
    """Read exactly version 2; version 1 remains on ``trace.read_trace``."""
    return validate_trace_v2_records(_read_jsonl_records(path))


def validate_versioned_trace_records(
    records: Iterable[Mapping[str, Any]],
) -> VersionedRoutingTrace:
    """Dispatch exactly once by the first metadata format/version pair."""
    materialized = tuple(records)
    metadata = _require_metadata_record(materialized)
    if metadata.get("format") != TRACE_FORMAT:
        raise ValueError("unsupported routing trace format")
    version = metadata.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("format_version must be an integer")
    if version == 1:
        return validate_trace_records(materialized)
    if version == TRACE_VERSION_V2:
        return validate_trace_v2_records(materialized)
    raise ValueError("unsupported routing trace format version")


def read_versioned_trace(path: str | Path) -> VersionedRoutingTrace:
    """Read canonical v1 or v2 with strict metadata-based dispatch."""
    return validate_versioned_trace_records(_read_jsonl_records(path))


def _validate_trace_v2_semantics(
    profiles: tuple[RoutingStageProfile, ...],
    events: tuple[RoutingEventV2, ...],
) -> None:
    by_stage = {profile.routing_stage: profile for profile in profiles}
    identities = [
        (event.routing_stage, event.token_position, event.layer) for event in events
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "routing trace contains a duplicate routing-stage/token-position/layer event"
        )

    decoder_started = False
    decoder_generated_started = False
    last_position: dict[tuple[str, str, int], int] = {}
    for event in events:
        profile = by_stage.get(event.routing_stage)
        if profile is None:
            raise ValueError("routing event stage has no metadata profile")
        if event.routing_stage == "decoder":
            decoder_started = True
            decoder_generated_started = (
                decoder_generated_started or event.phase == "decoder_generated"
            )
        elif decoder_started:
            raise ValueError("encoder routing events cannot follow decoder routing events")
        if decoder_generated_started and event.phase == "decoder_prompt":
            raise ValueError(
                "decoder_prompt routing events cannot follow decoder_generated events"
            )

        stream = event.routing_stage, event.phase, event.layer
        previous = last_position.get(stream)
        if previous is not None and event.token_position <= previous:
            raise ValueError(
                "token positions must be strictly increasing within each stage/phase/layer"
            )
        last_position[stream] = event.token_position

        if event.assignment_state == "assigned":
            if len(event.selected_experts) != profile.assigned_experts_per_token:
                raise ValueError(
                    "assigned event selection count does not match stage profile"
                )
        elif not profile.allows_unassigned:
            raise ValueError("stage profile does not allow unassigned events")
        if any(expert >= profile.num_experts for expert in event.selected_experts):
            raise ValueError("selected expert ID is outside stage num_experts")

    encoder_order = [
        (event.layer, event.token_position)
        for event in events
        if event.routing_stage == "encoder"
    ]
    if encoder_order != sorted(encoder_order):
        raise ValueError("encoder source events must use ascending layer-major order")
    decoder_prompt_order = [
        (event.layer, event.token_position)
        for event in events
        if event.phase == "decoder_prompt"
    ]
    if decoder_prompt_order != sorted(decoder_prompt_order):
        raise ValueError("decoder prompt events must use ascending layer-major order")
    decoder_generated_order = [
        (event.token_position, event.layer)
        for event in events
        if event.phase == "decoder_generated"
    ]
    if decoder_generated_order != sorted(decoder_generated_order):
        raise ValueError("decoder generated events must use ascending token-major order")

    prompt_positions = [
        event.token_position for event in events if event.phase == "decoder_prompt"
    ]
    generated_positions = [
        event.token_position for event in events if event.phase == "decoder_generated"
    ]
    if prompt_positions and generated_positions and (
        max(prompt_positions) >= min(generated_positions)
    ):
        raise ValueError(
            "all decoder prompt positions must precede decoder generated positions"
        )


def _require_metadata_record(
    records: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    if not records:
        raise ValueError("trace must contain at least one record")
    if not isinstance(records[0], Mapping):
        raise ValueError("trace line 1 must be a JSON object")
    if records[0].get("record_type") != "metadata":
        raise ValueError("trace must start with a metadata record")
    return records[0]


def _validate_v2_header(metadata: Mapping[str, Any]) -> None:
    if metadata["format"] != TRACE_FORMAT:
        raise ValueError("unsupported routing trace format")
    version = metadata["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("format_version must be an integer")
    if version != TRACE_VERSION_V2:
        raise ValueError("unsupported routing trace format version")
    if not isinstance(metadata["model_id"], str) or not metadata["model_id"]:
        raise ValueError("model_id must be a non-empty string")
    for name in (
        "source_text",
        "generated_text",
        "transformers_version",
        "model_revision",
    ):
        value = metadata.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be null or a string")
    for name in ("capture_method", "created_at"):
        if name in metadata and not isinstance(metadata[name], str):
            raise ValueError(f"{name} must be a string when provided")


def _validate_record_shape(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    line_number: int | str,
) -> None:
    if any(not isinstance(key, str) for key in record):
        raise ValueError(f"trace line {line_number} contains a non-string object key")
    required = frozenset(contract["required"])
    allowed = frozenset(contract["properties"])
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(
            f"trace line {line_number} is missing required field(s): {', '.join(missing)}"
        )
    unknown = sorted(set(record).difference(allowed))
    if unknown:
        raise ValueError(
            f"trace line {line_number} contains unknown field(s): {', '.join(unknown)}"
        )


def _read_jsonl_records(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line, object_pairs_hook=_unique_json_object))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid JSON at trace line {line_number}: {error}"
                ) from error
    return tuple(records)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


__all__ = [
    "RoutingEventV2",
    "RoutingExpertKeyV2",
    "RoutingStageProfile",
    "RoutingTraceV2",
    "TRACE_SCHEMA_RESOURCE_V2",
    "TRACE_VERSION_V2",
    "VersionedRoutingTrace",
    "load_trace_v2_schema",
    "read_trace_v2",
    "read_versioned_trace",
    "validate_trace_v2_records",
    "validate_versioned_trace_records",
    "write_trace_v2",
]
