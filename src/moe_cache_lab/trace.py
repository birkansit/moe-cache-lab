"""Versioned, line-oriented storage for observed MoE routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.resources import files
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

TRACE_FORMAT = "moe-cache-lab.routing-jsonl"
TRACE_VERSION = 1
TRACE_SCHEMA_RESOURCE = "schemas/routing-trace-v1.schema.json"
ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class RoutingEvent:
    """One measured router selection for a token at a routed model layer."""

    phase: str
    token_position: int
    layer: int
    selected_experts: tuple[int, ...]
    token_id: int | None = None
    selected_probabilities: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in {"prompt", "generated"}:
            raise ValueError("phase must be 'prompt' or 'generated'")
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
        if not self.selected_experts:
            raise ValueError("selected_experts cannot be empty")
        if any(isinstance(expert, bool) or not isinstance(expert, int) or expert < 0 for expert in self.selected_experts):
            raise ValueError("selected_experts must contain non-negative integer IDs")
        if len(set(self.selected_experts)) != len(self.selected_experts):
            raise ValueError("selected_experts must be unique within a routing event")
        if self.selected_probabilities and len(self.selected_probabilities) != len(self.selected_experts):
            raise ValueError("selected_probabilities must match selected_experts")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in self.selected_probabilities
        ):
            raise ValueError("selected_probabilities must be finite values between zero and one")


@dataclass(frozen=True)
class RoutingTrace:
    """Trace metadata plus immutable measured routing events.

    V0.1 collector order is authoritative: prompt/prefill is layer-major
    ``(layer, token_position)`` and generated/decode is token-major
    ``(token_position, layer)``.
    """

    model_id: str
    num_experts: int | None
    experts_per_token: int | None
    events: tuple[RoutingEvent, ...]
    source_text: str | None = None
    generated_text: str | None = None
    capture_method: str = "model.forward(output_router_logits=True)"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    transformers_version: str | None = None
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("a routing trace must contain at least one event")
        identities = [(event.token_position, event.layer) for event in self.events]
        if len(set(identities)) != len(identities):
            raise ValueError("routing trace contains a duplicate physical token-position/layer event")
        generated_started = False
        last_position_by_phase_layer: dict[tuple[str, int], int] = {}
        for event in self.events:
            generated_started = generated_started or event.phase == "generated"
            if generated_started and event.phase == "prompt":
                raise ValueError("prompt routing events cannot follow generated routing events")
            phase_layer = event.phase, event.layer
            previous = last_position_by_phase_layer.get(phase_layer)
            if previous is not None and event.token_position <= previous:
                raise ValueError("token positions must be strictly increasing within each phase/layer")
            last_position_by_phase_layer[phase_layer] = event.token_position
        prompt_positions = [event.token_position for event in self.events if event.phase == "prompt"]
        generated_positions = [event.token_position for event in self.events if event.phase == "generated"]
        if prompt_positions and generated_positions and max(prompt_positions) >= min(generated_positions):
            raise ValueError("all prompt token positions must precede generated token positions")
        prompt_order = [
            (event.layer, event.token_position) for event in self.events if event.phase == "prompt"
        ]
        if prompt_order != sorted(prompt_order):
            raise ValueError("prompt routing events must use ascending layer-major collector order")
        generated_order = [
            (event.token_position, event.layer) for event in self.events if event.phase == "generated"
        ]
        if generated_order != sorted(generated_order):
            raise ValueError("generated routing events must use ascending token-major collector order")
        if self.num_experts is not None:
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive when provided")
            if any(expert >= self.num_experts for event in self.events for expert in event.selected_experts):
                raise ValueError("selected expert ID is outside num_experts")
        if self.experts_per_token is not None:
            if self.experts_per_token <= 0:
                raise ValueError("experts_per_token must be positive when provided")
            if any(len(event.selected_experts) != self.experts_per_token for event in self.events):
                raise ValueError("routing event selection count does not match experts_per_token")

    @property
    def expert_requests(self) -> tuple[ExpertKey, ...]:
        """Layer-qualified expert-weight requests in chronological event order."""
        return tuple((event.layer, expert) for event in self.events for expert in event.selected_experts)

    def metadata(self) -> dict[str, Any]:
        return {
            "record_type": "metadata",
            "format": TRACE_FORMAT,
            "format_version": TRACE_VERSION,
            "model_id": self.model_id,
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
            "source_text": self.source_text,
            "generated_text": self.generated_text,
            "capture_method": self.capture_method,
            "created_at": self.created_at,
            "transformers_version": self.transformers_version,
            "model_revision": self.model_revision,
        }


def write_trace(path: str | Path, trace: RoutingTrace) -> Path:
    """Write a JSONL trace atomically enough for ordinary local use."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(trace.metadata(), sort_keys=True) + "\n")
        for event in trace.events:
            payload = {"record_type": "routing_selection", **asdict(event)}
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return destination


def load_trace_schema() -> dict[str, Any]:
    """Load the packaged machine-readable schema for canonical v1 records."""
    resource = files("moe_cache_lab").joinpath(TRACE_SCHEMA_RESOURCE)
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_trace_records(records: Iterable[Mapping[str, Any]]) -> RoutingTrace:
    """Validate canonical v1 JSONL records and return an immutable trace.

    Record shape is closed by the packaged JSON Schema. Cross-record chronology
    and routing invariants are enforced by :class:`RoutingTrace`; input order is
    authoritative and is never sorted or repaired.
    """
    materialized = tuple(records)
    if not materialized:
        raise ValueError("trace must contain at least one record")
    if not isinstance(materialized[0], Mapping):
        raise ValueError("trace line 1 must be a JSON object")
    if materialized[0].get("record_type") != "metadata":
        raise ValueError("trace must start with a metadata record")

    schema = load_trace_schema()
    metadata_contract = schema["$defs"]["metadata_record"]
    event_contract = schema["$defs"]["routing_selection_record"]
    metadata = materialized[0]
    _validate_record_shape(metadata, metadata_contract, 1)
    _validate_metadata_values(metadata)

    events: list[RoutingEvent] = []
    for line_number, record in enumerate(materialized[1:], start=2):
        if not isinstance(record, Mapping):
            raise ValueError(f"trace line {line_number} must be a JSON object")
        if record.get("record_type") != "routing_selection":
            raise ValueError(f"unexpected record type at trace line {line_number}")
        _validate_record_shape(record, event_contract, line_number)
        if not isinstance(record["selected_experts"], list):
            raise ValueError(
                f"selected_experts at trace line {line_number} must be a JSON array"
            )
        if "selected_probabilities" in record and not isinstance(
            record["selected_probabilities"], list
        ):
            raise ValueError(
                f"selected_probabilities at trace line {line_number} must be a JSON array"
            )
        events.append(
            RoutingEvent(
                phase=record["phase"],
                token_position=record["token_position"],
                layer=record["layer"],
                selected_experts=tuple(record["selected_experts"]),
                token_id=record.get("token_id"),
                selected_probabilities=tuple(record.get("selected_probabilities", ())),
            )
        )

    return RoutingTrace(
        model_id=metadata["model_id"],
        num_experts=metadata.get("num_experts"),
        experts_per_token=metadata.get("experts_per_token"),
        events=tuple(events),
        source_text=metadata.get("source_text"),
        generated_text=metadata.get("generated_text"),
        capture_method=metadata.get("capture_method", "unknown"),
        created_at=metadata.get("created_at"),
        transformers_version=metadata.get("transformers_version"),
        model_revision=metadata.get("model_revision"),
    )


def read_trace(path: str | Path) -> RoutingTrace:
    """Read and validate the routing JSONL format."""
    with Path(path).open(encoding="utf-8") as stream:
        records = []
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line, object_pairs_hook=_unique_json_object))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid JSON at trace line {line_number}: {error}") from error
    return validate_trace_records(records)


def _validate_record_shape(
    record: Mapping[str, Any], contract: Mapping[str, Any], line_number: int
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


def _validate_metadata_values(metadata: Mapping[str, Any]) -> None:
    if metadata["format"] != TRACE_FORMAT:
        raise ValueError("unsupported routing trace format")
    version = metadata["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("format_version must be an integer")
    if version != TRACE_VERSION:
        raise ValueError("unsupported routing trace format version")
    model_id = metadata["model_id"]
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    for name in ("num_experts", "experts_per_token"):
        value = metadata.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be null or a positive integer")
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


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def requests_from_events(events: Iterable[RoutingEvent]) -> tuple[ExpertKey, ...]:
    return tuple((event.layer, expert) for event in events for expert in event.selected_experts)
