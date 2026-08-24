"""Core-only canonical trace validation results and deterministic renderers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .producer import CanonicalRoutingTrace
from .trace import TRACE_FORMAT, RoutingTrace
from .trace_v2 import RoutingTraceV2, read_versioned_trace


TRACE_VALIDATION_FORMAT = "moe-cache-lab.trace-validation"
TRACE_VALIDATION_FORMAT_VERSION = 1
TRACE_VALIDATION_ERROR_CODES = (
    "unsupported_format_version",
    "invalid_json",
    "invalid_record_shape",
    "unknown_field",
    "duplicate_event_identity",
    "chronology_regression",
    "invalid_stage_or_phase",
    "invalid_assignment",
    "invalid_expert_identity",
    "invalid_trace_metadata",
    "empty_or_missing_trace",
    "io_error",
)

_CHRONOLOGY_MESSAGES = (
    " cannot follow ",
    "must use ascending",
    "must be strictly increasing",
    "must precede",
    "position rollback",
)
_ASSIGNMENT_MESSAGES = (
    "assignment_state",
    "assigned events",
    "unassigned events",
    "does not allow unassigned",
    "selection count does not match stage profile",
)
_STAGE_MESSAGES = (
    "routing_stage",
    "routing stage",
    "phase is incompatible",
    "event stage has no metadata profile",
)
_EXPERT_MESSAGES = (
    "selected expert",
    "selected_experts",
    "selected_probabilities",
)
_METADATA_MESSAGES = (
    "unsupported routing trace format",
    "format_version must be an integer",
    "model_id must be",
    "routing_stages must be",
    "routing_stages must contain",
    "routing_stages must have",
)
_TRACE_LINE = re.compile(r"trace line ([0-9]+)")


@dataclass(frozen=True)
class TraceValidationResult:
    """One immutable canonical-file validation result.

    This result records canonical trace validity only. It does not establish
    producer semantic validation or producer non-interference.
    """

    valid: bool
    trace_format_version: int | None
    summary: tuple[tuple[str, Any], ...] = ()
    trace: CanonicalRoutingTrace | None = None
    error_code: str | None = None
    message: str | None = None
    line_number: int | None = None

    def __post_init__(self) -> None:
        if self.valid:
            if self.trace is None or self.error_code is not None or self.message is not None:
                raise ValueError("valid trace result requires a trace and no error")
        elif (
            self.trace is not None
            or self.summary
            or self.error_code not in TRACE_VALIDATION_ERROR_CODES
            or not self.message
        ):
            raise ValueError("invalid trace result requires one bounded error")


def validate_trace_file(path: str | Path) -> TraceValidationResult:
    """Validate one trace with the authoritative single-dispatch reader."""

    source = Path(path)
    try:
        declared_version = _declared_trace_version(source)
    except OSError:
        return _failure("io_error", "could not read trace file")
    except UnicodeError:
        return _failure("invalid_json", "trace file must be UTF-8 text")

    try:
        trace = read_versioned_trace(source)
    except OSError:
        return _failure("io_error", "could not read trace file")
    except UnicodeError:
        return _failure(
            "invalid_json",
            "trace file must be UTF-8 text",
            trace_format_version=declared_version,
        )
    except ValueError as error:
        message = _bounded_message(str(error))
        return _failure(
            _classify_error(message),
            message,
            trace_format_version=declared_version,
            line_number=_line_number(message),
        )

    if isinstance(trace, RoutingTrace):
        summary = (
            ("model_id", trace.model_id),
            ("event_count", len(trace.events)),
            ("num_experts", trace.num_experts),
            ("experts_per_token", trace.experts_per_token),
            ("expert_request_count", len(trace.expert_requests)),
        )
        version = 1
    else:
        summary = (
            ("model_id", trace.model_id),
            ("event_count", len(trace.events)),
            (
                "routing_stages",
                tuple(profile.routing_stage for profile in trace.routing_stages),
            ),
            ("assigned_event_count", trace.assigned_event_count),
            ("unassigned_event_count", trace.unassigned_event_count),
            ("expert_request_count", len(trace.expert_requests)),
        )
        version = 2
    return TraceValidationResult(
        valid=True,
        trace_format_version=version,
        summary=summary,
        trace=trace,
    )


def render_trace_validation_human(result: TraceValidationResult) -> str:
    """Render one deterministic bounded human validation result."""

    if not result.valid:
        lines = [
            "Trace validation: INVALID",
            f"Error code: {result.error_code}",
        ]
        if result.trace_format_version is not None:
            lines.append(f"Trace format version: {result.trace_format_version}")
        if result.line_number is not None:
            lines.append(f"Line: {result.line_number}")
        lines.append(f"Message: {result.message}")
        return "\n".join(lines) + "\n"

    summary = dict(result.summary)
    lines = [
        "Trace validation: VALID",
        f"Trace format version: {result.trace_format_version}",
        f"Model ID: {summary['model_id']}",
        f"Events: {summary['event_count']}",
    ]
    if result.trace_format_version == 1:
        lines.extend(
            (
                f"Configured experts: {_human_optional(summary['num_experts'])}",
                f"Experts per token: {_human_optional(summary['experts_per_token'])}",
            )
        )
    else:
        lines.extend(
            (
                "Routing stages: " + ", ".join(summary["routing_stages"]),
                f"Assigned events: {summary['assigned_event_count']}",
                f"Unassigned events: {summary['unassigned_event_count']}",
            )
        )
    lines.append(f"Expert requests: {summary['expert_request_count']}")
    return "\n".join(lines) + "\n"


def render_trace_validation_json(result: TraceValidationResult) -> str:
    """Render one newline-terminated deterministic machine result object."""

    payload: dict[str, Any] = {
        "format": TRACE_VALIDATION_FORMAT,
        "format_version": TRACE_VALIDATION_FORMAT_VERSION,
        "valid": result.valid,
    }
    if result.trace_format_version is not None:
        payload["trace_format_version"] = result.trace_format_version
    if result.valid:
        payload["summary"] = dict(result.summary)
    else:
        payload["error_code"] = result.error_code
        payload["message"] = result.message
        if result.line_number is not None:
            payload["line_number"] = result.line_number
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _declared_trace_version(path: Path) -> int | None:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                metadata = json.loads(line, object_pairs_hook=_unique_json_object)
            except (json.JSONDecodeError, ValueError):
                return None
            if not isinstance(metadata, dict):
                return None
            if metadata.get("record_type") != "metadata":
                return None
            if metadata.get("format") != TRACE_FORMAT:
                return None
            version = metadata.get("format_version")
            if isinstance(version, bool) or not isinstance(version, int):
                return None
            return version
    return None


def _classify_error(message: str) -> str:
    lower = f" {message.lower()} "
    if message == "unsupported routing trace format version":
        return "unsupported_format_version"
    if message.startswith("invalid JSON at trace line"):
        return "invalid_json"
    if message == "trace must contain at least one record":
        return "empty_or_missing_trace"
    if "contains unknown field(s)" in message:
        return "unknown_field"
    if "duplicate" in lower and "event" in lower:
        return "duplicate_event_identity"
    if any(fragment in lower for fragment in _CHRONOLOGY_MESSAGES):
        return "chronology_regression"
    if any(fragment in lower for fragment in _ASSIGNMENT_MESSAGES):
        return "invalid_assignment"
    if any(fragment in lower for fragment in _STAGE_MESSAGES):
        return "invalid_stage_or_phase"
    if any(fragment in lower for fragment in _EXPERT_MESSAGES):
        return "invalid_expert_identity"
    if any(fragment in lower for fragment in _METADATA_MESSAGES):
        return "invalid_trace_metadata"
    return "invalid_record_shape"


def _failure(
    error_code: str,
    message: str,
    *,
    trace_format_version: int | None = None,
    line_number: int | None = None,
) -> TraceValidationResult:
    return TraceValidationResult(
        valid=False,
        trace_format_version=trace_format_version,
        error_code=error_code,
        message=message,
        line_number=line_number,
    )


def _line_number(message: str) -> int | None:
    match = _TRACE_LINE.search(message)
    return None if match is None else int(match.group(1))


def _bounded_message(message: str) -> str:
    single_line = " ".join(message.splitlines()).strip()
    if len(single_line) <= 400:
        return single_line
    return single_line[:397] + "..."


def _human_optional(value: Any) -> str:
    return "unavailable" if value is None else str(value)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


__all__ = [
    "TRACE_VALIDATION_ERROR_CODES",
    "TRACE_VALIDATION_FORMAT",
    "TRACE_VALIDATION_FORMAT_VERSION",
    "TraceValidationResult",
    "render_trace_validation_human",
    "render_trace_validation_json",
    "validate_trace_file",
]
