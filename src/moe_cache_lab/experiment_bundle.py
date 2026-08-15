"""Deterministic, offline experiment-bundle integrity container.

Bundles anchor caller-supplied artifacts and canonical trace references.  They
do not run analysis, change evidence classes, or establish scientific quality,
representativeness, performance, physical residency, or recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
from typing import Any

from . import __version__
from .trace import TRACE_FORMAT, RoutingTrace
from .trace_v2 import (
    RoutingTraceV2,
    read_versioned_trace,
    validate_versioned_trace_records,
)


BUNDLE_FORMAT = "moe-cache-lab.experiment-bundle"
BUNDLE_VERSION = 1

CONFIG_NAME = "bundle-config.json"
ENVIRONMENT_NAME = "environment.json"
REPORT_JSON_NAME = "report.json"
REPORT_MARKDOWN_NAME = "report.md"
MANIFEST_NAME = "bundle-manifest.json"
MANIFEST_SHA256_NAME = "bundle-manifest.sha256"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_ARTIFACTS = (
    ("config", CONFIG_NAME),
    ("environment", ENVIRONMENT_NAME),
    ("report_json", REPORT_JSON_NAME),
    ("report_markdown", REPORT_MARKDOWN_NAME),
)
_MANIFEST_KEYS = {
    "format",
    "format_version",
    "experiment_id",
    "moe_cache_lab_version",
    "artifacts",
    "traces",
}
_ARTIFACT_KEYS = {"role", "path", "size_bytes", "sha256"}


@dataclass(frozen=True)
class RuntimeProvenance:
    """Optional caller-known runtime facts; no ML package is inspected."""

    torch_version: str | None = None
    transformers_version: str | None = None
    device: str | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "torch_version",
            "transformers_version",
            "device",
            "dtype",
        ):
            _validate_optional_text(getattr(self, name), name)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "device": self.device,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class EmbeddedTrace:
    """Explicit request to copy one validated canonical trace byte-for-byte."""

    trace_id: str
    source_path: str | Path

    def __post_init__(self) -> None:
        _validate_id(self.trace_id, "trace_id")
        if not isinstance(self.source_path, (str, Path)):
            raise TypeError("embedded trace source_path must be str or Path")


@dataclass(frozen=True)
class ExternalTrace:
    """Metadata-only immutable reference; writer and verifier never fetch it."""

    trace_id: str
    reference: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_id(self.trace_id, "trace_id")
        _validate_external_reference(self.reference)
        _validate_sha256(self.sha256, "external trace sha256")


TraceInput = EmbeddedTrace | ExternalTrace


@dataclass(frozen=True)
class BundleCreationResult:
    bundle_root: Path
    experiment_id: str
    manifest_sha256: str
    artifact_count: int
    trace_count: int


@dataclass(frozen=True)
class BundleVerificationResult:
    experiment_id: str
    moe_cache_lab_version: str
    manifest_sha256: str
    artifact_count: int
    trace_count: int


def write_experiment_bundle(
    bundle_root: str | Path,
    *,
    experiment_id: str,
    config: dict[str, Any],
    report_json: str | bytes,
    report_markdown: str | bytes,
    traces: tuple[TraceInput, ...] = (),
    runtime_provenance: RuntimeProvenance | None = None,
) -> BundleCreationResult:
    """Write one deterministic bundle after validating every supplied input.

    Report ``str`` values are encoded as UTF-8 without newline normalization;
    report ``bytes`` must already be UTF-8 and are copied exactly.
    """

    _validate_id(experiment_id, "experiment_id")
    if not isinstance(config, dict):
        raise TypeError("bundle config must be a JSON object")
    if not isinstance(traces, tuple):
        raise TypeError("traces must be a tuple")
    if runtime_provenance is None:
        runtime_provenance = RuntimeProvenance()
    if not isinstance(runtime_provenance, RuntimeProvenance):
        raise TypeError("runtime_provenance must be RuntimeProvenance or None")

    config_bytes = _canonical_json_bytes(config, "bundle config")
    report_json_bytes = _utf8_bytes(report_json, "report_json")
    _strict_json_loads(report_json_bytes, "report JSON")
    report_markdown_bytes = _utf8_bytes(report_markdown, "report_markdown")

    environment = _environment_data(runtime_provenance)
    _validate_environment(environment, __version__)
    environment_bytes = _canonical_json_bytes(environment, "environment")
    prepared_traces = _prepare_traces(traces)

    root = Path(bundle_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError("bundle root must not already exist")

    base_payloads = (
        ("config", CONFIG_NAME, config_bytes),
        ("environment", ENVIRONMENT_NAME, environment_bytes),
        ("report_json", REPORT_JSON_NAME, report_json_bytes),
        ("report_markdown", REPORT_MARKDOWN_NAME, report_markdown_bytes),
    )
    artifact_records = [
        _artifact_record(role, path, payload)
        for role, path, payload in base_payloads
    ]
    trace_records: list[dict[str, Any]] = []
    for prepared in prepared_traces:
        trace_records.append(prepared["record"])
        if prepared["payload"] is not None:
            artifact_records.append(
                _artifact_record(
                    "embedded_trace", prepared["record"]["path"], prepared["payload"]
                )
            )

    manifest = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "experiment_id": experiment_id,
        "moe_cache_lab_version": __version__,
        "artifacts": artifact_records,
        "traces": trace_records,
    }
    manifest_bytes = _canonical_json_bytes(manifest, "bundle manifest")
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    sidecar_bytes = f"{manifest_sha256}  {MANIFEST_NAME}\n".encode("ascii")

    root.mkdir(parents=True)
    for _, path, payload in base_payloads:
        (root / path).write_bytes(payload)
    embedded = [item for item in prepared_traces if item["payload"] is not None]
    if embedded:
        (root / "traces").mkdir()
        for item in embedded:
            (root / item["record"]["path"]).write_bytes(item["payload"])
    (root / MANIFEST_NAME).write_bytes(manifest_bytes)
    (root / MANIFEST_SHA256_NAME).write_bytes(sidecar_bytes)

    verification = verify_experiment_bundle(root)
    return BundleCreationResult(
        root,
        verification.experiment_id,
        verification.manifest_sha256,
        verification.artifact_count,
        verification.trace_count,
    )


def verify_experiment_bundle(
    bundle_root: str | Path,
) -> BundleVerificationResult:
    """Strictly verify bundle integrity without fetching or repairing content."""

    root = Path(bundle_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("bundle root must be a regular directory")

    manifest_bytes = _read_owned_file(root, MANIFEST_NAME)
    sidecar_bytes = _read_owned_file(root, MANIFEST_SHA256_NAME)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    expected_sidecar = f"{manifest_sha256}  {MANIFEST_NAME}\n".encode("ascii")
    if sidecar_bytes != expected_sidecar:
        raise ValueError("bundle manifest SHA-256 sidecar mismatch")

    manifest = _strict_json_loads(manifest_bytes, "bundle manifest")
    if not isinstance(manifest, dict):
        raise ValueError("bundle manifest must be a JSON object")
    if _canonical_json_bytes(manifest, "bundle manifest") != manifest_bytes:
        raise ValueError("bundle manifest is not canonical JSON")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "bundle manifest")
    if manifest["format"] != BUNDLE_FORMAT:
        raise ValueError("unsupported experiment bundle format")
    if (
        isinstance(manifest["format_version"], bool)
        or not isinstance(manifest["format_version"], int)
        or manifest["format_version"] != BUNDLE_VERSION
    ):
        raise ValueError("unsupported experiment bundle format version")
    _validate_id(manifest["experiment_id"], "experiment_id")
    _validate_bounded_text(
        manifest["moe_cache_lab_version"], "moe_cache_lab_version"
    )

    artifacts = manifest["artifacts"]
    traces = manifest["traces"]
    if not isinstance(artifacts, list) or not isinstance(traces, list):
        raise ValueError("bundle manifest artifacts and traces must be arrays")

    artifact_by_path: dict[str, dict[str, Any]] = {}
    artifact_pairs: list[tuple[str, str]] = []
    path_casefolds: set[str] = set()
    for record in artifacts:
        _validate_artifact_record(record)
        path = record["path"]
        folded = path.casefold()
        if folded in path_casefolds:
            raise ValueError("bundle artifact paths collide")
        path_casefolds.add(folded)
        artifact_by_path[path] = record
        artifact_pairs.append((record["role"], path))

    trace_ids: set[str] = set()
    trace_id_casefolds: set[str] = set()
    embedded_records: list[dict[str, Any]] = []
    for record in traces:
        if not isinstance(record, dict):
            raise ValueError("bundle trace record must be a JSON object")
        trace_id = record.get("trace_id")
        _validate_id(trace_id, "trace_id")
        if trace_id in trace_ids or trace_id.casefold() in trace_id_casefolds:
            raise ValueError("bundle trace IDs must be unique")
        trace_ids.add(trace_id)
        trace_id_casefolds.add(trace_id.casefold())
        if record.get("mode") == "embedded":
            _validate_embedded_trace_record(record)
            embedded_records.append(record)
        elif record.get("mode") == "external":
            _validate_external_trace_record(record)
        else:
            raise ValueError("bundle trace mode must be embedded or external")

    expected_pairs = list(_REQUIRED_ARTIFACTS) + [
        ("embedded_trace", record["path"]) for record in embedded_records
    ]
    if artifact_pairs != expected_pairs:
        raise ValueError("bundle artifact roles/paths do not match the v1 contract")

    for record in embedded_records:
        artifact = artifact_by_path.get(record["path"])
        if artifact is None or any(
            artifact[name] != record[name] for name in ("size_bytes", "sha256")
        ):
            raise ValueError("embedded trace artifact metadata mismatch")

    payloads: dict[str, bytes] = {}
    for record in artifacts:
        payload = _read_owned_file(root, record["path"])
        if len(payload) != record["size_bytes"]:
            raise ValueError(f"bundle artifact size mismatch: {record['path']}")
        if _sha256_bytes(payload) != record["sha256"]:
            raise ValueError(f"bundle artifact SHA-256 mismatch: {record['path']}")
        payloads[record["path"]] = payload

    config = _strict_json_loads(payloads[CONFIG_NAME], "bundle config")
    if not isinstance(config, dict):
        raise ValueError("bundle config must be a JSON object")
    if _canonical_json_bytes(config, "bundle config") != payloads[CONFIG_NAME]:
        raise ValueError("bundle config is not canonical JSON")

    environment = _strict_json_loads(payloads[ENVIRONMENT_NAME], "environment")
    _validate_environment(environment, manifest["moe_cache_lab_version"])
    if _canonical_json_bytes(environment, "environment") != payloads[ENVIRONMENT_NAME]:
        raise ValueError("bundle environment is not canonical JSON")

    _strict_json_loads(payloads[REPORT_JSON_NAME], "report JSON")
    _decode_utf8(payloads[REPORT_MARKDOWN_NAME], "report Markdown")

    for record in embedded_records:
        trace = _validate_trace_bytes(payloads[record["path"]])
        version = 1 if isinstance(trace, RoutingTrace) else 2
        if not isinstance(trace, (RoutingTrace, RoutingTraceV2)):
            raise TypeError("versioned trace reader returned an unsupported trace")
        if record["trace_format"] != TRACE_FORMAT or record["trace_format_version"] != version:
            raise ValueError("embedded trace format/version metadata mismatch")

    expected_files = {
        MANIFEST_NAME,
        MANIFEST_SHA256_NAME,
        *(record["path"] for record in artifacts),
    }
    expected_dirs = {"traces"} if embedded_records else set()
    _validate_inventory(root, expected_files, expected_dirs)

    return BundleVerificationResult(
        experiment_id=manifest["experiment_id"],
        moe_cache_lab_version=manifest["moe_cache_lab_version"],
        manifest_sha256=manifest_sha256,
        artifact_count=len(artifacts),
        trace_count=len(traces),
    )


def _prepare_traces(traces: tuple[TraceInput, ...]) -> tuple[dict[str, Any], ...]:
    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_casefolds: set[str] = set()
    for item in traces:
        if not isinstance(item, (EmbeddedTrace, ExternalTrace)):
            raise TypeError("traces must contain EmbeddedTrace or ExternalTrace")
        if item.trace_id in seen_ids or item.trace_id.casefold() in seen_casefolds:
            raise ValueError("bundle trace IDs must be unique")
        seen_ids.add(item.trace_id)
        seen_casefolds.add(item.trace_id.casefold())
        if isinstance(item, EmbeddedTrace):
            source = Path(item.source_path)
            if not source.is_file():
                raise ValueError("embedded trace source must be an existing file")
            # Exercise the canonical file reader required by the public trace
            # contract, then validate the exact bytes that will be copied.  The
            # second validation closes a source-change window without repairing
            # or normalizing either chronology or serialization.
            read_versioned_trace(source)
            payload = source.read_bytes()
            trace = _validate_trace_bytes(payload)
            version = 1 if isinstance(trace, RoutingTrace) else 2
            relative_path = f"traces/{item.trace_id}.jsonl"
            record = {
                "trace_id": item.trace_id,
                "mode": "embedded",
                "trace_format": TRACE_FORMAT,
                "trace_format_version": version,
                "path": relative_path,
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            prepared.append({"record": record, "payload": payload})
        else:
            prepared.append(
                {
                    "record": {
                        "trace_id": item.trace_id,
                        "mode": "external",
                        "reference": item.reference,
                        "sha256": item.sha256,
                    },
                    "payload": None,
                }
            )
    return tuple(prepared)


def _environment_data(runtime: RuntimeProvenance) -> dict[str, Any]:
    return {
        "moe_cache_lab_version": __version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
        },
        "runtime": runtime.to_dict(),
    }


def _validate_environment(value: Any, tool_version: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("bundle environment must be a JSON object")
    _require_exact_keys(
        value,
        {"moe_cache_lab_version", "python", "platform", "runtime"},
        "bundle environment",
    )
    if value["moe_cache_lab_version"] != tool_version:
        raise ValueError("bundle environment tool version mismatch")
    _require_exact_keys(value["python"], {"version", "implementation"}, "python provenance")
    _require_exact_keys(value["platform"], {"system", "release"}, "platform provenance")
    _require_exact_keys(
        value["runtime"],
        {"torch_version", "transformers_version", "device", "dtype"},
        "runtime provenance",
    )
    for name, item in value["python"].items():
        _validate_bounded_text(item, f"python {name}")
    for name, item in value["platform"].items():
        _validate_bounded_text(item, f"platform {name}")
    for name, item in value["runtime"].items():
        _validate_optional_text(item, name)


def _artifact_record(role: str, path: str, payload: bytes) -> dict[str, Any]:
    return {
        "role": role,
        "path": path,
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _validate_artifact_record(record: Any) -> None:
    if not isinstance(record, dict):
        raise ValueError("bundle artifact record must be a JSON object")
    _require_exact_keys(record, _ARTIFACT_KEYS, "bundle artifact record")
    if record["role"] not in {
        "config",
        "environment",
        "report_json",
        "report_markdown",
        "embedded_trace",
    }:
        raise ValueError("unsupported bundle artifact role")
    _validate_safe_relative_path(record["path"])
    size = record["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("bundle artifact size_bytes must be a non-negative integer")
    _validate_sha256(record["sha256"], "bundle artifact sha256")


def _validate_embedded_trace_record(record: dict[str, Any]) -> None:
    _require_exact_keys(
        record,
        {
            "trace_id",
            "mode",
            "trace_format",
            "trace_format_version",
            "path",
            "size_bytes",
            "sha256",
        },
        "embedded trace record",
    )
    expected_path = f"traces/{record['trace_id']}.jsonl"
    if record["path"] != expected_path:
        raise ValueError("embedded trace path does not match trace_id")
    _validate_safe_relative_path(record["path"])
    if record["trace_format"] != TRACE_FORMAT:
        raise ValueError("unsupported embedded trace format")
    version = record["trace_format_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2):
        raise ValueError("unsupported embedded trace format version")
    size = record["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("embedded trace size_bytes must be a non-negative integer")
    _validate_sha256(record["sha256"], "embedded trace sha256")


def _validate_external_trace_record(record: dict[str, Any]) -> None:
    _require_exact_keys(
        record,
        {"trace_id", "mode", "reference", "sha256"},
        "external trace record",
    )
    _validate_external_reference(record["reference"])
    _validate_sha256(record["sha256"], "external trace sha256")


def _canonical_json_bytes(value: Any, label: str) -> bytes:
    _validate_json_value(value, label)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {error}") from error
    try:
        return (text + "\n").encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} contains invalid Unicode") from error


def _strict_json_loads(payload: bytes, label: str) -> Any:
    text = _decode_utf8(payload, label)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard numeric constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
        _validate_json_value(value, label)
        return value
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {error}") from error


def _validate_trace_bytes(payload: bytes) -> RoutingTrace | RoutingTraceV2:
    text = _decode_utf8(payload, "embedded trace")
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = _strict_json_loads(
                line.encode("utf-8"), f"embedded trace line {line_number}"
            )
        except ValueError as error:
            raise ValueError(
                f"invalid embedded trace JSON at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"embedded trace line {line_number} must be a JSON object")
        records.append(record)
    return validate_versioned_trace_records(records)


def _validate_json_value(
    value: Any, label: str, active_containers: set[int] | None = None
) -> None:
    if active_containers is None:
        active_containers = set()
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{label} contains NaN or infinity")
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{label} contains a circular JSON value")
        active_containers.add(identity)
        try:
            for item in value:
                _validate_json_value(item, label, active_containers)
        finally:
            active_containers.remove(identity)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{label} contains a circular JSON value")
        active_containers.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{label} contains a non-string object key")
                _validate_json_value(item, label, active_containers)
        finally:
            active_containers.remove(identity)
        return
    raise ValueError(f"{label} contains a non-JSON value: {type(value).__name__}")


def _utf8_bytes(value: str | bytes, label: str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytes):
        _decode_utf8(value, label)
        return value
    raise TypeError(f"{label} must be str or bytes")


def _decode_utf8(value: bytes, label: str) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be valid UTF-8") from error


def _validate_id(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{label} must be 1-128 safe ASCII letters, digits, dot, underscore, or hyphen"
        )


def _validate_external_reference(value: Any) -> None:
    _validate_bounded_text(value, "external trace reference", maximum=2048)
    if value.lower().startswith("file:"):
        raise ValueError("external trace reference cannot use file://")
    if PurePosixPath(value).is_absolute():
        raise ValueError("external trace reference cannot be an absolute local path")
    windows = PureWindowsPath(value)
    if windows.is_absolute() or windows.drive:
        raise ValueError("external trace reference cannot be an absolute local path")
    if ".." in PurePosixPath(value).parts or ".." in windows.parts:
        raise ValueError("external trace reference cannot contain local path traversal")


def _validate_bounded_text(value: Any, label: str, maximum: int = 256) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or not value.isprintable()
    ):
        raise ValueError(f"{label} must be non-empty bounded printable text")


def _validate_optional_text(value: Any, label: str) -> None:
    if value is not None:
        _validate_bounded_text(value, label)


def _validate_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")


def _validate_safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("bundle-owned path must be a normalized relative POSIX path")
    parts = value.split("/")
    windows = PureWindowsPath(value)
    if (
        value.startswith("/")
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(value).as_posix() != value
    ):
        raise ValueError("bundle-owned path must be a normalized relative POSIX path")
    return value


def _owned_path(root: Path, relative: str) -> Path:
    _validate_safe_relative_path(relative)
    candidate = root.joinpath(*relative.split("/"))
    root_resolved = root.resolve()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"bundle-owned file is missing: {relative}") from error
    if not resolved.is_relative_to(root_resolved):
        raise ValueError("bundle-owned path escapes bundle root")
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise ValueError("bundle-owned path cannot contain a symlink")
    if not candidate.is_file():
        raise ValueError(f"bundle-owned path is not a regular file: {relative}")
    return candidate


def _read_owned_file(root: Path, relative: str) -> bytes:
    return _owned_path(root, relative).read_bytes()


def _validate_inventory(
    root: Path, expected_files: set[str], expected_dirs: set[str]
) -> None:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("bundle inventory cannot contain symlinks")
        if path.is_file():
            files.add(relative)
        elif path.is_dir():
            directories.add(relative)
        else:
            raise ValueError("bundle inventory contains an unsupported entry")
    if files != expected_files or directories != expected_dirs:
        raise ValueError("bundle inventory does not match the manifest contract")


def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has missing or unknown fields")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "RuntimeProvenance",
    "EmbeddedTrace",
    "ExternalTrace",
    "BundleCreationResult",
    "BundleVerificationResult",
    "write_experiment_bundle",
    "verify_experiment_bundle",
]
