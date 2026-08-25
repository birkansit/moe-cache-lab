#!/usr/bin/env python3
"""Audit external raw->canonical trace preservation and two-run reproduction.

This is an evidence/reproduction helper, not product behavior. It deliberately
keeps the preservation oracle independent from the external converter: the
checker never imports converter code or converter helper functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


KIMI_RAW_ROWS = 300
KIMI_LAYERS_PER_ROW = 26
KIMI_EXPERTS_PER_EVENT = 8
KIMI_EVENT_COUNT = 7_800
KIMI_EXPERT_REQUEST_COUNT = 62_400

OLMOE_RAW_ROWS = 299
OLMOE_LAYERS_PER_ROW = 16
OLMOE_EXPERTS_PER_EVENT = 8
OLMOE_EVENT_COUNT = 4_784
OLMOE_EXPERT_REQUEST_COUNT = 38_272


class AuditFailure(ValueError):
    """Raised when evidence bytes/semantics disagree with the frozen spec."""


class ReproductionError(RuntimeError):
    """Raised for an operational failure that is not evidence disagreement."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise AuditFailure(f"{path}: blank line at {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditFailure(
                    f"{path}: invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise AuditFailure(f"{path}: line {line_number} is not an object")
            rows.append(value)
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _validate_expert_list(
    experts: Any, *, width: int, label: str
) -> list[int]:
    _require(isinstance(experts, list), f"{label}: expert list is not a list")
    _require(len(experts) == width, f"{label}: expected {width} experts")
    _require(
        all(isinstance(expert, int) and not isinstance(expert, bool) for expert in experts),
        f"{label}: expert IDs must be integers",
    )
    _require(len(set(experts)) == len(experts), f"{label}: duplicate expert ID")
    return experts


def audit_kimi(raw_path: Path, canonical_path: Path) -> dict[str, Any]:
    raw = _read_jsonl(raw_path)
    canonical = _read_jsonl(canonical_path)

    _require(len(raw) == KIMI_RAW_ROWS, f"Kimi: expected {KIMI_RAW_ROWS} raw rows")
    _require(bool(canonical), "Kimi: canonical trace is empty")
    metadata, records = canonical[0], canonical[1:]
    _require(metadata.get("record_type") == "metadata", "Kimi: missing metadata")
    _require(metadata.get("format") == "moe-cache-lab.routing-jsonl", "Kimi: wrong format")
    _require(metadata.get("format_version") == 2, "Kimi: wrong format version")

    profiles = metadata.get("routing_stages")
    _require(isinstance(profiles, list) and len(profiles) == 1, "Kimi: wrong routing-stage profile count")
    profile = profiles[0]
    _require(profile.get("routing_stage") == "decoder", "Kimi: wrong routing stage")
    _require(profile.get("num_experts") == 256, "Kimi: wrong expert universe")
    _require(profile.get("assigned_experts_per_token") == 8, "Kimi: wrong assigned width")
    _require(profile.get("allows_unassigned") is False, "Kimi: unassigned must be disabled")

    expected: list[tuple[int, int, list[int]]] = []
    for row_index, row in enumerate(raw):
        _require(row.get("tok") == row_index, f"Kimi: raw token chronology mismatch at row {row_index}")
        layers = row.get("layers")
        _require(isinstance(layers, list), f"Kimi: row {row_index} layers is not a list")
        _require(
            len(layers) == KIMI_LAYERS_PER_ROW,
            f"Kimi: row {row_index} expected {KIMI_LAYERS_PER_ROW} routed layers",
        )
        for ordinal, experts_value in enumerate(layers):
            experts = _validate_expert_list(
                experts_value,
                width=KIMI_EXPERTS_PER_EVENT,
                label=f"Kimi raw token {row_index} ordinal {ordinal}",
            )
            # Frozen audit specification, intentionally not imported from converter.
            expected.append((row_index, ordinal + 1, experts))

    _require(len(expected) == KIMI_EVENT_COUNT, "Kimi: internal expected event count mismatch")
    _require(len(records) == KIMI_EVENT_COUNT, f"Kimi: expected {KIMI_EVENT_COUNT} canonical events")

    for event_index, ((token_position, layer, experts), actual) in enumerate(zip(expected, records)):
        prefix = f"Kimi canonical event {event_index}"
        _require(actual.get("record_type") == "routing_selection", f"{prefix}: wrong record type")
        _require(actual.get("assignment_state") == "assigned", f"{prefix}: wrong assignment state")
        _require(actual.get("token_position") == token_position, f"{prefix}: token mismatch/reorder")
        _require(actual.get("layer") == layer, f"{prefix}: layer mapping mismatch")
        _require(actual.get("routing_stage") == "decoder", f"{prefix}: wrong routing stage")
        _require(actual.get("phase") == "decoder_generated", f"{prefix}: wrong phase")
        _require(actual.get("selected_experts") == experts, f"{prefix}: expert membership/list changed")
        _require(actual.get("selected_probabilities") == [], f"{prefix}: probabilities were reconstructed")

    expert_requests = sum(len(record["selected_experts"]) for record in records)
    layers = {record["layer"] for record in records}
    _require(expert_requests == KIMI_EXPERT_REQUEST_COUNT, "Kimi: expert-request count mismatch")
    _require(layers == set(range(1, 27)), "Kimi: canonical layer range is not 1..26")
    return {
        "model": "kimi",
        "raw_rows": len(raw),
        "canonical_events": len(records),
        "expert_requests": expert_requests,
        "layer_min": min(layers),
        "layer_max": max(layers),
        "preservation_audit": "PASS",
    }


def audit_olmoe(raw_path: Path, canonical_path: Path) -> dict[str, Any]:
    raw = _read_jsonl(raw_path)
    canonical = _read_jsonl(canonical_path)

    _require(len(raw) == OLMOE_RAW_ROWS, f"OLMoE: expected {OLMOE_RAW_ROWS} raw rows")
    _require(bool(canonical), "OLMoE: canonical trace is empty")
    metadata, records = canonical[0], canonical[1:]
    _require(metadata.get("record_type") == "metadata", "OLMoE: missing metadata")
    _require(metadata.get("format") == "moe-cache-lab.routing-jsonl", "OLMoE: wrong format")
    _require(metadata.get("format_version") == 1, "OLMoE: wrong format version")
    _require(metadata.get("num_experts") == 64, "OLMoE: wrong expert universe")
    _require(metadata.get("experts_per_token") == 8, "OLMoE: wrong expert width")

    expected: list[tuple[int, int, list[int]]] = []
    for row_index, row in enumerate(raw, 1):
        _require(row.get("tok") == row_index, f"OLMoE: raw token chronology mismatch at token {row_index}")
        layers = row.get("layers")
        _require(isinstance(layers, list), f"OLMoE: token {row_index} layers is not a list")
        _require(
            len(layers) == OLMOE_LAYERS_PER_ROW,
            f"OLMoE: token {row_index} expected {OLMOE_LAYERS_PER_ROW} routed layers",
        )
        for layer, experts_value in enumerate(layers):
            experts = _validate_expert_list(
                experts_value,
                width=OLMOE_EXPERTS_PER_EVENT,
                label=f"OLMoE raw token {row_index} layer {layer}",
            )
            expected.append((row_index, layer, experts))

    _require(len(expected) == OLMOE_EVENT_COUNT, "OLMoE: internal expected event count mismatch")
    _require(len(records) == OLMOE_EVENT_COUNT, f"OLMoE: expected {OLMOE_EVENT_COUNT} canonical events")

    for event_index, ((token_position, layer, experts), actual) in enumerate(zip(expected, records)):
        prefix = f"OLMoE canonical event {event_index}"
        _require(actual.get("record_type") == "routing_selection", f"{prefix}: wrong record type")
        _require(actual.get("token_position") == token_position, f"{prefix}: token mismatch/reorder")
        _require(actual.get("layer") == layer, f"{prefix}: layer mapping mismatch")
        _require(actual.get("phase") == "generated", f"{prefix}: wrong phase")
        _require(actual.get("selected_experts") == experts, f"{prefix}: expert membership/list changed")

    expert_requests = sum(len(record["selected_experts"]) for record in records)
    layers = {record["layer"] for record in records}
    _require(expert_requests == OLMOE_EXPERT_REQUEST_COUNT, "OLMoE: expert-request count mismatch")
    _require(layers == set(range(16)), "OLMoE: canonical layer range is not 0..15")
    return {
        "model": "olmoe",
        "raw_rows": len(raw),
        "canonical_events": len(records),
        "expert_requests": expert_requests,
        "layer_min": min(layers),
        "layer_max": max(layers),
        "preservation_audit": "PASS",
    }


def _run_external_converter(
    converter: Path,
    model: str,
    raw_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    command = [sys.executable, str(converter), model, str(raw_path), str(output_path)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ReproductionError(
            f"converter failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def reproduce(
    *,
    model: str,
    raw_path: Path,
    converter: Path,
    expected_raw_sha256: str,
    expected_canonical_sha256: str,
    workspace: Path,
) -> dict[str, Any]:
    actual_raw_sha = sha256_file(raw_path)
    _require(
        actual_raw_sha == expected_raw_sha256,
        f"{model}: raw SHA-256 disagreement: {actual_raw_sha} != {expected_raw_sha256}",
    )

    workspace.mkdir(parents=True, exist_ok=True)
    run1 = workspace / f"{model}-run1.jsonl"
    run2 = workspace / f"{model}-run2.jsonl"
    first = _run_external_converter(converter, model, raw_path, run1)
    second = _run_external_converter(converter, model, raw_path, run2)
    run1_sha = sha256_file(run1)
    run2_sha = sha256_file(run2)
    _require(run1_sha == run2_sha, f"{model}: two conversion runs differ")
    _require(
        run1_sha == expected_canonical_sha256,
        f"{model}: canonical SHA-256 disagreement: {run1_sha} != {expected_canonical_sha256}",
    )

    preservation = audit_kimi(raw_path, run1) if model == "kimi" else audit_olmoe(raw_path, run1)
    return {
        "model": model,
        "mechanical_reproduction": "AGREEMENT",
        "raw_sha256": actual_raw_sha,
        "converter_sha256": sha256_file(converter),
        "run1_sha256": run1_sha,
        "run2_sha256": run2_sha,
        "expected_canonical_sha256": expected_canonical_sha256,
        "preservation": preservation,
        "commands": [first["command"], second["command"]],
        "environment": {
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("kimi", "olmoe"), required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--expected-canonical-sha256", required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.workspace is None:
            with tempfile.TemporaryDirectory(prefix="mcl-external-repro-") as temporary:
                result = reproduce(
                    model=args.model,
                    raw_path=args.raw,
                    converter=args.converter,
                    expected_raw_sha256=args.expected_raw_sha256,
                    expected_canonical_sha256=args.expected_canonical_sha256,
                    workspace=Path(temporary),
                )
        else:
            result = reproduce(
                model=args.model,
                raw_path=args.raw,
                converter=args.converter,
                expected_raw_sha256=args.expected_raw_sha256,
                expected_canonical_sha256=args.expected_canonical_sha256,
                workspace=args.workspace,
            )
    except AuditFailure as exc:
        payload = {"mechanical_reproduction": "DISAGREEMENT", "message": str(exc), "model": args.model}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"DISAGREEMENT: {exc}", file=sys.stderr)
        return 2
    except (OSError, ReproductionError) as exc:
        payload = {"mechanical_reproduction": "NOT ESTABLISHED", "message": str(exc), "model": args.model, "status": "ERROR"}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"{args.model}: mechanical reproduction AGREEMENT")
        print(f"raw SHA-256: {result['raw_sha256']}")
        print(f"canonical SHA-256: {result['run1_sha256']}")
        print("raw->canonical preservation audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
