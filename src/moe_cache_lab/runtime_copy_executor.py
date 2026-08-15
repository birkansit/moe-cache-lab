"""Independent CPU copy executor for the bounded Switch validation experiment.

This module deliberately does not import the project's cache simulator or
hardware-cost model.  It observes only canonical routing events and the real
expert payload manifest, then implements the frozen LRU slot map independently.
The staging tensors are test storage and never participate in model inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .trace_v2 import RoutingEventV2, RoutingExpertKeyV2, read_trace_v2


EXECUTOR_INPUT_FORMAT = "moe-cache-lab.runtime-copy-executor-input/v1"
OBSERVATION_FORMAT = "moe-cache-lab.runtime-copy-observation/v1"


@dataclass(frozen=True)
class ExecutorInput:
    """Raw frozen inputs accepted by the independent executor.

    No prediction counters, miss sequence, eviction decisions, or predicted
    resident keys are representable in this API.
    """

    model_id: str
    revision: str
    torch_version: str
    transformers_version: str
    device: str
    dtype: str
    policy: str
    capacity_bytes: int
    slot_count: int
    trace_path: str
    trace_sha256: str
    payload_manifest_path: str
    payload_manifest_sha256: str
    output_path: str


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    contiguous: bool
    numel: int
    element_size: int
    logical_payload_bytes: int


@dataclass(frozen=True)
class ExpertPayloadSpec:
    key: RoutingExpertKeyV2
    components: tuple[ComponentSpec, ...]
    expert_payload_bytes: int


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(destination)
    return destination


def key_to_json(key: RoutingExpertKeyV2) -> list[str | int]:
    return [key[0], key[1], key[2]]


def key_from_json(value: object) -> RoutingExpertKeyV2:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or value[0] not in ("encoder", "decoder")
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value[1:])
    ):
        raise ValueError("invalid stage-qualified expert key")
    return value[0], value[1], value[2]


def load_payload_manifest(
    path: str | Path,
) -> tuple[dict[RoutingExpertKeyV2, ExpertPayloadSpec], Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != "moe-cache-lab.switch-payload-manifest/v1":
        raise ValueError("unsupported Switch payload manifest")
    experts_raw = payload.get("experts")
    if not isinstance(experts_raw, list) or not experts_raw:
        raise ValueError("payload manifest must contain expert records")
    result: dict[RoutingExpertKeyV2, ExpertPayloadSpec] = {}
    for raw in experts_raw:
        if not isinstance(raw, dict):
            raise ValueError("payload expert record must be an object")
        key = key_from_json(raw.get("key"))
        components_raw = raw.get("components")
        if not isinstance(components_raw, list) or not components_raw:
            raise ValueError("payload expert must contain components")
        components = []
        for component in components_raw:
            if not isinstance(component, dict):
                raise ValueError("payload component must be an object")
            shape = component.get("shape")
            if not isinstance(shape, list) or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            ):
                raise ValueError("payload component shape is invalid")
            spec = ComponentSpec(
                name=component["name"],
                shape=tuple(shape),
                dtype=component["dtype"],
                device=component["device"],
                contiguous=component["contiguous"],
                numel=component["numel"],
                element_size=component["element_size"],
                logical_payload_bytes=component["logical_payload_bytes"],
            )
            if spec.logical_payload_bytes != spec.numel * spec.element_size:
                raise ValueError("component logical byte count is inconsistent")
            components.append(spec)
        total = raw.get("expert_payload_bytes")
        if total != sum(item.logical_payload_bytes for item in components):
            raise ValueError("expert payload byte total is inconsistent")
        if key in result:
            raise ValueError("duplicate expert payload key")
        result[key] = ExpertPayloadSpec(key, tuple(components), total)
    if payload.get("expert_count") != len(result):
        raise ValueError("payload expert count is inconsistent")
    return dict(sorted(result.items())), payload


def _component_layout(spec: ExpertPayloadSpec) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (item.name, item.shape, item.dtype, item.device, item.logical_payload_bytes)
        for item in spec.components
    )


def execute_independent_replay(
    events: tuple[RoutingEventV2, ...],
    payload_specs: Mapping[RoutingExpertKeyV2, ExpertPayloadSpec],
    payload_sources: Mapping[RoutingExpertKeyV2, Mapping[str, Any]],
    *,
    capacity_bytes: int,
    slot_count: int,
    profile_copies: bool = True,
) -> dict[str, Any]:
    """Replay raw events once with an independent stage-qualified LRU.

    Each miss performs one explicit ``Tensor.copy_`` per actual parameter
    component.  Unassigned events do not advance the logical cache clock.
    """
    import torch

    if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
        raise ValueError("capacity_bytes must be a positive integer")
    if isinstance(slot_count, bool) or not isinstance(slot_count, int) or slot_count <= 0:
        raise ValueError("slot_count must be a positive integer")
    specs = dict(payload_specs)
    sources = dict(payload_sources)
    if not specs:
        raise ValueError("payload specifications cannot be empty")
    if set(specs) != set(sources):
        raise ValueError("payload source keys do not match payload manifest")
    template = specs[min(specs)]
    layout = _component_layout(template)
    if any(_component_layout(spec) != layout for spec in specs.values()):
        raise ValueError("all expert payloads must share one staging-slot layout")
    if template.expert_payload_bytes > capacity_bytes:
        raise ValueError("expert payload exceeds byte capacity")
    if slot_count * template.expert_payload_bytes > capacity_bytes:
        raise ValueError("preallocated staging slots exceed byte capacity")

    first_sources = sources[min(sources)]
    if tuple(sorted(first_sources)) != tuple(sorted(item.name for item in template.components)):
        raise ValueError("source components do not match manifest")
    slots: list[dict[str, Any]] = []
    for _ in range(slot_count):
        slots.append(
            {
                component.name: torch.empty_like(first_sources[component.name], device="cpu")
                for component in template.components
            }
        )

    resident_to_slot: dict[RoutingExpertKeyV2, int] = {}
    last_touch: dict[RoutingExpertKeyV2, int] = {}
    free_slots = list(range(slot_count))
    clock = 0
    hits = misses = evictions = 0
    completed_loads = 0
    source_operand_bytes = 0
    copy_records: list[dict[str, Any]] = []
    loaded_key_bytes: list[dict[str, Any]] = []

    activities = [torch.profiler.ProfilerActivity.CPU]
    profiler_context = torch.profiler.profile(activities=activities) if profile_copies else _NullProfiler()
    with profiler_context as profiler:
        for event in events:
            required = event.expert_requests
            if not required:
                continue
            if len(required) != 1:
                raise ValueError("bounded executor requires at most one assigned expert per event")
            key = required[0]
            spec = specs.get(key)
            source = sources.get(key)
            if spec is None or source is None:
                raise ValueError(f"missing payload for routed expert {key!r}")
            clock += 1
            if key in resident_to_slot:
                hits += 1
                last_touch[key] = clock
                continue
            misses += 1
            if free_slots:
                slot_index = free_slots.pop(0)
            else:
                victim = min(resident_to_slot, key=lambda item: (last_touch[item], item))
                slot_index = resident_to_slot.pop(victim)
                del last_touch[victim]
                evictions += 1
            destination = slots[slot_index]
            key_bytes = 0
            for component in spec.components:
                source_tensor = source.get(component.name)
                destination_tensor = destination.get(component.name)
                if source_tensor is None or destination_tensor is None:
                    raise ValueError("source or destination component is missing")
                source_bytes = source_tensor.numel() * source_tensor.element_size()
                destination_bytes = destination_tensor.numel() * destination_tensor.element_size()
                destination_tensor.copy_(source_tensor)
                equal = bool(torch.equal(destination_tensor, source_tensor))
                record = {
                    "key": key_to_json(key),
                    "component_name": component.name,
                    "source_shape": list(source_tensor.shape),
                    "destination_shape": list(destination_tensor.shape),
                    "source_dtype": str(source_tensor.dtype),
                    "destination_dtype": str(destination_tensor.dtype),
                    "source_device": str(source_tensor.device),
                    "destination_device": str(destination_tensor.device),
                    "source_logical_bytes": source_bytes,
                    "destination_logical_bytes": destination_bytes,
                    "content_equal": equal,
                }
                copy_records.append(record)
                source_operand_bytes += source_bytes
                key_bytes += source_bytes
            if key_bytes != spec.expert_payload_bytes:
                raise RuntimeError("copied expert bytes differ from payload manifest")
            completed_loads += 1
            loaded_key_bytes.append(
                {
                    "key": key_to_json(key),
                    "component_count": len(spec.components),
                    "copied_source_bytes": key_bytes,
                }
            )
            resident_to_slot[key] = slot_index
            last_touch[key] = clock

    profiler_copy_count = 0
    if profile_copies:
        profiler_copy_count = sum(
            event.count for event in profiler.key_averages() if event.key == "aten::copy_"
        )
    return {
        "event_count": len(events),
        "expert_request_count": sum(len(event.expert_requests) for event in events),
        "hits": hits,
        "misses": misses,
        "eviction_count": evictions,
        "completed_physical_logical_loads": completed_loads,
        "actual_component_copy_count": len(copy_records),
        "profiler_aten_copy_count": profiler_copy_count,
        "actual_source_operand_logical_bytes": source_operand_bytes,
        "all_copy_content_equal": all(record["content_equal"] for record in copy_records),
        "final_resident_bytes": sum(specs[key].expert_payload_bytes for key in resident_to_slot),
        "final_resident_keys": [key_to_json(key) for key in sorted(resident_to_slot)],
        "copy_records": copy_records,
        "loaded_key_bytes": loaded_key_bytes,
    }


class _NullProfiler:
    def __enter__(self) -> "_NullProfiler":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _load_executor_input(path: str | Path) -> ExecutorInput:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.pop("format", None) != EXECUTOR_INPUT_FORMAT:
        raise ValueError("unsupported executor input")
    expected = set(ExecutorInput.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("executor input fields do not match the frozen API")
    return ExecutorInput(**raw)


def _real_payload_sources(collector: object) -> dict[RoutingExpertKeyV2, dict[str, Any]]:
    from .switch_collector import _expert_index

    result: dict[RoutingExpertKeyV2, dict[str, Any]] = {}
    for stage, layer, mlp in collector._sparse_mlps:  # type: ignore[attr-defined]
        for name, expert in mlp.experts.items():
            key = stage, layer, _expert_index(name)
            result[key] = dict(sorted(expert.named_parameters(recurse=True)))
    return dict(sorted(result.items()))


def run_executor(input_path: str | Path) -> dict[str, Any]:
    from .switch_collector import (
        DEFAULT_SWITCH_MODEL,
        SWITCH_MODEL_REVISION,
        SwitchTraceCollector,
    )
    import torch
    import transformers

    inputs = _load_executor_input(input_path)
    if (
        inputs.model_id != DEFAULT_SWITCH_MODEL
        or inputs.revision != SWITCH_MODEL_REVISION
        or inputs.torch_version != torch.__version__
        or inputs.transformers_version != transformers.__version__
        or inputs.device != "cpu"
        or inputs.dtype != "float32"
        or inputs.policy != "lru"
        or inputs.slot_count != 8
    ):
        raise RuntimeError("executor input does not match the frozen runtime protocol")
    torch.set_num_threads(4)
    if torch.get_num_threads() != 4:
        raise RuntimeError("executor could not establish four Torch intra-op threads")
    if sha256_file(inputs.trace_path) != inputs.trace_sha256:
        raise RuntimeError("executor trace hash changed")
    if sha256_file(inputs.payload_manifest_path) != inputs.payload_manifest_sha256:
        raise RuntimeError("executor payload manifest hash changed")
    trace = read_trace_v2(inputs.trace_path)
    specs, manifest = load_payload_manifest(inputs.payload_manifest_path)
    if manifest.get("model_id") != inputs.model_id or manifest.get("revision") != inputs.revision:
        raise RuntimeError("payload manifest identity does not match executor input")
    collector = SwitchTraceCollector(local_files_only=True)
    sources = _real_payload_sources(collector)
    observed = execute_independent_replay(
        trace.events,
        specs,
        sources,
        capacity_bytes=inputs.capacity_bytes,
        slot_count=inputs.slot_count,
        profile_copies=True,
    )
    result = {
        "format": OBSERVATION_FORMAT,
        "model_id": inputs.model_id,
        "revision": inputs.revision,
        "torch_version": inputs.torch_version,
        "transformers_version": inputs.transformers_version,
        "device": inputs.device,
        "dtype": inputs.dtype,
        "policy": inputs.policy,
        "capacity_bytes": inputs.capacity_bytes,
        "slot_count": inputs.slot_count,
        "trace_sha256": inputs.trace_sha256,
        "payload_manifest_sha256": inputs.payload_manifest_sha256,
        **observed,
    }
    write_canonical_json(inputs.output_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private independent Switch CPU copy executor")
    parser.add_argument("--input", required=True)
    arguments = parser.parse_args(argv)
    run_executor(arguments.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
