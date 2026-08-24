"""Predictor, comparer, and evidence helpers for bounded CPU copy validation.

The predictor is the only role that imports the canonical byte-cache
simulation.  The independent executor lives in :mod:`runtime_copy_executor`
and is intentionally not imported here during its fresh-process execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .byte_cache import simulate_versioned_byte_cache
from .runtime_copy_executor import (
    key_from_json,
    key_to_json,
    sha256_file,
    write_canonical_json,
)
from .trace_v2 import read_trace_v2, write_trace_v2


MODEL_ID = "google/switch-base-8"
MODEL_REVISION = "92fe2d22b024d9937146fe097ba3d3a7ba146e1b"
PROMPT_SHA256 = "30752b0dfe1d6680cc2568d56b6a924c18fd6460ad332e8afdac8de4ddc17e77"
CAPACITY_BYTES = 150_994_944
SLOT_COUNT = 8
EXPERT_PAYLOAD_BYTES = 18_874_368
COMPONENT_NAMES = ("wi.weight", "wo.weight")
PREDICTION_FORMAT = "moe-cache-lab.runtime-copy-prediction/v1"
COMPARISON_FORMAT = "moe-cache-lab.runtime-copy-comparison/v1"
ARTIFACT_MANIFEST_FORMAT = "moe-cache-lab.runtime-copy-artifact-manifest/v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_prompt(prompt: str) -> str:
    digest = sha256_bytes(prompt.encode("utf-8"))
    if digest != PROMPT_SHA256:
        raise RuntimeError("frozen prompt bytes do not match the required SHA-256")
    return digest


def _component_record(name: str, parameter: Any) -> dict[str, Any]:
    numel = parameter.numel()
    element_size = parameter.element_size()
    return {
        "name": name,
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "contiguous": bool(parameter.is_contiguous()),
        "numel": numel,
        "element_size": element_size,
        "logical_payload_bytes": numel * element_size,
    }


def build_real_payload_manifest(collector: object) -> dict[str, Any]:
    """Describe every real stage-qualified expert parameter component."""
    from .switch_collector import _expert_index

    experts: list[dict[str, Any]] = []
    for stage, layer, mlp in collector._sparse_mlps:  # type: ignore[attr-defined]
        for module_name, expert in mlp.experts.items():
            key = stage, layer, _expert_index(module_name)
            named = dict(expert.named_parameters(recurse=True))
            if tuple(sorted(named)) != tuple(sorted(COMPONENT_NAMES)):
                raise RuntimeError("real Switch expert components differ from frozen protocol")
            components = [_component_record(name, named[name]) for name in COMPONENT_NAMES]
            expert_bytes = sum(item["logical_payload_bytes"] for item in components)
            experts.append(
                {
                    "key": key_to_json(key),
                    "components": components,
                    "expert_payload_bytes": expert_bytes,
                }
            )
    experts.sort(key=lambda item: tuple(item["key"]))
    payload_sizes = {item["expert_payload_bytes"] for item in experts}
    component_counts = {len(item["components"]) for item in experts}
    if len(experts) != 96 or payload_sizes != {EXPERT_PAYLOAD_BYTES} or component_counts != {2}:
        raise RuntimeError("real Switch payload manifest differs from frozen protocol")
    return {
        "format": "moe-cache-lab.switch-payload-manifest/v1",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "expert_count": len(experts),
        "component_names": list(COMPONENT_NAMES),
        "components_per_expert": 2,
        "expert_payload_bytes": EXPERT_PAYLOAD_BYTES,
        "experts": experts,
    }


def _trace_counts(trace: object) -> dict[str, Any]:
    counter = Counter(
        (event.routing_stage, event.phase, event.assignment_state)
        for event in trace.events
    )
    by_stage_phase = [
        {
            "routing_stage": stage,
            "phase": phase,
            "assigned": counter[(stage, phase, "assigned")],
            "unassigned": counter[(stage, phase, "unassigned")],
        }
        for stage, phase in (
            ("encoder", "source"),
            ("decoder", "decoder_prompt"),
            ("decoder", "decoder_generated"),
        )
    ]
    return {
        "event_count": len(trace.events),
        "expert_request_count": len(trace.expert_requests),
        "assigned_event_count": trace.assigned_event_count,
        "unassigned_event_count": trace.unassigned_event_count,
        "by_stage_phase": by_stage_phase,
    }


def run_predictor(attempt_dir: str | Path, prompt_path: str | Path) -> dict[str, Any]:
    """Collect a fresh trace, manifest real payloads, and freeze one simulation."""
    from .switch_collector import (
        DEFAULT_SWITCH_MODEL,
        SWITCH_MODEL_REVISION,
        SwitchTraceCollector,
    )
    import torch
    import transformers

    attempt = Path(attempt_dir)
    attempt.mkdir(parents=True, exist_ok=True)
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    prompt_hash = verify_prompt(prompt)
    if DEFAULT_SWITCH_MODEL != MODEL_ID or SWITCH_MODEL_REVISION != MODEL_REVISION:
        raise RuntimeError("collector constants differ from frozen experiment identity")
    if torch.__version__ != "2.12.0+cpu" or transformers.__version__ != "5.12.0":
        raise RuntimeError("installed runtime differs from frozen experiment runtime")
    torch.set_num_threads(4)
    if torch.get_num_threads() != 4:
        raise RuntimeError("predictor could not establish four Torch intra-op threads")

    collector = SwitchTraceCollector(local_files_only=True)
    outcome = collector.collect_and_verify(prompt, max_fed_back_non_eos_steps=4)
    trace_path = attempt / "routing-trace-v2.jsonl"
    write_trace_v2(trace_path, outcome.trace)
    round_trip = read_trace_v2(trace_path)
    if round_trip != outcome.trace:
        raise RuntimeError("fresh trace failed exact serialization round trip")
    trace_hash = sha256_file(trace_path)

    manifest = build_real_payload_manifest(collector)
    manifest_path = attempt / "payload-manifest.json"
    write_canonical_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    sizes = {key_from_json(item["key"]): item["expert_payload_bytes"] for item in manifest["experts"]}

    # Frozen protocol: exactly one call to the canonical simulation.
    simulation = simulate_versioned_byte_cache(
        round_trip,
        capacity_bytes=CAPACITY_BYTES,
        expert_sizes_bytes=sizes,
        policy_name="lru",
    )
    prediction = {
        "format": PREDICTION_FORMAT,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "cpu",
        "dtype": "float32",
        "prompt_sha256": prompt_hash,
        "trace_sha256": trace_hash,
        "payload_manifest_sha256": manifest_hash,
        "policy": simulation.policy,
        "capacity_bytes": simulation.capacity_bytes,
        "event_count": simulation.event_count,
        "expert_request_count": simulation.expert_request_count,
        "hits": simulation.hits,
        "misses": simulation.misses,
        "simulated_demand_load_bytes": simulation.simulated_demand_load_bytes,
        "eviction_count": simulation.eviction_count,
        "simulated_evicted_bytes": simulation.simulated_evicted_bytes,
        "peak_resident_bytes": simulation.peak_resident_bytes,
        "final_resident_bytes": simulation.final_resident_bytes,
        "final_resident_keys": [key_to_json(key) for key in simulation.final_resident_keys],
        "trace_counts": _trace_counts(round_trip),
    }
    prediction_path = attempt / "frozen-prediction.json"
    write_canonical_json(prediction_path, prediction)
    result = {
        "trace_path": trace_path.name,
        "trace_sha256": trace_hash,
        "payload_manifest_path": manifest_path.name,
        "payload_manifest_sha256": manifest_hash,
        "prediction_path": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
    }
    write_canonical_json(attempt / "predictor-result.json", result)
    return result


def _comparison_checks(
    prediction: Mapping[str, Any],
    observation: Mapping[str, Any],
    payload_manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def exact(name: str, predicted: Any, observed: Any) -> None:
        checks.append(
            {"name": name, "passed": predicted == observed, "predicted": predicted, "observed": observed}
        )

    for field in ("hits", "misses", "eviction_count", "final_resident_keys"):
        exact(field, prediction[field], observation[field])
    exact("completed_loads_equal_misses", prediction["misses"], observation["completed_physical_logical_loads"])
    exact(
        "source_operand_bytes",
        prediction["simulated_demand_load_bytes"],
        observation["actual_source_operand_logical_bytes"],
    )
    exact(
        "profiler_copy_count",
        observation["actual_component_copy_count"],
        observation["profiler_aten_copy_count"],
    )
    exact("copy_content_equal", True, observation["all_copy_content_equal"])
    copy_records = observation.get("copy_records", [])
    loaded_key_bytes = observation.get("loaded_key_bytes", [])
    exact("copy_record_count", observation["actual_component_copy_count"], len(copy_records))
    exact(
        "copy_operand_metadata_equal",
        True,
        all(
            record["source_shape"] == record["destination_shape"]
            and record["source_dtype"] == record["destination_dtype"]
            and record["source_device"] == record["destination_device"]
            and record["source_logical_bytes"] == record["destination_logical_bytes"]
            for record in copy_records
        ),
    )
    exact("loaded_key_record_count", observation["completed_physical_logical_loads"], len(loaded_key_bytes))
    if payload_manifest is not None:
        manifest_experts = {
            tuple(item["key"]): item for item in payload_manifest["experts"]
        }
        exact(
            "loaded_key_bytes_match_manifest",
            True,
            all(
                record["copied_source_bytes"]
                == manifest_experts[tuple(record["key"])]["expert_payload_bytes"]
                and record["component_count"]
                == len(manifest_experts[tuple(record["key"])]["components"])
                for record in loaded_key_bytes
            ),
        )
        component_specs = {
            (tuple(expert["key"]), component["name"]): component
            for expert in payload_manifest["experts"]
            for component in expert["components"]
        }
        exact(
            "copy_source_metadata_matches_manifest",
            True,
            all(
                record["source_shape"] == component_specs[(tuple(record["key"]), record["component_name"])]["shape"]
                and record["source_dtype"] == component_specs[(tuple(record["key"]), record["component_name"])]["dtype"]
                and record["source_device"] == component_specs[(tuple(record["key"]), record["component_name"])]["device"]
                and record["source_logical_bytes"]
                == component_specs[(tuple(record["key"]), record["component_name"])]["logical_payload_bytes"]
                for record in copy_records
            ),
        )
    for field in (
        "model_id",
        "revision",
        "torch_version",
        "transformers_version",
        "device",
        "dtype",
        "policy",
        "capacity_bytes",
        "trace_sha256",
        "payload_manifest_sha256",
        "event_count",
        "expert_request_count",
        "final_resident_bytes",
    ):
        exact(field, prediction[field], observation[field])
    return checks


def compare_prediction_observation(
    prediction: Mapping[str, Any],
    observation: Mapping[str, Any],
    payload_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply only exact post-execution equality checks; no tolerance or repair."""
    if prediction.get("format") != PREDICTION_FORMAT:
        raise ValueError("unsupported frozen prediction")
    if observation.get("format") != "moe-cache-lab.runtime-copy-observation/v1":
        raise ValueError("unsupported runtime observation")
    checks = _comparison_checks(prediction, observation, payload_manifest)
    return {
        "format": COMPARISON_FORMAT,
        "outcome": "AGREEMENT" if all(item["passed"] for item in checks) else "DISAGREEMENT",
        "checks": checks,
    }


def compare_attempt(attempt_dir: str | Path) -> dict[str, Any]:
    attempt = Path(attempt_dir)
    predictor_result = json.loads((attempt / "predictor-result.json").read_text(encoding="utf-8"))
    prediction_path = attempt / predictor_result["prediction_path"]
    if sha256_file(prediction_path) != predictor_result["prediction_sha256"]:
        raise RuntimeError("frozen prediction hash changed before comparison")
    observation_path = attempt / "runtime-observation.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    payload_manifest_path = attempt / predictor_result["payload_manifest_path"]
    if sha256_file(payload_manifest_path) != predictor_result["payload_manifest_sha256"]:
        raise RuntimeError("payload manifest hash changed before comparison")
    payload_manifest = json.loads(payload_manifest_path.read_text(encoding="utf-8"))
    result = compare_prediction_observation(prediction, observation, payload_manifest)
    result.update(
        {
            "prediction_sha256": predictor_result["prediction_sha256"],
            "observation_sha256": sha256_file(observation_path),
        }
    )
    write_canonical_json(attempt / "comparison.json", result)
    return result


def write_artifact_manifest(attempt_dir: str | Path) -> tuple[Path, str]:
    attempt = Path(attempt_dir)
    names = sorted(
        path.name
        for path in attempt.iterdir()
        if path.is_file()
        and path.name not in (
            "artifact-manifest.json",
            "run-result.json",
        )
    )
    records = [
        {"path": name, "size_bytes": (attempt / name).stat().st_size, "sha256": sha256_file(attempt / name)}
        for name in names
    ]
    payload = {"format": ARTIFACT_MANIFEST_FORMAT, "files": records}
    path = write_canonical_json(attempt / "artifact-manifest.json", payload)
    return path, sha256_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private Switch runtime-copy predictor")
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--prompt-file", required=True)
    arguments = parser.parse_args(argv)
    run_predictor(arguments.attempt_dir, arguments.prompt_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
