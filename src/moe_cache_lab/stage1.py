"""Frozen Stage 1 repetition collection, validation, and orchestration.

This module is intentionally separate from the V0.2 manifest workflow.  It
does not simulate or report cache policies and never changes measured traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
from uuid import UUID, uuid4

import torch
import transformers

from . import __version__
from .collector import (
    DEFAULT_MODEL,
    STAGE1_MAX_DECODE_INPUT_STEPS,
    GraniteTraceCollector,
    Stage1CollectionOutcome,
)
from .trace import RoutingEvent, RoutingTrace, read_trace, write_trace
from .views import (
    PREFILL_LAYER_UNION_ATOMIC,
    TOKEN_LAYER_ATOMIC,
    PromptTraceSource,
    adapt_prompt_traces,
    summarize_bundles,
)
from .workflow import (
    CORPUS_VERSION,
    CorpusDefinition,
    CorpusPrompt,
    file_sha256,
    load_corpus,
    load_suite_inputs,
    sha256_bytes,
)

STAGE1_REPETITION_FORMAT = "moe-cache-lab.stage1-repetition-manifest"
STAGE1_SET_FORMAT = "moe-cache-lab.stage1-set-manifest"
STAGE1_MANIFEST_VERSION = 1
STAGE1_REPETITIONS = 3
STAGE1_TORCH_THREADS = 4
STAGE1_DEADLINE_SECONDS = 900
STAGE1_MODEL_ID = DEFAULT_MODEL
STAGE1_MODEL_REVISION = "0da7a48b0276d500ce5922fd2b33944091fc6c09"
STAGE1_CORPUS_SHA256 = "67144988f37ae14134aa83ca4b447f5119e96597dec4ab7158fc1f2592e6be2c"
TRUSTED_V02_MANIFEST_SHA256 = "d5ca0013b1dfee96b353bb964666eaf9f6335494b0279362f536b4e8bc359e30"
STAGE1_ROUTED_LAYERS = tuple(range(24))
STAGE1_CAPTURE_METHOD = (
    "PyTorch forward hooks on Granite MoE router modules; greedy autoregressive "
    "forward calls for generated tokens"
)


@dataclass(frozen=True)
class Stage1PromptInput:
    prompt: CorpusPrompt
    record: dict[str, Any]
    trace: RoutingTrace


@dataclass(frozen=True)
class Stage1RepetitionInputs:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    corpus: CorpusDefinition
    prompts: tuple[Stage1PromptInput, ...]


@dataclass(frozen=True)
class Stage1SetInputs:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    repetitions: tuple[Stage1RepetitionInputs, ...]


def collect_stage1_repetition(
    corpus_path: str | Path,
    output_directory: str | Path,
    repetition_number: int,
    process_uuid: str,
) -> Path:
    """Private child operation: one cold collector, all twelve prompts."""

    if repetition_number not in range(1, STAGE1_REPETITIONS + 1):
        raise ValueError("Stage 1 repetition number must be 1, 2, or 3")
    _parse_uuid(process_uuid, "process UUID")
    corpus = _load_frozen_corpus(corpus_path)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    traces_directory = output / "traces"
    traces_directory.mkdir()
    shutil.copyfile(corpus.path, output / "corpus.json")
    torch.set_num_threads(STAGE1_TORCH_THREADS)
    started_wall = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    collector = GraniteTraceCollector(
        STAGE1_MODEL_ID,
        revision=STAGE1_MODEL_REVISION,
        local_files_only=True,
    )
    if collector.resolved_revision != STAGE1_MODEL_REVISION:
        raise ValueError("Stage 1 collector did not resolve the frozen model revision")

    prompt_records: list[dict[str, Any]] = []
    outcomes: list[tuple[CorpusPrompt, Stage1CollectionOutcome, str, str]] = []
    expected_layers: tuple[int, ...] | None = None
    for prompt in corpus.prompts:
        outcome = collector.collect_stage1(
            prompt.text,
            max_decode_input_steps=STAGE1_MAX_DECODE_INPUT_STEPS,
        )
        expected_layers = _validate_stage1_trace(
            outcome.trace,
            outcome.actual_decode_input_steps_routed,
            outcome.routed_non_eos_token_ids,
            expected_layers,
        )
        relative = Path("traces") / f"{prompt.order:02d}-{prompt.id}.jsonl"
        trace_path = _contained_path(output, relative, f"trace for {prompt.id}")
        write_trace(trace_path, outcome.trace)
        outcomes.append((prompt, outcome, relative.as_posix(), file_sha256(trace_path)))

    if expected_layers != STAGE1_ROUTED_LAYERS:
        raise ValueError("Stage 1 routed layers do not match the frozen Granite layer set")
    for prompt, outcome, relative_path, trace_hash in outcomes:
        prompt_records.append(_prompt_record(
            prompt, outcome, relative_path, trace_hash
        ))
    finished_wall = datetime.now(timezone.utc)
    elapsed = time.monotonic() - started_clock
    manifest = {
        "format": STAGE1_REPETITION_FORMAT,
        "format_version": STAGE1_MANIFEST_VERSION,
        "created_at": finished_wall.isoformat(),
        "repetition_number": repetition_number,
        "corpus": {
            "path": "corpus.json",
            "version": corpus.version,
            "canonical_json_sha256": corpus.sha256,
        },
        "model_id": STAGE1_MODEL_ID,
        "model_revision": {
            "requested": STAGE1_MODEL_REVISION,
            "resolved": collector.resolved_revision,
        },
        "routing": {
            "experts_per_layer": 32,
            "selected_experts_per_token": 8,
            "routed_layer_ids": list(expected_layers),
            "routed_layer_count": len(expected_layers),
            "capture_method": STAGE1_CAPTURE_METHOD,
        },
        "environment": {
            "moe_cache_lab": __version__,
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": "float32",
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "local_files_only": True,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "hf_hub_disable_telemetry": os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1",
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "generation": {
            "strategy": "greedy",
            "token_selection": "argmax",
            "do_sample": False,
            "use_cache": True,
            "protocol": "candidate-before-feed-eos-v1",
            "max_decode_input_steps": STAGE1_MAX_DECODE_INPUT_STEPS,
        },
        "process": {
            "uuid": process_uuid,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "started_at": started_wall.isoformat(),
            "finished_at": finished_wall.isoformat(),
            "elapsed_seconds": elapsed,
        },
        "prompts": prompt_records,
    }
    manifest_path = output / "stage1-repetition-manifest.json"
    _write_json(manifest_path, manifest)
    load_stage1_repetition_manifest(manifest_path)
    return manifest_path


def collect_stage1_set(
    corpus_path: str | Path,
    output_directory: str | Path,
    *,
    run_process: Callable[..., Any] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    trusted_v02_manifest: str | Path = Path("results/v0.2-corpus-v1/manifest-v1.json"),
) -> Path:
    """Run exactly three fresh sequential children under one 900-second deadline."""

    corpus = _load_frozen_corpus(corpus_path)
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Stage 1 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_uuid = str(uuid4())
    staging = output.parent / f".{output.name}.staging-{run_uuid}"
    staging.mkdir()
    shutil.copyfile(corpus.path, staging / "corpus.json")
    started_wall = datetime.now(timezone.utc)
    started_clock = monotonic()
    repetition_records: list[dict[str, Any]] = []
    environment = dict(os.environ)
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "OMP_NUM_THREADS": str(STAGE1_TORCH_THREADS),
        "MKL_NUM_THREADS": str(STAGE1_TORCH_THREADS),
    })
    for repetition_number in range(1, STAGE1_REPETITIONS + 1):
        remaining = STAGE1_DEADLINE_SECONDS - (monotonic() - started_clock)
        if remaining <= 0:
            raise TimeoutError("Stage 1 collection exceeded the shared 900-second deadline")
        child_uuid = str(uuid4())
        repetition_relative = Path(f"repetition-{repetition_number}")
        repetition_directory = staging / repetition_relative
        command = [
            sys.executable,
            "-m",
            "moe_cache_lab.cli",
            "_collect-stage1-child",
            "--corpus",
            str(corpus.path),
            "--output-dir",
            str(repetition_directory),
            "--repetition-number",
            str(repetition_number),
            "--process-uuid",
            child_uuid,
        ]
        try:
            completed = run_process(
                command,
                env=environment,
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                "Stage 1 collection exceeded the shared 900-second deadline"
            ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                f"Stage 1 child repetition {repetition_number} failed with exit code "
                f"{completed.returncode}"
            )
        repetition_manifest = repetition_directory / "stage1-repetition-manifest.json"
        loaded = load_stage1_repetition_manifest(repetition_manifest)
        if (
            loaded.manifest["repetition_number"] != repetition_number
            or loaded.manifest["process"]["uuid"] != child_uuid
        ):
            raise ValueError("Stage 1 child identity does not match its parent assignment")
        repetition_records.append({
            "repetition_number": repetition_number,
            "process_uuid": child_uuid,
            "process_pid": loaded.manifest["process"]["pid"],
            "manifest_path": (
                repetition_relative / "stage1-repetition-manifest.json"
            ).as_posix(),
            "manifest_sha256": loaded.manifest_sha256,
        })
    finished_wall = datetime.now(timezone.utc)
    set_manifest = {
        "format": STAGE1_SET_FORMAT,
        "format_version": STAGE1_MANIFEST_VERSION,
        "created_at": finished_wall.isoformat(),
        "run_uuid": run_uuid,
        "parent_pid": os.getpid(),
        "started_at": started_wall.isoformat(),
        "finished_at": finished_wall.isoformat(),
        "elapsed_seconds": monotonic() - started_clock,
        "deadline_seconds": STAGE1_DEADLINE_SECONDS,
        "repetition_count": STAGE1_REPETITIONS,
        "corpus": {
            "path": "corpus.json",
            "version": corpus.version,
            "canonical_json_sha256": corpus.sha256,
        },
        "model_id": STAGE1_MODEL_ID,
        "model_revision": STAGE1_MODEL_REVISION,
        "max_decode_input_steps": STAGE1_MAX_DECODE_INPUT_STEPS,
        "torch_num_threads": STAGE1_TORCH_THREADS,
        "repetitions": repetition_records,
    }
    set_path = staging / "stage1-set-manifest.json"
    _write_json(set_path, set_manifest)
    loaded_set = load_stage1_set_manifest(set_path)
    validate_v02_prefix(loaded_set.repetitions[0], trusted_v02_manifest)
    staging.rename(output)
    return output / "stage1-set-manifest.json"


def load_stage1_repetition_manifest(path: str | Path) -> Stage1RepetitionInputs:
    manifest_path = Path(path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("format") != STAGE1_REPETITION_FORMAT
        or manifest.get("format_version") != STAGE1_MANIFEST_VERSION
    ):
        raise ValueError("unsupported Stage 1 repetition manifest")
    _require_exact_keys(manifest, {
        "format", "format_version", "created_at", "repetition_number", "corpus",
        "model_id", "model_revision", "routing", "environment", "generation",
        "process", "prompts",
    }, "repetition manifest")
    created_at = _require_iso(manifest.get("created_at"), "created_at")
    number = manifest.get("repetition_number")
    if (
        isinstance(number, bool)
        or not isinstance(number, int)
        or number not in range(1, STAGE1_REPETITIONS + 1)
    ):
        raise ValueError("invalid Stage 1 repetition number")
    _validate_frozen_model_environment(manifest)
    corpus = _load_manifest_corpus(manifest_path, manifest)
    routing = manifest.get("routing")
    if routing != {
        "experts_per_layer": 32,
        "selected_experts_per_token": 8,
        "routed_layer_ids": list(STAGE1_ROUTED_LAYERS),
        "routed_layer_count": len(STAGE1_ROUTED_LAYERS),
        "capture_method": STAGE1_CAPTURE_METHOD,
    }:
        raise ValueError("Stage 1 routing metadata does not match the frozen model")
    process = manifest.get("process")
    if not isinstance(process, dict):
        raise ValueError("Stage 1 process metadata is missing")
    _require_exact_keys(process, {
        "uuid", "pid", "parent_pid", "started_at", "finished_at", "elapsed_seconds"
    }, "process")
    _parse_uuid(process.get("uuid"), "process UUID")
    for name in ("pid", "parent_pid"):
        if isinstance(process.get(name), bool) or not isinstance(process.get(name), int) or process[name] <= 0:
            raise ValueError(f"Stage 1 process {name} must be a positive integer")
    started_at = _require_iso(process.get("started_at"), "process started_at")
    finished_at = _require_iso(process.get("finished_at"), "process finished_at")
    _require_duration(process.get("elapsed_seconds"), "process elapsed_seconds")
    if not started_at <= finished_at == created_at:
        raise ValueError(
            "Stage 1 repetition timestamps must satisfy "
            "started_at <= finished_at == created_at"
        )
    records = manifest.get("prompts")
    if not isinstance(records, list) or len(records) != len(corpus.prompts):
        raise ValueError("Stage 1 prompt records do not match the frozen corpus")
    prompts: list[Stage1PromptInput] = []
    seen_paths: set[Path] = set()
    for prompt, record in zip(corpus.prompts, records):
        _validate_prompt_metadata(prompt, record)
        trace_path = _manifest_relative_path(
            manifest_path, record.get("trace_path"), f"trace for {prompt.id}"
        )
        if trace_path in seen_paths:
            raise ValueError("Stage 1 trace paths must be unique")
        seen_paths.add(trace_path)
        if file_sha256(trace_path) != record.get("trace_sha256"):
            raise ValueError(f"Stage 1 trace hash mismatch for {prompt.id}")
        trace = read_trace(trace_path)
        _validate_trace_metadata(trace, prompt, manifest)
        eos = _validate_eos_record(record.get("eos"), trace)
        _validate_stage1_trace(
            trace,
            eos["actual_decode_input_steps_routed"],
            tuple(eos["routed_non_eos_token_ids"]),
            STAGE1_ROUTED_LAYERS,
        )
        _validate_record_counts(prompt, record, trace)
        prompts.append(Stage1PromptInput(prompt, record, trace))
    eos_lists = {
        tuple(item.record["eos"]["normalized_eos_token_ids"]) for item in prompts
    }
    if len(eos_lists) != 1:
        raise ValueError("Stage 1 normalized EOS token IDs must match across all prompts")
    return Stage1RepetitionInputs(
        manifest_path,
        sha256_bytes(manifest_bytes),
        manifest,
        corpus,
        tuple(prompts),
    )


def load_stage1_set_manifest(path: str | Path) -> Stage1SetInputs:
    manifest_path = Path(path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("format") != STAGE1_SET_FORMAT
        or manifest.get("format_version") != STAGE1_MANIFEST_VERSION
    ):
        raise ValueError("unsupported Stage 1 set manifest")
    _require_exact_keys(manifest, {
        "format", "format_version", "created_at", "run_uuid", "parent_pid",
        "started_at", "finished_at", "elapsed_seconds", "deadline_seconds",
        "repetition_count", "corpus", "model_id", "model_revision",
        "max_decode_input_steps", "torch_num_threads", "repetitions",
    }, "set manifest")
    _parse_uuid(manifest.get("run_uuid"), "run UUID")
    created_at = _require_iso(manifest.get("created_at"), "created_at")
    started_at = _require_iso(manifest.get("started_at"), "started_at")
    finished_at = _require_iso(manifest.get("finished_at"), "finished_at")
    if (
        isinstance(manifest.get("parent_pid"), bool)
        or not isinstance(manifest.get("parent_pid"), int)
        or manifest["parent_pid"] <= 0
    ):
        raise ValueError("Stage 1 set parent PID or timing metadata is invalid")
    _require_duration(
        manifest.get("elapsed_seconds"),
        "set elapsed_seconds",
        maximum=STAGE1_DEADLINE_SECONDS,
    )
    if not started_at <= finished_at == created_at:
        raise ValueError(
            "Stage 1 set timestamps must satisfy "
            "started_at <= finished_at == created_at"
        )
    if (
        manifest.get("model_id") != STAGE1_MODEL_ID
        or manifest.get("model_revision") != STAGE1_MODEL_REVISION
        or manifest.get("max_decode_input_steps") != STAGE1_MAX_DECODE_INPUT_STEPS
        or manifest.get("torch_num_threads") != STAGE1_TORCH_THREADS
        or manifest.get("deadline_seconds") != STAGE1_DEADLINE_SECONDS
        or manifest.get("repetition_count") != STAGE1_REPETITIONS
    ):
        raise ValueError("Stage 1 set metadata does not match the frozen protocol")
    corpus = _load_manifest_corpus(manifest_path, manifest)
    records = manifest.get("repetitions")
    if not isinstance(records, list) or len(records) != STAGE1_REPETITIONS:
        raise ValueError("Stage 1 set must contain exactly three repetitions")
    loaded: list[Stage1RepetitionInputs] = []
    uuids: set[str] = set()
    paths: set[Path] = set()
    previous_finished_at: datetime | None = None
    for expected_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError("Stage 1 repetition set record must be an object")
        _require_exact_keys(record, {
            "repetition_number", "process_uuid", "process_pid", "manifest_path",
            "manifest_sha256",
        }, "repetition set record")
        if record.get("repetition_number") != expected_number:
            raise ValueError("Stage 1 repetitions must be ordered 1, 2, 3")
        process_uuid = record.get("process_uuid")
        _parse_uuid(process_uuid, "repetition process UUID")
        pid = record.get("process_pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("repetition process PID must be positive")
        child_path = _manifest_relative_path(
            manifest_path, record.get("manifest_path"), "repetition manifest"
        )
        if process_uuid in uuids or child_path in paths:
            raise ValueError("Stage 1 repetitions require unique process UUIDs and paths")
        uuids.add(process_uuid)
        paths.add(child_path)
        if file_sha256(child_path) != record.get("manifest_sha256"):
            raise ValueError("Stage 1 repetition manifest hash mismatch")
        repetition = load_stage1_repetition_manifest(child_path)
        # On Windows, a virtual-environment python.exe redirector may be the
        # immediate observed parent instead of this orchestrator process.
        # Child parent_pid remains validated positive informational metadata;
        # freshness is established by the launched records, UUIDs, child PIDs,
        # paths, and ordered non-overlapping intervals below.
        if (
            repetition.manifest["repetition_number"] != expected_number
            or repetition.manifest["process"]["uuid"] != process_uuid
            or repetition.manifest["process"]["pid"] != pid
            or repetition.corpus.sha256 != corpus.sha256
        ):
            raise ValueError("Stage 1 repetition identity or corpus mismatch")
        child_started_at = _require_iso(
            repetition.manifest["process"]["started_at"],
            "repetition process started_at",
        )
        child_finished_at = _require_iso(
            repetition.manifest["process"]["finished_at"],
            "repetition process finished_at",
        )
        if not started_at <= child_started_at <= child_finished_at <= finished_at:
            raise ValueError("Stage 1 repetition interval must be contained in the parent interval")
        if previous_finished_at is not None and child_started_at < previous_finished_at:
            raise ValueError("Stage 1 repetition intervals must be ordered and non-overlapping")
        previous_finished_at = child_finished_at
        loaded.append(repetition)
    environments = [item.manifest["environment"] for item in loaded]
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("Stage 1 repetition environments must match exactly")
    compare_stage1_repetitions(tuple(loaded))
    return Stage1SetInputs(
        manifest_path,
        sha256_bytes(manifest_bytes),
        manifest,
        tuple(loaded),
    )


def validate_stage1_set(
    path: str | Path,
    trusted_v02_manifest: str | Path = Path("results/v0.2-corpus-v1/manifest-v1.json"),
) -> Stage1SetInputs:
    loaded = load_stage1_set_manifest(path)
    validate_v02_prefix(loaded.repetitions[0], trusted_v02_manifest)
    return loaded


def compare_stage1_repetitions(
    repetitions: tuple[Stage1RepetitionInputs, ...],
    probability_tolerance: float = 1e-6,
) -> None:
    if len(repetitions) != STAGE1_REPETITIONS:
        raise ValueError("semantic comparison requires exactly three repetitions")
    if not math.isfinite(probability_tolerance) or probability_tolerance < 0:
        raise ValueError("probability tolerance must be finite and non-negative")
    baseline = repetitions[0]
    for candidate in repetitions[1:]:
        for left, right in zip(baseline.prompts, candidate.prompts):
            left_eos = left.record["eos"]
            right_eos = right.record["eos"]
            semantic_fields = (
                "max_decode_input_steps_requested",
                "actual_decode_input_steps_routed",
                "routed_non_eos_token_ids",
                "emitted_non_eos_token_ids",
                "emitted_non_eos_text",
                "normalized_eos_token_ids",
                "terminal_eos_token_id",
                "terminal_eos_candidate_position",
                "terminal_eos_text",
                "eos_emitted",
                "horizon_exhausted",
            )
            if any(left_eos[field] != right_eos[field] for field in semantic_fields):
                raise ValueError(f"Stage 1 EOS/token repeatability mismatch for {left.prompt.id}")
            if len(left.trace.events) != len(right.trace.events):
                raise ValueError(f"Stage 1 event-count repeatability mismatch for {left.prompt.id}")
            for left_event, right_event in zip(left.trace.events, right.trace.events):
                if _event_identity(left_event) != _event_identity(right_event):
                    raise ValueError(f"Stage 1 routing repeatability mismatch for {left.prompt.id}")
                if len(left_event.selected_probabilities) != len(right_event.selected_probabilities):
                    raise ValueError(f"Stage 1 probability-count mismatch for {left.prompt.id}")
                for left_value, right_value in zip(
                    left_event.selected_probabilities, right_event.selected_probabilities
                ):
                    if (
                        not math.isfinite(left_value)
                        or not math.isfinite(right_value)
                        or abs(left_value - right_value) > probability_tolerance
                    ):
                        raise ValueError(f"Stage 1 probability repeatability mismatch for {left.prompt.id}")


def validate_v02_prefix(
    repetition: Stage1RepetitionInputs,
    trusted_manifest_path: str | Path,
) -> None:
    trusted_path = Path(trusted_manifest_path).resolve()
    if file_sha256(trusted_path) != TRUSTED_V02_MANIFEST_SHA256:
        raise ValueError("trusted V0.2 manifest hash mismatch")
    trusted = load_suite_inputs(trusted_path)
    trusted_prompts = {
        item.prompt.id: item for item in (*trusted.calibration, *trusted.evaluation)
    }
    for prompt in repetition.prompts:
        old = trusted_prompts[prompt.prompt.id]
        current_prompt = tuple(event for event in prompt.trace.events if event.phase == "prompt")
        old_prompt = tuple(event for event in old.trace.events if event.phase == "prompt")
        if tuple(map(_prefix_identity, current_prompt)) != tuple(map(_prefix_identity, old_prompt)):
            raise ValueError(f"Stage 1 prompt prefix mismatch for {prompt.prompt.id}")
        actual = prompt.record["eos"]["actual_decode_input_steps_routed"]
        prefix_steps = min(actual, 2)
        prompt_length = len({event.token_position for event in current_prompt})
        current_decode = tuple(
            event for event in prompt.trace.events
            if event.phase == "generated" and event.token_position < prompt_length + prefix_steps
        )
        old_decode = tuple(
            event for event in old.trace.events
            if event.phase == "generated" and event.token_position < prompt_length + prefix_steps
        )
        if tuple(map(_prefix_identity, current_decode)) != tuple(map(_prefix_identity, old_decode)):
            raise ValueError(f"Stage 1 decode prefix mismatch for {prompt.prompt.id}")


def _prompt_record(
    prompt: CorpusPrompt,
    outcome: Stage1CollectionOutcome,
    relative_path: str,
    trace_hash: str,
) -> dict[str, Any]:
    source = PromptTraceSource(prompt.id, prompt.order, outcome.trace)
    token_summary = summarize_bundles(adapt_prompt_traces((source,), TOKEN_LAYER_ATOMIC))
    grouped_summary = summarize_bundles(adapt_prompt_traces(
        (source,), PREFILL_LAYER_UNION_ATOMIC
    ))
    return {
        "id": prompt.id,
        "category": prompt.category,
        "split": prompt.split,
        "order": prompt.order,
        "prompt_sha256": sha256_bytes(prompt.text.encode("utf-8")),
        "trace_path": relative_path,
        "trace_sha256": trace_hash,
        "eos": {
            "max_decode_input_steps_requested": outcome.max_decode_input_steps_requested,
            "actual_decode_input_steps_routed": outcome.actual_decode_input_steps_routed,
            "routed_non_eos_token_ids": list(outcome.routed_non_eos_token_ids),
            "emitted_non_eos_token_ids": list(outcome.emitted_non_eos_token_ids),
            "emitted_non_eos_text": outcome.emitted_non_eos_text,
            "normalized_eos_token_ids": list(outcome.normalized_eos_token_ids),
            "terminal_eos_token_id": outcome.terminal_eos_token_id,
            "terminal_eos_candidate_position": outcome.terminal_eos_candidate_position,
            "terminal_eos_text": outcome.terminal_eos_text,
            "eos_emitted": outcome.eos_emitted,
            "horizon_exhausted": outcome.horizon_exhausted,
        },
        "measured": {
            "routing_event_count": len(outcome.trace.events),
            "expert_request_count": len(outcome.trace.expert_requests),
            "prompt_event_count": sum(event.phase == "prompt" for event in outcome.trace.events),
            "generated_event_count": sum(event.phase == "generated" for event in outcome.trace.events),
        },
        "views": {
            TOKEN_LAYER_ATOMIC: _summary_record(token_summary),
            PREFILL_LAYER_UNION_ATOMIC: _summary_record(grouped_summary),
        },
    }


def _summary_record(summary: Any) -> dict[str, Any]:
    def scope(value: Any) -> dict[str, int]:
        return {
            "bundles": value.bundles,
            "requests": value.requests,
            "source_events": value.source_events,
            "source_assignments": value.source_assignments,
            "max_bundle_size": value.max_bundle_size,
        }
    return {
        "combined": scope(summary.combined),
        "prompt": scope(summary.prompt),
        "generated": scope(summary.generated),
    }


def _validate_stage1_trace(
    trace: RoutingTrace,
    actual_steps: int,
    routed_ids: tuple[int, ...],
    expected_layers: tuple[int, ...] | None,
) -> tuple[int, ...]:
    prompt_events = tuple(event for event in trace.events if event.phase == "prompt")
    if not prompt_events:
        raise ValueError("Stage 1 traces require a nonempty prompt phase")
    prompt_positions = sorted({event.token_position for event in prompt_events})
    if prompt_positions != list(range(len(prompt_positions))):
        raise ValueError("Stage 1 prompt positions must be contiguous from zero")
    generated = tuple(event for event in trace.events if event.phase == "generated")
    generated_positions = sorted({event.token_position for event in generated})
    expected_generated = list(range(len(prompt_positions), len(prompt_positions) + actual_steps))
    if generated_positions != expected_generated:
        raise ValueError("Stage 1 generated positions do not reconcile with actual routed steps")
    if len(routed_ids) != actual_steps:
        raise ValueError("Stage 1 routed token count does not match actual steps")
    if any(
        len(event.selected_probabilities) != 8
        or any(not math.isfinite(value) for value in event.selected_probabilities)
        for event in trace.events
    ):
        raise ValueError("Stage 1 selected probabilities must contain eight finite values")
    layers = tuple(sorted({event.layer for event in trace.events}))
    if expected_layers is not None and layers != expected_layers:
        raise ValueError("Stage 1 routed-layer set is inconsistent")
    for phase, positions in (("prompt", prompt_positions), ("generated", generated_positions)):
        for position in positions:
            events = tuple(
                event for event in trace.events
                if event.phase == phase and event.token_position == position
            )
            if tuple(sorted(event.layer for event in events)) != layers:
                raise ValueError("Stage 1 token position lacks the complete routed-layer set")
            token_ids = {event.token_id for event in events}
            if len(token_ids) != 1 or None in token_ids:
                raise ValueError("Stage 1 event token IDs must agree across routed layers")
            if phase == "generated":
                offset = position - len(prompt_positions)
                if token_ids != {routed_ids[offset]}:
                    raise ValueError("Stage 1 generated event token ID does not match routed IDs")
    return layers


def _validate_frozen_model_environment(manifest: dict[str, Any]) -> None:
    if manifest.get("model_id") != STAGE1_MODEL_ID or manifest.get("model_revision") != {
        "requested": STAGE1_MODEL_REVISION,
        "resolved": STAGE1_MODEL_REVISION,
    }:
        raise ValueError("Stage 1 model/revision metadata does not match the frozen protocol")
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("Stage 1 environment metadata is missing")
    _require_exact_keys(environment, {
        "moe_cache_lab", "python", "transformers", "torch", "dtype", "device",
        "torch_num_threads", "local_files_only", "hf_hub_offline",
        "transformers_offline", "hf_hub_disable_telemetry", "omp_num_threads",
        "mkl_num_threads",
    }, "environment")
    for name in ("moe_cache_lab", "python", "transformers", "torch"):
        if not isinstance(environment.get(name), str) or not environment[name]:
            raise ValueError(f"Stage 1 environment is missing {name}")
    if (
        environment.get("dtype") != "float32"
        or environment.get("device") != "cpu"
        or environment.get("torch_num_threads") != STAGE1_TORCH_THREADS
        or environment.get("local_files_only") is not True
        or environment.get("hf_hub_offline") is not True
        or environment.get("transformers_offline") is not True
        or environment.get("hf_hub_disable_telemetry") is not True
        or environment.get("omp_num_threads") != str(STAGE1_TORCH_THREADS)
        or environment.get("mkl_num_threads") != str(STAGE1_TORCH_THREADS)
    ):
        raise ValueError("Stage 1 CPU/thread/offline environment metadata is invalid")
    generation = manifest.get("generation")
    if generation != {
        "strategy": "greedy",
        "token_selection": "argmax",
        "do_sample": False,
        "use_cache": True,
        "protocol": "candidate-before-feed-eos-v1",
        "max_decode_input_steps": STAGE1_MAX_DECODE_INPUT_STEPS,
    }:
        raise ValueError("Stage 1 generation protocol metadata is invalid")


def _load_frozen_corpus(path: str | Path) -> CorpusDefinition:
    corpus = load_corpus(path)
    if corpus.version != CORPUS_VERSION or corpus.sha256 != STAGE1_CORPUS_SHA256:
        raise ValueError("Stage 1 requires the exact frozen corpus v1 content")
    return corpus


def _load_manifest_corpus(
    manifest_path: Path, manifest: dict[str, Any]
) -> CorpusDefinition:
    record = manifest.get("corpus")
    if not isinstance(record, dict):
        raise ValueError("Stage 1 corpus metadata is missing")
    corpus_path = _manifest_relative_path(manifest_path, record.get("path"), "corpus")
    corpus = _load_frozen_corpus(corpus_path)
    if record != {
        "path": record.get("path"),
        "version": corpus.version,
        "canonical_json_sha256": corpus.sha256,
    }:
        raise ValueError("Stage 1 corpus metadata is inconsistent")
    return corpus


def _validate_prompt_metadata(prompt: CorpusPrompt, record: Any) -> None:
    if not isinstance(record, dict):
        raise ValueError("Stage 1 prompt record must be an object")
    _require_exact_keys(record, {
        "id", "category", "split", "order", "prompt_sha256", "trace_path",
        "trace_sha256", "eos", "measured", "views",
    }, "prompt record")
    if (
        record.get("id"), record.get("category"), record.get("split"), record.get("order")
    ) != (prompt.id, prompt.category, prompt.split, prompt.order):
        raise ValueError("Stage 1 prompt metadata/order mismatch")
    if record.get("prompt_sha256") != sha256_bytes(prompt.text.encode("utf-8")):
        raise ValueError(f"Stage 1 prompt hash mismatch for {prompt.id}")


def _validate_trace_metadata(
    trace: RoutingTrace,
    prompt: CorpusPrompt,
    manifest: dict[str, Any],
) -> None:
    if (
        trace.model_id != STAGE1_MODEL_ID
        or trace.model_revision != STAGE1_MODEL_REVISION
        or trace.transformers_version != manifest["environment"]["transformers"]
        or trace.source_text != prompt.text
        or trace.num_experts != 32
        or trace.experts_per_token != 8
        or trace.capture_method != STAGE1_CAPTURE_METHOD
    ):
        raise ValueError(f"Stage 1 trace metadata mismatch for {prompt.id}")


def _validate_eos_record(record: Any, trace: RoutingTrace) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Stage 1 EOS metadata is missing")
    required = {
        "max_decode_input_steps_requested",
        "actual_decode_input_steps_routed",
        "routed_non_eos_token_ids",
        "emitted_non_eos_token_ids",
        "emitted_non_eos_text",
        "normalized_eos_token_ids",
        "terminal_eos_token_id",
        "terminal_eos_candidate_position",
        "terminal_eos_text",
        "eos_emitted",
        "horizon_exhausted",
    }
    if set(record) != required:
        raise ValueError("Stage 1 EOS metadata fields are invalid")
    actual = record.get("actual_decode_input_steps_routed")
    routed = record.get("routed_non_eos_token_ids")
    emitted = record.get("emitted_non_eos_token_ids")
    eos_ids = record.get("normalized_eos_token_ids")
    if (
        record.get("max_decode_input_steps_requested") != STAGE1_MAX_DECODE_INPUT_STEPS
        or isinstance(actual, bool)
        or not isinstance(actual, int)
        or not 0 <= actual <= STAGE1_MAX_DECODE_INPUT_STEPS
        or not isinstance(routed, list)
        or emitted != routed
        or len(routed) != actual
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in routed)
        or not isinstance(eos_ids, list)
        or eos_ids != sorted(set(eos_ids))
        or not eos_ids
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in eos_ids)
        or any(token in eos_ids for token in routed)
        or not isinstance(record.get("emitted_non_eos_text"), str)
        or trace.generated_text != record.get("emitted_non_eos_text")
    ):
        raise ValueError("Stage 1 EOS/token accounting is invalid")
    eos_emitted = record.get("eos_emitted")
    horizon = record.get("horizon_exhausted")
    if eos_emitted is True and horizon is False:
        if (
            record.get("terminal_eos_token_id") not in eos_ids
            or record.get("terminal_eos_candidate_position") != actual
            or not isinstance(record.get("terminal_eos_text"), str)
        ):
            raise ValueError("Stage 1 terminal EOS metadata is invalid")
    elif eos_emitted is False and horizon is True:
        if (
            actual != STAGE1_MAX_DECODE_INPUT_STEPS
            or record.get("terminal_eos_token_id") is not None
            or record.get("terminal_eos_candidate_position") is not None
            or record.get("terminal_eos_text") is not None
        ):
            raise ValueError("Stage 1 horizon metadata is invalid")
    else:
        raise ValueError("Stage 1 EOS/horizon flags are invalid")
    return record


def _validate_record_counts(
    prompt: CorpusPrompt, record: dict[str, Any], trace: RoutingTrace
) -> None:
    measured = record.get("measured")
    expected_measured = {
        "routing_event_count": len(trace.events),
        "expert_request_count": len(trace.expert_requests),
        "prompt_event_count": sum(event.phase == "prompt" for event in trace.events),
        "generated_event_count": sum(event.phase == "generated" for event in trace.events),
    }
    if measured != expected_measured:
        raise ValueError(f"Stage 1 measured counts mismatch for {prompt.id}")
    source = PromptTraceSource(prompt.id, prompt.order, trace)
    expected_views = {
        TOKEN_LAYER_ATOMIC: _summary_record(summarize_bundles(
            adapt_prompt_traces((source,), TOKEN_LAYER_ATOMIC)
        )),
        PREFILL_LAYER_UNION_ATOMIC: _summary_record(summarize_bundles(
            adapt_prompt_traces((source,), PREFILL_LAYER_UNION_ATOMIC)
        )),
    }
    if record.get("views") != expected_views:
        raise ValueError(f"Stage 1 derived-view counts mismatch for {prompt.id}")


def _event_identity(event: RoutingEvent) -> tuple[Any, ...]:
    return (
        event.phase,
        event.token_position,
        event.layer,
        event.token_id,
        event.selected_experts,
    )


def _prefix_identity(event: RoutingEvent) -> tuple[Any, ...]:
    return _event_identity(event)


def _manifest_relative_path(
    manifest_path: Path, relative: Any, label: str
) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Stage 1 {label} path must be relative")
    return _contained_path(manifest_path.parent, Path(relative), label)


def _contained_path(base: Path, relative: Path, label: str) -> Path:
    destination = (base / relative).resolve()
    try:
        destination.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(f"Stage 1 {label} path escapes its manifest directory") from error
    return destination


def _parse_uuid(value: Any, label: str) -> UUID:
    if not isinstance(value, str):
        raise ValueError(f"Stage 1 {label} must be a UUID string")
    try:
        return UUID(value)
    except ValueError as error:
        raise ValueError(f"Stage 1 {label} must be a valid UUID") from error


def _require_iso(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Stage 1 {label} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Stage 1 {label} must be a timezone-aware ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Stage 1 {label} must be a timezone-aware ISO timestamp")
    return parsed


def _require_duration(value: Any, label: str, *, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        if maximum is None:
            requirement = "finite and non-negative"
        else:
            requirement = f"finite and between zero and {maximum:g} inclusive"
        raise ValueError(f"Stage 1 {label} must be {requirement}")
    return float(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"Stage 1 {label} fields are invalid")
