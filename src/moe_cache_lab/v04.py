"""Frozen V0.4 CPU trace-profiler overhead workflow.

The public helpers are deliberately dependency-injected so unit tests never
load the model, wait for PDH, or launch child processes.  Real collection is
offline/local-only and remains gated by independent review.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Protocol
from uuid import UUID, uuid4

import torch
import transformers

from . import __version__
from .v04_aggregation import (
    EVALUATION_IDS,
    PROFILED_MODE,
    REFERENCE_MODE,
    V04_CHILD_COUNT,
    V04_FORMAT_VERSION,
    V04_RESULT_FORMAT,
    _child_number,
    _cv,
    aggregate_children,
    evaluation_schedule,
)
from .collector import (
    GraniteTraceCollector,
    _RouterHookCapture,
    _events_from_router_logits,
    _normalize_eos_token_ids,
)
from .stage1 import (
    STAGE1_CORPUS_SHA256,
    STAGE1_MAX_DECODE_INPUT_STEPS,
    STAGE1_MODEL_ID,
    STAGE1_MODEL_REVISION,
    STAGE1_TORCH_THREADS,
    load_stage1_repetition_manifest,
)
from .trace import RoutingEvent
from .workflow import CorpusPrompt, file_sha256, load_corpus, sha256_bytes

V04_DEADLINE_SECONDS = 1200
V04_CHILD_FORMAT = "moe-cache-lab.v04-child"
V04_SET_FORMAT = "moe-cache-lab.v04-set"
V04_MODES = (REFERENCE_MODE, PROFILED_MODE)
CPU_COUNTER_PATH = r"\Processor(_Total)\% Processor Time"
DISK_COUNTER_PATH = r"\PhysicalDisk(_Total)\% Disk Time"
ADMISSION_SAMPLE_COUNT = 15
ADMISSION_INTERVAL_SECONDS = 1.0
MIN_AVAILABLE_BYTES = 8_589_934_592
ABORT_AVAILABLE_BYTES = 2 * 1024**3
TRUSTED_STAGE1_SET_SHA256 = "f7e624d95368b1aa976f3e31412fc44e86dbf46439e5c15534032c14446e34d1"
TRUSTED_STAGE1_REP1_SHA256 = "c752c0e3128786bb6ae40c3cdf2d67b6510b4fb369053be550b4cafdca85b94c"
V04_SPEC_SHA256 = "0540f58fb6267a041e3ef235e14798dc6c7896d58011aa0564ae1550d485ed2f"
EXPECTED_PARAMETER_COUNT = 1_334_628_352
TRUSTED_STAGE1_SET = Path("results/stage1-runtime-fidelity-v1/stage1-set-manifest.json")
TRUSTED_STAGE1_REP1 = Path(
    "results/stage1-runtime-fidelity-v1/repetition-1/stage1-repetition-manifest.json"
)
CALIBRATION_IDS = (
    "cal-factual-01", "cal-coding-01", "cal-math-01", "cal-conversation-01",
)


@dataclass(frozen=True)
class ChildProcessOutcome:
    pid: int
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    started_at: str
    finished_at: str


def run_child_process(
    command: list[str], *, env: dict[str, str], timeout: float,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> ChildProcessOutcome:
    """Run one child while retaining its launch PID and timeout evidence."""
    started = datetime.now(timezone.utc)
    process = popen_factory(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    pid = int(process.pid)
    if pid <= 0:
        raise RuntimeError("V0.4 subprocess runner received an invalid PID")
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        process.kill()
        tail_stdout, tail_stderr = process.communicate()
        stdout = _output_text(error.stdout) + _output_text(tail_stdout)
        stderr = _output_text(error.stderr) + _output_text(tail_stderr)
    return ChildProcessOutcome(
        pid, None if timed_out else process.returncode, _output_text(stdout), _output_text(stderr),
        timed_out, started.isoformat(), datetime.now(timezone.utc).isoformat(),
    )


def calibration_schedule(child_number: int) -> tuple[tuple[str, str], ...]:
    """Return all-C/all-mode warmup ordering frozen by the specification."""
    _child_number(child_number)
    modes = V04_MODES if child_number in (1, 3) else tuple(reversed(V04_MODES))
    return tuple((prompt_id, mode) for mode in modes for prompt_id in CALIBRATION_IDS)


@dataclass(frozen=True)
class NestedTiming:
    primary_total_ns: int
    hook_setup_ns: int
    prefill_forward_ns: int
    decode_forward_ns: tuple[int, ...]
    event_extraction_ns: tuple[int, ...]
    hook_removal_ns: int
    lifecycle_assertion_ns: int
    accounted_ns: int
    unaccounted_ns: int

    def __post_init__(self) -> None:
        values = (
            self.primary_total_ns, self.hook_setup_ns, self.prefill_forward_ns,
            *self.decode_forward_ns, *self.event_extraction_ns,
            self.hook_removal_ns, self.lifecycle_assertion_ns,
            self.accounted_ns, self.unaccounted_ns,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("V0.4 timer values must be nonnegative integer nanoseconds")
        component_sum = (
            self.hook_setup_ns + self.prefill_forward_ns + sum(self.decode_forward_ns)
            + sum(self.event_extraction_ns) + self.hook_removal_ns
            + self.lifecycle_assertion_ns
        )
        if self.accounted_ns != component_sum or self.primary_total_ns != self.accounted_ns + self.unaccounted_ns:
            raise ValueError("V0.4 nested timer accounting is inconsistent")


@dataclass(frozen=True)
class SemanticRecord:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    candidate_token_ids: tuple[int, ...]
    candidate_texts: tuple[str, ...]
    routed_non_eos_token_ids: tuple[int, ...]
    routed_non_eos_text: str
    terminal_eos_token_id: int | None
    terminal_eos_text: str | None
    terminal_eos_candidate_position: int | None
    actual_decode_input_steps: int
    eos_emitted: bool
    horizon_exhausted: bool
    validation_only_horizon_candidate_id: int | None
    validation_only_horizon_candidate_text: str | None

    def __post_init__(self) -> None:
        if isinstance(self.actual_decode_input_steps, bool) or not isinstance(self.actual_decode_input_steps, int) or not 0 <= self.actual_decode_input_steps <= STAGE1_MAX_DECODE_INPUT_STEPS:
            raise ValueError("V0.4 actual decode-input steps must be an integer from zero through sixteen")
        integer_sequences = (
            self.input_ids, self.attention_mask, self.candidate_token_ids,
            self.routed_non_eos_token_ids,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for sequence in integer_sequences for value in sequence
        ):
            raise ValueError("V0.4 semantic token/mask fields require nonnegative integers")
        if not self.input_ids or len(self.input_ids) != len(self.attention_mask):
            raise ValueError("V0.4 input IDs/mask must be nonempty and aligned")
        if any(value not in (0, 1) for value in self.attention_mask):
            raise ValueError("V0.4 attention mask must contain only zero/one integers")
        if any(not isinstance(value, str) for value in self.candidate_texts) or not isinstance(self.routed_non_eos_text, str):
            raise ValueError("V0.4 semantic text fields must be strings")
        if type(self.eos_emitted) is not bool or type(self.horizon_exhausted) is not bool:
            raise ValueError("V0.4 EOS/horizon flags must be booleans")
        for value in (self.terminal_eos_token_id, self.terminal_eos_candidate_position, self.validation_only_horizon_candidate_id):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError("V0.4 optional semantic IDs/positions are invalid")
        for value in (self.terminal_eos_text, self.validation_only_horizon_candidate_text):
            if value is not None and not isinstance(value, str):
                raise ValueError("V0.4 optional semantic text fields are invalid")
        if len(self.candidate_token_ids) != len(self.candidate_texts):
            raise ValueError("V0.4 candidate IDs/texts do not reconcile")
        if self.actual_decode_input_steps != len(self.routed_non_eos_token_ids):
            raise ValueError("V0.4 routed token count does not reconcile")
        if self.eos_emitted == self.horizon_exhausted:
            raise ValueError("V0.4 requires exactly one EOS/horizon state")
        if self.horizon_exhausted:
            if self.actual_decode_input_steps != STAGE1_MAX_DECODE_INPUT_STEPS:
                raise ValueError("V0.4 horizon requires exactly sixteen routed steps")
            if self.validation_only_horizon_candidate_id != self.candidate_token_ids[-1]:
                raise ValueError("V0.4 horizon candidate must be retained for validation")
            if self.validation_only_horizon_candidate_text != self.candidate_texts[-1]:
                raise ValueError("V0.4 horizon candidate text must be retained")
            if self.terminal_eos_token_id is not None or self.terminal_eos_text is not None or self.terminal_eos_candidate_position is not None:
                raise ValueError("V0.4 horizon result cannot have terminal EOS metadata")
        elif self.validation_only_horizon_candidate_id is not None:
            raise ValueError("V0.4 EOS result cannot have a horizon candidate")
        elif (
            self.terminal_eos_token_id != self.candidate_token_ids[-1]
            or self.terminal_eos_text != self.candidate_texts[-1]
            or self.terminal_eos_candidate_position != self.actual_decode_input_steps
            or self.validation_only_horizon_candidate_text is not None
        ):
            raise ValueError("V0.4 terminal EOS metadata is inconsistent")
        if self.candidate_token_ids[:-1] != self.routed_non_eos_token_ids:
            raise ValueError("V0.4 candidates before terminal/horizon must equal routed tokens")


@dataclass(frozen=True)
class PromptRun:
    prompt_id: str
    mode: str
    semantic: SemanticRecord
    timing: NestedTiming
    events: tuple[RoutingEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in V04_MODES:
            raise ValueError("unsupported V0.4 mode")
        if len(self.timing.decode_forward_ns) != self.semantic.actual_decode_input_steps:
            raise ValueError("V0.4 decode timer count does not match semantic steps")
        if self.mode == REFERENCE_MODE and self.events:
            raise ValueError("reference mode must not produce routing events")
        if self.mode == REFERENCE_MODE and self.timing.event_extraction_ns:
            raise ValueError("reference mode must not contain extraction timers")
        if self.mode == PROFILED_MODE and not self.events:
            raise ValueError("profiled mode must produce routing events")
        if self.mode == PROFILED_MODE and len(self.timing.event_extraction_ns) != 1 + self.semantic.actual_decode_input_steps:
            raise ValueError("profiled extraction timer count does not match prompt/decode forwards")


def run_prompt(
    collector: Any,
    prompt: CorpusPrompt,
    mode: str,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    capture_factory: Callable[[Any], Any] = _RouterHookCapture,
    event_extractor: Callable[..., list[RoutingEvent]] = _events_from_router_logits,
    expected_router_layers: tuple[int, ...] | None = tuple(range(24)),
) -> PromptRun:
    """Run the shared candidate-before-feed driver for one prompt/mode."""
    if mode not in V04_MODES:
        raise ValueError("unsupported V0.4 mode")
    encoded = collector.tokenizer(prompt.text, return_tensors="pt")
    encoded = {name: value.detach().clone().to("cpu") for name, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention = encoded.get("attention_mask", torch.ones_like(input_ids))
    eos_ids = _normalize_eos_token_ids(collector.tokenizer)
    router_modules = _router_modules(collector.model)
    if expected_router_layers is not None and tuple(router_modules) != expected_router_layers:
        raise RuntimeError("V0.4 requires exactly the expected 24 Granite router modules")
    baseline_hooks = _repository_hook_snapshot(router_modules)
    baseline_all_hooks = _all_hook_snapshot(router_modules)
    if any(baseline_hooks.values()):
        raise RuntimeError("V0.4 prompt began with router hook carryover")
    events: list[RoutingEvent] = []
    candidates: list[int] = []
    candidate_texts: list[str] = []
    routed: list[int] = []
    decode_ns: list[int] = []
    extraction_ns: list[int] = []
    capture: Any = None

    primary_start = clock_ns()
    setup_start = clock_ns()
    try:
        if mode == PROFILED_MODE:
            capture = capture_factory(collector.model)
            capture.__enter__()
            if expected_router_layers is not None:
                installed = _repository_hook_snapshot(router_modules)
                if any(len(installed[layer]) != 1 for layer in expected_router_layers):
                    raise RuntimeError("V0.4 did not install exactly one repository hook per router")
                handle_ids = tuple(sorted(int(handle.id) for handle in capture._handles))
                installed_ids = tuple(sorted(value for values in installed.values() for value in values))
                if handle_ids != installed_ids:
                    raise RuntimeError("V0.4 hook handle identities do not match registered hooks")
    except BaseException as original_error:
        if capture is not None:
            try:
                capture.__exit__(type(original_error), original_error, original_error.__traceback__)
            except BaseException as cleanup_error:
                raise RuntimeError(f"V0.4 hook setup failed and cleanup also failed: {cleanup_error}") from original_error
        raise
    setup_ns = _elapsed(setup_start, clock_ns())
    removal_ns = 0
    try:
        with torch.inference_mode():
            forward_start = clock_ns()
            outputs = collector.model(**encoded, use_cache=True)
            prefill_ns = _elapsed(forward_start, clock_ns())
            if capture is not None:
                extraction_start = clock_ns()
                events.extend(event_extractor(
                    capture.take(), input_ids, "prompt", 0, collector.experts_per_token
                ))
                if getattr(capture, "_logits_by_layer", {}):
                    raise RuntimeError("V0.4 capture was not cleared after prompt extraction")
                extraction_ns.append(_elapsed(extraction_start, clock_ns()))
            past = outputs.past_key_values
            candidate = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            terminal: int | None = None
            terminal_text: str | None = None
            terminal_position: int | None = None
            while True:
                candidate_id = int(candidate[0, 0])
                candidate_text = collector.tokenizer.decode([candidate_id], skip_special_tokens=False)
                candidates.append(candidate_id)
                candidate_texts.append(candidate_text)
                if len(routed) == STAGE1_MAX_DECODE_INPUT_STEPS:
                    break
                if candidate_id in eos_ids:
                    terminal, terminal_text, terminal_position = candidate_id, candidate_text, len(routed)
                    break
                routed.append(candidate_id)
                decode_start = clock_ns()
                generated = collector.model(input_ids=candidate, past_key_values=past, use_cache=True)
                decode_ns.append(_elapsed(decode_start, clock_ns()))
                if capture is not None:
                    extraction_start = clock_ns()
                    events.extend(event_extractor(
                        capture.take(), candidate, "generated",
                        input_ids.shape[1] + len(routed) - 1, collector.experts_per_token,
                    ))
                    if getattr(capture, "_logits_by_layer", {}):
                        raise RuntimeError("V0.4 capture was not cleared after decode extraction")
                    extraction_ns.append(_elapsed(extraction_start, clock_ns()))
                past = generated.past_key_values
                candidate = generated.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finally:
        removal_start = clock_ns()
        if capture is not None:
            capture.__exit__(None, None, None)
        removal_ns = _elapsed(removal_start, clock_ns())
    lifecycle_start = clock_ns()
    if _repository_hook_snapshot(router_modules) != baseline_hooks or _all_hook_snapshot(router_modules) != baseline_all_hooks:
        raise RuntimeError("V0.4 router hook lifecycle/carryover assertion failed")
    lifecycle_ns = _elapsed(lifecycle_start, clock_ns())
    primary_ns = _elapsed(primary_start, clock_ns())
    accounted = setup_ns + prefill_ns + sum(decode_ns) + sum(extraction_ns) + removal_ns + lifecycle_ns
    if accounted > primary_ns:
        raise ValueError("V0.4 nested component time exceeds primary time")
    horizon = terminal is None
    semantic = SemanticRecord(
        input_ids=tuple(int(value) for value in input_ids[0]),
        attention_mask=tuple(int(value) for value in attention[0]),
        candidate_token_ids=tuple(candidates),
        candidate_texts=tuple(candidate_texts),
        routed_non_eos_token_ids=tuple(routed),
        routed_non_eos_text=collector.tokenizer.decode(routed, skip_special_tokens=True),
        terminal_eos_token_id=terminal,
        terminal_eos_text=terminal_text,
        terminal_eos_candidate_position=terminal_position,
        actual_decode_input_steps=len(routed),
        eos_emitted=terminal is not None,
        horizon_exhausted=horizon,
        validation_only_horizon_candidate_id=candidates[-1] if horizon else None,
        validation_only_horizon_candidate_text=candidate_texts[-1] if horizon else None,
    )
    return PromptRun(
        prompt.id, mode, semantic,
        NestedTiming(
            primary_ns, setup_ns, prefill_ns, tuple(decode_ns), tuple(extraction_ns),
            removal_ns, lifecycle_ns, accounted, primary_ns - accounted,
        ),
        tuple(events),
    )


def _elapsed(start: int, end: int) -> int:
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise ValueError("V0.4 clock must return monotonic integer nanoseconds")
    return end - start


def _router_modules(model: Any) -> dict[int, Any]:
    modules: dict[int, Any] = {}
    for name, module in model.named_modules():
        if name.endswith(".block_sparse_moe.router"):
            parts = name.split(".")
            try:
                layer = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError) as error:
                raise RuntimeError(f"cannot determine V0.4 router layer from {name}") from error
            if layer in modules:
                raise RuntimeError("duplicate V0.4 router layer")
            modules[layer] = module
    return dict(sorted(modules.items()))


def _repository_hook_snapshot(modules: dict[int, Any]) -> dict[int, tuple[int, ...]]:
    return {
        layer: tuple(sorted(
            int(hook_id) for hook_id, hook in getattr(module, "_forward_hooks", {}).items()
            if getattr(hook, "_moe_cache_lab_router_hook", False)
        ))
        for layer, module in modules.items()
    }


def _all_hook_snapshot(modules: dict[int, Any]) -> dict[int, tuple[int, ...]]:
    return {
        layer: tuple(sorted(int(hook_id) for hook_id in getattr(module, "_forward_hooks", {})))
        for layer, module in modules.items()
    }


def validate_prompt_pair(reference: PromptRun, profiled: PromptRun) -> None:
    if reference.prompt_id != profiled.prompt_id or reference.mode != REFERENCE_MODE or profiled.mode != PROFILED_MODE:
        raise ValueError("V0.4 paired prompt identity/modes are invalid")
    if reference.semantic != profiled.semantic:
        raise ValueError(f"V0.4 observer changed semantic output for {reference.prompt_id}")


@dataclass(frozen=True)
class ResourceSample:
    cpu_percent: float
    disk_percent: float
    available_physical_bytes: int
    commit_total_pages: int
    commit_limit_pages: int
    page_size_bytes: int
    cpu_counter_path: str = CPU_COUNTER_PATH
    disk_counter_path: str = DISK_COUNTER_PATH
    timestamp: str = "2000-01-01T00:00:00+00:00"
    collect_status: int = 0
    cpu_formatted_api_status: int = 0
    disk_formatted_api_status: int = 0
    cpu_data_status: int = 0
    disk_data_status: int = 0
    cpu_raw_api_status: int = 0
    disk_raw_api_status: int = 0
    cpu_raw_data_status: int = 0
    disk_raw_data_status: int = 0
    cpu_raw_first_value: int = 0
    cpu_raw_second_value: int = 0
    cpu_raw_multi_count: int = 1
    disk_raw_first_value: int = 0
    disk_raw_second_value: int = 0
    disk_raw_multi_count: int = 1
    global_memory_status: int = 1
    performance_info_status: int = 1
    api_status: str = "ok"
    monotonic_ns: int = 0

    def __post_init__(self) -> None:
        for value in (self.cpu_percent, self.disk_percent):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("V0.4 PDH values must be finite percentages in [0,100]")
        for value in (self.available_physical_bytes, self.commit_total_pages, self.commit_limit_pages, self.page_size_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("V0.4 memory counters must be positive integers")
        if self.commit_limit_pages <= 0 or self.cpu_counter_path != CPU_COUNTER_PATH or self.disk_counter_path != DISK_COUNTER_PATH or self.api_status != "ok":
            raise ValueError("V0.4 resource sample metadata is invalid")
        _aware_timestamp(self.timestamp)
        statuses = (
            self.collect_status, self.cpu_formatted_api_status,
            self.disk_formatted_api_status, self.cpu_raw_api_status,
            self.disk_raw_api_status,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value != 0 for value in statuses):
            raise ValueError("V0.4 PDH API statuses must be success")
        if any(isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1) for value in (
            self.cpu_data_status, self.disk_data_status,
            self.cpu_raw_data_status, self.disk_raw_data_status,
        )):
            raise ValueError("V0.4 PDH data statuses must be VALID_DATA or NEW_DATA")
        if any(isinstance(value, bool) or not isinstance(value, int) or value != 1 for value in (self.global_memory_status, self.performance_info_status)):
            raise ValueError("V0.4 memory API statuses must be successful BOOL values")
        raw_values = (
            self.cpu_raw_first_value, self.cpu_raw_second_value,
            self.cpu_raw_multi_count, self.disk_raw_first_value,
            self.disk_raw_second_value, self.disk_raw_multi_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_values):
            raise ValueError("V0.4 PDH raw values must be integers")
        if self.cpu_raw_multi_count < 0 or self.disk_raw_multi_count < 0:
            raise ValueError("V0.4 PDH raw multi-counts must be nonnegative")
        if isinstance(self.monotonic_ns, bool) or not isinstance(self.monotonic_ns, int) or self.monotonic_ns < 0:
            raise ValueError("V0.4 resource monotonic timestamp is invalid")

    @property
    def commit_percent(self) -> float:
        return 100.0 * self.commit_total_pages / self.commit_limit_pages

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update({
            "commit_numerator_bytes": self.commit_total_pages * self.page_size_bytes,
            "commit_denominator_bytes": self.commit_limit_pages * self.page_size_bytes,
            "commit_percent": self.commit_percent,
        })
        return record


class ResourceProbe(Protocol):
    def prime(self) -> "ResourcePrime": ...
    def sample(self) -> ResourceSample: ...


@dataclass(frozen=True)
class ResourcePrime:
    timestamp: str
    collect_status: int
    cpu_counter_path: str = CPU_COUNTER_PATH
    disk_counter_path: str = DISK_COUNTER_PATH
    monotonic_ns: int = 0

    def __post_init__(self) -> None:
        _aware_timestamp(self.timestamp)
        if isinstance(self.collect_status, bool) or not isinstance(self.collect_status, int) or self.collect_status != 0 or self.cpu_counter_path != CPU_COUNTER_PATH or self.disk_counter_path != DISK_COUNTER_PATH:
            raise ValueError("invalid V0.4 PDH prime record")
        if isinstance(self.monotonic_ns, bool) or not isinstance(self.monotonic_ns, int) or self.monotonic_ns < 0:
            raise ValueError("invalid V0.4 PDH prime monotonic timestamp")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    prime: ResourcePrime | None
    baseline: ResourceSample | None
    samples: tuple[ResourceSample, ...]
    errors: tuple[str, ...]
    reason: str
    audit: dict[str, Any]
    started_at: str
    finished_at: str
    started_monotonic_ns: int
    finished_monotonic_ns: int


def collect_admission(
    probe: ResourceProbe,
    *,
    sleep: Callable[[float], None] = time.sleep,
    wall_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> AdmissionResult:
    """Prime rate counters, record one excluded baseline, then 15 samples."""
    started_at = wall_now().isoformat()
    started_monotonic_ns = monotonic_ns()
    prime: ResourcePrime | None = None
    baseline: ResourceSample | None = None
    samples: list[ResourceSample] = []
    errors: list[str] = []
    try:
        prime = probe.prime()
    except Exception as error:  # persisted as deferred evidence
        errors.append(f"prime: {type(error).__name__}: {error}")
    if prime is not None:
        try:
            baseline = probe.sample()
        except Exception as error:
            errors.append(f"baseline: {type(error).__name__}: {error}")
    for index in range(ADMISSION_SAMPLE_COUNT):
        sleep(ADMISSION_INTERVAL_SECONDS)
        try:
            samples.append(probe.sample())
        except Exception as error:
            errors.append(f"sample {index + 1}: {type(error).__name__}: {error}")
    finished_at = wall_now().isoformat()
    finished_monotonic_ns = monotonic_ns()
    audit = _probe_audit(probe)
    if prime is None or baseline is None or errors or len(samples) != ADMISSION_SAMPLE_COUNT:
        return AdmissionResult(False, prime, baseline, tuple(samples), tuple(errors), "resource API/sample invalid", audit, started_at, finished_at, started_monotonic_ns, finished_monotonic_ns)
    cpu = [float(sample.cpu_percent) for sample in samples]
    disk = [float(sample.disk_percent) for sample in samples]
    admitted = (
        statistics.median(cpu) <= 10 and max(cpu) <= 25
        and all(sample.available_physical_bytes >= MIN_AVAILABLE_BYTES for sample in samples)
        and all(sample.commit_percent <= 75 for sample in samples)
        and statistics.median(disk) <= 10 and max(disk) <= 50
    )
    return AdmissionResult(admitted, prime, baseline, tuple(samples), (), "admitted" if admitted else "host busy or memory gate failed", audit, started_at, finished_at, started_monotonic_ns, finished_monotonic_ns)


def _probe_audit(probe: Any) -> dict[str, Any]:
    method = getattr(probe, "audit_record", None)
    if callable(method):
        return method()
    return {
        "open_query_status": 0, "add_cpu_status": 0, "add_disk_status": 0,
        "close_query_status": None, "constructor_error": None,
    }


@dataclass(frozen=True)
class ProcessMemory:
    working_set_size: int
    private_usage: int
    peak_working_set_size: int
    available_physical_bytes: int
    timestamp: str = "2000-01-01T00:00:00+00:00"
    process_memory_status: int = 1
    global_memory_status: int = 1
    api_status: str = "ok"

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (
            self.working_set_size, self.private_usage, self.peak_working_set_size,
            self.available_physical_bytes,
        )) or self.api_status != "ok":
            raise ValueError("invalid V0.4 process-memory snapshot")
        _aware_timestamp(self.timestamp)
        if any(isinstance(value, bool) or not isinstance(value, int) or value != 1 for value in (self.process_memory_status, self.global_memory_status)):
            raise ValueError("V0.4 process/global memory API status is invalid")


@dataclass(frozen=True)
class ProcessTimes:
    user_ns: int
    system_ns: int

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.user_ns, self.system_ns)):
            raise ValueError("invalid V0.4 process CPU times")


class _V04FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class ResourceProbeConstructionError(OSError):
    def __init__(self, message: str, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


class WindowsResourceProbe:
    """Typed Windows PDH/PSAPI boundary; construction performs no sampling."""

    PDH_FMT_DOUBLE = 0x00000200

    class _PdhValue(ctypes.Structure):
        _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

    _FileTime = _V04FileTime

    class _RawCounter(ctypes.Structure):
        _fields_ = [
            ("CStatus", wintypes.DWORD), ("TimeStamp", _V04FileTime),
            ("FirstValue", ctypes.c_longlong), ("SecondValue", ctypes.c_longlong),
            ("MultiCount", wintypes.DWORD),
        ]

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class _PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize",
        )] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]

    class _ProcessCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows PDH resource probe is available only on Windows")
        self._pdh = ctypes.WinDLL("pdh.dll", use_last_error=True)
        self._kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        self._pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self._pdh.PdhOpenQueryW.restype = wintypes.LONG
        self._pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self._pdh.PdhAddEnglishCounterW.restype = wintypes.LONG
        self._pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        self._pdh.PdhCollectQueryData.restype = wintypes.LONG
        self._pdh.PdhGetFormattedCounterValue.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(self._PdhValue)]
        self._pdh.PdhGetFormattedCounterValue.restype = wintypes.LONG
        self._pdh.PdhGetRawCounterValue.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(self._RawCounter)]
        self._pdh.PdhGetRawCounterValue.restype = wintypes.LONG
        self._pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        self._pdh.PdhCloseQuery.restype = wintypes.LONG
        self._kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(self._MemoryStatus)]
        self._kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
        self._kernel.GetCurrentProcess.argtypes = []
        self._kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self._kernel.GetProcessTimes.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(self._FileTime), ctypes.POINTER(self._FileTime),
            ctypes.POINTER(self._FileTime), ctypes.POINTER(self._FileTime),
        ]
        self._kernel.GetProcessTimes.restype = wintypes.BOOL
        self._kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
        self._kernel.GetPriorityClass.restype = wintypes.DWORD
        self._kernel.K32GetPerformanceInfo.argtypes = [ctypes.POINTER(self._PerformanceInfo), wintypes.DWORD]
        self._kernel.K32GetPerformanceInfo.restype = wintypes.BOOL
        self._kernel.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(self._ProcessCounters), wintypes.DWORD]
        self._kernel.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        self._query = wintypes.HANDLE()
        self._cpu = wintypes.HANDLE()
        self._disk = wintypes.HANDLE()
        self._open_query_status = int(self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)))
        self._add_cpu_status = None
        self._add_disk_status = None
        self._close_query_status = None
        if self._open_query_status != 0:
            raise ResourceProbeConstructionError(
                f"PdhOpenQueryW failed with PDH status 0x{self._open_query_status & 0xffffffff:08x}",
                self.audit_record(),
            )
        try:
            self._add_cpu_status = int(self._pdh.PdhAddEnglishCounterW(self._query, CPU_COUNTER_PATH, 0, ctypes.byref(self._cpu)))
            self._check(self._add_cpu_status, "CPU counter")
            self._add_disk_status = int(self._pdh.PdhAddEnglishCounterW(self._query, DISK_COUNTER_PATH, 0, ctypes.byref(self._disk)))
            self._check(self._add_disk_status, "disk counter")
        except BaseException as original_error:
            close_status = int(self._pdh.PdhCloseQuery(self._query))
            self._close_query_status = close_status
            self._query = None
            message = f"PDH constructor failed: {original_error}"
            if close_status != 0:
                message += f"; query rollback failed with status 0x{close_status & 0xffffffff:08x}"
            raise ResourceProbeConstructionError(message, self.audit_record()) from original_error

    @staticmethod
    def _check(status: int, label: str) -> None:
        if status != 0:
            raise OSError(f"{label} failed with PDH status 0x{status & 0xffffffff:08x}")

    def prime(self) -> ResourcePrime:
        status = int(self._pdh.PdhCollectQueryData(self._query))
        self._check(status, "PdhCollectQueryData prime")
        return ResourcePrime(
            datetime.now(timezone.utc).isoformat(), status,
            monotonic_ns=time.perf_counter_ns(),
        )

    def sample(self) -> ResourceSample:
        collect_status = int(self._pdh.PdhCollectQueryData(self._query))
        self._check(collect_status, "PdhCollectQueryData")
        values: list[float] = []
        formatted_statuses: list[int] = []
        data_statuses: list[int] = []
        raw_statuses: list[int] = []
        raw_data_statuses: list[int] = []
        raw_values: list[tuple[int, int, int]] = []
        for counter in (self._cpu, self._disk):
            value = self._PdhValue()
            formatted_status = int(self._pdh.PdhGetFormattedCounterValue(counter, self.PDH_FMT_DOUBLE, None, ctypes.byref(value)))
            self._check(formatted_status, "PdhGetFormattedCounterValue")
            if value.CStatus not in (0, 1):  # PDH_CSTATUS_VALID_DATA / NEW_DATA
                raise OSError(f"PDH formatted counter status 0x{value.CStatus:08x}")
            raw = self._RawCounter()
            raw_status = int(self._pdh.PdhGetRawCounterValue(counter, None, ctypes.byref(raw)))
            self._check(raw_status, "PdhGetRawCounterValue")
            if raw.CStatus not in (0, 1):
                raise OSError(f"PDH raw counter status 0x{raw.CStatus:08x}")
            values.append(float(value.doubleValue))
            formatted_statuses.append(formatted_status)
            data_statuses.append(int(value.CStatus))
            raw_statuses.append(raw_status)
            raw_data_statuses.append(int(raw.CStatus))
            raw_values.append((int(raw.FirstValue), int(raw.SecondValue), int(raw.MultiCount)))
        memory = self._MemoryStatus()
        memory.dwLength = ctypes.sizeof(memory)
        memory_status = int(self._kernel.GlobalMemoryStatusEx(ctypes.byref(memory)))
        if not memory_status:
            raise ctypes.WinError()
        performance = self._PerformanceInfo()
        performance.cb = ctypes.sizeof(performance)
        performance_status = int(self._kernel.K32GetPerformanceInfo(ctypes.byref(performance), performance.cb))
        if not performance_status:
            raise ctypes.WinError()
        return ResourceSample(
            values[0], values[1], int(memory.ullAvailPhys), int(performance.CommitTotal),
            int(performance.CommitLimit), int(performance.PageSize),
            timestamp=datetime.now(timezone.utc).isoformat(), collect_status=collect_status,
            monotonic_ns=time.perf_counter_ns(),
            cpu_formatted_api_status=formatted_statuses[0], disk_formatted_api_status=formatted_statuses[1],
            cpu_data_status=data_statuses[0], disk_data_status=data_statuses[1],
            cpu_raw_api_status=raw_statuses[0], disk_raw_api_status=raw_statuses[1],
            cpu_raw_data_status=raw_data_statuses[0], disk_raw_data_status=raw_data_statuses[1],
            cpu_raw_first_value=raw_values[0][0], cpu_raw_second_value=raw_values[0][1], cpu_raw_multi_count=raw_values[0][2],
            disk_raw_first_value=raw_values[1][0], disk_raw_second_value=raw_values[1][1], disk_raw_multi_count=raw_values[1][2],
            global_memory_status=memory_status, performance_info_status=performance_status,
        )

    def process_memory(self) -> ProcessMemory:
        counters = self._ProcessCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = self._kernel.GetCurrentProcess()
        process_status = int(self._kernel.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb))
        if not process_status:
            raise ctypes.WinError()
        memory = self._MemoryStatus()
        memory.dwLength = ctypes.sizeof(memory)
        global_status = int(self._kernel.GlobalMemoryStatusEx(ctypes.byref(memory)))
        if not global_status:
            raise ctypes.WinError()
        return ProcessMemory(
            int(counters.WorkingSetSize), int(counters.PrivateUsage),
            int(counters.PeakWorkingSetSize), int(memory.ullAvailPhys),
            timestamp=datetime.now(timezone.utc).isoformat(),
            process_memory_status=process_status, global_memory_status=global_status,
        )

    def process_times(self) -> ProcessTimes:
        created, exited, kernel, user = (self._FileTime() for _ in range(4))
        if not self._kernel.GetProcessTimes(
            self._kernel.GetCurrentProcess(), ctypes.byref(created), ctypes.byref(exited),
            ctypes.byref(kernel), ctypes.byref(user),
        ):
            raise ctypes.WinError()
        def nanoseconds(value: WindowsResourceProbe._FileTime) -> int:
            return ((int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)) * 100
        return ProcessTimes(nanoseconds(user), nanoseconds(kernel))

    def priority_class(self) -> int:
        value = int(self._kernel.GetPriorityClass(self._kernel.GetCurrentProcess()))
        if value == 0:
            raise ctypes.WinError()
        return value

    def close(self) -> None:
        if getattr(self, "_query", None):
            status = int(self._pdh.PdhCloseQuery(self._query))
            self._close_query_status = status
            self._query = None
            self._check(status, "PdhCloseQuery")

    def audit_record(self) -> dict[str, Any]:
        return {
            "open_query_status": self._open_query_status,
            "add_cpu_status": self._add_cpu_status,
            "add_disk_status": self._add_disk_status,
            "close_query_status": self._close_query_status,
            "constructor_error": None,
        }

    def __enter__(self) -> "WindowsResourceProbe":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid4()}")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return destination


def load_v04_json(path: str | Path, *, expected_format: str | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonfinite {token}")))
    if not isinstance(value, dict) or value.get("format_version") != V04_FORMAT_VERSION:
        raise ValueError("unsupported V0.4 JSON format/version")
    if expected_format is not None and value.get("format") != expected_format:
        raise ValueError("unexpected V0.4 JSON format")
    return value


def render_markdown(result: dict[str, Any]) -> str:
    if result.get("format") != V04_RESULT_FORMAT:
        raise ValueError("Markdown requires a validated V0.4 result")
    lines = [
        "# V0.4 CPU trace-profiler overhead", "",
        "Measured CPU wall-time; process-memory snapshots are descriptive and order-conditioned.",
        "No GPU, offload, cache speedup, or production-throughput claim.", "",
        f"Decision: **{result['decision']}**", "",
        "| Child | Reference ns | Profiled ns | Delta ns | Ratio |", "|---:|---:|---:|---:|---:|",
    ]
    for row in result["children"]:
        lines.append(f"| {row['child_number']} | {row['sum_reference_ns']} | {row['sum_profiled_ns']} | {row['delta_ns']} | {row['ratio']:.6f} |")
    lines.extend([
        "", "Each child row sums all 8 evaluation prompts. Repetitions are paired repeatability checks, not independent workload samples.",
        f"Across-child ratio stats (n={result['child_ratio_stats']['n']}): mean {result['child_ratio_stats']['mean']:.6f}, sample SD {result['child_ratio_stats']['sample_sd']:.6f}, min {result['child_ratio_stats']['min']:.6f}, max {result['child_ratio_stats']['max']:.6f}.",
        f"Across-child delta stats (n={result['child_delta_stats']['n']}): mean {result['child_delta_stats']['mean']:.3f} ns, sample SD {result['child_delta_stats']['sample_sd']:.3f}, min {result['child_delta_stats']['min']:.3f}, max {result['child_delta_stats']['max']:.3f}.",
        f"CV reference/profiled/ratio: {result['cv_sum_reference']:.6f} / {result['cv_sum_profiled']:.6f} / {result['cv_paired_ratio']:.6f}",
        f"Ratio span: {result['paired_ratio_span']:.6f}", "",
    ])
    lines.extend(["", "## Within-child prompt-ratio summaries", "", "| Child | n prompts | Mean | Sample SD | Min | Max |", "|---:|---:|---:|---:|---:|---:|"])
    for row in result["children"]:
        stats = row["prompt_ratio_stats"]
        lines.append(f"| {row['child_number']} | {stats['n']} | {stats['mean']:.6f} | {stats['sample_sd']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |")
    lines.extend(["", "## Per-prompt paired wall-time ratio", "", "| Prompt | n children | Mean | Sample SD | Min | Max |", "|---|---:|---:|---:|---:|---:|"])
    for row in result["prompt_rows"]:
        lines.append(f"| {row['prompt_id']} | {row['n']} | {row['mean']:.6f} | {row['sample_sd']:.6f} | {row['min']:.6f} | {row['max']:.6f} |")
    macro = result["prompt_macro"]
    lines.extend([
        "", f"Workload macro across exactly n={macro['n']} prompt means: mean {macro['mean']:.6f}, sample SD {macro['sample_sd']:.6f}, min {macro['min']:.6f}, max {macro['max']:.6f}.",
        "", "## Descriptive process-memory observations", "",
        result["descriptive_memory"]["claim"] + ". Process lifetime PeakWorkingSetSize is retained only in raw child snapshots and is not aggregated as per-mode peak overhead.",
        "", "| Prompt | Metric | n children | Mean bytes | Sample SD | Min | Max |", "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in result["descriptive_memory"]["prompt_rows"]:
        for metric in (
            "reference_working_set_delta", "profiled_working_set_delta",
            "reference_private_delta", "profiled_private_delta",
        ):
            stats = row[metric]
            lines.append(f"| {row['prompt_id']} | {metric} | {stats['n']} | {stats['mean']:.3f} | {stats['sample_sd']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |")
    lines.extend(["", "Memory workload macros use exactly n=8 prompt means:", ""])
    for metric in (
        "reference_working_set_delta", "profiled_working_set_delta",
        "reference_private_delta", "profiled_private_delta",
    ):
        stats = result["descriptive_memory"]["prompt_macro"][metric]
        lines.append(f"- {metric}: mean {stats['mean']:.3f}, sample SD {stats['sample_sd']:.3f}, min {stats['min']:.3f}, max {stats['max']:.3f} (n={stats['n']})")
    lines.append("")
    return "\n".join(lines)


def validate_trusted_stage1_paths(
    set_manifest: str | Path = TRUSTED_STAGE1_SET,
    repetition_manifest: str | Path = TRUSTED_STAGE1_REP1,
) -> None:
    if file_sha256(set_manifest) != TRUSTED_STAGE1_SET_SHA256 or file_sha256(repetition_manifest) != TRUSTED_STAGE1_REP1_SHA256:
        raise ValueError("V0.4 canonical trusted Stage 1 hash mismatch")
    repetition = load_stage1_repetition_manifest(repetition_manifest)
    if tuple(item.prompt.id for item in repetition.prompts) != CALIBRATION_IDS + EVALUATION_IDS:
        raise ValueError("V0.4 canonical Stage 1 prompt mapping mismatch")


def transition_state(
    current: str,
    event: str,
    *,
    repeated_same_root_cause: bool = False,
    rerun_index: int = 0,
) -> str:
    """Pure frozen attempt-state transition used by orchestration and tests."""
    if event == "admission_deferred" and current in {"planned", "running"}:
        return "deferred"
    if event == "prerequisite_blocked" and current in {"planned", "deferred", "invalid"}:
        return "blocked"
    if event == "invalid" and current in {"planned", "running"}:
        return "blocked" if repeated_same_root_cause else "invalid"
    if event == "stable" and current == "running":
        return "valid_stable"
    if event == "inconclusive" and current == "running":
        return "inconclusive_final" if rerun_index == 1 else "valid_inconclusive"
    raise ValueError(f"invalid V0.4 state transition: {current} + {event}")


def _collect_v04_child_impl(
    corpus_path: str | Path,
    output_directory: str | Path,
    child_number: int,
    process_uuid: str,
    *,
    collector_factory: Callable[..., Any] = GraniteTraceCollector,
    memory_probe_factory: Callable[[], Any] = WindowsResourceProbe,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    progress: dict[str, Any] | None = None,
) -> Path:
    """Private fresh-process child. Public orchestration is the only caller."""
    _child_number(child_number)
    UUID(process_uuid)
    corpus = load_corpus(corpus_path)
    if corpus.sha256 != STAGE1_CORPUS_SHA256:
        raise ValueError("V0.4 requires the frozen corpus")
    output = Path(output_directory).resolve()
    if not output.is_dir():
        raise ValueError("V0.4 child output directory was not prepared")
    probe = memory_probe_factory()
    started = datetime.now(timezone.utc)
    warm_runs: list[PromptRun] = []
    evaluation_rows: list[dict[str, Any]] = []
    if progress is None:
        progress = {
            "model_load": None, "completed_calibration_runs": 0,
            "completed_evaluation_prompts": 0, "last_resource_snapshot": None,
            "failure_phase": "model_load",
        }
    try:
        torch.set_num_threads(STAGE1_TORCH_THREADS)
        load_started_at = datetime.now(timezone.utc)
        load_memory_before = probe.process_memory()
        if load_memory_before.available_physical_bytes < ABORT_AVAILABLE_BYTES:
            raise MemoryError("V0.4 available RAM fell below the 2 GiB abort floor before model load")
        load_times_before = probe.process_times()
        load_started = clock_ns()
        collector = collector_factory(
            STAGE1_MODEL_ID, revision=STAGE1_MODEL_REVISION, local_files_only=True
        )
        model_load_ns = clock_ns() - load_started
        load_times_after = probe.process_times()
        load_memory_after = probe.process_memory()
        load_finished_at = datetime.now(timezone.utc)
        if load_memory_after.available_physical_bytes < ABORT_AVAILABLE_BYTES:
            raise MemoryError("V0.4 available RAM fell below the 2 GiB abort floor after model load")
        if collector.resolved_revision != STAGE1_MODEL_REVISION:
            raise ValueError("V0.4 collector revision mismatch")
        parameter_count = _validate_loaded_model(collector.model)
        if load_times_after.user_ns < load_times_before.user_ns or load_times_after.system_ns < load_times_before.system_ns:
            raise ValueError("V0.4 model-load process CPU times moved backwards")
        model_load = {
            "elapsed_ns": model_load_ns,
            "started_at": load_started_at.isoformat(),
            "finished_at": load_finished_at.isoformat(),
            "process_cpu_user_ns": load_times_after.user_ns - load_times_before.user_ns,
            "process_cpu_system_ns": load_times_after.system_ns - load_times_before.system_ns,
            "memory_before": asdict(load_memory_before),
            "memory_after": asdict(load_memory_after),
            "parameter_count": parameter_count,
            "device": "cpu", "dtype": "float32", "nonmeta": True,
            "eval_mode": True,
        }
        progress["model_load"] = model_load
        progress["failure_phase"] = "calibration"
        prompts = {prompt.id: prompt for prompt in corpus.prompts}
        for prompt_id, mode in calibration_schedule(child_number):
            warm_runs.append(_run_with_memory(
                collector, prompts[prompt_id], mode, probe, clock_ns=clock_ns
            ))
            progress["completed_calibration_runs"] = len(warm_runs)
            progress["last_resource_snapshot"] = asdict(getattr(warm_runs[-1], "_memory_after"))
        for prompt_id in CALIBRATION_IDS:
            reference = next(run for run in warm_runs if run.prompt_id == prompt_id and run.mode == REFERENCE_MODE)
            profiled = next(run for run in warm_runs if run.prompt_id == prompt_id and run.mode == PROFILED_MODE)
            validate_prompt_pair(reference, profiled)
        for prompt_id, modes in evaluation_schedule(child_number):
            progress["failure_phase"] = "evaluation"
            runs: dict[str, PromptRun] = {}
            records: list[dict[str, Any]] = []
            for mode in modes:
                run = _run_with_memory(
                    collector, prompts[prompt_id], mode, probe, clock_ns=clock_ns
                )
                runs[mode] = run
                records.append(_prompt_run_record(run))
            validate_prompt_pair(runs[REFERENCE_MODE], runs[PROFILED_MODE])
            evaluation_rows.append({
                "prompt_id": prompt_id,
                "mode_order": list(modes),
                "reference_total_ns": runs[REFERENCE_MODE].timing.primary_total_ns,
                "profiled_total_ns": runs[PROFILED_MODE].timing.primary_total_ns,
                "reference_working_set_delta": _memory_delta(runs[REFERENCE_MODE], "working_set_size"),
                "profiled_working_set_delta": _memory_delta(runs[PROFILED_MODE], "working_set_size"),
                "reference_private_delta": _memory_delta(runs[REFERENCE_MODE], "private_usage"),
                "profiled_private_delta": _memory_delta(runs[PROFILED_MODE], "private_usage"),
                "runs": records,
            })
            progress["completed_evaluation_prompts"] = len(evaluation_rows)
            progress["last_resource_snapshot"] = records[-1]["memory_after"]
        profiled_runs = tuple(
            run for run in warm_runs if run.mode == PROFILED_MODE
        ) + tuple(
            _prompt_run_from_record(record)
            for row in evaluation_rows for record in row["runs"]
            if record["mode"] == PROFILED_MODE
        )
        progress["failure_phase"] = "validation"
        _validate_profiled_against_trusted(profiled_runs)
    finally:
        close = getattr(probe, "close", None)
        if callable(close):
            close()
    finished = datetime.now(timezone.utc)
    manifest = {
        "format": V04_CHILD_FORMAT,
        "format_version": V04_FORMAT_VERSION,
        "state": "valid",
        "child_number": child_number,
        "process_uuid": process_uuid,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "model_id": STAGE1_MODEL_ID,
        "model_revision": STAGE1_MODEL_REVISION,
        "corpus_sha256": corpus.sha256,
        "model_load": model_load,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "moe_cache_lab": __version__,
            "device": "cpu", "dtype": "float32", "torch_threads": torch.get_num_threads(),
            "local_files_only": True,
            "os_name": platform.system(), "os_release": platform.release(),
            "os_version": platform.version(), "windows_build": platform.version(),
            "logical_cpu_count": os.cpu_count(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "process_priority_class": probe.priority_class() if hasattr(probe, "priority_class") else None,
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "calibration_schedule": [list(item) for item in calibration_schedule(child_number)],
        "evaluation_schedule": [[prompt_id, list(modes)] for prompt_id, modes in evaluation_schedule(child_number)],
        "calibration": [_prompt_run_record(run) for run in warm_runs],
        "evaluation": evaluation_rows,
    }
    path = write_json_atomic(output / "v04-child.json", manifest)
    _validate_child_record(load_v04_json(path, expected_format=V04_CHILD_FORMAT))
    return path


def _validate_loaded_model(model: Any) -> int:
    if getattr(model, "training", None) is not False:
        raise ValueError("V0.4 model must be in eval mode")
    total = 0
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("V0.4 model exposes no parameters")
    for parameter in parameters:
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32 or parameter.is_meta:
            raise ValueError("V0.4 requires all model parameters CPU FP32 and non-meta")
        total += int(parameter.numel())
    if total != EXPECTED_PARAMETER_COUNT:
        raise ValueError("V0.4 exact model parameter count mismatch")
    return total


def collect_v04_child(
    corpus_path: str | Path,
    output_directory: str | Path,
    child_number: int,
    process_uuid: str,
    *,
    collector_factory: Callable[..., Any] = GraniteTraceCollector,
    memory_probe_factory: Callable[[], Any] = WindowsResourceProbe,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> Path:
    """Private child that preserves a terminal invalid manifest on any failure."""
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    progress = {
        "model_load": None, "completed_calibration_runs": 0,
        "completed_evaluation_prompts": 0, "last_resource_snapshot": None,
        "failure_phase": "model_load",
    }
    try:
        return _collect_v04_child_impl(
            corpus_path, output, child_number, process_uuid,
            collector_factory=collector_factory,
            memory_probe_factory=memory_probe_factory,
            clock_ns=clock_ns,
            progress=progress,
        )
    except BaseException as error:
        invalid = {
            "format": V04_CHILD_FORMAT, "format_version": V04_FORMAT_VERSION,
            "state": "invalid", "child_number": child_number,
            "process_uuid": process_uuid, "pid": os.getpid(),
            "parent_pid": os.getppid(), "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "reason": f"{type(error).__name__}: {error}",
            "reason_type": type(error).__name__, "reason_message": str(error),
            "failure_phase": progress["failure_phase"],
            "partial": {name: progress[name] for name in (
                "model_load", "completed_calibration_runs",
                "completed_evaluation_prompts", "last_resource_snapshot",
            )},
        }
        write_json_atomic(output / "v04-child.json", invalid)
        raise


def _run_with_memory(
    collector: Any,
    prompt: CorpusPrompt,
    mode: str,
    probe: Any,
    *,
    clock_ns: Callable[[], int],
) -> PromptRun:
    before = probe.process_memory()
    times_before = probe.process_times() if hasattr(probe, "process_times") else ProcessTimes(0, 0)
    if before.available_physical_bytes < ABORT_AVAILABLE_BYTES:
        raise MemoryError("V0.4 available RAM fell below the 2 GiB abort floor")
    run = run_prompt(collector, prompt, mode, clock_ns=clock_ns)
    after = probe.process_memory()
    times_after = probe.process_times() if hasattr(probe, "process_times") else times_before
    if after.available_physical_bytes < ABORT_AVAILABLE_BYTES:
        raise MemoryError("V0.4 available RAM fell below the 2 GiB abort floor")
    object.__setattr__(run, "_memory_before", before)
    object.__setattr__(run, "_memory_after", after)
    if times_after.user_ns < times_before.user_ns or times_after.system_ns < times_before.system_ns:
        raise ValueError("V0.4 process CPU times moved backwards")
    object.__setattr__(run, "_process_user_ns", times_after.user_ns - times_before.user_ns)
    object.__setattr__(run, "_process_system_ns", times_after.system_ns - times_before.system_ns)
    return run


def _memory_delta(run: PromptRun, field: str) -> int:
    return int(getattr(getattr(run, "_memory_after"), field) - getattr(getattr(run, "_memory_before"), field))


def collect_v04(
    corpus_path: str | Path,
    output_root: str | Path,
    *,
    run_process: Callable[..., ChildProcessOutcome] = run_child_process,
    probe_factory: Callable[[], Any] = WindowsResourceProbe,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    rerun_of: str | Path | None = None,
    retry_of: str | Path | None = None,
) -> Path:
    """Collect one immutable attempt using four sequential admitted children."""
    corpus = load_corpus(corpus_path)
    if corpus.sha256 != STAGE1_CORPUS_SHA256:
        raise ValueError("V0.4 requires the frozen corpus")
    validate_trusted_stage1_paths()
    if rerun_of is not None and retry_of is not None:
        raise ValueError("V0.4 statistical rerun and recovery retry are mutually exclusive")
    rerun_index = 0
    rerun_of_uuid: str | None = None
    rerun_source_sha256: str | None = None
    rerun_source_path: str | None = None
    retry_of_uuid: str | None = None
    retry_source_sha256: str | None = None
    retry_source_path: str | None = None
    retry_source_state: str | None = None
    retry_source_reason: str | None = None
    retry_source_root_digest: str | None = None
    prior_path: Path | None = None
    if rerun_of is not None:
        prior_path = validate_v04(rerun_of)
        prior = load_v04_json(prior_path, expected_format=V04_SET_FORMAT)
        if prior.get("state") != "valid_inconclusive" or prior.get("rerun_index") != 0 or prior.get("rerun_of") is not None:
            raise ValueError("V0.4 permits one rerun only from the first valid-inconclusive attempt")
        rerun_index = 1
        rerun_of_uuid = prior["attempt_uuid"]
        rerun_source_sha256 = file_sha256(prior_path)
    elif retry_of is not None:
        prior_path = validate_v04(retry_of)
        prior = load_v04_json(prior_path, expected_format=V04_SET_FORMAT)
        if prior.get("state") not in {"deferred", "invalid", "blocked"}:
            raise ValueError("V0.4 retry source must be deferred, invalid, or blocked")
        retry_of_uuid = prior["attempt_uuid"]
        retry_source_sha256 = file_sha256(prior_path)
        retry_source_state = prior["state"]
        retry_source_reason = prior["reason"]
        retry_source_root_digest = prior["root_cause"]["digest"]
    attempt_uuid = str(uuid4())
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if prior_path is None:
        unresolved = _unresolved_retry_sources(root)
        if unresolved:
            raise ValueError(
                "V0.4 unresolved prior terminal attempt requires explicit --retry-of: "
                + unresolved[0].as_posix()
            )
    if prior_path is not None:
        if prior_path.parent.parent.resolve() != root:
            raise ValueError("V0.4 lineage source must be an existing sibling attempt")
        relative_prior = prior_path.relative_to(root).as_posix()
        _reject_consumed_lineage(root, (rerun_of_uuid or retry_of_uuid), prior_path)
        if rerun_of_uuid:
            rerun_source_path = relative_prior
        else:
            retry_source_path = relative_prior
    final = root / f"v0.4-cpu-profiler-attempt-{attempt_uuid}"
    staging = root / f".v0.4-cpu-profiler-attempt-{attempt_uuid}.staging"
    if final.exists() or staging.exists():
        raise FileExistsError("V0.4 attempt path already exists")
    staging.mkdir()
    started_wall = datetime.now(timezone.utc)
    started_clock: float | None = None
    experiment_started_wall: datetime | None = None
    child_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    admission_records: list[dict[str, Any]] = []
    environment = dict(os.environ)
    environment.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    })
    state = "planned"
    try:
        for child_number in range(1, 5):
            try:
                probe = probe_factory()
            except Exception as error:
                prerequisite_missing = isinstance(error, OSError)
                construction_audit = getattr(error, "audit", None) or {
                    "open_query_status": None, "add_cpu_status": None,
                    "add_disk_status": None, "close_query_status": None,
                    "constructor_error": f"{type(error).__name__}: {error}",
                }
                construction_audit = dict(construction_audit)
                construction_audit["constructor_error"] = f"{type(error).__name__}: {error}"
                admission = AdmissionResult(
                    False, None, None, (), (f"probe: {type(error).__name__}: {error}",),
                    "resource prerequisite unavailable" if prerequisite_missing else "resource probe construction failed",
                    construction_audit,
                    datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat(),
                    time.perf_counter_ns(), time.perf_counter_ns(),
                )
                admission_path = write_json_atomic(
                    staging / f"admission-{child_number}.json",
                    _admission_record(child_number, admission),
                )
                admission_records.append({
                    "child_number": child_number, "path": admission_path.name,
                    "sha256": file_sha256(admission_path),
                    "size": admission_path.stat().st_size,
                })
                current_root = _admission_root_cause(admission, "resource_probe")
                if retry_source_state in {"deferred", "blocked"} and retry_source_root_digest == current_root["digest"]:
                    state = transition_state(state, "prerequisite_blocked")
                else:
                    state = transition_state(state, "admission_deferred")
                return _finalize_nonvalid(
                    staging, final, attempt_uuid, state, started_wall,
                    admission.reason, admission_records, child_records, failure_records,
                    rerun_index, rerun_of_uuid, rerun_source_sha256,
                    elapsed_seconds=0.0,
                    experiment_started_at=None,
                    rerun_source_path=rerun_source_path,
                    retry_of=retry_of_uuid, retry_source_sha256=retry_source_sha256,
                    retry_source_path=retry_source_path, retry_source_state=retry_source_state,
                    retry_source_reason=retry_source_reason,
                    retry_source_root_digest=retry_source_root_digest,
                    root_cause=current_root,
                )
            close_error: BaseException | None = None
            try:
                admission = collect_admission(probe, sleep=sleep)
            finally:
                close = getattr(probe, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as error:
                        close_error = error
            admission = replace(admission, audit=_probe_audit(probe))
            if close_error is not None:
                admission = replace(
                    admission, admitted=False,
                    errors=admission.errors + (f"close: {type(close_error).__name__}: {close_error}",),
                    reason="resource API/sample invalid",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    finished_monotonic_ns=time.perf_counter_ns(),
                )
            admission_record = _admission_record(child_number, admission)
            admission_path = write_json_atomic(staging / f"admission-{child_number}.json", admission_record)
            admission_records.append({
                "child_number": child_number,
                "path": admission_path.name,
                "sha256": file_sha256(admission_path),
                "size": admission_path.stat().st_size,
            })
            if not admission.admitted:
                current_root = _admission_root_cause(admission, "resource_admission")
                if (
                    retry_source_state in {"deferred", "blocked"}
                    and retry_source_root_digest == current_root["digest"]
                    and current_root["category"] != "busy_threshold"
                ):
                    state = transition_state(state, "prerequisite_blocked")
                else:
                    state = transition_state(state, "admission_deferred")
                return _finalize_nonvalid(
                    staging, final, attempt_uuid, state, started_wall,
                    admission.reason, admission_records, child_records, failure_records,
                    rerun_index, rerun_of_uuid, rerun_source_sha256,
                    elapsed_seconds=0.0 if started_clock is None else monotonic() - started_clock,
                    experiment_started_at=experiment_started_wall,
                    rerun_source_path=rerun_source_path,
                    retry_of=retry_of_uuid, retry_source_sha256=retry_source_sha256,
                    retry_source_path=retry_source_path, retry_source_state=retry_source_state,
                    retry_source_reason=retry_source_reason,
                    retry_source_root_digest=retry_source_root_digest,
                    root_cause=current_root,
                )
            if started_clock is None:
                # The frozen 20-minute interval begins only after child 1's
                # admission succeeds, immediately before its launch.
                experiment_started_wall = datetime.now(timezone.utc)
                started_clock = monotonic()
            state = "running"
            remaining = V04_DEADLINE_SECONDS - (monotonic() - started_clock)
            if remaining <= 0:
                raise TimeoutError("V0.4 attempt exceeded 20-minute deadline")
            child_uuid = str(uuid4())
            child_relative = Path(f"child-{child_number}")
            child_directory = staging / child_relative
            command = [
                sys.executable, "-m", "moe_cache_lab.cli", "_collect-v04-child",
                "--corpus", str(corpus.path), "--output-dir", str(child_directory),
                "--child-number", str(child_number), "--process-uuid", child_uuid,
            ]
            launch_attempt_started = datetime.now(timezone.utc)
            try:
                completed = run_process(command, env=environment, timeout=remaining)
            except BaseException as launch_error:
                failure_records.append(_write_launch_failure(
                    staging, child_number, child_uuid, child_relative,
                    launch_attempt_started, datetime.now(timezone.utc), None,
                    False, "", "",
                    f"launch exception: {type(launch_error).__name__}: {launch_error}",
                    actual_pid=None,
                ))
                raise RuntimeError(
                    f"V0.4 child {child_number} process launch failed: "
                    f"{type(launch_error).__name__}: {launch_error}"
                ) from launch_error
            child_path = child_directory / "v04-child.json"
            launch_started = _aware_timestamp(completed.started_at)
            launch_finished = _aware_timestamp(completed.finished_at)
            if completed.timed_out or completed.returncode != 0:
                failure_records.append(_write_launch_failure(
                    staging, child_number, child_uuid, child_relative,
                    launch_started, launch_finished, completed.returncode,
                    completed.timed_out, completed.stdout, completed.stderr,
                    "subprocess timeout" if completed.timed_out else "nonzero subprocess exit",
                    actual_pid=completed.pid,
                ))
                if completed.timed_out:
                    raise TimeoutError("V0.4 attempt exceeded 20-minute deadline")
                raise RuntimeError(f"V0.4 child {child_number} failed with exit code {completed.returncode}")
            try:
                child = load_v04_json(child_path, expected_format=V04_CHILD_FORMAT)
                _validate_child_record(child)
                if child["child_number"] != child_number or child["process_uuid"] != child_uuid:
                    raise ValueError("V0.4 child identity mismatch")
            except Exception as child_error:
                failure_records.append(_write_launch_failure(
                    staging, child_number, child_uuid, child_relative,
                    launch_started, launch_finished, completed.returncode, False,
                    completed.stdout, completed.stderr,
                    f"zero-exit child artifact invalid: {type(child_error).__name__}: {child_error}",
                    actual_pid=completed.pid,
                ))
                raise RuntimeError(f"V0.4 child {child_number} produced no valid terminal child manifest") from child_error
            child_records.append(_child_reference(
                child_number, child_uuid, child_relative, child_path, child,
                completed.stdout, completed.stderr, completed.pid,
            ))
        children = [load_v04_json(staging / record["path"], expected_format=V04_CHILD_FORMAT) for record in child_records]
        aggregate_input = [
            {
                "child_number": child["child_number"],
                "evaluation": [{
                    "prompt_id": row["prompt_id"], "mode_order": row["mode_order"],
                    "reference_total_ns": row["reference_total_ns"],
                    "profiled_total_ns": row["profiled_total_ns"],
                    "reference_working_set_delta": row["reference_working_set_delta"],
                    "profiled_working_set_delta": row["profiled_working_set_delta"],
                    "reference_private_delta": row["reference_private_delta"],
                    "profiled_private_delta": row["profiled_private_delta"],
                } for row in child["evaluation"]],
            }
            for child in children
        ]
        result = aggregate_children(aggregate_input)
        result_path = write_json_atomic(staging / "v04-result.json", result)
        markdown_path = staging / "v04-report.md"
        markdown_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
        state = result["decision"]
        if rerun_index == 1 and state == "valid_inconclusive":
            state = "inconclusive_final"
            result["decision"] = state
            write_json_atomic(result_path, result)
            markdown_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
        assert started_clock is not None and experiment_started_wall is not None
        elapsed_seconds = monotonic() - started_clock
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0 or elapsed_seconds > V04_DEADLINE_SECONDS:
            raise TimeoutError("V0.4 attempt exceeded 20-minute deadline during finalization")
        manifest = {
            "format": V04_SET_FORMAT, "format_version": V04_FORMAT_VERSION,
            "attempt_uuid": attempt_uuid, "state": state, "rerun_index": rerun_index,
            "rerun_of": rerun_of_uuid, "rerun_source_sha256": rerun_source_sha256,
            "rerun_source_path": rerun_source_path,
            "retry_of": retry_of_uuid, "retry_source_sha256": retry_source_sha256,
            "retry_source_path": retry_source_path, "retry_source_state": retry_source_state,
            "retry_source_reason": retry_source_reason,
            "retry_source_root_digest": retry_source_root_digest,
            "attempt_directory": final.name, "spec_sha256": V04_SPEC_SHA256,
            "started_at": started_wall.isoformat(),
            "experiment_started_at": experiment_started_wall.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "deadline_seconds": V04_DEADLINE_SECONDS,
            "elapsed_seconds": elapsed_seconds,
            "corpus_sha256": corpus.sha256,
            "trusted_stage1_set_sha256": TRUSTED_STAGE1_SET_SHA256,
            "trusted_stage1_repetition_sha256": TRUSTED_STAGE1_REP1_SHA256,
            "admissions": admission_records, "children": child_records,
            "failures": failure_records,
            "result": {"path": result_path.name, "sha256": file_sha256(result_path), "size": result_path.stat().st_size},
            "markdown": {"path": markdown_path.name, "sha256": file_sha256(markdown_path), "size": markdown_path.stat().st_size},
        }
        staging_manifest = write_json_atomic(staging / "v04-set.json", manifest)
        # A complete attempt is validated while it is still staging.  The
        # atomic rename is the publication step, never the validation step.
        validate_v04(staging_manifest)
        if monotonic() - started_clock > V04_DEADLINE_SECONDS:
            raise TimeoutError("V0.4 attempt exceeded 20-minute deadline during finalization")
        staging.replace(final)
        return final / "v04-set.json"
    except BaseException as error:
        if staging.exists():
            failure_reason = f"{type(error).__name__}: {error}"
            failure_root = _root_cause(_failure_category(error), "orchestrator", failure_reason)
            if retry_source_state in {"invalid", "blocked"} and retry_source_root_digest == failure_root["digest"]:
                state = transition_state("invalid", "prerequisite_blocked")
            else:
                state = transition_state("running" if state == "running" else "planned", "invalid") if state not in {"deferred", "blocked"} else state
            _finalize_nonvalid(
                staging, final, attempt_uuid, state, started_wall,
                failure_reason, admission_records,
                child_records, failure_records, rerun_index, rerun_of_uuid, rerun_source_sha256,
                elapsed_seconds=0.0 if started_clock is None else monotonic() - started_clock,
                experiment_started_at=experiment_started_wall,
                rerun_source_path=rerun_source_path,
                retry_of=retry_of_uuid, retry_source_sha256=retry_source_sha256,
                retry_source_path=retry_source_path, retry_source_state=retry_source_state,
                retry_source_reason=retry_source_reason,
                retry_source_root_digest=retry_source_root_digest,
                root_cause=failure_root,
            )
        raise


def _finalize_nonvalid(
    staging: Path,
    final: Path,
    attempt_uuid: str,
    state: str,
    started: datetime,
    reason: str,
    admissions: list[dict[str, Any]],
    children: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    rerun_index: int,
    rerun_of: str | None,
    rerun_source_sha256: str | None,
    *,
    elapsed_seconds: float = 0.0,
    experiment_started_at: datetime | None = None,
    rerun_source_path: str | None = None,
    retry_of: str | None = None,
    retry_source_sha256: str | None = None,
    retry_source_path: str | None = None,
    retry_source_state: str | None = None,
    retry_source_reason: str | None = None,
    retry_source_root_digest: str | None = None,
    root_cause: dict[str, Any] | None = None,
) -> Path:
    finished = datetime.now(timezone.utc)
    manifest = {
        "format": V04_SET_FORMAT, "format_version": V04_FORMAT_VERSION,
        "attempt_uuid": attempt_uuid, "state": state, "rerun_index": rerun_index,
        "rerun_of": rerun_of, "rerun_source_sha256": rerun_source_sha256,
        "rerun_source_path": rerun_source_path,
        "retry_of": retry_of, "retry_source_sha256": retry_source_sha256,
        "retry_source_path": retry_source_path, "retry_source_state": retry_source_state,
        "retry_source_reason": retry_source_reason,
        "retry_source_root_digest": retry_source_root_digest,
        "attempt_directory": final.name, "spec_sha256": V04_SPEC_SHA256,
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "experiment_started_at": experiment_started_at.isoformat() if experiment_started_at else None,
        "elapsed_seconds": elapsed_seconds,
        "corpus_sha256": STAGE1_CORPUS_SHA256,
        "trusted_stage1_set_sha256": TRUSTED_STAGE1_SET_SHA256,
        "trusted_stage1_repetition_sha256": TRUSTED_STAGE1_REP1_SHA256,
        "reason": reason, "admissions": admissions, "children": children,
        "failures": failures,
        "root_cause": root_cause or _root_cause("unknown", "orchestrator", reason),
    }
    manifest_path = write_json_atomic(staging / "v04-set.json", manifest)
    validate_v04(manifest_path)
    staging.replace(final)
    return final / "v04-set.json"


def _child_reference(
    child_number: int,
    process_uuid: str,
    relative: Path,
    path: Path,
    child: dict[str, Any],
    stdout: Any,
    stderr: Any,
    launch_pid: int,
) -> dict[str, Any]:
    return {
        "child_number": child_number, "process_uuid": process_uuid,
        "state": child["state"],
        "launch_pid": launch_pid,
        "path": (relative / "v04-child.json").as_posix(),
        "sha256": file_sha256(path), "size": path.stat().st_size,
        "started_at": child["started_at"], "finished_at": child["finished_at"],
        "stdout": _output_text(stdout), "stderr": _output_text(stderr),
    }


def _output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


def _write_launch_failure(
    staging: Path,
    child_number: int,
    process_uuid: str,
    relative: Path,
    started: datetime,
    finished: datetime,
    returncode: int | None,
    timed_out: bool,
    stdout: Any,
    stderr: Any,
    reason: str,
    *,
    actual_pid: int | None,
) -> dict[str, Any]:
    artifact = {
        "format": "moe-cache-lab.v04-launch-failure", "format_version": 1,
        "child_number": child_number, "process_uuid": process_uuid,
        "actual_pid": actual_pid,
        "assigned_path": relative.as_posix(), "started_at": started.isoformat(),
        "finished_at": finished.isoformat(), "returncode": returncode,
        "timed_out": timed_out, "stdout": _output_text(stdout),
        "stderr": _output_text(stderr), "reason": reason,
        "failure_category": (
            "launch_exception" if actual_pid is None else
            ("timeout" if timed_out else ("child_exit" if returncode not in (None, 0) else "child_artifact"))
        ),
        "terminal_manifest_path": None, "terminal_manifest_sha256": None,
        "terminal_manifest_size": None,
    }
    terminal = staging / relative / "v04-child.json"
    if terminal.is_file():
        artifact["terminal_manifest_path"] = (relative / "v04-child.json").as_posix()
        artifact["terminal_manifest_sha256"] = file_sha256(terminal)
        artifact["terminal_manifest_size"] = terminal.stat().st_size
    path = write_json_atomic(staging / f"launch-failure-{child_number}.json", artifact)
    return {
        "child_number": child_number, "process_uuid": process_uuid,
        "actual_pid": actual_pid,
        "path": path.name, "sha256": file_sha256(path), "size": path.stat().st_size,
        "started_at": artifact["started_at"], "finished_at": artifact["finished_at"],
    }


def _validate_invalid_child_record(child: dict[str, Any]) -> None:
    _require_exact_keys(child, {
        "format", "format_version", "state", "child_number", "process_uuid",
        "pid", "parent_pid", "started_at", "finished_at", "reason",
        "reason_type", "reason_message", "failure_phase", "partial",
    }, "invalid child")
    if child.get("format") != V04_CHILD_FORMAT or child.get("format_version") != 1 or child.get("state") != "invalid":
        raise ValueError("invalid V0.4 child terminal manifest")
    _child_number(child.get("child_number"))
    UUID(child.get("process_uuid", ""))
    child_start, child_finish = _validate_interval(child["started_at"], child["finished_at"])
    if any(isinstance(child.get(name), bool) or not isinstance(child.get(name), int) or child[name] <= 0 for name in ("pid", "parent_pid")):
        raise ValueError("invalid V0.4 child PIDs must be positive integers")
    if not isinstance(child.get("reason"), str) or not child["reason"]:
        raise ValueError("invalid V0.4 child requires a reason")
    if any(not isinstance(child.get(name), str) or not child[name] for name in ("reason_type", "failure_phase")) or not isinstance(child.get("reason_message"), str):
        raise ValueError("invalid V0.4 child requires structured failure fields")
    if child["reason"] != f"{child['reason_type']}: {child['reason_message']}" or child["failure_phase"] not in {"model_load", "calibration", "evaluation", "validation", "cleanup"}:
        raise ValueError("invalid V0.4 child reason/phase reconciliation failed")
    _require_exact_keys(child.get("partial"), {
        "model_load", "completed_calibration_runs", "completed_evaluation_prompts",
        "last_resource_snapshot",
    }, "invalid child partial")
    if any(isinstance(child["partial"][name], bool) or not isinstance(child["partial"][name], int) or child["partial"][name] < 0 for name in ("completed_calibration_runs", "completed_evaluation_prompts")):
        raise ValueError("invalid V0.4 child partial counts are invalid")
    if child["partial"]["completed_calibration_runs"] > 8 or child["partial"]["completed_evaluation_prompts"] > 8:
        raise ValueError("invalid V0.4 child partial counts exceed frozen schedules")
    calibration_count = child["partial"]["completed_calibration_runs"]
    evaluation_count = child["partial"]["completed_evaluation_prompts"]
    if child["partial"]["model_load"] is None:
        if calibration_count or evaluation_count or child["partial"]["last_resource_snapshot"] is not None or child["failure_phase"] != "model_load":
            raise ValueError("invalid V0.4 child pre-model partial dependencies are impossible")
    else:
        if evaluation_count and calibration_count != 8:
            raise ValueError("invalid V0.4 child evaluation progress requires complete calibration")
        if (calibration_count or evaluation_count) and child["partial"]["last_resource_snapshot"] is None:
            raise ValueError("invalid V0.4 child completed run requires a resource snapshot")
        if child["failure_phase"] == "calibration" and evaluation_count != 0:
            raise ValueError("invalid V0.4 child calibration phase cannot contain evaluation progress")
        if child["failure_phase"] in {"evaluation", "validation"} and calibration_count != 8:
            raise ValueError("invalid V0.4 child later phase requires complete calibration")
        if child["failure_phase"] == "validation" and evaluation_count != 8:
            raise ValueError("invalid V0.4 child validation phase requires complete evaluation")
    if child["partial"]["model_load"] is not None:
        _validate_model_load_record(child["partial"]["model_load"])
        load_start, load_finish = _validate_interval(
            child["partial"]["model_load"]["started_at"],
            child["partial"]["model_load"]["finished_at"],
        )
        if load_start < child_start or load_finish > child_finish:
            raise ValueError("invalid V0.4 child model-load interval is outside child")
    snapshot = child["partial"]["last_resource_snapshot"]
    if snapshot is not None:
        _require_exact_keys(snapshot, set(ProcessMemory.__dataclass_fields__), "invalid child resource snapshot")
        ProcessMemory(**snapshot)
        if not child_start <= _aware_timestamp(snapshot["timestamp"]) <= child_finish:
            raise ValueError("invalid V0.4 child resource snapshot is outside child interval")


def validate_v04(path: str | Path, *, _seen: set[Path] | None = None) -> Path:
    source = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if source in seen:
        raise ValueError("V0.4 lineage cycle detected")
    seen.add(source)
    value = load_v04_json(source, expected_format=V04_SET_FORMAT)
    UUID(value.get("attempt_uuid", ""))
    _validate_attempt_binding(source, value)
    _validate_rerun(value, source, seen)
    state = value.get("state")
    _validate_elapsed(value.get("elapsed_seconds"), require_within_deadline=state not in {"deferred", "invalid", "blocked"})
    if state in {"deferred", "invalid", "blocked"}:
        _require_exact_keys(value, {
            "format", "format_version", "attempt_uuid", "state", "rerun_index",
            "rerun_of", "rerun_source_sha256", "rerun_source_path",
            "retry_of", "retry_source_sha256", "retry_source_path",
            "retry_source_state", "retry_source_reason", "retry_source_root_digest",
            "attempt_directory", "spec_sha256",
            "started_at", "experiment_started_at", "finished_at", "elapsed_seconds", "reason", "admissions",
            "children", "failures", "corpus_sha256", "trusted_stage1_set_sha256",
            "trusted_stage1_repetition_sha256", "root_cause",
        }, "nonvalid set")
        if not isinstance(value.get("reason"), str) or not value["reason"]:
            raise ValueError("nonvalid V0.4 attempt requires a reason")
        if value.get("corpus_sha256") != STAGE1_CORPUS_SHA256 or value.get("trusted_stage1_set_sha256") != TRUSTED_STAGE1_SET_SHA256 or value.get("trusted_stage1_repetition_sha256") != TRUSTED_STAGE1_REP1_SHA256:
            raise ValueError("nonvalid V0.4 frozen input hashes mismatch")
        root_cause = _validate_root_cause(value.get("root_cause"))
        attempt_start, attempt_finish = _validate_interval(value["started_at"], value["finished_at"])
        experiment_start_raw = value.get("experiment_started_at")
        if experiment_start_raw is not None:
            experiment_start = _aware_timestamp(experiment_start_raw)
            if experiment_start < attempt_start or experiment_start > attempt_finish:
                raise ValueError("V0.4 experiment start is outside the attempt interval")
        elif value["elapsed_seconds"] != 0:
            raise ValueError("V0.4 pre-experiment terminal attempt must have zero elapsed time")
        admissions = value.get("admissions")
        children = value.get("children")
        failures = value.get("failures")
        if not isinstance(admissions, list) or len(admissions) > 4 or not isinstance(children, list) or len(children) > 4 or not isinstance(failures, list) or len(failures) > 1:
            raise ValueError("nonvalid V0.4 partial artifact lists are invalid")
        admission_intervals: dict[int, tuple[datetime, datetime]] = {}
        admission_states: dict[int, bool] = {}
        for index, record in enumerate(admissions, 1):
            _validate_artifact_reference(source.parent, record, index, "admission")
            artifact = load_v04_json(_contained(source.parent, record["path"]))
            _validate_admission_artifact(artifact, index, require_admitted=False)
            admission_intervals[index] = _validate_interval(artifact["started_at"], artifact["finished_at"])
            admission_states[index] = artifact["admitted"]
        previous_finish = attempt_start
        child_states: dict[int, str] = {}
        for index, record in enumerate(children, 1):
            child = _validate_child_reference(source.parent, record, index, allow_invalid=True)
            child_states[index] = child["state"]
            child_start, child_finish = _validate_interval(record["started_at"], record["finished_at"])
            if child_start < previous_finish or child_start < attempt_start or child_finish > attempt_finish:
                raise ValueError("V0.4 child intervals are outside or overlap the parent interval")
            if admission_intervals[index][1] > child_start:
                raise ValueError("V0.4 admission must finish before its child")
            previous_finish = child_finish
        for record in failures:
            _validate_failure_reference(source.parent, record, attempt_start, attempt_finish)
            if record["child_number"] not in admission_intervals or admission_intervals[record["child_number"]][1] > _aware_timestamp(record["started_at"]):
                raise ValueError("V0.4 admission must finish before launch failure")
        admission_numbers = [record["child_number"] for record in admissions]
        terminal_numbers = [record["child_number"] for record in children] + [record["child_number"] for record in failures]
        if admission_numbers != list(range(1, len(admission_numbers) + 1)) or any(number not in admission_numbers for number in terminal_numbers) or len(set(terminal_numbers)) != len(terminal_numbers):
            raise ValueError("nonvalid V0.4 admission/child/failure reconciliation mismatch")
        terminal_uuids = [record["process_uuid"] for record in children + failures]
        if len(set(terminal_uuids)) != len(terminal_uuids):
            raise ValueError("nonvalid V0.4 process UUIDs must be unique")
        if state == "blocked" and value.get("retry_of") is None:
            raise ValueError("blocked V0.4 attempt requires bound retry lineage")
        deferred_history = state == "deferred" or (state == "blocked" and root_cause["category"] in {"missing_prerequisite", "busy_threshold"})
        if deferred_history:
            if not admissions or admission_states[len(admissions)] is not False or failures or any(item != "valid" for item in child_states.values()):
                raise ValueError("deferred/blocked V0.4 history must end in one rejected admission without invalid launch evidence")
            expected_children = list(range(1, len(admissions)))
            if list(child_states) != expected_children or any(not admission_states[number] for number in expected_children):
                raise ValueError("deferred/blocked V0.4 earlier admitted children do not reconcile")
        else:
            if state not in {"invalid", "blocked"} or not admissions or not all(admission_states.values()):
                raise ValueError("invalid/blocked V0.4 launch history requires admitted launch evidence")
            terminal_number = len(admissions)
            has_invalid_child = child_states.get(terminal_number) == "invalid"
            has_failure = len(failures) == 1 and failures[0]["child_number"] == terminal_number
            if has_invalid_child == has_failure or any(child_states.get(number) != "valid" for number in range(1, terminal_number)):
                raise ValueError("invalid/blocked V0.4 history requires exactly one final invalid-child or launch-failure reference")
            if set(child_states) - set(range(1, terminal_number + 1)):
                raise ValueError("invalid/blocked V0.4 child history contains an impossible child number")
        return source
    if state not in {"valid_stable", "valid_inconclusive", "inconclusive_final"}:
        raise ValueError("invalid V0.4 attempt state")
    _require_exact_keys(value, {
        "format", "format_version", "attempt_uuid", "state", "rerun_index",
        "rerun_of", "rerun_source_sha256", "rerun_source_path",
        "retry_of", "retry_source_sha256", "retry_source_path",
        "retry_source_state", "retry_source_reason", "retry_source_root_digest",
        "attempt_directory", "spec_sha256",
        "started_at", "experiment_started_at", "finished_at", "deadline_seconds", "elapsed_seconds",
        "corpus_sha256", "trusted_stage1_set_sha256",
        "trusted_stage1_repetition_sha256", "admissions", "children",
        "failures", "result", "markdown",
    }, "valid set")
    attempt_start, attempt_finish = _validate_interval(value["started_at"], value["finished_at"])
    experiment_start = _aware_timestamp(value.get("experiment_started_at"))
    if experiment_start < attempt_start or experiment_start > attempt_finish or (attempt_finish - experiment_start).total_seconds() > V04_DEADLINE_SECONDS:
        raise ValueError("valid V0.4 wall interval exceeds the frozen deadline")
    if (
        value.get("deadline_seconds") != V04_DEADLINE_SECONDS
        or value.get("corpus_sha256") != STAGE1_CORPUS_SHA256
        or value.get("trusted_stage1_set_sha256") != TRUSTED_STAGE1_SET_SHA256
        or value.get("trusted_stage1_repetition_sha256") != TRUSTED_STAGE1_REP1_SHA256
    ):
        raise ValueError("V0.4 set frozen metadata mismatch")
    admissions = value.get("admissions")
    if not isinstance(admissions, list) or len(admissions) != 4:
        raise ValueError("valid V0.4 set requires four admissions")
    admission_intervals: list[tuple[datetime, datetime]] = []
    for index, record in enumerate(admissions, 1):
        admission_path = _validate_artifact_reference(source.parent, record, index, "admission")
        admission_value = load_v04_json(admission_path)
        _validate_admission_artifact(admission_value, index, require_admitted=True)
        admission_intervals.append(_validate_interval(admission_value["started_at"], admission_value["finished_at"]))
    children_records = value.get("children")
    if not isinstance(children_records, list) or len(children_records) != 4:
        raise ValueError("valid V0.4 set requires four children")
    if value.get("failures") != []:
        raise ValueError("valid V0.4 set cannot contain launch failures")
    children = []
    previous_finish = attempt_start
    for index, record in enumerate(children_records, 1):
        child = _validate_child_reference(source.parent, record, index, allow_invalid=False)
        child_start, child_finish = _validate_interval(record["started_at"], record["finished_at"])
        if child_start < previous_finish or child_start < attempt_start or child_finish > attempt_finish:
            raise ValueError("V0.4 child intervals are outside or overlap the parent interval")
        admission_start, admission_finish = admission_intervals[index - 1]
        if admission_finish > child_start or (index == 1 and not admission_finish <= experiment_start <= child_start) or (index > 1 and admission_start < previous_finish):
            raise ValueError("V0.4 admission/child chronology is invalid")
        previous_finish = child_finish
        children.append(child)
    _validate_cross_child_semantics(children)
    if len({child["process_uuid"] for child in children}) != 4:
        raise ValueError("valid V0.4 child process UUIDs must be unique")
    aggregate_input = [{
        "child_number": child["child_number"],
        "evaluation": [{
            "prompt_id": row["prompt_id"], "mode_order": row["mode_order"],
            "reference_total_ns": row["reference_total_ns"], "profiled_total_ns": row["profiled_total_ns"],
            "reference_working_set_delta": row["reference_working_set_delta"],
            "profiled_working_set_delta": row["profiled_working_set_delta"],
            "reference_private_delta": row["reference_private_delta"],
            "profiled_private_delta": row["profiled_private_delta"],
        } for row in child["evaluation"]],
    } for child in children]
    expected = aggregate_children(aggregate_input)
    result_path = _validate_named_reference(source.parent, value["result"], "result")
    result = load_v04_json(result_path, expected_format=V04_RESULT_FORMAT)
    if state == "inconclusive_final":
        expected["decision"] = "inconclusive_final"
    if result != expected or state != result["decision"]:
        raise ValueError("V0.4 aggregate result mismatch")
    markdown_path = _validate_named_reference(source.parent, value["markdown"], "Markdown")
    if markdown_path.read_text(encoding="utf-8") != render_markdown(result):
        raise ValueError("V0.4 Markdown hash/content mismatch")
    return source


def _validate_attempt_binding(source: Path, value: dict[str, Any]) -> None:
    expected = value.get("attempt_directory")
    if not isinstance(expected, str) or expected != f"v0.4-cpu-profiler-attempt-{value.get('attempt_uuid')}":
        raise ValueError("V0.4 attempt directory is not bound to its UUID")
    if source.parent.name not in {expected, f".{expected}.staging"}:
        raise ValueError("V0.4 manifest is not stored in its bound attempt directory")
    if value.get("spec_sha256") != V04_SPEC_SHA256:
        raise ValueError("V0.4 frozen specification hash mismatch")
    specification = Path(__file__).resolve().parents[2] / "V04_EXPERIMENT.md"
    if not specification.is_file() or file_sha256(specification) != V04_SPEC_SHA256:
        raise ValueError("V0.4 local frozen specification bytes do not match the pinned hash")


def _validate_rerun(value: dict[str, Any], manifest_path: Path, seen: set[Path]) -> None:
    index = value.get("rerun_index")
    rerun_of = value.get("rerun_of")
    source_hash = value.get("rerun_source_sha256")
    source_path = value.get("rerun_source_path")
    retry_of = value.get("retry_of")
    retry_hash = value.get("retry_source_sha256")
    retry_path = value.get("retry_source_path")
    retry_state = value.get("retry_source_state")
    retry_reason = value.get("retry_source_reason")
    retry_root_digest = value.get("retry_source_root_digest")
    if retry_of is not None and rerun_of is not None:
        raise ValueError("V0.4 retry and statistical rerun lineage are mutually exclusive")
    if index == 0 and rerun_of is None and source_hash is None and source_path is None:
        pass
    elif index == 1 and isinstance(rerun_of, str) and isinstance(source_hash, str) and isinstance(source_path, str):
        UUID(rerun_of)
        _sha256_text(source_hash)
    else:
        raise ValueError("V0.4 rerun provenance is invalid")
    if retry_of is None:
        if any(item is not None for item in (retry_hash, retry_path, retry_state, retry_reason, retry_root_digest)):
            raise ValueError("V0.4 retry provenance is incomplete")
    else:
        UUID(retry_of)
        _sha256_text(retry_hash)
        _sha256_text(retry_root_digest)
        if not isinstance(retry_path, str) or retry_state not in {"deferred", "invalid", "blocked"} or not isinstance(retry_reason, str) or not retry_reason:
            raise ValueError("V0.4 retry provenance is invalid")
    lineage_path = source_path if rerun_of is not None else retry_path
    lineage_uuid = rerun_of if rerun_of is not None else retry_of
    lineage_hash = source_hash if rerun_of is not None else retry_hash
    if lineage_path is not None:
        root = manifest_path.parent.parent.resolve()
        prior_path = _contained(root, lineage_path)
        if prior_path.resolve() == manifest_path.resolve() or not prior_path.is_file() or file_sha256(prior_path) != lineage_hash:
            raise ValueError("V0.4 lineage source path/hash is invalid")
        validate_v04(prior_path, _seen=seen)
        prior = load_v04_json(prior_path, expected_format=V04_SET_FORMAT)
        if prior.get("attempt_uuid") != lineage_uuid:
            raise ValueError("V0.4 lineage UUID does not bind its source")
        if rerun_of is not None:
            if prior.get("state") != "valid_inconclusive" or prior.get("rerun_index") != 0 or prior.get("rerun_of") is not None:
                raise ValueError("V0.4 statistical rerun source is invalid")
        elif prior.get("state") != retry_state or prior.get("reason") != retry_reason or prior.get("root_cause", {}).get("digest") != retry_root_digest:
            raise ValueError("V0.4 recovery retry source is invalid")
        consumers = 0
        for candidate in root.glob("*v0.4-cpu-profiler-attempt-*/v04-set.json"):
            try:
                item = load_v04_json(candidate, expected_format=V04_SET_FORMAT)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if item.get("rerun_of") == lineage_uuid or item.get("retry_of") == lineage_uuid:
                consumers += 1
        if consumers > 1:
            raise ValueError("V0.4 lineage source has multiple sibling consumers")


def _sha256_text(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("V0.4 SHA-256 text is invalid")
    return value


def _root_cause(category: str, component: str, message: str, status: str | None = None) -> dict[str, Any]:
    if category not in {
        "missing_prerequisite", "busy_threshold", "probe_failure", "timeout",
        "child_exit", "child_artifact", "low_memory", "correctness", "unknown",
    } or not component or not message:
        raise ValueError("V0.4 root-cause fields are invalid")
    core = {"category": category, "component": component, "status": status, "message": message}
    digest = sha256_bytes(json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return {**core, "digest": digest}


def _admission_root_cause(admission: AdmissionResult, component: str) -> dict[str, Any]:
    """Fingerprint the underlying admission failure, not its generic disposition."""
    category = "busy_threshold" if admission.reason == "host busy or memory gate failed" else "missing_prerequisite"
    identity = {
        "reason": admission.reason,
        "errors": list(admission.errors),
        "audit": admission.audit,
    }
    return _root_cause(
        category, component,
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    )


def _failure_category(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, MemoryError):
        return "low_memory"
    text = str(error).lower()
    if "exit code" in text:
        return "child_exit"
    if "terminal child manifest" in text or "child artifact" in text:
        return "child_artifact"
    if "probe" in text:
        return "probe_failure"
    if isinstance(error, (ValueError, RuntimeError)):
        return "correctness"
    return "unknown"


def _validate_root_cause(value: Any) -> dict[str, Any]:
    _require_exact_keys(value, {"category", "component", "status", "message", "digest"}, "root cause")
    expected = _root_cause(value["category"], value["component"], value["message"], value["status"])
    if value != expected:
        raise ValueError("V0.4 root-cause fingerprint mismatch")
    return value


def _reject_consumed_lineage(root: Path, source_uuid: str | None, source_path: Path) -> None:
    if source_uuid is None:
        return
    for manifest_path in root.glob("*v0.4-cpu-profiler-attempt-*/v04-set.json"):
        if manifest_path.resolve() == source_path.resolve():
            continue
        try:
            record = load_v04_json(manifest_path, expected_format=V04_SET_FORMAT)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if record.get("rerun_of") == source_uuid or record.get("retry_of") == source_uuid:
            raise ValueError("V0.4 lineage source has already been consumed by a sibling attempt")


def _unresolved_retry_sources(root: Path) -> list[Path]:
    manifests = sorted(root.glob("v0.4-cpu-profiler-attempt-*/v04-set.json"))
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in manifests:
        validate_v04(path)
        records.append((path, load_v04_json(path, expected_format=V04_SET_FORMAT)))
    consumed = {
        value
        for _, record in records
        for value in (record.get("retry_of"), record.get("rerun_of"))
        if value is not None
    }
    return [
        path for path, record in records
        if record.get("state") in {"deferred", "invalid", "blocked"}
        and record.get("attempt_uuid") not in consumed
        and record.get("root_cause", {}).get("category") != "busy_threshold"
    ]


def _validate_elapsed(value: Any, *, require_within_deadline: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (require_within_deadline and value > V04_DEADLINE_SECONDS):
        raise ValueError("V0.4 elapsed time is invalid")


def _validate_named_reference(base: Path, record: Any, label: str) -> Path:
    _require_exact_keys(record, {"path", "sha256", "size"}, f"{label} reference")
    path = _contained(base, record["path"])
    if not path.is_file() or file_sha256(path) != record["sha256"] or path.stat().st_size != record["size"]:
        raise ValueError(f"V0.4 {label} hash/size mismatch")
    return path


def _validate_artifact_reference(base: Path, record: Any, number: int, label: str) -> Path:
    _require_exact_keys(record, {"child_number", "path", "sha256", "size"}, f"{label} reference")
    if record.get("child_number") != number:
        raise ValueError(f"V0.4 {label} order mismatch")
    return _validate_named_reference(base, {name: record[name] for name in ("path", "sha256", "size")}, label)


def _validate_child_reference(base: Path, record: Any, number: int, *, allow_invalid: bool) -> dict[str, Any]:
    _require_exact_keys(record, {
        "child_number", "process_uuid", "state", "path", "sha256", "size",
        "started_at", "finished_at", "stdout", "stderr", "launch_pid",
    }, "child reference")
    if record.get("child_number") != number or not isinstance(record.get("stdout"), str) or not isinstance(record.get("stderr"), str):
        raise ValueError("V0.4 child reference identity/order mismatch")
    UUID(record.get("process_uuid", ""))
    if isinstance(record.get("launch_pid"), bool) or not isinstance(record.get("launch_pid"), int) or record["launch_pid"] <= 0:
        raise ValueError("V0.4 child launch PID is invalid")
    path = _validate_named_reference(base, {name: record[name] for name in ("path", "sha256", "size")}, "child")
    child = load_v04_json(path, expected_format=V04_CHILD_FORMAT)
    if child.get("state") != record.get("state") or child.get("child_number") != number or child.get("process_uuid") != record.get("process_uuid"):
        raise ValueError("V0.4 child reference does not bind the child artifact")
    if child.get("started_at") != record.get("started_at") or child.get("finished_at") != record.get("finished_at"):
        raise ValueError("V0.4 child reference interval mismatch")
    if child.get("state") == "valid":
        _validate_child_record(child)
    elif allow_invalid:
        _validate_invalid_child_record(child)
    else:
        raise ValueError("valid V0.4 set contains invalid child")
    return child


def _validate_failure_reference(
    base: Path, record: Any, attempt_start: datetime, attempt_finish: datetime,
) -> None:
    _require_exact_keys(record, {
        "child_number", "process_uuid", "path", "sha256", "size",
        "started_at", "finished_at", "actual_pid",
    }, "launch-failure reference")
    number = _child_number(record.get("child_number"))
    UUID(record.get("process_uuid", ""))
    if record.get("actual_pid") is not None and (
        isinstance(record.get("actual_pid"), bool)
        or not isinstance(record.get("actual_pid"), int)
        or record["actual_pid"] <= 0
    ):
        raise ValueError("V0.4 launch-failure PID is invalid")
    path = _validate_named_reference(base, {name: record[name] for name in ("path", "sha256", "size")}, "launch failure")
    artifact = load_v04_json(path)
    _require_exact_keys(artifact, {
        "format", "format_version", "child_number", "process_uuid",
        "actual_pid", "assigned_path", "started_at", "finished_at", "returncode",
        "timed_out", "stdout", "stderr", "reason", "terminal_manifest_path",
        "terminal_manifest_sha256", "terminal_manifest_size", "failure_category",
    }, "launch-failure artifact")
    if artifact.get("format") != "moe-cache-lab.v04-launch-failure" or artifact.get("child_number") != number or artifact.get("process_uuid") != record.get("process_uuid") or artifact.get("actual_pid") != record.get("actual_pid"):
        raise ValueError("V0.4 launch-failure identity mismatch")
    if artifact.get("started_at") != record.get("started_at") or artifact.get("finished_at") != record.get("finished_at"):
        raise ValueError("V0.4 launch-failure interval reference mismatch")
    started, finished = _validate_interval(artifact["started_at"], artifact["finished_at"])
    if started < attempt_start or finished > attempt_finish:
        raise ValueError("V0.4 launch-failure interval is outside attempt")
    if not isinstance(artifact.get("assigned_path"), str) or artifact["assigned_path"] != f"child-{number}" or type(artifact.get("timed_out")) is not bool:
        raise ValueError("V0.4 launch-failure assigned path/state mismatch")
    returncode = artifact.get("returncode")
    launch_exception = artifact.get("failure_category") == "launch_exception"
    if launch_exception:
        if artifact.get("actual_pid") is not None or returncode is not None or artifact["timed_out"]:
            raise ValueError("V0.4 prelaunch failure must not invent a PID or return code")
    elif artifact.get("actual_pid") is None:
        raise ValueError("launched V0.4 failure requires a positive actual PID")
    elif artifact["timed_out"]:
        if returncode is not None:
            raise ValueError("timed-out V0.4 launch failure cannot have a return code")
    elif isinstance(returncode, bool) or not isinstance(returncode, int):
        raise ValueError("non-timeout V0.4 launch failure requires an integer return code")
    expected_category = (
        "launch_exception" if launch_exception else
        ("timeout" if artifact["timed_out"] else ("child_exit" if returncode != 0 else "child_artifact"))
    )
    if artifact.get("failure_category") != expected_category:
        raise ValueError("V0.4 launch-failure category is inconsistent")
    if any(not isinstance(artifact.get(name), str) for name in ("stdout", "stderr", "reason")) or not artifact["reason"]:
        raise ValueError("V0.4 launch-failure text fields are invalid")
    terminal_path = artifact["terminal_manifest_path"]
    if terminal_path is None:
        if artifact["terminal_manifest_sha256"] is not None or artifact["terminal_manifest_size"] is not None:
            raise ValueError("V0.4 launch-failure terminal manifest reference is incomplete")
    else:
        if terminal_path != f"child-{number}/v04-child.json":
            raise ValueError("V0.4 launch-failure terminal manifest path is not the assigned child path")
        terminal = _contained(base, terminal_path)
        if file_sha256(terminal) != artifact["terminal_manifest_sha256"] or terminal.stat().st_size != artifact["terminal_manifest_size"]:
            raise ValueError("V0.4 launch-failure terminal manifest hash/size mismatch")


def _validate_cross_child_semantics(children: list[dict[str, Any]]) -> None:
    baseline: dict[tuple[str, str], PromptRun] = {}
    for child in children:
        records = list(child["calibration"]) + [record for row in child["evaluation"] for record in row["runs"]]
        for record in records:
            run = _prompt_run_from_record(record)
            key = (run.prompt_id, run.mode)
            prior = baseline.setdefault(key, run)
            if run.semantic != prior.semantic:
                raise ValueError("V0.4 cross-child semantic repeatability mismatch")
            if len(run.events) != len(prior.events):
                raise ValueError("V0.4 cross-child routing count mismatch")
            for actual, expected in zip(run.events, prior.events):
                if (
                    actual.phase, actual.token_position, actual.layer, actual.token_id,
                    actual.selected_experts,
                ) != (
                    expected.phase, expected.token_position, expected.layer, expected.token_id,
                    expected.selected_experts,
                ) or len(actual.selected_probabilities) != len(expected.selected_probabilities):
                    raise ValueError("V0.4 cross-child routing identity mismatch")
                if any(abs(left - right) > 1e-6 for left, right in zip(actual.selected_probabilities, expected.selected_probabilities)):
                    raise ValueError("V0.4 cross-child routing probability mismatch")


def _contained(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("V0.4 artifact path must be relative")
    path = (base / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError("V0.4 artifact path escapes attempt") from error
    return path


def _admission_record(child_number: int, admission: AdmissionResult) -> dict[str, Any]:
    return {
        "format": "moe-cache-lab.v04-admission", "format_version": 1,
        "child_number": child_number, "admitted": admission.admitted,
        "started_at": admission.started_at, "finished_at": admission.finished_at,
        "started_monotonic_ns": admission.started_monotonic_ns,
        "finished_monotonic_ns": admission.finished_monotonic_ns,
        "interval_ns": admission.finished_monotonic_ns - admission.started_monotonic_ns,
        "prime": admission.prime.to_record() if admission.prime else None,
        "baseline": admission.baseline.to_record() if admission.baseline else None,
        "samples": [sample.to_record() for sample in admission.samples],
        "errors": list(admission.errors), "reason": admission.reason,
        "audit": admission.audit,
    }


def _prompt_run_record(run: PromptRun) -> dict[str, Any]:
    memory_before = getattr(run, "_memory_before", None)
    memory_after = getattr(run, "_memory_after", None)
    return {
        "prompt_id": run.prompt_id, "mode": run.mode,
        "semantic": asdict(run.semantic), "timing": asdict(run.timing),
        "events": [_event_record(event) for event in run.events],
        "event_count": len(run.events),
        "assignment_count": sum(len(event.selected_experts) for event in run.events),
        "memory_before": asdict(memory_before) if memory_before else None,
        "memory_after": asdict(memory_after) if memory_after else None,
        "process_cpu_user_ns": getattr(run, "_process_user_ns", 0),
        "process_cpu_system_ns": getattr(run, "_process_system_ns", 0),
    }


def _event_record(event: RoutingEvent) -> dict[str, Any]:
    return {
        "phase": event.phase, "token_position": event.token_position,
        "layer": event.layer, "token_id": event.token_id,
        "selected_experts": list(event.selected_experts),
        "selected_probabilities": list(event.selected_probabilities),
    }


def _event_from_record(record: dict[str, Any]) -> RoutingEvent:
    _require_exact_keys(record, {
        "phase", "token_position", "layer", "token_id", "selected_experts",
        "selected_probabilities",
    }, "routing event")
    return RoutingEvent(
        phase=record["phase"], token_position=record["token_position"],
        layer=record["layer"], token_id=record["token_id"],
        selected_experts=tuple(record["selected_experts"]),
        selected_probabilities=tuple(record["selected_probabilities"]),
    )


def _validate_profiled_against_trusted(runs: tuple[PromptRun, ...]) -> None:
    validate_trusted_stage1_paths()
    trusted = load_stage1_repetition_manifest(TRUSTED_STAGE1_REP1)
    by_id = {item.prompt.id: item for item in trusted.prompts}
    if tuple(run.prompt_id for run in runs) != CALIBRATION_IDS + EVALUATION_IDS:
        raise ValueError("V0.4 profiled prompt mapping does not match canonical Stage 1")
    for run in runs:
        item = by_id[run.prompt_id]
        prompt_token_ids = {
            event.token_position: event.token_id
            for event in item.trace.events if event.phase == "prompt"
        }
        if run.semantic.input_ids != tuple(prompt_token_ids[position] for position in sorted(prompt_token_ids)):
            raise ValueError(f"V0.4 input IDs mismatch canonical Stage 1 for {run.prompt_id}")
        eos = item.record["eos"]
        expected_semantic = (
            tuple(eos["routed_non_eos_token_ids"]), eos["emitted_non_eos_text"],
            eos["terminal_eos_token_id"], eos["terminal_eos_text"],
            eos["terminal_eos_candidate_position"], eos["actual_decode_input_steps_routed"],
            eos["eos_emitted"], eos["horizon_exhausted"],
        )
        actual_semantic = (
            run.semantic.routed_non_eos_token_ids, run.semantic.routed_non_eos_text,
            run.semantic.terminal_eos_token_id, run.semantic.terminal_eos_text,
            run.semantic.terminal_eos_candidate_position, run.semantic.actual_decode_input_steps,
            run.semantic.eos_emitted, run.semantic.horizon_exhausted,
        )
        if actual_semantic != expected_semantic:
            raise ValueError(f"V0.4 semantic mismatch canonical Stage 1 for {run.prompt_id}")
        if len(run.events) != len(item.trace.events):
            raise ValueError(f"V0.4 routing count mismatch canonical Stage 1 for {run.prompt_id}")
        for actual, expected in zip(run.events, item.trace.events):
            if (
                actual.phase, actual.token_position, actual.layer, actual.token_id,
                actual.selected_experts,
            ) != (
                expected.phase, expected.token_position, expected.layer, expected.token_id,
                expected.selected_experts,
            ):
                raise ValueError(f"V0.4 routing mismatch canonical Stage 1 for {run.prompt_id}")
            if len(actual.selected_probabilities) != len(expected.selected_probabilities) or any(
                not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > 1e-6
                for left, right in zip(actual.selected_probabilities, expected.selected_probabilities)
            ):
                raise ValueError(f"V0.4 probability mismatch canonical Stage 1 for {run.prompt_id}")


def _prompt_run_from_record(record: dict[str, Any]) -> PromptRun:
    _require_exact_keys(record, {
        "prompt_id", "mode", "semantic", "timing", "events",
        "event_count", "assignment_count",
        "memory_before", "memory_after",
        "process_cpu_user_ns", "process_cpu_system_ns",
    }, "prompt run")
    semantic_raw = dict(record["semantic"])
    _require_exact_keys(semantic_raw, set(SemanticRecord.__dataclass_fields__), "semantic record")
    for field in ("input_ids", "attention_mask", "candidate_token_ids", "candidate_texts", "routed_non_eos_token_ids"):
        semantic_raw[field] = tuple(semantic_raw[field])
    timing_raw = dict(record["timing"])
    _require_exact_keys(timing_raw, set(NestedTiming.__dataclass_fields__), "timing record")
    timing_raw["decode_forward_ns"] = tuple(timing_raw["decode_forward_ns"])
    timing_raw["event_extraction_ns"] = tuple(timing_raw["event_extraction_ns"])
    memories: dict[str, ProcessMemory] = {}
    for label in ("memory_before", "memory_after"):
        memory = record[label]
        if not isinstance(memory, dict):
            raise ValueError(f"V0.4 {label} is missing")
        _require_exact_keys(memory, set(ProcessMemory.__dataclass_fields__), label)
        memories[label] = ProcessMemory(**memory)
    if _aware_timestamp(memories["memory_after"].timestamp) < _aware_timestamp(memories["memory_before"].timestamp):
        raise ValueError("V0.4 prompt memory timestamps moved backwards")
    events = tuple(_event_from_record(event) for event in record["events"])
    if (
        isinstance(record["event_count"], bool) or record["event_count"] != len(events)
        or isinstance(record["assignment_count"], bool)
        or record["assignment_count"] != sum(len(event.selected_experts) for event in events)
    ):
        raise ValueError("V0.4 event/assignment counts do not reconcile")
    run = PromptRun(record["prompt_id"], record["mode"], SemanticRecord(**semantic_raw), NestedTiming(**timing_raw), events)
    object.__setattr__(run, "_memory_before", memories["memory_before"])
    object.__setattr__(run, "_memory_after", memories["memory_after"])
    for name in ("process_cpu_user_ns", "process_cpu_system_ns"):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("V0.4 process CPU deltas must be nonnegative integers")
        target = "_process_user_ns" if name == "process_cpu_user_ns" else "_process_system_ns"
        object.__setattr__(run, target, value)
    return run


def _validate_child_record(child: dict[str, Any]) -> None:
    _require_exact_keys(child, {
        "format", "format_version", "state", "child_number", "process_uuid", "pid",
        "parent_pid", "started_at", "finished_at", "model_id",
        "model_revision", "corpus_sha256", "model_load", "environment",
        "calibration_schedule", "evaluation_schedule", "calibration",
        "evaluation",
    }, "child")
    if child.get("format") != V04_CHILD_FORMAT or child.get("format_version") != 1 or child.get("state") != "valid":
        raise ValueError("invalid V0.4 child format")
    number = _child_number(child.get("child_number"))
    UUID(child.get("process_uuid", ""))
    child_start, child_finish = _validate_interval(child.get("started_at"), child.get("finished_at"))
    if any(isinstance(child.get(name), bool) or not isinstance(child.get(name), int) or child[name] <= 0 for name in ("pid", "parent_pid")):
        raise ValueError("V0.4 child PIDs must be positive integers")
    if child.get("model_id") != STAGE1_MODEL_ID or child.get("model_revision") != STAGE1_MODEL_REVISION or child.get("corpus_sha256") != STAGE1_CORPUS_SHA256:
        raise ValueError("V0.4 child frozen metadata mismatch")
    if child.get("calibration_schedule") != [list(item) for item in calibration_schedule(number)] or child.get("evaluation_schedule") != [[prompt_id, list(modes)] for prompt_id, modes in evaluation_schedule(number)]:
        raise ValueError("V0.4 child schedule mismatch")
    environment = child.get("environment")
    _require_exact_keys(environment, {
        "python", "torch", "transformers", "moe_cache_lab", "device", "dtype",
        "torch_threads", "local_files_only", "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "os_name", "os_release", "os_version", "windows_build",
        "logical_cpu_count", "torch_interop_threads", "process_priority_class",
    }, "child environment")
    if (
        any(not isinstance(environment.get(name), str) or not environment[name] for name in ("python", "torch", "transformers", "moe_cache_lab"))
        or environment.get("device") != "cpu" or environment.get("dtype") != "float32"
        or environment.get("torch_threads") != 4 or environment.get("local_files_only") is not True
        or any(environment.get(name) != "1" for name in (
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            "HF_HUB_DISABLE_TELEMETRY",
        ))
        or environment.get("OMP_NUM_THREADS") != "4" or environment.get("MKL_NUM_THREADS") != "4"
        or environment.get("os_name") != "Windows"
        or any(not isinstance(environment.get(name), str) or not environment[name] for name in ("os_release", "os_version", "windows_build"))
        or any(isinstance(environment.get(name), bool) or not isinstance(environment.get(name), int) or environment[name] <= 0 for name in ("logical_cpu_count", "torch_interop_threads", "process_priority_class"))
    ):
        raise ValueError("V0.4 child environment mismatch")
    _validate_model_load_record(child.get("model_load"))
    load_start, load_finish = _validate_interval(child["model_load"]["started_at"], child["model_load"]["finished_at"])
    if load_start < child_start or load_finish > child_finish:
        raise ValueError("V0.4 model-load interval is outside child interval")
    calibration = child.get("calibration")
    if not isinstance(calibration, list) or len(calibration) != 8:
        raise ValueError("V0.4 child requires eight calibration runs")
    cal_runs = [_prompt_run_from_record(record) for record in calibration]
    calibration_finish = _validate_run_memory_containment(calibration, load_finish, child_finish)
    if tuple((run.prompt_id, run.mode) for run in cal_runs) != calibration_schedule(number):
        raise ValueError("V0.4 calibration run order mismatch")
    for prompt_id in CALIBRATION_IDS:
        validate_prompt_pair(
            next(run for run in cal_runs if run.prompt_id == prompt_id and run.mode == REFERENCE_MODE),
            next(run for run in cal_runs if run.prompt_id == prompt_id and run.mode == PROFILED_MODE),
        )
    evaluation = child.get("evaluation")
    if not isinstance(evaluation, list) or tuple(row.get("prompt_id") for row in evaluation) != EVALUATION_IDS:
        raise ValueError("V0.4 child evaluation order mismatch")
    expected = dict(evaluation_schedule(number))
    _validate_run_memory_containment(
        [record for row in evaluation for record in row.get("runs", [])],
        calibration_finish, child_finish,
    )
    for row in evaluation:
        _require_exact_keys(row, {
            "prompt_id", "mode_order", "reference_total_ns",
            "profiled_total_ns", "reference_working_set_delta",
            "profiled_working_set_delta", "reference_private_delta",
            "profiled_private_delta", "runs",
        }, "evaluation row")
        if tuple(row.get("mode_order", ())) != expected[row["prompt_id"]] or not isinstance(row.get("runs"), list) or len(row["runs"]) != 2:
            raise ValueError("V0.4 evaluation pair structure mismatch")
        ordered_runs = [_prompt_run_from_record(record) for record in row["runs"]]
        if tuple(run.mode for run in ordered_runs) != tuple(row["mode_order"]) or any(run.prompt_id != row["prompt_id"] for run in ordered_runs):
            raise ValueError("V0.4 evaluation run identity/order mismatch")
        runs = {run.mode: run for run in ordered_runs}
        validate_prompt_pair(runs[REFERENCE_MODE], runs[PROFILED_MODE])
        if row.get("reference_total_ns") != runs[REFERENCE_MODE].timing.primary_total_ns or row.get("profiled_total_ns") != runs[PROFILED_MODE].timing.primary_total_ns:
            raise ValueError("V0.4 evaluation timing reconciliation mismatch")
        expected_memory = {
            "reference_working_set_delta": _memory_delta(runs[REFERENCE_MODE], "working_set_size"),
            "profiled_working_set_delta": _memory_delta(runs[PROFILED_MODE], "working_set_size"),
            "reference_private_delta": _memory_delta(runs[REFERENCE_MODE], "private_usage"),
            "profiled_private_delta": _memory_delta(runs[PROFILED_MODE], "private_usage"),
        }
        if any(isinstance(row.get(name), bool) or not isinstance(row.get(name), int) or row[name] != value for name, value in expected_memory.items()):
            raise ValueError("V0.4 descriptive memory reconciliation mismatch")
    _validate_profiled_against_trusted(tuple(
        [run for run in cal_runs if run.mode == PROFILED_MODE]
        + [
            _prompt_run_from_record(record)
            for row in evaluation for record in row["runs"]
            if record["mode"] == PROFILED_MODE
        ]
    ))


def _validate_run_memory_containment(records: list[dict[str, Any]], start: datetime, finish: datetime) -> datetime:
    previous = start
    for record in records:
        before = _aware_timestamp(record["memory_before"]["timestamp"])
        after = _aware_timestamp(record["memory_after"]["timestamp"])
        if before < previous or after < before or after > finish:
            raise ValueError("V0.4 prompt memory timestamps are not ordered/contained")
        previous = after
    return previous


def _validate_model_load_record(record: Any) -> None:
    _require_exact_keys(record, {
        "elapsed_ns", "process_cpu_user_ns", "process_cpu_system_ns",
        "started_at", "finished_at",
        "memory_before", "memory_after", "parameter_count", "device", "dtype",
        "nonmeta", "eval_mode",
    }, "model-load record")
    for name in ("elapsed_ns", "process_cpu_user_ns", "process_cpu_system_ns"):
        if isinstance(record[name], bool) or not isinstance(record[name], int) or record[name] < 0:
            raise ValueError("V0.4 model-load timing values are invalid")
    load_start, load_finish = _validate_interval(record["started_at"], record["finished_at"])
    if record["parameter_count"] != EXPECTED_PARAMETER_COUNT or record["device"] != "cpu" or record["dtype"] != "float32" or record["nonmeta"] is not True or record["eval_mode"] is not True:
        raise ValueError("V0.4 model-load invariant record mismatch")
    for name in ("memory_before", "memory_after"):
        _require_exact_keys(record[name], set(ProcessMemory.__dataclass_fields__), f"model-load {name}")
        ProcessMemory(**record[name])
    before_time = _aware_timestamp(record["memory_before"]["timestamp"])
    after_time = _aware_timestamp(record["memory_after"]["timestamp"])
    if not load_start <= before_time <= after_time <= load_finish:
        raise ValueError("V0.4 model-load memory timestamps are not ordered/contained")


def _validate_admission_artifact(value: dict[str, Any], child_number: int, *, require_admitted: bool) -> None:
    _require_exact_keys(value, {
        "format", "format_version", "child_number", "admitted", "started_at",
        "finished_at", "started_monotonic_ns", "finished_monotonic_ns", "interval_ns",
        "prime", "baseline", "samples", "errors", "reason",
        "audit",
    }, "admission artifact")
    if value.get("format") != "moe-cache-lab.v04-admission" or value.get("format_version") != 1 or value.get("child_number") != child_number or type(value.get("admitted")) is not bool:
        raise ValueError("V0.4 admission identity is invalid")
    if require_admitted and value["admitted"] is not True:
        raise ValueError("valid V0.4 set contains a rejected admission")
    admission_start, admission_finish = _validate_interval(value.get("started_at"), value.get("finished_at"))
    start_mono = value.get("started_monotonic_ns")
    finish_mono = value.get("finished_monotonic_ns")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (start_mono, finish_mono)) or finish_mono < start_mono or value.get("interval_ns") != finish_mono - start_mono:
        raise ValueError("V0.4 admission monotonic interval is invalid")
    prime_raw = value.get("prime")
    baseline = value.get("baseline")
    samples = value.get("samples")
    errors = value.get("errors")
    reason = value.get("reason")
    if not isinstance(samples, list) or len(samples) > 15 or not isinstance(errors, list) or any(not isinstance(item, str) or not item for item in errors) or not isinstance(reason, str) or not reason:
        raise ValueError("V0.4 admission terminal evidence is invalid")
    prime = None
    if prime_raw is not None:
        _require_exact_keys(prime_raw, set(ResourcePrime.__dataclass_fields__), "resource prime")
        prime = ResourcePrime(**prime_raw)
        prime_time = _aware_timestamp(prime.timestamp)
        if not admission_start <= prime_time <= admission_finish:
            raise ValueError("V0.4 prime timestamp is outside admission interval")
        if not start_mono <= prime.monotonic_ns <= finish_mono:
            raise ValueError("V0.4 prime monotonic timestamp is outside admission interval")
    baseline_sample = None
    if baseline is not None:
        baseline_sample = _resource_from_record(baseline)
    loaded_samples = tuple(_resource_from_record(record) for record in samples)
    audit = value.get("audit")
    _require_exact_keys(audit, {
        "open_query_status", "add_cpu_status", "add_disk_status",
        "close_query_status", "constructor_error",
    }, "PDH audit")
    for name in ("open_query_status", "add_cpu_status", "add_disk_status", "close_query_status"):
        status = audit[name]
        if status is not None and (isinstance(status, bool) or not isinstance(status, int)):
            raise ValueError("V0.4 PDH constructor/close status is invalid")
    if audit["constructor_error"] is not None and (not isinstance(audit["constructor_error"], str) or not audit["constructor_error"]):
        raise ValueError("V0.4 PDH constructor error record is invalid")
    ordered = ([baseline_sample] if baseline_sample else []) + list(loaded_samples)
    previous = _aware_timestamp(prime.timestamp) if prime else admission_start
    previous_mono = prime.monotonic_ns if prime else start_mono
    for index, sample in enumerate(ordered):
        current = _aware_timestamp(sample.timestamp)
        # Windows wall-clock timestamps can legitimately repeat at this scale.
        # Monotonic nanoseconds are authoritative for strict order and cadence;
        # wall timestamps remain aware, nondecreasing, and contained.
        if current < previous or current > admission_finish or sample.monotonic_ns <= previous_mono or sample.monotonic_ns > finish_mono:
            raise ValueError("V0.4 admission wall timestamps are not nondecreasing/contained or monotonic timestamps are not strictly ordered")
        if index > 0:
            cadence_ns = sample.monotonic_ns - previous_mono
            if not 500_000_000 <= cadence_ns <= 1_500_000_000:
                raise ValueError("V0.4 admission sample cadence is not one second")
        previous = current
        previous_mono = sample.monotonic_ns
    if not value["admitted"]:
        if require_admitted:
            raise ValueError("valid V0.4 set contains a rejected admission")
        if reason == "admitted" or (prime is not None and baseline_sample is not None and len(loaded_samples) == 15 and not errors and _samples_pass(loaded_samples)):
            raise ValueError("rejected V0.4 admission is inconsistent with its evidence")
        return
    if prime is None or baseline_sample is None or len(loaded_samples) != 15 or errors or reason != "admitted":
        raise ValueError("valid V0.4 admission requires a successful prime and fifteen samples")
    if any(audit[name] != 0 for name in ("open_query_status", "add_cpu_status", "add_disk_status", "close_query_status")) or audit["constructor_error"] is not None:
        raise ValueError("admitted V0.4 admission has unsuccessful PDH lifecycle status")
    if not _samples_pass(loaded_samples):
        raise ValueError("V0.4 admission values do not pass frozen thresholds")


def _samples_pass(samples: tuple[ResourceSample, ...]) -> bool:
    return bool(samples) and (
        statistics.median(sample.cpu_percent for sample in samples) <= 10
        and max(sample.cpu_percent for sample in samples) <= 25
        and all(sample.available_physical_bytes >= MIN_AVAILABLE_BYTES and sample.commit_percent <= 75 for sample in samples)
        and statistics.median(sample.disk_percent for sample in samples) <= 10
        and max(sample.disk_percent for sample in samples) <= 50
    )


def _resource_from_record(record: dict[str, Any]) -> ResourceSample:
    _require_exact_keys(
        record,
        set(ResourceSample.__dataclass_fields__) | {
            "commit_numerator_bytes", "commit_denominator_bytes", "commit_percent",
        },
        "resource sample",
    )
    sample = ResourceSample(**{name: record[name] for name in ResourceSample.__dataclass_fields__})
    if (
        record["commit_numerator_bytes"] != sample.commit_total_pages * sample.page_size_bytes
        or record["commit_denominator_bytes"] != sample.commit_limit_pages * sample.page_size_bytes
        or not isinstance(record["commit_percent"], (int, float))
        or isinstance(record["commit_percent"], bool)
        or not math.isfinite(record["commit_percent"])
        or record["commit_percent"] != sample.commit_percent
    ):
        raise ValueError("V0.4 commit-memory arithmetic mismatch")
    return sample


def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"V0.4 {label} fields are invalid")


def _validate_interval(start: Any, finish: Any) -> tuple[datetime, datetime]:
    try:
        left, right = datetime.fromisoformat(start), datetime.fromisoformat(finish)
    except (TypeError, ValueError) as error:
        raise ValueError("V0.4 timestamps must be timezone-aware ISO values") from error
    if left.tzinfo is None or right.tzinfo is None or left.utcoffset() is None or right.utcoffset() is None or right < left:
        raise ValueError("V0.4 timestamps must be aware and ordered")
    return left, right


def _aware_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("V0.4 timestamp must be a timezone-aware ISO string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("V0.4 timestamp must be a timezone-aware ISO string") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("V0.4 timestamp must be timezone-aware")
    return parsed
