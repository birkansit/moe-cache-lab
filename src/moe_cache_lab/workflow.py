"""Versioned corpus collection and reproducible multi-trace cache benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Any, Iterable

import torch
import transformers

from . import __version__
from .cache import CacheSimulation, simulate
from .collector import DEFAULT_MODEL, GraniteTraceCollector
from .trace import RoutingEvent, RoutingTrace, read_trace, write_trace

CORPUS_FORMAT = "moe-cache-lab.prompt-corpus"
CORPUS_VERSION = "1.0"
MANIFEST_FORMAT = "moe-cache-lab.collection-manifest"
MANIFEST_VERSION = 1
DEFAULT_CAPACITIES = (8, 16, 32, 64, 128, 256)
SUITE_POLICIES = (
    "lru",
    "lfu",
    "calibrated_static_frequency",
    "offline_oracle_frequency",
)
SAFE_PROMPT_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class CorpusPrompt:
    id: str
    category: str
    split: str
    text: str
    order: int


@dataclass(frozen=True)
class CorpusDefinition:
    path: Path
    version: str
    sha256: str
    prompts: tuple[CorpusPrompt, ...]


@dataclass(frozen=True)
class LoadedPromptTrace:
    prompt: CorpusPrompt
    relative_path: str
    sha256: str
    trace: RoutingTrace


@dataclass(frozen=True)
class SuiteInputs:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    corpus: CorpusDefinition
    calibration: tuple[LoadedPromptTrace, ...]
    evaluation: tuple[LoadedPromptTrace, ...]


@dataclass(frozen=True)
class SuiteBenchmark:
    inputs: SuiteInputs
    capacities: tuple[int, ...]
    simulations: tuple[CacheSimulation, ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON used for stable corpus hashing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus(path: str | Path) -> CorpusDefinition:
    corpus_path = Path(path).resolve()
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    if data.get("format") != CORPUS_FORMAT or data.get("version") != CORPUS_VERSION:
        raise ValueError("unsupported benchmark corpus format or version")
    records = data.get("prompts")
    if not isinstance(records, list):
        raise ValueError("corpus prompts must be a list")
    prompts: list[CorpusPrompt] = []
    for order, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"id", "category", "split", "text"}:
            raise ValueError("each corpus prompt must define exactly id, category, split, and text")
        if record["split"] not in {"calibration", "evaluation"}:
            raise ValueError("prompt split must be calibration or evaluation")
        if any(not isinstance(record[name], str) or not record[name].strip() for name in record):
            raise ValueError("corpus prompt fields must be non-empty strings")
        if not SAFE_PROMPT_ID.fullmatch(record["id"]):
            raise ValueError("corpus prompt IDs must be safe lowercase filename slugs")
        prompts.append(CorpusPrompt(order=order, **record))
    if len({prompt.id for prompt in prompts}) != len(prompts):
        raise ValueError("corpus prompt IDs must be unique")
    if sum(prompt.split == "calibration" for prompt in prompts) != 4:
        raise ValueError("corpus must contain exactly 4 calibration prompts")
    if sum(prompt.split == "evaluation" for prompt in prompts) != 8:
        raise ValueError("corpus must contain exactly 8 evaluation prompts")
    return CorpusDefinition(
        path=corpus_path,
        version=data["version"],
        sha256=sha256_bytes(canonical_json_bytes(data)),
        prompts=tuple(prompts),
    )


def collect_corpus(
    corpus_path: str | Path,
    output_directory: str | Path,
    *,
    model_id: str = DEFAULT_MODEL,
    revision: str | None = None,
    max_new_tokens: int = 2,
) -> Path:
    """Load Granite once and collect one independent trace per prompt."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    corpus = load_corpus(corpus_path)
    output = Path(output_directory).resolve()
    traces_directory = output / "traces"
    traces_directory.mkdir(parents=True, exist_ok=True)
    # Keep each run self-contained and avoid cross-drive relative-path problems
    # on Windows while retaining the canonical hash of the tracked source.
    (output / "corpus.json").write_bytes(corpus.path.read_bytes())
    collector = GraniteTraceCollector(model_id, revision=revision)
    prompt_records: list[dict[str, Any]] = []
    collected_traces: list[RoutingTrace] = []
    routed_layers: tuple[int, ...] | None = None
    routing_shape: tuple[int | None, int | None, str] | None = None
    for prompt in corpus.prompts:
        trace = collector.collect(prompt.text, max_new_tokens=max_new_tokens)
        if trace.model_revision != collector.resolved_revision:
            raise ValueError("collector trace revision does not match its pinned model revision")
        routed_layers = _validate_trace_structure(trace, routed_layers, max_new_tokens)
        current_shape = (trace.num_experts, trace.experts_per_token, trace.capture_method)
        if routing_shape is None:
            routing_shape = current_shape
        elif current_shape != routing_shape:
            raise ValueError("collected trace routing metadata is inconsistent")
        relative_trace = Path("traces") / f"{prompt.order:02d}-{prompt.id}.jsonl"
        trace_path = (output / relative_trace).resolve()
        try:
            trace_path.relative_to(output)
        except ValueError as error:
            raise ValueError("trace destination escapes the collection output directory") from error
        write_trace(trace_path, trace)
        collected_traces.append(trace)
        prompt_records.append({
            "id": prompt.id,
            "category": prompt.category,
            "split": prompt.split,
            "order": prompt.order,
            "prompt_sha256": sha256_bytes(prompt.text.encode("utf-8")),
            "trace_path": relative_trace.as_posix(),
            "trace_sha256": file_sha256(trace_path),
            "routing_event_count": len(trace.events),
            "expert_request_count": len(trace.expert_requests),
        })
    if not collected_traces or routed_layers is None or routing_shape is None:
        raise AssertionError("validated corpus collection produced no traces")
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "path": "corpus.json",
            "version": corpus.version,
            "canonical_json_sha256": corpus.sha256,
        },
        "model_id": model_id,
        "model_revision": {
            "requested": revision,
            "resolved": collector.resolved_revision,
        },
        "routing": {
            "experts_per_layer": routing_shape[0],
            "selected_experts_per_token": routing_shape[1],
            "routed_layer_ids": list(routed_layers),
            "routed_layer_count": len(routed_layers),
            "capture_method": routing_shape[2],
        },
        "environment": {
            "moe_cache_lab": __version__,
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": "float32",
            "device": "cpu",
        },
        "generation": {
            "strategy": "greedy",
            "token_selection": "argmax",
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        },
        "prompts": prompt_records,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest-v1.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def load_suite_inputs(manifest_path: str | Path) -> SuiteInputs:
    """Validate manifest, corpus, prompt metadata, trace hashes, and trace metadata."""
    path = Path(manifest_path).resolve()
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported collection manifest format or version")
    try:
        datetime.fromisoformat(manifest["created_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("manifest created_at must be an ISO-8601 timestamp") from error
    if not isinstance(manifest.get("model_id"), str) or not manifest["model_id"].strip():
        raise ValueError("manifest model_id must be a non-empty string")
    revision_record = manifest.get("model_revision")
    if not isinstance(revision_record, dict):
        raise ValueError("manifest is missing model revision metadata")
    requested_revision = revision_record.get("requested")
    if requested_revision is not None and (
        not isinstance(requested_revision, str) or not requested_revision.strip()
    ):
        raise ValueError("requested model revision must be null or a non-empty string")
    resolved_revision = revision_record.get("resolved")
    if not isinstance(resolved_revision, str) or not resolved_revision.strip():
        raise ValueError("resolved model revision must be non-empty")
    corpus_record = manifest.get("corpus")
    if not isinstance(corpus_record, dict):
        raise ValueError("manifest is missing corpus metadata")
    if (
        not isinstance(corpus_record.get("path"), str)
        or Path(corpus_record["path"]).is_absolute()
    ):
        raise ValueError("manifest corpus path must be relative")
    resolved_corpus_path = (path.parent / corpus_record["path"]).resolve()
    try:
        resolved_corpus_path.relative_to(path.parent)
    except ValueError as error:
        raise ValueError("manifest corpus path escapes the manifest directory") from error
    corpus = load_corpus(resolved_corpus_path)
    if corpus_record.get("version") != corpus.version:
        raise ValueError("manifest corpus version does not match corpus")
    if corpus_record.get("canonical_json_sha256") != corpus.sha256:
        raise ValueError("corpus hash mismatch")
    environment = manifest.get("environment", {})
    generation = manifest.get("generation", {})
    for name in ("moe_cache_lab", "python", "transformers", "torch"):
        if not isinstance(environment.get(name), str) or not environment[name].strip():
            raise ValueError(f"manifest environment is missing {name} version")
    if environment.get("dtype") != "float32" or environment.get("device") != "cpu":
        raise ValueError("manifest must describe CPU float32 collection")
    if (
        generation.get("strategy") != "greedy"
        or generation.get("token_selection") != "argmax"
        or generation.get("do_sample") is not False
    ):
        raise ValueError("manifest must describe deterministic greedy generation")
    if generation.get("use_cache") is not True:
        raise ValueError("manifest must describe KV-cache generation")
    if (
        isinstance(generation.get("max_new_tokens"), bool)
        or not isinstance(generation.get("max_new_tokens"), int)
        or generation["max_new_tokens"] < 0
    ):
        raise ValueError("manifest max_new_tokens must be a non-negative integer")
    records = manifest.get("prompts")
    if not isinstance(records, list) or len(records) != len(corpus.prompts):
        raise ValueError("manifest prompt list does not match corpus size")
    routing_record = manifest.get("routing")
    if not isinstance(routing_record, dict):
        raise ValueError("manifest is missing routing metadata")
    experts_per_layer = routing_record.get("experts_per_layer")
    selected_per_token = routing_record.get("selected_experts_per_token")
    routed_layer_ids = routing_record.get("routed_layer_ids")
    if (
        not isinstance(experts_per_layer, int) or experts_per_layer <= 0
        or not isinstance(selected_per_token, int) or selected_per_token <= 0
        or not isinstance(routed_layer_ids, list)
        or not routed_layer_ids
        or any(not isinstance(layer, int) or layer < 0 for layer in routed_layer_ids)
        or routed_layer_ids != sorted(set(routed_layer_ids))
        or routing_record.get("routed_layer_count") != len(routed_layer_ids)
        or not isinstance(routing_record.get("capture_method"), str)
        or not routing_record["capture_method"].strip()
    ):
        raise ValueError("manifest routing metadata is invalid")
    expected_layers = tuple(routed_layer_ids)
    loaded: list[LoadedPromptTrace] = []
    seen_trace_paths: set[Path] = set()
    routing_shape: tuple[int | None, int | None] | None = None
    for prompt, record in zip(corpus.prompts, records):
        expected = (prompt.id, prompt.category, prompt.split, prompt.order)
        actual = (record.get("id"), record.get("category"), record.get("split"), record.get("order"))
        if actual != expected:
            raise ValueError("manifest prompt metadata/order does not match corpus")
        if record.get("prompt_sha256") != sha256_bytes(prompt.text.encode("utf-8")):
            raise ValueError(f"prompt hash mismatch for {prompt.id}")
        relative_path = record.get("trace_path")
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
            raise ValueError(f"trace path for {prompt.id} must be relative")
        trace_path = (path.parent / relative_path).resolve()
        try:
            trace_path.relative_to(path.parent)
        except ValueError as error:
            raise ValueError(f"trace path for {prompt.id} escapes the manifest directory") from error
        if trace_path in seen_trace_paths:
            raise ValueError("manifest trace paths must be unique")
        seen_trace_paths.add(trace_path)
        if file_sha256(trace_path) != record.get("trace_sha256"):
            raise ValueError(f"trace hash mismatch for {prompt.id}")
        trace = read_trace(trace_path)
        if trace.model_id != manifest.get("model_id") or trace.source_text != prompt.text:
            raise ValueError(f"trace metadata mismatch for {prompt.id}")
        if trace.transformers_version != environment["transformers"]:
            raise ValueError(f"trace Transformers version mismatch for {prompt.id}")
        if trace.model_revision != resolved_revision:
            raise ValueError(f"trace model revision mismatch for {prompt.id}")
        current_shape = trace.num_experts, trace.experts_per_token
        if routing_shape is None:
            routing_shape = current_shape
        elif current_shape != routing_shape:
            raise ValueError("trace routing metadata must be consistent across the corpus")
        if current_shape != (experts_per_layer, selected_per_token):
            raise ValueError(f"trace routing shape mismatch for {prompt.id}")
        if trace.capture_method != routing_record["capture_method"]:
            raise ValueError(f"trace capture method mismatch for {prompt.id}")
        _validate_trace_structure(trace, expected_layers, generation["max_new_tokens"])
        if len(trace.events) != record.get("routing_event_count"):
            raise ValueError(f"routing event count mismatch for {prompt.id}")
        if len(trace.expert_requests) != record.get("expert_request_count"):
            raise ValueError(f"expert request count mismatch for {prompt.id}")
        loaded.append(LoadedPromptTrace(prompt, relative_path, record["trace_sha256"], trace))
    calibration = tuple(item for item in loaded if item.prompt.split == "calibration")
    evaluation = tuple(item for item in loaded if item.prompt.split == "evaluation")
    return SuiteInputs(
        manifest_path=path,
        manifest_sha256=sha256_bytes(manifest_bytes),
        manifest=manifest,
        corpus=corpus,
        calibration=calibration,
        evaluation=evaluation,
    )


def validate_capacities(capacities: Iterable[int]) -> tuple[int, ...]:
    values = tuple(capacities)
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("capacities must be positive integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("capacities must be unique ascending integers")
    return values


def _validate_trace_structure(
    trace: RoutingTrace,
    expected_layers: tuple[int, ...] | None,
    max_new_tokens: int,
) -> tuple[int, ...]:
    """Require a complete, consistent routed-layer set at every token position."""
    prompt_events = tuple(event for event in trace.events if event.phase == "prompt")
    if not prompt_events:
        raise ValueError("each trace must contain a nonempty prompt/prefill phase")
    prompt_positions = sorted({event.token_position for event in prompt_events})
    if prompt_positions != list(range(len(prompt_positions))):
        raise ValueError("trace prompt token positions must be contiguous from zero")
    generated_positions = sorted({
        event.token_position for event in trace.events if event.phase == "generated"
    })
    expected_generated_positions = list(range(
        len(prompt_positions), len(prompt_positions) + max_new_tokens
    ))
    if generated_positions != expected_generated_positions:
        raise ValueError(
            "trace generated token positions must be contiguous at the immediate decode boundary"
        )
    observed_layers = tuple(sorted({event.layer for event in trace.events}))
    if expected_layers is not None and observed_layers != expected_layers:
        raise ValueError("trace routed-layer set does not match the manifest/corpus")
    full_layer_set = set(observed_layers)
    for phase in ("prompt", "generated"):
        positions = sorted({event.token_position for event in trace.events if event.phase == phase})
        for position in positions:
            position_layers = {
                event.layer
                for event in trace.events
                if event.phase == phase and event.token_position == position
            }
            if position_layers != full_layer_set:
                raise ValueError(
                    f"trace {phase} token position {position} does not contain the complete routed-layer set"
                )
    return observed_layers


def benchmark_suite(
    manifest_path: str | Path,
    capacities: Iterable[int] = DEFAULT_CAPACITIES,
) -> SuiteBenchmark:
    """Use calibration only for calibrated targets and evaluation only for results."""
    inputs = load_suite_inputs(manifest_path)
    values = validate_capacities(capacities)
    calibration_events = tuple(
        event for item in inputs.calibration for event in item.trace.events
    )
    evaluation_events = tuple(
        event for item in inputs.evaluation for event in item.trace.events
    )
    largest_bundle = max((len(event.selected_experts) for event in evaluation_events), default=0)
    if values[0] < largest_bundle:
        raise ValueError(
            f"all capacities must be at least the largest evaluation event bundle {largest_bundle}"
        )
    simulations: list[CacheSimulation] = []
    for capacity in values:
        for policy in SUITE_POLICIES:
            simulations.append(simulate(
                evaluation_events,
                capacity,
                policy,
                calibration_events=calibration_events if policy == "calibrated_static_frequency" else None,
            ))
    return SuiteBenchmark(inputs, values, tuple(simulations))
