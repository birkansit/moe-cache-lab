import json
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import torch
import transformers

from moe_cache_lab import __version__
from moe_cache_lab.collector import GraniteTraceCollector, Stage1CollectionOutcome
from moe_cache_lab.stage1 import (
    STAGE1_CAPTURE_METHOD,
    STAGE1_CORPUS_SHA256,
    STAGE1_DEADLINE_SECONDS,
    STAGE1_MANIFEST_VERSION,
    STAGE1_MAX_DECODE_INPUT_STEPS,
    STAGE1_MODEL_ID,
    STAGE1_MODEL_REVISION,
    STAGE1_REPETITION_FORMAT,
    STAGE1_REPETITIONS,
    STAGE1_ROUTED_LAYERS,
    STAGE1_SET_FORMAT,
    STAGE1_TORCH_THREADS,
    TOKEN_LAYER_ATOMIC,
    PREFILL_LAYER_UNION_ATOMIC,
    _prompt_record,
    collect_stage1_repetition,
    collect_stage1_set,
    compare_stage1_repetitions,
    load_stage1_repetition_manifest,
    load_stage1_set_manifest,
    validate_v02_prefix,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace, write_trace
from moe_cache_lab.workflow import file_sha256, load_corpus, load_suite_inputs, sha256_bytes


ROOT = Path(__file__).parents[1]
SOURCE_CORPUS = ROOT / "benchmarks" / "corpus-v1.json"
TRUSTED_V02 = ROOT / "results" / "v0.2-corpus-v1" / "manifest-v1.json"


class _FakeTokenizer:
    eos_token_id = [10, 9]
    eos_token_ids = (9,)

    def __call__(self, _: str, return_tensors: str):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens: bool):
        prefix = "clean" if skip_special_tokens else "raw"
        return prefix + ":" + ",".join(str(token) for token in token_ids)


class _FakeModel:
    def __init__(self, candidates: list[int]) -> None:
        self.candidates = candidates
        self.calls = 0

    def __call__(self, **_: object):
        candidate = self.candidates[self.calls]
        self.calls += 1
        logits = torch.full((1, 1, 32), -10.0)
        logits[0, 0, candidate] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=(self.calls,))


class _FakeCapture:
    def __init__(self) -> None:
        self.takes = 0

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def take(self):
        length = 2 if self.takes == 0 else 1
        self.takes += 1
        return {0: torch.arange(length * 4, dtype=torch.float32).reshape(length, 4)}


def _collector(candidates: list[int]) -> tuple[GraniteTraceCollector, _FakeModel]:
    model = _FakeModel(candidates)
    collector = GraniteTraceCollector.__new__(GraniteTraceCollector)
    collector.model_id = STAGE1_MODEL_ID
    collector.requested_revision = STAGE1_MODEL_REVISION
    collector.resolved_revision = STAGE1_MODEL_REVISION
    collector.tokenizer = _FakeTokenizer()
    collector.model = model
    collector.num_experts = 4
    collector.experts_per_token = 1
    return collector, model


def _event(
    phase: str,
    position: int,
    layer: int,
    token_id: int,
    probability_delta: float = 0.0,
    expert_mismatch: bool = False,
) -> RoutingEvent:
    experts = (8, 1, 2, 3, 4, 5, 6, 7) if expert_mismatch else tuple(range(8))
    probabilities = (0.125 + probability_delta,) + (0.125,) * 7
    return RoutingEvent(
        phase,
        position,
        layer,
        experts,
        token_id=token_id,
        selected_probabilities=probabilities,
    )


def _trace_for_prompt(
    prompt,
    actual_steps: int,
    *,
    trusted_prefix: bool = False,
    probability_delta: float = 0.0,
    expert_mismatch: bool = False,
    token_mismatch: bool = False,
) -> RoutingTrace:
    if trusted_prefix:
        trusted = load_suite_inputs(TRUSTED_V02)
        item = next(
            item for item in (*trusted.calibration, *trusted.evaluation)
            if item.prompt.id == prompt.id
        )
        prompt_events = [event for event in item.trace.events if event.phase == "prompt"]
        old_generated = [event for event in item.trace.events if event.phase == "generated"]
        prompt_length = len({event.token_position for event in prompt_events})
        generated: list[RoutingEvent] = []
        for step in range(actual_steps):
            if step < 2:
                generated.extend(
                    event for event in old_generated
                    if event.token_position == prompt_length + step
                )
            else:
                template = tuple(
                    event for event in old_generated
                    if event.token_position == prompt_length + 1
                )
                generated.extend(RoutingEvent(
                    "generated",
                    prompt_length + step,
                    event.layer,
                    event.selected_experts,
                    token_id=300 + step,
                    selected_probabilities=event.selected_probabilities,
                ) for event in template)
        events = list(prompt_events) + generated
    else:
        prompt_length = 1
        events = [_event("prompt", 0, layer, 101) for layer in STAGE1_ROUTED_LAYERS]
        events.extend(
            _event("generated", 1 + step, layer, 200 + step)
            for step in range(actual_steps)
            for layer in STAGE1_ROUTED_LAYERS
        )
    first = events[0]
    if probability_delta:
        events[0] = RoutingEvent(
            first.phase, first.token_position, first.layer, first.selected_experts,
            token_id=first.token_id,
            selected_probabilities=(first.selected_probabilities[0] + probability_delta,)
            + first.selected_probabilities[1:],
        )
    if expert_mismatch:
        events[0] = RoutingEvent(
            first.phase, first.token_position, first.layer,
            (8,) + first.selected_experts[1:], token_id=first.token_id,
            selected_probabilities=first.selected_probabilities,
        )
    if token_mismatch:
        events = [RoutingEvent(
            event.phase,
            event.token_position,
            event.layer,
            event.selected_experts,
            token_id=(event.token_id or 0) + 1,
            selected_probabilities=event.selected_probabilities,
        ) if event.phase == "prompt" and event.token_position == 0 else event for event in events]
    return RoutingTrace(
        STAGE1_MODEL_ID,
        32,
        8,
        tuple(events),
        source_text=prompt.text,
        generated_text=f"generated-{actual_steps}",
        capture_method=STAGE1_CAPTURE_METHOD,
        created_at="2026-01-01T00:00:00+00:00",
        transformers_version=transformers.__version__,
        model_revision=STAGE1_MODEL_REVISION,
    )


def _outcome(trace: RoutingTrace, actual_steps: int) -> Stage1CollectionOutcome:
    prompt_length = len({event.token_position for event in trace.events if event.phase == "prompt"})
    routed = tuple(
        next(
            event.token_id for event in trace.events
            if event.phase == "generated" and event.token_position == prompt_length + step
        )
        for step in range(actual_steps)
    )
    eos = actual_steps < 16
    return Stage1CollectionOutcome(
        trace=trace,
        max_decode_input_steps_requested=16,
        actual_decode_input_steps_routed=actual_steps,
        routed_non_eos_token_ids=routed,
        emitted_non_eos_token_ids=routed,
        emitted_non_eos_text=trace.generated_text or "",
        normalized_eos_token_ids=(999999,),
        terminal_eos_token_id=999999 if eos else None,
        terminal_eos_candidate_position=actual_steps if eos else None,
        terminal_eos_text="<eos>" if eos else None,
        eos_emitted=eos,
        horizon_exhausted=not eos,
    )


def _write_repetition(
    directory: Path,
    number: int,
    process_uuid: str,
    pid: int,
    *,
    actual_steps: int | dict[int, int] = 0,
    trusted_prefix: bool = False,
    probability_delta: float = 0.0,
    expert_mismatch: bool = False,
    token_mismatch: bool = False,
    parent_pid: int = 1,
) -> Path:
    directory.mkdir(parents=True)
    shutil.copyfile(SOURCE_CORPUS, directory / "corpus.json")
    traces = directory / "traces"
    traces.mkdir()
    corpus = load_corpus(SOURCE_CORPUS)
    records = []
    for prompt in corpus.prompts:
        actual = actual_steps[prompt.order] if isinstance(actual_steps, dict) else actual_steps
        trace = _trace_for_prompt(
            prompt,
            actual,
            trusted_prefix=trusted_prefix,
            probability_delta=probability_delta,
            expert_mismatch=expert_mismatch,
            token_mismatch=token_mismatch,
        )
        relative = Path("traces") / f"{prompt.order:02d}-{prompt.id}.jsonl"
        trace_path = write_trace(directory / relative, trace)
        records.append(_prompt_record(
            prompt,
            _outcome(trace, actual),
            relative.as_posix(),
            file_sha256(trace_path),
        ))
    started_at = f"2026-01-01T00:00:0{number - 1}+00:00"
    finished_at = f"2026-01-01T00:00:0{number}+00:00"
    manifest = {
        "format": STAGE1_REPETITION_FORMAT,
        "format_version": STAGE1_MANIFEST_VERSION,
        "created_at": finished_at,
        "repetition_number": number,
        "corpus": {
            "path": "corpus.json",
            "version": "1.0",
            "canonical_json_sha256": STAGE1_CORPUS_SHA256,
        },
        "model_id": STAGE1_MODEL_ID,
        "model_revision": {"requested": STAGE1_MODEL_REVISION, "resolved": STAGE1_MODEL_REVISION},
        "routing": {
            "experts_per_layer": 32,
            "selected_experts_per_token": 8,
            "routed_layer_ids": list(STAGE1_ROUTED_LAYERS),
            "routed_layer_count": 24,
            "capture_method": STAGE1_CAPTURE_METHOD,
        },
        "environment": {
            "moe_cache_lab": __version__,
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": "float32",
            "device": "cpu",
            "torch_num_threads": 4,
            "local_files_only": True,
            "hf_hub_offline": True,
            "transformers_offline": True,
            "hf_hub_disable_telemetry": True,
            "omp_num_threads": "4",
            "mkl_num_threads": "4",
        },
        "generation": {
            "strategy": "greedy",
            "token_selection": "argmax",
            "do_sample": False,
            "use_cache": True,
            "protocol": "candidate-before-feed-eos-v1",
            "max_decode_input_steps": 16,
        },
        "process": {
            "uuid": process_uuid,
            "pid": pid,
            "parent_pid": parent_pid,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": 1.0,
        },
        "prompts": records,
    }
    path = directory / "stage1-repetition-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def _write_set(directory: Path, *, actual_steps: int = 0, trusted_prefix: bool = False) -> Path:
    directory.mkdir()
    shutil.copyfile(SOURCE_CORPUS, directory / "corpus.json")
    repetitions = []
    for index in range(3):
        number = index + 1
        child_uuid = str(uuid4())
        relative = Path(f"repetition-{number}") / "stage1-repetition-manifest.json"
        path = _write_repetition(
            directory / f"repetition-{number}", number, child_uuid, 100,
            actual_steps=actual_steps, trusted_prefix=trusted_prefix, parent_pid=50,
        )
        repetitions.append({
            "repetition_number": number,
            "process_uuid": child_uuid,
            "process_pid": 100,
            "manifest_path": relative.as_posix(),
            "manifest_sha256": file_sha256(path),
        })
    manifest = {
        "format": STAGE1_SET_FORMAT,
        "format_version": 1,
        "created_at": "2026-01-01T00:00:03+00:00",
        "run_uuid": str(uuid4()),
        "parent_pid": 50,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:03+00:00",
        "elapsed_seconds": 3.0,
        "deadline_seconds": 900,
        "repetition_count": 3,
        "corpus": {"path": "corpus.json", "version": "1.0", "canonical_json_sha256": STAGE1_CORPUS_SHA256},
        "model_id": STAGE1_MODEL_ID,
        "model_revision": STAGE1_MODEL_REVISION,
        "max_decode_input_steps": 16,
        "torch_num_threads": 4,
        "repetitions": repetitions,
    }
    path = directory / "stage1-set-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def _mutate_set_child(path: Path, index: int, mutation) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    record = manifest["repetitions"][index]
    child_path = path.parent / record["manifest_path"]
    child = json.loads(child_path.read_text(encoding="utf-8"))
    mutation(child)
    child_path.write_text(json.dumps(child), encoding="utf-8")
    record["manifest_sha256"] = file_sha256(child_path)
    path.write_text(json.dumps(manifest), encoding="utf-8")


class Stage1CollectorTests(unittest.TestCase):
    def test_candidate_before_feed_eos_state_machine_edges(self) -> None:
        cases = (
            ([9], 0, True),
            ([3, 9], 1, True),
            (list(range(11, 26)) + [9], 15, True),
            (list(range(11, 28)), 16, False),
        )
        for candidates, expected_steps, eos_emitted in cases:
            collector, model = _collector(candidates)
            capture = _FakeCapture()
            with self.subTest(expected_steps=expected_steps), patch(
                "moe_cache_lab.collector._RouterHookCapture", return_value=capture
            ):
                outcome = collector.collect_stage1("prompt")
            self.assertEqual(outcome.actual_decode_input_steps_routed, expected_steps)
            self.assertEqual(model.calls, 1 + expected_steps)
            self.assertEqual(capture.takes, 1 + expected_steps)
            self.assertEqual(outcome.eos_emitted, eos_emitted)
            self.assertEqual(outcome.horizon_exhausted, not eos_emitted)
            self.assertEqual(outcome.normalized_eos_token_ids, (9, 10))
            self.assertEqual(outcome.routed_non_eos_token_ids, outcome.emitted_non_eos_token_ids)
            self.assertEqual(len([event for event in outcome.trace.events if event.phase == "generated"]), expected_steps)
            if eos_emitted:
                self.assertEqual(outcome.terminal_eos_candidate_position, expected_steps)
                self.assertEqual(outcome.terminal_eos_text, "raw:9")
            else:
                self.assertIsNone(outcome.terminal_eos_token_id)
                self.assertNotIn(candidates[-1], outcome.emitted_non_eos_token_ids)

    def test_stage1_method_rejects_changed_horizon_and_old_collect_api_remains(self) -> None:
        collector, _ = _collector([9])
        with self.assertRaisesRegex(ValueError, "frozen at 16"):
            collector.collect_stage1("prompt", 15)
        self.assertTrue(callable(collector.collect))


class Stage1ManifestTests(unittest.TestCase):
    def test_child_uses_one_offline_collector_for_all_twelve_prompts(self) -> None:
        corpus = load_corpus(SOURCE_CORPUS)
        prompts_by_text = {prompt.text: prompt for prompt in corpus.prompts}

        class FakeCollector:
            instances = 0
            calls = 0

            def __init__(self, model_id, revision, *, local_files_only):
                if (model_id, revision, local_files_only) != (
                    STAGE1_MODEL_ID, STAGE1_MODEL_REVISION, True
                ):
                    raise AssertionError("child did not use the frozen offline collector")
                self.resolved_revision = STAGE1_MODEL_REVISION
                FakeCollector.instances += 1

            def collect_stage1(self, text, max_decode_input_steps):
                if max_decode_input_steps != 16:
                    raise AssertionError("child changed the frozen decode horizon")
                self.__class__.calls += 1
                return _outcome(_trace_for_prompt(prompts_by_text[text], 0), 0)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "moe_cache_lab.stage1.GraniteTraceCollector", FakeCollector
        ), patch("moe_cache_lab.stage1.torch.set_num_threads"), patch(
            "moe_cache_lab.stage1.torch.get_num_threads", return_value=4
        ), patch.dict(os.environ, {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }):
            path = collect_stage1_repetition(
                SOURCE_CORPUS, Path(temporary) / "rep", 1, str(uuid4())
            )
            loaded = load_stage1_repetition_manifest(path)
        self.assertEqual((FakeCollector.instances, FakeCollector.calls), (1, 12))
        self.assertEqual(len(loaded.prompts), 12)

    def test_repetition_loader_accepts_all_eos_and_reconciles_empty_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_repetition(Path(temporary) / "rep", 1, str(uuid4()), 100)
            loaded = load_stage1_repetition_manifest(path)
        self.assertEqual(len(loaded.prompts), 12)
        for prompt in loaded.prompts:
            self.assertEqual(prompt.record["eos"]["actual_decode_input_steps_routed"], 0)
            self.assertEqual(prompt.record["measured"]["generated_event_count"], 0)
            self.assertEqual(
                prompt.record["views"][PREFILL_LAYER_UNION_ATOMIC]["generated"]["bundles"], 0
            )

    def test_no_eos_counts_match_frozen_formulas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_repetition(
                Path(temporary) / "rep", 1, str(uuid4()), 100,
                actual_steps=16, trusted_prefix=True,
            )
            loaded = load_stage1_repetition_manifest(path)
        calibration = [item for item in loaded.prompts if item.prompt.split == "calibration"]
        evaluation = [item for item in loaded.prompts if item.prompt.split == "evaluation"]
        self.assertEqual(sum(item.record["measured"]["routing_event_count"] for item in calibration), 2664)
        self.assertEqual(sum(item.record["measured"]["expert_request_count"] for item in calibration), 21312)
        self.assertEqual(sum(item.record["measured"]["routing_event_count"] for item in evaluation), 5832)
        self.assertEqual(sum(item.record["measured"]["expert_request_count"] for item in evaluation), 46656)
        self.assertEqual(sum(
            item.record["views"][PREFILL_LAYER_UNION_ATOMIC]["combined"]["bundles"]
            for item in evaluation
        ), 3264)
        self.assertEqual(sum(
            item.record["views"][PREFILL_LAYER_UNION_ATOMIC]["combined"]["requests"]
            for item in evaluation
        ), 29937)

    def test_manifest_rejects_revision_thread_offline_hash_path_count_and_token_tamper(self) -> None:
        mutations = (
            lambda manifest: manifest["model_revision"].update(resolved="wrong"),
            lambda manifest: manifest["environment"].update(torch_num_threads=3),
            lambda manifest: manifest["environment"].update(local_files_only=False),
            lambda manifest: manifest["prompts"][0].update(trace_sha256="0" * 64),
            lambda manifest: manifest["prompts"][0].update(trace_path="../escape.jsonl"),
            lambda manifest: manifest["prompts"][0]["views"][TOKEN_LAYER_ATOMIC]["combined"].update(bundles=99),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                path = _write_repetition(Path(temporary) / "rep", 1, str(uuid4()), 100)
                manifest = json.loads(path.read_text(encoding="utf-8"))
                mutation(manifest)
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_stage1_repetition_manifest(path)

        with tempfile.TemporaryDirectory() as temporary:
            path = _write_repetition(
                Path(temporary) / "rep", 1, str(uuid4()), 100,
                actual_steps=1,
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
            eos = manifest["prompts"][0]["eos"]
            eos["routed_non_eos_token_ids"][0] += 1
            eos["emitted_non_eos_token_ids"][0] += 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "token ID"):
                load_stage1_repetition_manifest(path)

    def test_manifest_rejects_hashed_boolean_and_nan_trace_tamper(self) -> None:
        mutations = (
            lambda event: event.update(token_position=True),
            lambda event: event["selected_probabilities"].__setitem__(0, float("nan")),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                path = _write_repetition(Path(temporary) / "rep", 1, str(uuid4()), 100)
                manifest = json.loads(path.read_text(encoding="utf-8"))
                record = manifest["prompts"][0]
                trace_path = path.parent / record["trace_path"]
                trace_records = [
                    json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
                ]
                mutation(trace_records[1])
                trace_path.write_text(
                    "\n".join(json.dumps(item, sort_keys=True) for item in trace_records) + "\n",
                    encoding="utf-8",
                )
                record["trace_sha256"] = file_sha256(trace_path)
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_stage1_repetition_manifest(path)

    def test_repetition_timing_requires_aware_ordered_finite_values(self) -> None:
        mutations = (
            lambda manifest: manifest["process"].update(elapsed_seconds=True),
            lambda manifest: manifest["process"].update(elapsed_seconds=float("nan")),
            lambda manifest: manifest["process"].update(elapsed_seconds=-1.0),
            lambda manifest: manifest["process"].update(started_at="2026-01-01T00:00:00"),
            lambda manifest: manifest["process"].update(started_at="2026-01-01T00:00:02+00:00"),
            lambda manifest: manifest.update(created_at="2026-01-01T00:00:02+00:00"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                path = _write_repetition(Path(temporary) / "rep", 1, str(uuid4()), 100)
                manifest = json.loads(path.read_text(encoding="utf-8"))
                mutation(manifest)
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_stage1_repetition_manifest(path)

    def test_semantic_comparator_tolerance_and_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = load_stage1_repetition_manifest(_write_repetition(root / "a", 1, str(uuid4()), 100))
            near = load_stage1_repetition_manifest(_write_repetition(
                root / "b", 2, str(uuid4()), 101, probability_delta=5e-7
            ))
            same = load_stage1_repetition_manifest(_write_repetition(root / "c", 3, str(uuid4()), 102))
            compare_stage1_repetitions((base, near, same))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = load_stage1_repetition_manifest(_write_repetition(root / "a", 1, str(uuid4()), 100))
            far = load_stage1_repetition_manifest(_write_repetition(
                root / "b", 2, str(uuid4()), 101, probability_delta=2e-6
            ))
            same = load_stage1_repetition_manifest(_write_repetition(root / "c", 3, str(uuid4()), 102))
            with self.assertRaisesRegex(ValueError, "probability"):
                compare_stage1_repetitions((base, far, same))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = load_stage1_repetition_manifest(_write_repetition(root / "a", 1, str(uuid4()), 100))
            changed = load_stage1_repetition_manifest(_write_repetition(
                root / "b", 2, str(uuid4()), 101, expert_mismatch=True
            ))
            same = load_stage1_repetition_manifest(_write_repetition(root / "c", 3, str(uuid4()), 102))
            with self.assertRaisesRegex(ValueError, "routing"):
                compare_stage1_repetitions((base, changed, same))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = load_stage1_repetition_manifest(_write_repetition(root / "a", 1, str(uuid4()), 100))
            changed = load_stage1_repetition_manifest(_write_repetition(
                root / "b", 2, str(uuid4()), 101, token_mismatch=True
            ))
            same = load_stage1_repetition_manifest(_write_repetition(root / "c", 3, str(uuid4()), 102))
            with self.assertRaisesRegex(ValueError, "routing"):
                compare_stage1_repetitions((base, changed, same))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = load_stage1_repetition_manifest(_write_repetition(root / "a", 1, str(uuid4()), 100))
            changed = load_stage1_repetition_manifest(_write_repetition(
                root / "b", 2, str(uuid4()), 101, actual_steps=1
            ))
            same = load_stage1_repetition_manifest(_write_repetition(root / "c", 3, str(uuid4()), 102))
            with self.assertRaisesRegex(ValueError, "EOS/token"):
                compare_stage1_repetitions((base, changed, same))

    def test_v02_prefix_gate_accepts_zero_one_and_two_decode_steps(self) -> None:
        actual = {order: order % 3 for order in range(12)}
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_repetition(
                Path(temporary) / "rep", 1, str(uuid4()), 100,
                actual_steps=actual, trusted_prefix=True,
            )
            loaded = load_stage1_repetition_manifest(path)
            validate_v02_prefix(loaded, TRUSTED_V02)

    def test_set_loader_allows_pid_reuse_but_requires_unique_uuid_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_set(Path(temporary) / "set")
            loaded = load_stage1_set_manifest(path)
            self.assertEqual(len(loaded.repetitions), 3)
            self.assertEqual(
                {item.manifest["process"]["pid"] for item in loaded.repetitions},
                {100},
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["repetitions"][1]["process_uuid"] = manifest["repetitions"][0]["process_uuid"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_stage1_set_manifest(path)

    def test_set_loader_accepts_windows_redirector_parent_pids_but_rejects_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_set(Path(temporary) / "set")
            for index, observed_parent_pid in enumerate((701, 702, 703)):
                _mutate_set_child(
                    path,
                    index,
                    lambda child, value=observed_parent_pid: child["process"].update(
                        parent_pid=value
                    ),
                )
            loaded = load_stage1_set_manifest(path)
            self.assertEqual(
                [item.manifest["process"]["parent_pid"] for item in loaded.repetitions],
                [701, 702, 703],
            )
            self.assertEqual(loaded.manifest["parent_pid"], 50)

        for invalid_parent_pid in (0, -1, True):
            with self.subTest(parent_pid=invalid_parent_pid), tempfile.TemporaryDirectory() as temporary:
                path = _write_set(Path(temporary) / "set")
                _mutate_set_child(
                    path,
                    0,
                    lambda child, value=invalid_parent_pid: child["process"].update(
                        parent_pid=value
                    ),
                )
                with self.assertRaisesRegex(ValueError, "parent_pid must be a positive integer"):
                    load_stage1_set_manifest(path)

    def test_set_timing_rejects_invalid_parent_and_child_intervals(self) -> None:
        parent_mutations = (
            lambda manifest: manifest.update(elapsed_seconds=True),
            lambda manifest: manifest.update(elapsed_seconds=float("inf")),
            lambda manifest: manifest.update(elapsed_seconds=-1.0),
            lambda manifest: manifest.update(elapsed_seconds=901.0),
            lambda manifest: manifest.update(started_at="2026-01-01T00:00:00"),
            lambda manifest: manifest.update(started_at="2026-01-01T00:00:04+00:00"),
            lambda manifest: manifest.update(created_at="2026-01-01T00:00:04+00:00"),
        )
        for mutation in parent_mutations:
            with self.subTest(parent_mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                path = _write_set(Path(temporary) / "set")
                manifest = json.loads(path.read_text(encoding="utf-8"))
                mutation(manifest)
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_stage1_set_manifest(path)

        child_mutations = (
            lambda manifest: manifest["process"].update(
                started_at="2025-12-31T23:59:58+00:00",
                finished_at="2025-12-31T23:59:59+00:00",
            ) or manifest.update(created_at="2025-12-31T23:59:59+00:00"),
            lambda manifest: manifest["process"].update(
                started_at="2026-01-01T00:00:00.500000+00:00",
            ),
            lambda manifest: manifest["process"].update(
                started_at="2026-01-01T00:00:01",
            ),
        )
        for index, mutation in enumerate(child_mutations):
            with self.subTest(child_mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                path = _write_set(Path(temporary) / "set")
                _mutate_set_child(path, min(index, 1), mutation)
                with self.assertRaises(ValueError):
                    load_stage1_set_manifest(path)


class Stage1OrchestrationTests(unittest.TestCase):
    def test_parent_runs_three_fresh_offline_children_with_shared_deadline(self) -> None:
        calls = []

        def fake_run(command, *, env, timeout, check):
            calls.append((command, env, timeout, check))
            output = Path(command[command.index("--output-dir") + 1])
            number = int(command[command.index("--repetition-number") + 1])
            process_uuid = command[command.index("--process-uuid") + 1]
            child_path = _write_repetition(
                output, number, process_uuid, 200, trusted_prefix=True,
                parent_pid=os.getpid(),
            )
            child = json.loads(child_path.read_text(encoding="utf-8"))
            timestamp = datetime.now(timezone.utc).isoformat()
            child["created_at"] = timestamp
            child["process"].update(
                started_at=timestamp,
                finished_at=timestamp,
                elapsed_seconds=0.0,
            )
            child_path.write_text(json.dumps(child), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        clock_values = iter((0.0, 1.0, 2.0, 3.0, 4.0))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "final"
            manifest = collect_stage1_set(
                SOURCE_CORPUS,
                output,
                run_process=fake_run,
                monotonic=lambda: next(clock_values),
            )
            loaded = load_stage1_set_manifest(manifest)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({call[0][call[0].index("--process-uuid") + 1] for call in calls}), 3)
        self.assertTrue(all(call[1]["HF_HUB_OFFLINE"] == "1" for call in calls))
        self.assertTrue(all(call[1]["TRANSFORMERS_OFFLINE"] == "1" for call in calls))
        self.assertTrue(all(call[1]["HF_HUB_DISABLE_TELEMETRY"] == "1" for call in calls))
        self.assertTrue(all(call[1]["OMP_NUM_THREADS"] == "4" for call in calls))
        self.assertTrue(all(call[1]["MKL_NUM_THREADS"] == "4" for call in calls))
        self.assertEqual(
            [int(call[0][call[0].index("--repetition-number") + 1]) for call in calls],
            [1, 2, 3],
        )
        self.assertEqual([call[2] for call in calls], [899.0, 898.0, 897.0])
        self.assertEqual(len(loaded.repetitions), 3)

    def test_parent_stops_on_child_failure_timeout_and_expired_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                collect_stage1_set(
                    SOURCE_CORPUS,
                    Path(temporary) / "failed",
                    run_process=lambda *args, **kwargs: SimpleNamespace(returncode=7),
                    monotonic=iter((0.0, 1.0)).__next__,
                )
        with tempfile.TemporaryDirectory() as temporary:
            def timeout(*args, **kwargs):
                raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
            with self.assertRaisesRegex(TimeoutError, "900-second"):
                collect_stage1_set(
                    SOURCE_CORPUS,
                    Path(temporary) / "timeout",
                    run_process=timeout,
                    monotonic=iter((0.0, 1.0)).__next__,
                )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TimeoutError, "900-second"):
                collect_stage1_set(
                    SOURCE_CORPUS,
                    Path(temporary) / "expired",
                    run_process=lambda *args, **kwargs: self.fail("child should not run"),
                    monotonic=iter((0.0, 901.0)).__next__,
                )


if __name__ == "__main__":
    unittest.main()
