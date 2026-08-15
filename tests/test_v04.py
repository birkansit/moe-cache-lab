from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
import ctypes
import tempfile
import unittest
from unittest import mock
from uuid import uuid4

import torch
import moe_cache_lab.v04 as v04

from moe_cache_lab.collector import _RouterHookCapture
from moe_cache_lab.v04 import (
    CALIBRATION_IDS,
    ChildProcessOutcome,
    CPU_COUNTER_PATH,
    DISK_COUNTER_PATH,
    EVALUATION_IDS,
    PROFILED_MODE,
    REFERENCE_MODE,
    ResourceSample,
    ResourcePrime,
    ResourceProbeConstructionError,
    AdmissionResult,
    ProcessMemory,
    WindowsResourceProbe,
    V04_SPEC_SHA256,
    _admission_record,
    _finalize_nonvalid,
    _prompt_run_from_record,
    _prompt_run_record,
    _run_with_memory,
    _samples_pass,
    _validate_admission_artifact,
    _validate_cross_child_semantics,
    _write_launch_failure,
    aggregate_children,
    calibration_schedule,
    collect_admission,
    collect_v04,
    evaluation_schedule,
    load_v04_json,
    run_prompt,
    render_markdown,
    transition_state,
    validate_v04,
    validate_prompt_pair,
    write_json_atomic,
)
from moe_cache_lab.workflow import CorpusPrompt


class _Raises:
    def __init__(self, exception, match=None):
        self.exception = exception
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is None or not issubclass(exception_type, self.exception):
            return False
        if self.match is not None and self.match not in str(exception):
            raise AssertionError(f"{self.match!r} not found in {str(exception)!r}")
        return True


class _RaisesFactory:
    @staticmethod
    def raises(exception, match=None):
        return _Raises(exception, match)


pytest = _RaisesFactory()


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(value)) for value in ids)


class FakeModel:
    def __init__(self, candidates):
        self.candidates = iter(candidates)

    def named_modules(self):
        return ()

    def __call__(self, **kwargs):
        candidate = next(self.candidates)
        logits = torch.full((1, 1, 128), -10.0)
        logits[0, 0, candidate] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=(object(),))


class FakeCapture:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def take(self):
        return object()


def fake_events(_raw, token_ids, phase, offset, _top_k):
    from moe_cache_lab.trace import RoutingEvent
    return [
        RoutingEvent(
            phase=phase, token_position=offset + index, layer=0,
            selected_experts=(0,), token_id=int(token_ids[0, index]),
            selected_probabilities=(1.0,),
        )
        for index in range(token_ids.shape[1])
    ]


class TickClock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 10
        return self.value


def collector(candidates):
    return SimpleNamespace(
        tokenizer=FakeTokenizer(), model=FakeModel(candidates), experts_per_token=1
    )


def prompt():
    return CorpusPrompt("eval-factual-01", "factual", "evaluation", "x", 4)


def test_exact_counterbalance_and_warmup_order():
    for child in (1, 3):
        assert evaluation_schedule(child)[0][1] == (REFERENCE_MODE, PROFILED_MODE)
        assert evaluation_schedule(child)[1][1] == (PROFILED_MODE, REFERENCE_MODE)
        assert calibration_schedule(child) == tuple(
            (item, mode) for mode in (REFERENCE_MODE, PROFILED_MODE) for item in CALIBRATION_IDS
        )
    for child in (2, 4):
        assert evaluation_schedule(child)[0][1] == (PROFILED_MODE, REFERENCE_MODE)
        assert calibration_schedule(child)[0] == (CALIBRATION_IDS[0], PROFILED_MODE)
    assert all(tuple(item for item, _ in evaluation_schedule(child)) == EVALUATION_IDS for child in range(1, 5))


def test_driver_horizon_records_validation_candidate_and_nested_timers():
    candidates = list(range(10, 27))
    reference = run_prompt(collector(candidates), prompt(), REFERENCE_MODE, clock_ns=TickClock(), expected_router_layers=None)
    profiled = run_prompt(
        collector(candidates), prompt(), PROFILED_MODE, clock_ns=TickClock(),
        capture_factory=FakeCapture, event_extractor=fake_events, expected_router_layers=None,
    )
    validate_prompt_pair(reference, profiled)
    assert reference.semantic.actual_decode_input_steps == 16
    assert reference.semantic.validation_only_horizon_candidate_id == 26
    assert len(reference.semantic.candidate_token_ids) == 17
    assert len(profiled.events) == 18
    assert profiled.timing.primary_total_ns == profiled.timing.accounted_ns + profiled.timing.unaccounted_ns
    assert len(profiled.timing.decode_forward_ns) == 16
    assert len(profiled.timing.event_extraction_ns) == 17
    horizon_eos = run_prompt(
        collector(list(range(10, 26)) + [99]), prompt(), REFERENCE_MODE,
        clock_ns=TickClock(), expected_router_layers=None,
    )
    assert horizon_eos.semantic.horizon_exhausted
    assert not horizon_eos.semantic.eos_emitted
    assert horizon_eos.semantic.validation_only_horizon_candidate_id == 99


def test_driver_y0_eos_and_semantic_mismatch():
    left = run_prompt(collector([99]), prompt(), REFERENCE_MODE, clock_ns=TickClock(), expected_router_layers=None)
    right = run_prompt(
        collector([99]), prompt(), PROFILED_MODE, clock_ns=TickClock(),
        capture_factory=FakeCapture, event_extractor=fake_events, expected_router_layers=None,
    )
    validate_prompt_pair(left, right)
    assert left.semantic.actual_decode_input_steps == 0
    assert left.semantic.eos_emitted
    changed = run_prompt(
        collector([98, 99]), prompt(), PROFILED_MODE, clock_ns=TickClock(),
        capture_factory=FakeCapture, event_extractor=fake_events, expected_router_layers=None,
    )
    with pytest.raises(ValueError, match="changed semantic"):
        validate_prompt_pair(left, changed)


class FakeProbe:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.calls = 0

    def sample(self):
        self.calls += 1
        value = next(self.samples)
        if isinstance(value, Exception):
            raise value
        return value

    def prime(self):
        self.calls += 1
        return ResourcePrime("2000-01-01T00:00:00+00:00", 0)


def sample(cpu=1.0, disk=1.0, available=9_000_000_000, commit=50):
    return ResourceSample(cpu, disk, available, commit, 100, 4096)


def test_admission_exact_baseline_plus_fifteen_and_paths():
    probe = FakeProbe([sample()] * 16)
    sleeps = []
    result = collect_admission(probe, sleep=sleeps.append)
    assert result.admitted
    assert probe.calls == 17
    assert sleeps == [1.0] * 15
    assert len(result.samples) == 15
    assert result.prime is not None and result.baseline is not None
    assert result.samples[0].cpu_counter_path == CPU_COUNTER_PATH
    assert result.samples[0].disk_counter_path == DISK_COUNTER_PATH


def test_admission_invalid_sample_is_deferred_not_replaced():
    values = [sample()] * 5 + [RuntimeError("missing counter")] + [sample()] * 10
    probe = FakeProbe(values)
    result = collect_admission(probe, sleep=lambda _: None)
    assert not result.admitted
    assert probe.calls == 17
    assert len(result.samples) == 14
    assert "missing counter" in result.errors[0]
    with pytest.raises(ValueError):
        ResourceSample(float("nan"), 0, 1, 1, 1, 1)


def child(number, r=100, p=105):
    return {
        "child_number": number,
        "evaluation": [
            {
                "prompt_id": prompt_id,
                "mode_order": list(modes),
                "reference_total_ns": r,
                "profiled_total_ns": p,
                "reference_working_set_delta": 0,
                "profiled_working_set_delta": 0,
                "reference_private_delta": 0,
                "profiled_private_delta": 0,
            }
            for prompt_id, modes in evaluation_schedule(number)
        ],
    }


def test_hierarchical_aggregation_stable_and_exact_boundaries():
    result = aggregate_children([child(number) for number in range(1, 5)])
    assert result["decision"] == "valid_stable"
    assert result["child_count"] == 4
    assert result["workload_prompt_count"] == 8
    assert all(row["n"] == 4 for row in result["prompt_rows"])
    assert result["prompt_macro"]["n"] == 8
    markdown = render_markdown(result)
    assert "n=8 prompt means" in markdown
    assert "n children" in markdown
    assert "PeakWorkingSetSize" in markdown


def test_aggregation_rejects_order_and_marks_variability_inconclusive():
    bad = child(1)
    bad["evaluation"][0]["prompt_id"] = EVALUATION_IDS[1]
    with pytest.raises(ValueError, match="prompt order"):
        aggregate_children([bad, child(2), child(3), child(4)])
    result = aggregate_children([
        child(1, 100, 100), child(2, 100, 100),
        child(3, 100, 130), child(4, 100, 130),
    ])
    assert result["decision"] == "valid_inconclusive"


def test_attempt_state_transitions_are_explicit():
    assert transition_state("planned", "admission_deferred") == "deferred"
    assert transition_state("running", "invalid") == "invalid"
    assert transition_state("running", "invalid", repeated_same_root_cause=True) == "blocked"
    assert transition_state("running", "stable") == "valid_stable"
    assert transition_state("running", "inconclusive") == "valid_inconclusive"
    assert transition_state("running", "inconclusive", rerun_index=1) == "inconclusive_final"
    with pytest.raises(ValueError):
        transition_state("valid_stable", "stable")


def test_json_is_deterministic_and_rejects_nonfinite(tmp_path: Path):
    path = write_json_atomic(tmp_path / "value.json", {"format": "x", "format_version": 1, "z": 1})
    assert load_v04_json(path)["z"] == 1
    with pytest.raises(ValueError):
        write_json_atomic(tmp_path / "nan.json", {"value": float("nan")})


def test_router_capture_constructor_removes_partial_handles():
    removed = []

    class Handle:
        def remove(self):
            removed.append(True)

    class Module:
        def __init__(self, fail=False):
            self.fail = fail

        def register_forward_hook(self, hook):
            if self.fail:
                raise RuntimeError("register failed")
            return Handle()

    model = SimpleNamespace(named_modules=lambda: (
        ("model.layers.0.block_sparse_moe.router", Module()),
        ("model.layers.1.block_sparse_moe.router", Module(True)),
    ))
    with pytest.raises(RuntimeError, match="register failed"):
        _RouterHookCapture(model)
    assert removed == [True]


def test_pdh_prime_is_collect_only_and_abi_is_aligned():
    calls = []

    class Pdh:
        def PdhCollectQueryData(self, query):
            calls.append(("collect", query))
            return 0

        def PdhGetFormattedCounterValue(self, *args):
            raise AssertionError("prime must not format a rate counter")

    probe = WindowsResourceProbe.__new__(WindowsResourceProbe)
    probe._pdh = Pdh()
    probe._query = 7
    prime = probe.prime()
    assert prime.collect_status == 0
    assert calls == [("collect", 7)]
    assert ctypes.sizeof(WindowsResourceProbe._PdhValue) == 16
    assert ctypes.sizeof(WindowsResourceProbe._RawCounter) >= 32


def _attach_memory(run):
    memory = ProcessMemory(1, 1, 1, 9_000_000_000)
    object.__setattr__(run, "_memory_before", memory)
    object.__setattr__(run, "_memory_after", memory)
    object.__setattr__(run, "_process_user_ns", 0)
    object.__setattr__(run, "_process_system_ns", 0)
    return run


def test_timer_reconciliation_and_cross_child_horizon_semantics():
    run = _attach_memory(run_prompt(
        collector(list(range(10, 27))), prompt(), REFERENCE_MODE,
        clock_ns=TickClock(), expected_router_layers=None,
    ))
    record = _prompt_run_record(run)
    record["timing"]["decode_forward_ns"] = record["timing"]["decode_forward_ns"][:-1]
    with pytest.raises(ValueError):
        _prompt_run_from_record(record)

    left = _prompt_run_record(run)
    right = _prompt_run_record(run)
    right["semantic"]["validation_only_horizon_candidate_id"] += 1
    candidates = list(right["semantic"]["candidate_token_ids"])
    candidates[-1] += 1
    right["semantic"]["candidate_token_ids"] = candidates
    children = [
        {"calibration": [left], "evaluation": []},
        {"calibration": [left], "evaluation": []},
        {"calibration": [right], "evaluation": []},
        {"calibration": [left], "evaluation": []},
    ]
    with pytest.raises(ValueError, match="cross-child semantic"):
        _validate_cross_child_semantics(children)


def test_nonvalid_attempt_is_validated_before_atomic_publication(tmp_path: Path):
    attempt_uuid = str(uuid4())
    final = tmp_path / f"v0.4-cpu-profiler-attempt-{attempt_uuid}"
    staging = tmp_path / f".v0.4-cpu-profiler-attempt-{attempt_uuid}.staging"
    staging.mkdir()
    now = datetime.now(timezone.utc).isoformat()
    admission = AdmissionResult(
        False, None, None, (), ("probe: missing API",),
        "resource prerequisite unavailable",
        {"open_query_status": None, "add_cpu_status": None, "add_disk_status": None, "close_query_status": None, "constructor_error": "OSError: missing"},
        now, now, 0, 0,
    )
    admission_path = write_json_atomic(staging / "admission-1.json", _admission_record(1, admission))
    import hashlib
    ref = {
        "child_number": 1, "path": admission_path.name,
        "sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "size": admission_path.stat().st_size,
    }
    result = _finalize_nonvalid(
        staging, final, attempt_uuid, "deferred", datetime.now(timezone.utc),
        "missing API", [ref], [], [], 0, None, None, elapsed_seconds=0.0,
    )
    assert result == final / "v04-set.json"
    assert final.is_dir() and not staging.exists()
    validate_v04(result)

    bad_uuid = str(uuid4())
    bad_final = tmp_path / f"v0.4-cpu-profiler-attempt-{bad_uuid}"
    bad_staging = tmp_path / f".v0.4-cpu-profiler-attempt-{bad_uuid}.staging"
    bad_staging.mkdir()
    with pytest.raises(ValueError):
        _finalize_nonvalid(
            bad_staging, bad_final, bad_uuid, "invalid", datetime.now(timezone.utc),
            "bad reference", [{**ref, "path": "missing.json"}], [], [], 0, None, None,
        )
    assert bad_staging.is_dir() and not bad_final.exists()


def test_first_prerequisite_defer_is_outside_timer_and_second_is_blocked(tmp_path: Path):
    calls = []

    def no_monotonic():
        calls.append(True)
        raise AssertionError("first-admission deferral must remain outside experiment timer")

    def missing_probe():
        raise OSError("English PDH counter missing")

    corpus_path = Path("benchmarks/corpus-v1.json")
    first = collect_v04(
        corpus_path, tmp_path, probe_factory=missing_probe,
        monotonic=no_monotonic,
    )
    first_record = load_v04_json(first)
    assert first_record["state"] == "deferred"
    assert first_record["experiment_started_at"] is None
    assert first_record["elapsed_seconds"] == 0
    assert calls == []
    with pytest.raises(ValueError, match="requires explicit --retry-of"):
        collect_v04(
            corpus_path, tmp_path, probe_factory=missing_probe,
            monotonic=no_monotonic,
        )

    second = collect_v04(
        corpus_path, tmp_path, probe_factory=missing_probe,
        retry_of=first, monotonic=no_monotonic,
    )
    second_record = load_v04_json(second)
    assert second_record["state"] == "blocked"
    assert second_record["retry_of"] == first_record["attempt_uuid"]

    def changed_probe():
        raise OSError("different English PDH counter missing")

    resumed = collect_v04(
        corpus_path, tmp_path, probe_factory=changed_probe,
        retry_of=second, monotonic=no_monotonic,
    )
    resumed_record = load_v04_json(resumed)
    assert resumed_record["state"] == "deferred"
    assert resumed_record["retry_of"] == second_record["attempt_uuid"]
    assert resumed_record["root_cause"]["digest"] != second_record["root_cause"]["digest"]
    assert list(tmp_path.glob("*.staging")) == []
    with pytest.raises(ValueError, match="already been consumed"):
        collect_v04(
            corpus_path, tmp_path, probe_factory=missing_probe,
            retry_of=first, monotonic=no_monotonic,
        )

    different_root = tmp_path / "different-root"
    different_root.mkdir()
    original = collect_v04(
        corpus_path, different_root, probe_factory=missing_probe,
        monotonic=no_monotonic,
    )

    def different_probe():
        raise OSError("different English PDH failure")

    changed = collect_v04(
        corpus_path, different_root, probe_factory=different_probe,
        retry_of=original, monotonic=no_monotonic,
    )
    assert load_v04_json(changed)["state"] == "deferred"

    structured_root = tmp_path / "structured-constructor"
    structured_root.mkdir()
    constructor_audit = {
        "open_query_status": 0, "add_cpu_status": 0xC0000BB8,
        "add_disk_status": None, "close_query_status": 0,
        "constructor_error": None,
    }

    def structured_probe():
        raise ResourceProbeConstructionError("CPU counter add failed", constructor_audit)

    structured = collect_v04(
        corpus_path, structured_root, probe_factory=structured_probe,
        monotonic=no_monotonic,
    )
    structured_manifest = load_v04_json(structured)
    admission_ref = structured_manifest["admissions"][0]
    admission_value = load_v04_json(structured.parent / admission_ref["path"])
    assert admission_value["audit"]["add_cpu_status"] == 0xC0000BB8
    assert admission_value["audit"]["close_query_status"] == 0


def test_parent_records_zero_exit_artifact_failures_and_timeout_pid(tmp_path: Path):
    class ParentProbe(FakeProbe):
        def __init__(self):
            super().__init__([sample()] * 16)

        def close(self):
            return None

        def audit_record(self):
            return {"open_query_status": 0, "add_cpu_status": 0, "add_disk_status": 0, "close_query_status": 0, "constructor_error": None}

    def execute(case: str, timed_out: bool = False):
        root = tmp_path / case
        root.mkdir()

        def runner(command, **kwargs):
            output = Path(command[command.index("--output-dir") + 1])
            number = int(command[command.index("--child-number") + 1])
            process_uuid = command[command.index("--process-uuid") + 1]
            now = datetime.now(timezone.utc).isoformat()
            if case == "launch-exception":
                raise OSError("CreateProcess failed")
            if case != "missing" and not timed_out:
                output.mkdir()
                if case == "malformed":
                    (output / "v04-child.json").write_text("{bad", encoding="utf-8")
                else:
                    write_json_atomic(output / "v04-child.json", {
                        "format": v04.V04_CHILD_FORMAT, "format_version": 1,
                        "state": "valid", "child_number": number,
                        "process_uuid": str(uuid4()), "started_at": now,
                        "finished_at": now,
                    })
            return ChildProcessOutcome(
                9001, None if timed_out else 0, "out", "err", timed_out,
                now, now,
            )

        tick = iter(float(value) for value in range(100))
        with mock.patch("moe_cache_lab.v04.validate_trusted_stage1_paths"), mock.patch(
            "moe_cache_lab.v04._validate_child_record"
        ), mock.patch("moe_cache_lab.v04.validate_v04", side_effect=lambda path: Path(path).resolve()):
            with pytest.raises((RuntimeError, TimeoutError)):
                collect_v04(
                    Path("benchmarks/corpus-v1.json"), root,
                    run_process=runner, probe_factory=ParentProbe,
                    sleep=lambda _: None, monotonic=lambda: next(tick),
                )
        manifests = list(root.glob("v0.4-cpu-profiler-attempt-*/v04-set.json"))
        assert len(manifests) == 1
        record = load_v04_json(manifests[0])
        assert record["children"] == [] and len(record["failures"]) == 1
        failure_ref = record["failures"][0]
        assert failure_ref["actual_pid"] == (None if case == "launch-exception" else 9001)
        artifact = load_v04_json(manifests[0].parent / failure_ref["path"])
        assert artifact["actual_pid"] == (None if case == "launch-exception" else 9001)
        assert artifact["stdout"] == ("" if case == "launch-exception" else "out")
        assert artifact["failure_category"] == (
            "launch_exception" if case == "launch-exception" else
            ("timeout" if timed_out else "child_artifact")
        )
        assert list(root.glob("*.staging")) == []
        return artifact

    assert execute("missing")["terminal_manifest_path"] is None
    assert execute("malformed")["terminal_manifest_path"] == "child-1/v04-child.json"
    assert execute("wrong-identity")["terminal_manifest_path"] == "child-1/v04-child.json"
    assert execute("timeout", timed_out=True)["timed_out"] is True
    launch = execute("launch-exception")
    assert launch["returncode"] is None
    assert launch["reason"] == "launch exception: OSError: CreateProcess failed"


def test_popen_runner_preserves_pid_and_timeout_evidence():
    class FakeProcess:
        pid = 8765
        returncode = -9

        def __init__(self):
            self.calls = 0
            self.killed = False

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise v04.subprocess.TimeoutExpired(
                    cmd=["python", "-m", "child"], timeout=timeout,
                    output="partial-out", stderr="partial-err",
                )
            return "tail-out", "tail-err"

        def kill(self):
            self.killed = True

    process = FakeProcess()
    outcome = v04.run_child_process(
        ["python", "-m", "child"], env={"FLAG": "1"}, timeout=0.25,
        popen_factory=lambda *args, **kwargs: process,
    )
    assert process.killed
    assert outcome.pid == 8765
    assert outcome.timed_out
    assert outcome.returncode is None
    assert outcome.stdout == "partial-outtail-out"
    assert outcome.stderr == "partial-errtail-err"


def test_launched_failure_categories_reject_null_pid_with_recomputed_reference(tmp_path: Path):
    cases = (
        ("timeout", None, True),
        ("child-exit", 7, False),
        ("child-artifact", 0, False),
    )
    for case, returncode, timed_out in cases:
        base = tmp_path / case
        base.mkdir()
        started = datetime.now(timezone.utc) - timedelta(seconds=2)
        finished = started + timedelta(seconds=1)
        reference = _write_launch_failure(
            base, 1, str(uuid4()), Path("child-1"), started, finished,
            returncode, timed_out, "stdout", "stderr", case, actual_pid=1234,
        )
        artifact_path = base / reference["path"]
        artifact = load_v04_json(artifact_path)
        artifact["actual_pid"] = None
        write_json_atomic(artifact_path, artifact)
        reference["actual_pid"] = None
        reference["sha256"] = v04.file_sha256(artifact_path)
        reference["size"] = artifact_path.stat().st_size
        with pytest.raises(ValueError, match="positive actual PID"):
            v04._validate_failure_reference(
                base, reference, started - timedelta(seconds=1),
                finished + timedelta(seconds=1),
            )

    path_base = tmp_path / "shifted-terminal"
    (path_base / "child-1").mkdir(parents=True)
    terminal = path_base / "child-1" / "v04-child.json"
    terminal.write_text('{"terminal":true}\n', encoding="utf-8")
    started = datetime.now(timezone.utc) - timedelta(seconds=2)
    finished = started + timedelta(seconds=1)
    reference = _write_launch_failure(
        path_base, 1, str(uuid4()), Path("child-1"), started, finished,
        0, False, "", "", "invalid child artifact", actual_pid=1234,
    )
    artifact_path = path_base / reference["path"]
    artifact = load_v04_json(artifact_path)
    decoy = path_base / "decoy-child.json"
    decoy.write_bytes(terminal.read_bytes())
    artifact["terminal_manifest_path"] = decoy.name
    write_json_atomic(artifact_path, artifact)
    reference["sha256"] = v04.file_sha256(artifact_path)
    reference["size"] = artifact_path.stat().st_size
    with pytest.raises(ValueError, match="assigned child path"):
        v04._validate_failure_reference(
            path_base, reference, started - timedelta(seconds=1),
            finished + timedelta(seconds=1),
        )


def test_recursive_lineage_and_invalid_partial_dependencies(tmp_path: Path):
    corpus_path = Path("benchmarks/corpus-v1.json")

    def missing_probe():
        raise OSError("counter missing")

    prior = collect_v04(corpus_path, tmp_path, probe_factory=missing_probe)
    prior_value = load_v04_json(prior)
    prior_value["root_cause"]["message"] = "tampered without digest update"
    write_json_atomic(prior, prior_value)
    with pytest.raises(ValueError, match="root-cause fingerprint"):
        collect_v04(corpus_path, tmp_path, probe_factory=missing_probe, retry_of=prior)

    now = datetime.now(timezone.utc).isoformat()
    invalid = {
        "format": v04.V04_CHILD_FORMAT, "format_version": 1, "state": "invalid",
        "child_number": 1, "process_uuid": str(uuid4()), "pid": 1,
        "parent_pid": 1, "started_at": now, "finished_at": now,
        "reason": "RuntimeError: x", "reason_type": "RuntimeError",
        "reason_message": "x", "failure_phase": "evaluation",
        "partial": {
            "model_load": None, "completed_calibration_runs": 0,
            "completed_evaluation_prompts": 1, "last_resource_snapshot": None,
        },
    }
    with pytest.raises(ValueError, match="pre-model partial dependencies"):
        v04._validate_invalid_child_record(invalid)


def _complete_admission(start: datetime) -> AdmissionResult:
    mono = 1_000_000_000
    prime = ResourcePrime((start + timedelta(milliseconds=10)).isoformat(), 0, monotonic_ns=mono + 10_000_000)
    baseline = sample(cpu=10, disk=10, available=8_589_934_592, commit=75)
    object.__setattr__(baseline, "timestamp", (start + timedelta(milliseconds=100)).isoformat())
    object.__setattr__(baseline, "monotonic_ns", mono + 100_000_000)
    samples = []
    for index in range(15):
        item = sample(
            cpu=25 if index == 14 else 10,
            disk=50 if index == 14 else 10,
            available=8_589_934_592, commit=75,
        )
        object.__setattr__(item, "timestamp", (start + timedelta(seconds=index + 1, milliseconds=100)).isoformat())
        object.__setattr__(item, "monotonic_ns", mono + (index + 1) * 1_000_000_000 + 100_000_000)
        samples.append(item)
    return AdmissionResult(
        True, prime, baseline, tuple(samples), (), "admitted",
        {"open_query_status": 0, "add_cpu_status": 0, "add_disk_status": 0, "close_query_status": 0, "constructor_error": None},
        start.isoformat(), (start + timedelta(seconds=16)).isoformat(),
        mono, mono + 16_000_000_000,
    )


def test_admission_threshold_equality_cadence_and_timeout_failure_artifact(tmp_path: Path):
    start = datetime.now(timezone.utc) - timedelta(seconds=30)
    admission = _complete_admission(start)
    assert _samples_pass(admission.samples)
    record = _admission_record(1, admission)
    _validate_admission_artifact(record, 1, require_admitted=True)
    equal_wall = dict(record)
    equal_wall["baseline"] = dict(record["baseline"])
    equal_wall["baseline"]["timestamp"] = record["prime"]["timestamp"]
    _validate_admission_artifact(equal_wall, 1, require_admitted=True)
    bad = dict(record)
    bad["samples"] = [dict(item) for item in record["samples"]]
    prior_wall = datetime.fromisoformat(bad["samples"][1]["timestamp"])
    bad["samples"][2]["timestamp"] = (prior_wall - timedelta(microseconds=1)).isoformat()
    with pytest.raises(ValueError, match="nondecreasing"):
        _validate_admission_artifact(bad, 1, require_admitted=True)

    attempt_uuid = str(uuid4())
    final = tmp_path / f"v0.4-cpu-profiler-attempt-{attempt_uuid}"
    staging = tmp_path / f".v0.4-cpu-profiler-attempt-{attempt_uuid}.staging"
    staging.mkdir()
    admission_path = write_json_atomic(staging / "admission-1.json", record)
    import hashlib
    admission_ref = {
        "child_number": 1, "path": admission_path.name,
        "sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "size": admission_path.stat().st_size,
    }
    launch_start = start + timedelta(seconds=17)
    failure_ref = _write_launch_failure(
        staging, 1, str(uuid4()), Path("child-1"), launch_start,
        launch_start + timedelta(seconds=1), None, True, b"partial out",
        b"partial err", "timeout without child manifest",
        actual_pid=4321,
    )
    result = _finalize_nonvalid(
        staging, final, attempt_uuid, "invalid", start, "TimeoutError: deadline",
        [admission_ref], [], [failure_ref], 0, None, None,
        elapsed_seconds=2.0, experiment_started_at=start + timedelta(seconds=16, milliseconds=500),
    )
    validate_v04(result)
    failure = load_v04_json(final / failure_ref["path"])
    assert failure["timed_out"] and failure["stdout"] == "partial out"

    launch_uuid = str(uuid4())
    launch_final = tmp_path / f"v0.4-cpu-profiler-attempt-{launch_uuid}"
    launch_staging = tmp_path / f".v0.4-cpu-profiler-attempt-{launch_uuid}.staging"
    launch_staging.mkdir()
    # Regression for the observed Windows clock-resolution case: equal prime
    # and excluded-baseline wall timestamps still finalize atomically because
    # their monotonic timestamps are strictly ordered.
    launch_admission_path = write_json_atomic(launch_staging / "admission-1.json", equal_wall)
    launch_admission_ref = {
        "child_number": 1, "path": launch_admission_path.name,
        "sha256": hashlib.sha256(launch_admission_path.read_bytes()).hexdigest(),
        "size": launch_admission_path.stat().st_size,
    }
    launch_failure_ref = _write_launch_failure(
        launch_staging, 1, str(uuid4()), Path("child-1"), launch_start,
        launch_start + timedelta(seconds=1), None, False, "", "",
        "launch exception: OSError: CreateProcess failed", actual_pid=None,
    )
    launch_result = _finalize_nonvalid(
        launch_staging, launch_final, launch_uuid, "invalid", start,
        "RuntimeError: child process launch failed", [launch_admission_ref], [],
        [launch_failure_ref], 0, None, None, elapsed_seconds=2.0,
        experiment_started_at=start + timedelta(seconds=16, milliseconds=500),
    )
    validate_v04(launch_result)
    assert not launch_staging.exists()


def test_hook_forward_failure_removal_error_and_low_memory_abort():
    exits = []

    class FailingModel:
        def named_modules(self):
            return ()

        def __call__(self, **kwargs):
            raise RuntimeError("forward failed")

    class TrackingCapture(FakeCapture):
        def __exit__(self, *args):
            exits.append(args[0])

    failing = SimpleNamespace(tokenizer=FakeTokenizer(), model=FailingModel(), experts_per_token=1)
    with pytest.raises(RuntimeError, match="forward failed"):
        run_prompt(
            failing, prompt(), PROFILED_MODE, clock_ns=TickClock(),
            capture_factory=TrackingCapture, event_extractor=fake_events,
            expected_router_layers=None,
        )
    assert exits == [None]

    class RemovalFailure(FakeCapture):
        def __exit__(self, *args):
            raise RuntimeError("removal failed")

    with pytest.raises(RuntimeError, match="removal failed"):
        run_prompt(
            collector([99]), prompt(), PROFILED_MODE, clock_ns=TickClock(),
            capture_factory=RemovalFailure, event_extractor=fake_events,
            expected_router_layers=None,
        )

    class LowMemory:
        def process_memory(self):
            return ProcessMemory(1, 1, 1, 2 * 1024**3 - 1)

        def process_times(self):
            return SimpleNamespace(user_ns=0, system_ns=0)

    with pytest.raises(MemoryError, match="2 GiB"):
        _run_with_memory(
            collector([99]), prompt(), REFERENCE_MODE, LowMemory(),
            clock_ns=TickClock(),
        )

    def repository_hook(*args):
        return None

    repository_hook._moe_cache_lab_router_hook = True
    router = SimpleNamespace(_forward_hooks={7: repository_hook})
    carryover_model = SimpleNamespace(named_modules=lambda: (
        ("model.layers.0.block_sparse_moe.router", router),
    ))
    carryover = SimpleNamespace(
        tokenizer=FakeTokenizer(), model=carryover_model, experts_per_token=1,
    )
    with pytest.raises(RuntimeError, match="carryover"):
        run_prompt(
            carryover, prompt(), REFERENCE_MODE, clock_ns=TickClock(),
            expected_router_layers=(0,),
        )


def test_cross_child_probability_tolerance_and_exact_stability_boundaries():
    profiled = _attach_memory(run_prompt(
        collector(list(range(10, 27))), prompt(), PROFILED_MODE,
        clock_ns=TickClock(), capture_factory=FakeCapture,
        event_extractor=fake_events, expected_router_layers=None,
    ))
    original = _prompt_run_record(profiled)
    within = _prompt_run_record(profiled)
    within["events"][0]["selected_probabilities"][0] -= 0.5e-6
    children = [
        {"calibration": [original], "evaluation": []},
        {"calibration": [within], "evaluation": []},
        {"calibration": [original], "evaluation": []},
        {"calibration": [original], "evaluation": []},
    ]
    _validate_cross_child_semantics(children)
    over = _prompt_run_record(profiled)
    over["events"][0]["selected_probabilities"][0] -= 1.1e-6
    children[1] = {"calibration": [over], "evaluation": []}
    with pytest.raises(ValueError, match="probability"):
        _validate_cross_child_semantics(children)

    exact = aggregate_children([
        child(1, 1_000_000, 1_000_000), child(2, 1_000_000, 1_000_000),
        child(3, 1_000_000, 1_000_000), child(4, 1_000_000, 1_150_000),
    ])
    assert exact["decision"] == "valid_stable"
    above = aggregate_children([
        child(1, 1_000_000, 1_000_000), child(2, 1_000_000, 1_000_000),
        child(3, 1_000_000, 1_000_000), child(4, 1_000_000, 1_150_001),
    ])
    assert above["decision"] == "valid_inconclusive"
    assert render_markdown(exact) == render_markdown(exact)
    assert exact["child_delta_stats"]["n"] == 4
    assert all(row["prompt_ratio_stats"]["n"] == 8 for row in exact["children"])
    deviation = 0.1 * (3 ** 0.5) / 2
    cv_boundary_values = (1 - deviation, 1 - deviation, 1 + deviation, 1 + deviation)
    cv_over_values = (1 - deviation * 1.000001, 1 - deviation * 1.000001, 1 + deviation * 1.000001, 1 + deviation * 1.000001)
    # The same frozen <=10% formula is applied independently to sum(R),
    # sum(P), and the paired child ratio.
    for gate in ("cv_sum_reference", "cv_sum_profiled", "cv_paired_ratio"):
        boundary = v04._cv(cv_boundary_values)
        over = v04._cv(cv_over_values)
        assert abs(boundary - 0.10) < 1e-12, gate
        assert boundary <= 0.10 + 1e-12 and over > 0.10, gate


def test_full_fake_valid_parent_orchestration(tmp_path: Path):
    launches = []
    probe_instances = []

    class ParentProbe(FakeProbe):
        def __init__(self):
            super().__init__([sample()] * 16)
            probe_instances.append(self)

        def close(self):
            return None

        def audit_record(self):
            return {"open_query_status": 0, "add_cpu_status": 0, "add_disk_status": 0, "close_query_status": 0, "constructor_error": None}

    def run_process(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        number = int(command[command.index("--child-number") + 1])
        process_uuid = command[command.index("--process-uuid") + 1]
        output.mkdir()
        now = datetime.now(timezone.utc).isoformat()
        value = {
            "format": "moe-cache-lab.v04-child", "format_version": 1,
            "state": "valid", "child_number": number,
            "process_uuid": process_uuid, "started_at": now, "finished_at": now,
            "evaluation": child(number)["evaluation"],
        }
        write_json_atomic(output / "v04-child.json", value)
        launches.append((number, process_uuid, kwargs))
        return ChildProcessOutcome(
            1000 + number, 0, f"child {number}", "", False, now, now,
        )

    tick = iter(float(value) for value in range(100))
    with mock.patch("moe_cache_lab.v04.validate_trusted_stage1_paths"), mock.patch(
        "moe_cache_lab.v04._validate_child_record"
    ), mock.patch("moe_cache_lab.v04.validate_v04", side_effect=lambda path: Path(path).resolve()):
        manifest = collect_v04(
            Path("benchmarks/corpus-v1.json"), tmp_path,
            run_process=run_process, probe_factory=ParentProbe,
            sleep=lambda _: None, monotonic=lambda: next(tick),
        )
    record = load_v04_json(manifest)
    assert record["state"] == "valid_stable"
    assert [item[0] for item in launches] == [1, 2, 3, 4]
    assert len({item[1] for item in launches}) == 4
    assert all("timeout" in item[2] and "env" in item[2] for item in launches)
    assert len(probe_instances) == 4 and all(instance.calls == 17 for instance in probe_instances)
    assert record["experiment_started_at"] is not None and record["elapsed_seconds"] > 0


def _strict_run(item, mode, prompt_id, before: datetime, total_ns: int):
    prompt_ids = {
        event.token_position: event.token_id
        for event in item.trace.events if event.phase == "prompt"
    }
    eos = item.record["eos"]
    routed = tuple(eos["routed_non_eos_token_ids"])
    semantic = v04.SemanticRecord(
        input_ids=tuple(prompt_ids[position] for position in sorted(prompt_ids)),
        attention_mask=tuple(1 for _ in prompt_ids),
        candidate_token_ids=routed + (0,),
        candidate_texts=tuple(str(token) for token in routed) + ("0",),
        routed_non_eos_token_ids=routed,
        routed_non_eos_text=eos["emitted_non_eos_text"],
        terminal_eos_token_id=None, terminal_eos_text=None,
        terminal_eos_candidate_position=None,
        actual_decode_input_steps=16, eos_emitted=False,
        horizon_exhausted=True,
        validation_only_horizon_candidate_id=0,
        validation_only_horizon_candidate_text="0",
    )
    extraction = tuple([1] * 17) if mode == PROFILED_MODE else ()
    setup = removal = lifecycle = 1 if mode == PROFILED_MODE else 0
    accounted = setup + 20 + 16 + sum(extraction) + removal + lifecycle
    timing = v04.NestedTiming(
        total_ns, setup, 20, tuple([1] * 16), extraction, removal,
        lifecycle, accounted, total_ns - accounted,
    )
    events = tuple(item.trace.events) if mode == PROFILED_MODE else ()
    run = v04.PromptRun(prompt_id, mode, semantic, timing, events)
    memory_before = ProcessMemory(100, 100, 100, 9_000_000_000, timestamp=before.isoformat())
    memory_after = ProcessMemory(100, 100, 100, 9_000_000_000, timestamp=(before + timedelta(milliseconds=500)).isoformat())
    object.__setattr__(run, "_memory_before", memory_before)
    object.__setattr__(run, "_memory_after", memory_after)
    object.__setattr__(run, "_process_user_ns", 1)
    object.__setattr__(run, "_process_system_ns", 1)
    return run


def _strict_child(number: int, started: datetime):
    trusted = v04.load_stage1_repetition_manifest(v04.TRUSTED_STAGE1_REP1)
    items = {item.prompt.id: item for item in trusted.prompts}
    load_start = started + timedelta(milliseconds=100)
    load_before = ProcessMemory(1, 1, 1, 9_000_000_000, timestamp=(load_start + timedelta(milliseconds=10)).isoformat())
    load_after = ProcessMemory(2, 2, 2, 9_000_000_000, timestamp=(load_start + timedelta(milliseconds=20)).isoformat())
    model_load = {
        "elapsed_ns": 10, "started_at": load_start.isoformat(),
        "finished_at": (load_start + timedelta(milliseconds=30)).isoformat(),
        "process_cpu_user_ns": 1, "process_cpu_system_ns": 1,
        "memory_before": v04.asdict(load_before), "memory_after": v04.asdict(load_after),
        "parameter_count": v04.EXPECTED_PARAMETER_COUNT, "device": "cpu",
        "dtype": "float32", "nonmeta": True, "eval_mode": True,
    }
    cursor = load_start + timedelta(seconds=1)
    calibration = []
    for prompt_id, mode in calibration_schedule(number):
        calibration.append(v04._prompt_run_record(_strict_run(items[prompt_id], mode, prompt_id, cursor, 200 if mode == PROFILED_MODE else 100)))
        cursor += timedelta(seconds=1)
    evaluation = []
    for prompt_id, modes in evaluation_schedule(number):
        records = []
        runs = {}
        for mode in modes:
            run = _strict_run(items[prompt_id], mode, prompt_id, cursor, 200 if mode == PROFILED_MODE else 100)
            runs[mode] = run
            records.append(v04._prompt_run_record(run))
            cursor += timedelta(seconds=1)
        evaluation.append({
            "prompt_id": prompt_id, "mode_order": list(modes),
            "reference_total_ns": 100, "profiled_total_ns": 200,
            "reference_working_set_delta": 0, "profiled_working_set_delta": 0,
            "reference_private_delta": 0, "profiled_private_delta": 0,
            "runs": records,
        })
    finished = cursor + timedelta(seconds=1)
    return {
        "format": v04.V04_CHILD_FORMAT, "format_version": 1, "state": "valid",
        "child_number": number, "process_uuid": str(uuid4()), "pid": 2000 + number,
        "parent_pid": 1000, "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "model_id": v04.STAGE1_MODEL_ID, "model_revision": v04.STAGE1_MODEL_REVISION,
        "corpus_sha256": v04.STAGE1_CORPUS_SHA256, "model_load": model_load,
        "environment": {
            "python": "3.10", "torch": "2", "transformers": "5", "moe_cache_lab": "0.3",
            "device": "cpu", "dtype": "float32", "torch_threads": 4,
            "local_files_only": True, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "os_name": "Windows",
            "os_release": "11", "os_version": "10.0", "windows_build": "26200",
            "logical_cpu_count": 12, "torch_interop_threads": 1, "process_priority_class": 32,
        },
        "calibration_schedule": [list(item) for item in calibration_schedule(number)],
        "evaluation_schedule": [[prompt_id, list(modes)] for prompt_id, modes in evaluation_schedule(number)],
        "calibration": calibration, "evaluation": evaluation,
    }


def test_strict_valid_fixture_validates_in_staging_then_final(tmp_path: Path):
    attempt_uuid = str(uuid4())
    final = tmp_path / f"v0.4-cpu-profiler-attempt-{attempt_uuid}"
    staging = tmp_path / f".v0.4-cpu-profiler-attempt-{attempt_uuid}.staging"
    staging.mkdir()
    attempt_start = datetime.now(timezone.utc) - timedelta(minutes=10)
    admission_refs = []
    child_refs = []
    children = []
    cursor = attempt_start + timedelta(seconds=1)
    experiment_start = None
    for number in range(1, 5):
        admission = _complete_admission(cursor)
        admission_path = write_json_atomic(staging / f"admission-{number}.json", _admission_record(number, admission))
        admission_refs.append({"child_number": number, "path": admission_path.name, "sha256": v04.file_sha256(admission_path), "size": admission_path.stat().st_size})
        child_start = cursor + timedelta(seconds=17)
        if experiment_start is None:
            experiment_start = cursor + timedelta(seconds=16, milliseconds=500)
        child = _strict_child(number, child_start)
        child_dir = staging / f"child-{number}"
        child_dir.mkdir()
        child_path = write_json_atomic(child_dir / "v04-child.json", child)
        child_refs.append(v04._child_reference(number, child["process_uuid"], Path(f"child-{number}"), child_path, child, "", "", 3000 + number))
        children.append(child)
        cursor = datetime.fromisoformat(child["finished_at"]) + timedelta(seconds=1)
    aggregate_input = [{
        "child_number": child["child_number"],
        "evaluation": [{name: row[name] for name in (
            "prompt_id", "mode_order", "reference_total_ns", "profiled_total_ns",
            "reference_working_set_delta", "profiled_working_set_delta",
            "reference_private_delta", "profiled_private_delta",
        )} for row in child["evaluation"]],
    } for child in children]
    result = aggregate_children(aggregate_input)
    result_path = write_json_atomic(staging / "v04-result.json", result)
    markdown_path = staging / "v04-report.md"
    markdown_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    finished = cursor + timedelta(seconds=1)
    manifest = {
        "format": v04.V04_SET_FORMAT, "format_version": 1, "attempt_uuid": attempt_uuid,
        "state": "valid_stable", "rerun_index": 0, "rerun_of": None,
        "rerun_source_sha256": None, "rerun_source_path": None, "retry_of": None,
        "retry_source_sha256": None, "retry_source_path": None, "retry_source_state": None,
        "retry_source_reason": None, "retry_source_root_digest": None,
        "attempt_directory": final.name, "spec_sha256": v04.V04_SPEC_SHA256,
        "started_at": attempt_start.isoformat(), "experiment_started_at": experiment_start.isoformat(),
        "finished_at": finished.isoformat(), "deadline_seconds": v04.V04_DEADLINE_SECONDS,
        "elapsed_seconds": (finished - experiment_start).total_seconds(),
        "corpus_sha256": v04.STAGE1_CORPUS_SHA256,
        "trusted_stage1_set_sha256": v04.TRUSTED_STAGE1_SET_SHA256,
        "trusted_stage1_repetition_sha256": v04.TRUSTED_STAGE1_REP1_SHA256,
        "admissions": admission_refs, "children": child_refs, "failures": [],
        "result": {"path": result_path.name, "sha256": v04.file_sha256(result_path), "size": result_path.stat().st_size},
        "markdown": {"path": markdown_path.name, "sha256": v04.file_sha256(markdown_path), "size": markdown_path.stat().st_size},
    }
    manifest_path = write_json_atomic(staging / "v04-set.json", manifest)
    validate_v04(manifest_path)
    staging.replace(final)
    validate_v04(final / "v04-set.json")


class TestV04(unittest.TestCase):
    def test_counterbalance(self):
        test_exact_counterbalance_and_warmup_order()

    def test_driver_horizon(self):
        test_driver_horizon_records_validation_candidate_and_nested_timers()

    def test_driver_eos(self):
        test_driver_y0_eos_and_semantic_mismatch()

    def test_admission(self):
        test_admission_exact_baseline_plus_fifteen_and_paths()

    def test_admission_invalid(self):
        test_admission_invalid_sample_is_deferred_not_replaced()

    def test_aggregate(self):
        test_hierarchical_aggregation_stable_and_exact_boundaries()

    def test_aggregate_invalid(self):
        test_aggregation_rejects_order_and_marks_variability_inconclusive()

    def test_states(self):
        test_attempt_state_transitions_are_explicit()

    def test_json(self):
        with tempfile.TemporaryDirectory() as directory:
            test_json_is_deterministic_and_rejects_nonfinite(Path(directory))

    def test_capture_cleanup(self):
        test_router_capture_constructor_removes_partial_handles()

    def test_pdh_prime(self):
        test_pdh_prime_is_collect_only_and_abi_is_aligned()

    def test_timer_and_cross_child(self):
        test_timer_reconciliation_and_cross_child_horizon_semantics()

    def test_nonvalid_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            test_nonvalid_attempt_is_validated_before_atomic_publication(Path(directory))

    def test_retry_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            test_first_prerequisite_defer_is_outside_timer_and_second_is_blocked(Path(directory))

    def test_parent_terminal_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            test_parent_records_zero_exit_artifact_failures_and_timeout_pid(Path(directory))

    def test_popen_runner(self):
        test_popen_runner_preserves_pid_and_timeout_evidence()

    def test_launched_failure_pid_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            test_launched_failure_categories_reject_null_pid_with_recomputed_reference(Path(directory))

    def test_recursive_lineage_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            test_recursive_lineage_and_invalid_partial_dependencies(Path(directory))

    def test_admission_boundary_and_timeout_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            test_admission_threshold_equality_cadence_and_timeout_failure_artifact(Path(directory))

    def test_hook_failure_and_low_memory(self):
        test_hook_forward_failure_removal_error_and_low_memory_abort()

    def test_probability_and_boundaries(self):
        test_cross_child_probability_tolerance_and_exact_stability_boundaries()

    def test_fake_valid_orchestration(self):
        with tempfile.TemporaryDirectory() as directory:
            test_full_fake_valid_parent_orchestration(Path(directory))

    def test_strict_valid_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            test_strict_valid_fixture_validates_in_staging_then_final(Path(directory))
