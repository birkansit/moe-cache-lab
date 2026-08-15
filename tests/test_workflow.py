import json
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

import torch
import transformers

from moe_cache_lab import __version__
from moe_cache_lab.cache import simulate
from moe_cache_lab.cli import main
from moe_cache_lab.report import prefill_grouping_metrics, render_suite_report
from moe_cache_lab.trace import RoutingEvent, RoutingTrace, write_trace
from moe_cache_lab.workflow import (
    CORPUS_FORMAT,
    CORPUS_VERSION,
    DEFAULT_CAPACITIES,
    MANIFEST_FORMAT,
    MANIFEST_VERSION,
    benchmark_suite,
    canonical_json_bytes,
    collect_corpus,
    file_sha256,
    load_corpus,
    load_suite_inputs,
    sha256_bytes,
    validate_capacities,
)


SOURCE_CORPUS = Path(__file__).parents[1] / "benchmarks" / "corpus-v1.json"


def _make_suite(directory: Path) -> Path:
    corpus_data = json.loads(SOURCE_CORPUS.read_text(encoding="utf-8"))
    corpus_path = directory / "corpus.json"
    corpus_path.write_text(json.dumps(corpus_data), encoding="utf-8")
    traces = directory / "traces"
    traces.mkdir()
    records = []
    for order, prompt in enumerate(corpus_data["prompts"]):
        if prompt["split"] == "calibration":
            prompt_expert, generated_expert = 0, 0
        else:
            prompt_expert = 0 if prompt["id"] == "eval-factual-01" else 1
            generated_expert = 1
        trace = RoutingTrace(
            "example/moe",
            4,
            1,
            (
                RoutingEvent("prompt", 0, 0, (prompt_expert,)),
                RoutingEvent("generated", 1, 0, (generated_expert,)),
            ),
            source_text=prompt["text"],
            transformers_version=transformers.__version__,
            model_revision="resolved-test-revision",
        )
        relative = Path("traces") / f"{order:02d}-{prompt['id']}.jsonl"
        trace_path = write_trace(directory / relative, trace)
        records.append({
            "id": prompt["id"],
            "category": prompt["category"],
            "split": prompt["split"],
            "order": order,
            "prompt_sha256": sha256_bytes(prompt["text"].encode("utf-8")),
            "trace_path": relative.as_posix(),
            "trace_sha256": file_sha256(trace_path),
            "routing_event_count": 2,
            "expert_request_count": 2,
        })
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "created_at": "2026-01-01T00:00:00+00:00",
        "corpus": {
            "path": "corpus.json",
            "version": CORPUS_VERSION,
            "canonical_json_sha256": sha256_bytes(canonical_json_bytes(corpus_data)),
        },
        "model_id": "example/moe",
        "model_revision": {
            "requested": "requested-test-revision",
            "resolved": "resolved-test-revision",
        },
        "routing": {
            "experts_per_layer": 4,
            "selected_experts_per_token": 1,
            "routed_layer_ids": [0],
            "routed_layer_count": 1,
            "capture_method": "model.forward(output_router_logits=True)",
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
            "max_new_tokens": 1,
            "use_cache": True,
        },
        "prompts": records,
    }
    manifest_path = directory / "manifest-v1.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


class WorkflowTests(unittest.TestCase):
    def test_tracked_corpus_has_deterministic_versioned_split_and_order(self) -> None:
        raw = json.loads(SOURCE_CORPUS.read_text(encoding="utf-8"))
        self.assertEqual(raw["format"], CORPUS_FORMAT)
        first = load_corpus(SOURCE_CORPUS)
        second = load_corpus(SOURCE_CORPUS)
        self.assertEqual((first.version, first.sha256), (CORPUS_VERSION, second.sha256))
        self.assertEqual(len(first.prompts), 12)
        self.assertEqual([item.split for item in first.prompts].count("calibration"), 4)
        self.assertEqual([item.split for item in first.prompts].count("evaluation"), 8)
        self.assertEqual([item.order for item in first.prompts], list(range(12)))

    def test_collection_rejects_path_like_prompt_id_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = json.loads(SOURCE_CORPUS.read_text(encoding="utf-8"))
            data["prompts"][0]["id"] = "../escape"
            corpus = directory / "unsafe.json"
            corpus.write_text(json.dumps(data), encoding="utf-8")
            output = directory / "output"
            with self.assertRaisesRegex(ValueError, "safe lowercase filename slugs"):
                collect_corpus(corpus, output)
            self.assertFalse(output.exists())
            self.assertFalse((directory / "escape.jsonl").exists())

    def test_calibration_is_disjoint_and_cannot_see_evaluation_future(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = benchmark_suite(_make_suite(Path(temporary)), [1])
        calibrated = next(item for item in benchmark.simulations if item.policy == "calibrated_static_frequency")
        oracle = next(item for item in benchmark.simulations if item.policy == "offline_oracle_frequency")
        self.assertEqual(calibrated.fixed_entries, frozenset({(0, 0)}))
        self.assertEqual(oracle.fixed_entries, frozenset({(0, 1)}))
        self.assertEqual(calibrated.combined.samples, 16)  # Evaluation only, not 8 calibration events.
        self.assertEqual((calibrated.prompt.samples, calibrated.generated.samples), (8, 8))
        lru = next(item for item in benchmark.simulations if item.policy == "lru")
        self.assertEqual((lru.combined.hits, lru.combined.misses), (14, 2))
        self.assertEqual(lru.prompt.hits, 7)  # Cache residency crosses prompt boundaries.

    def test_calibrated_static_uses_hard_capacity_restoration_accounting(self) -> None:
        calibration = (RoutingEvent("prompt", 0, 0, (0,)),)
        evaluation = (RoutingEvent("prompt", 0, 0, (1,)),)
        result = simulate(
            evaluation,
            1,
            "calibrated_static_frequency",
            calibration_events=calibration,
        )
        self.assertEqual(result.fixed_entries, frozenset({(0, 0)}))
        self.assertEqual(
            (result.combined.prewarm_loads, result.combined.demand_loads, result.combined.evictions),
            (1, 2, 2),
        )

    def test_capacity_validation_and_complete_curves(self) -> None:
        self.assertEqual(DEFAULT_CAPACITIES, (8, 16, 32, 64, 128, 256))
        self.assertEqual(validate_capacities([1, 2, 4]), (1, 2, 4))
        for invalid in ([], [0], [2, 1], [1, 1], [True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_capacities(invalid)
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = benchmark_suite(_make_suite(Path(temporary)), [1, 2])
        self.assertEqual(len(benchmark.simulations), 8)
        self.assertEqual(
            [(item.capacity, item.policy) for item in benchmark.simulations],
            [(capacity, policy) for capacity in (1, 2) for policy in (
                "lru", "lfu", "calibrated_static_frequency", "offline_oracle_frequency"
            )],
        )

    def test_suite_rejects_capacity_below_evaluation_atomic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = _make_suite(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            corpus_data = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
            for record, prompt in zip(manifest["prompts"], corpus_data["prompts"]):
                trace_path = directory / record["trace_path"]
                write_trace(
                    trace_path,
                    RoutingTrace(
                        "example/moe",
                        4,
                        2,
                        (
                            RoutingEvent("prompt", 0, 0, (0, 1)),
                            RoutingEvent("generated", 1, 0, (0, 1)),
                        ),
                        source_text=prompt["text"],
                        transformers_version=transformers.__version__,
                        model_revision="resolved-test-revision",
                    ),
                )
                record["trace_sha256"] = file_sha256(trace_path)
                record["expert_request_count"] = 4
            manifest["routing"]["selected_experts_per_token"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "largest evaluation event bundle 2"):
                benchmark_suite(manifest_path, [1])

    def test_manifest_detects_trace_and_corpus_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = _make_suite(directory)
            load_suite_inputs(manifest_path)
            trace_path = directory / "traces" / "00-cal-factual-01.jsonl"
            trace_path.write_bytes(trace_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "trace hash mismatch"):
                load_suite_inputs(manifest_path)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = _make_suite(directory)
            corpus_path = directory / "corpus.json"
            data = json.loads(corpus_path.read_text(encoding="utf-8"))
            data["prompts"][0]["text"] += " tampered"
            corpus_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corpus hash mismatch"):
                load_suite_inputs(manifest_path)

    def test_manifest_rejects_prompt_order_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = _make_suite(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["prompts"][0], manifest["prompts"][1] = (
                manifest["prompts"][1], manifest["prompts"][0]
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata/order"):
                load_suite_inputs(manifest_path)

    def test_manifest_rejects_trace_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = _make_suite(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model_revision"]["resolved"] = "different-commit"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model revision mismatch"):
                load_suite_inputs(manifest_path)

    def test_structural_validation_rejects_missing_layer_with_valid_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = _make_suite(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            corpus = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
            manifest["routing"]["routed_layer_ids"] = [0, 1]
            manifest["routing"]["routed_layer_count"] = 2
            for index, (record, prompt) in enumerate(zip(manifest["prompts"], corpus["prompts"])):
                events = [
                    RoutingEvent("prompt", 0, 0, (0,)),
                    RoutingEvent("prompt", 0, 1, (0,)),
                    RoutingEvent("generated", 1, 0, (1,)),
                ]
                if index != 0:
                    events.append(RoutingEvent("generated", 1, 1, (1,)))
                trace_path = directory / record["trace_path"]
                write_trace(trace_path, RoutingTrace(
                    "example/moe", 4, 1, tuple(events), source_text=prompt["text"],
                    transformers_version=transformers.__version__,
                    model_revision="resolved-test-revision",
                ))
                record["trace_sha256"] = file_sha256(trace_path)
                record["routing_event_count"] = len(events)
                record["expert_request_count"] = len(events)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete routed-layer set"):
                load_suite_inputs(manifest_path)

    def test_structural_validation_rejects_generated_boundary_gap_with_valid_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = _make_suite(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = manifest["prompts"][0]
            prompt = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))["prompts"][0]
            trace_path = directory / record["trace_path"]
            write_trace(trace_path, RoutingTrace(
                "example/moe",
                4,
                1,
                (
                    RoutingEvent("prompt", 0, 0, (0,)),
                    RoutingEvent("generated", 2, 0, (1,)),
                ),
                source_text=prompt["text"], transformers_version=transformers.__version__,
                model_revision="resolved-test-revision",
            ))
            record["trace_sha256"] = file_sha256(trace_path)
            record["routing_event_count"] = 2
            record["expert_request_count"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immediate decode boundary"):
                load_suite_inputs(manifest_path)

    def test_structural_validation_rejects_missing_prompt_position_with_valid_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = _make_suite(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = manifest["prompts"][0]
            prompt = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))["prompts"][0]
            trace_path = directory / record["trace_path"]
            write_trace(trace_path, RoutingTrace(
                "example/moe",
                4,
                1,
                (
                    RoutingEvent("prompt", 0, 0, (0,)),
                    RoutingEvent("prompt", 2, 0, (0,)),
                    RoutingEvent("generated", 3, 0, (1,)),
                ),
                source_text=prompt["text"],
                transformers_version=transformers.__version__,
                model_revision="resolved-test-revision",
            ))
            record["trace_sha256"] = file_sha256(trace_path)
            record["routing_event_count"] = 3
            record["expert_request_count"] = 3
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt token positions must be contiguous"):
                load_suite_inputs(manifest_path)

    def test_structural_validation_rejects_layer_set_inconsistency_with_valid_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = _make_suite(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = manifest["prompts"][0]
            prompt = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))["prompts"][0]
            trace_path = directory / record["trace_path"]
            write_trace(trace_path, RoutingTrace(
                "example/moe",
                4,
                1,
                (
                    RoutingEvent("prompt", 0, 1, (0,)),
                    RoutingEvent("generated", 1, 1, (1,)),
                ),
                source_text=prompt["text"],
                transformers_version=transformers.__version__,
                model_revision="resolved-test-revision",
            ))
            record["trace_sha256"] = file_sha256(trace_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "routed-layer set"):
                load_suite_inputs(manifest_path)

    def test_suite_report_has_boundaries_reproducibility_and_no_semantic_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = benchmark_suite(_make_suite(Path(temporary)), [1, 2])
            report = render_suite_report(benchmark)
        for expected in (
            "MEASURED", "SIMULATED", "ESTIMATED", "Reproducibility",
            "canonical JSON SHA-256", "Manifest", "max_new_tokens",
            "Calibration prompts: 4", "Evaluation prompts: 8",
            "do not identify expert meaning", "calibrated_static_frequency",
            "resolved model/tokenizer commit", "Calibration measured routing samples/requests",
            "unique active `(prompt, layer, expert)`", "groups assignments from all prompt tokens",
            "do not demonstrate cache reuse", "no validated runtime scheduling interpretation",
        ):
            self.assertIn(expected, report)

    def test_prefill_grouping_math_respects_independent_prompt_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = load_suite_inputs(_make_suite(Path(temporary)))
            metrics = prefill_grouping_metrics(inputs.evaluation)
        self.assertEqual(metrics, (8, 8, 1, 1))

    def test_collection_constructs_one_reusable_collector_for_all_prompts(self) -> None:
        class FakeCollector:
            instances = 0
            calls = 0

            def __init__(self, model_id: str, revision: str | None = None) -> None:
                self.model_id = model_id
                self.resolved_revision = "fake-resolved-revision"
                FakeCollector.instances += 1

            def collect(self, prompt: str, max_new_tokens: int) -> RoutingTrace:
                FakeCollector.calls += 1
                events = [RoutingEvent("prompt", 0, 0, (0,))]
                events.extend(
                    RoutingEvent("generated", offset + 1, 0, (0,))
                    for offset in range(max_new_tokens)
                )
                return RoutingTrace(
                    self.model_id, 4, 1, tuple(events),
                    source_text=prompt, transformers_version=transformers.__version__,
                    model_revision=self.resolved_revision,
                )

        with tempfile.TemporaryDirectory() as temporary, patch(
            "moe_cache_lab.workflow.GraniteTraceCollector", FakeCollector
        ):
            manifest = collect_corpus(SOURCE_CORPUS, Path(temporary) / "output", max_new_tokens=2)
            loaded = load_suite_inputs(manifest)
        self.assertEqual((FakeCollector.instances, FakeCollector.calls), (1, 12))
        self.assertEqual(len(loaded.evaluation), 8)
        self.assertEqual(loaded.manifest["corpus"]["path"], "corpus.json")
        self.assertEqual(loaded.manifest["generation"]["max_new_tokens"], 2)
        self.assertEqual(
            loaded.manifest["model_revision"],
            {"requested": None, "resolved": "fake-resolved-revision"},
        )
        self.assertEqual(
            [record["order"] for record in loaded.manifest["prompts"]],
            list(range(12)),
        )

    def test_cli_runs_synthetic_manifest_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = _make_suite(directory)
            output = directory / "report.md"
            with patch("sys.argv", [
                "moe-cache-lab", "benchmark-suite", str(manifest),
                "--capacities", "1", "2", "--output", str(output),
            ]):
                main()
            self.assertIn("SIMULATED", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
