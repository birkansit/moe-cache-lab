import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.cache import FixedTargetPlan, build_fixed_target_plan, simulate_bundles
from moe_cache_lab.cli import main
from moe_cache_lab.stage1 import (
    Stage1PromptInput,
    Stage1RepetitionInputs,
    Stage1SetInputs,
)
from moe_cache_lab.stage1_benchmark import (
    STAGE1_CAPACITIES,
    _target_overlap_records,
    benchmark_stage1,
    evaluate_continuation_rule,
    render_stage1_markdown,
    sample_statistics,
    write_stage1_json,
    write_stage1_markdown,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.views import (
    PREFILL_LAYER_UNION_ATOMIC,
    TOKEN_LAYER_ATOMIC,
    PromptTraceSource,
    adapt_prompt_traces,
)
from moe_cache_lab.workflow import CorpusDefinition, CorpusPrompt


def _trace(prompt_id: str, decode_steps: int) -> RoutingTrace:
    events = []
    for layer in (0, 1):
        for position in range(4):
            start = position * 8
            events.append(RoutingEvent(
                "prompt", position, layer, tuple(range(start, start + 8)),
                token_id=position + 1,
                selected_probabilities=(0.125,) * 8,
            ))
    for offset in range(decode_steps):
        position = 4 + offset
        for layer in (0, 1):
            events.append(RoutingEvent(
                "generated", position, layer, tuple(range(8)),
                token_id=50 + offset,
                selected_probabilities=(0.125,) * 8,
            ))
    return RoutingTrace(
        "ibm-granite/granite-3.1-1b-a400m-instruct",
        32,
        8,
        tuple(events),
        source_text=prompt_id,
        model_revision="0da7a48b0276d500ce5922fd2b33944091fc6c09",
    )


def _stage1_inputs(*, decode_steps: int = 2) -> Stage1SetInputs:
    prompts = tuple(
        CorpusPrompt(
            id=f"{'cal' if order < 4 else 'eval'}-{order}",
            category="synthetic",
            split="calibration" if order < 4 else "evaluation",
            text=f"prompt-{order}",
            order=order,
        )
        for order in range(12)
    )
    corpus = CorpusDefinition(Path("corpus.json"), "1.0", "c" * 64, prompts)
    repetitions = []
    for number in range(1, 4):
        loaded = []
        for prompt in prompts:
            trace = _trace(prompt.id, decode_steps)
            eos = {
                "max_decode_input_steps_requested": 16,
                "actual_decode_input_steps_routed": decode_steps,
                "routed_non_eos_token_ids": [50 + offset for offset in range(decode_steps)],
                "emitted_non_eos_token_ids": [50 + offset for offset in range(decode_steps)],
                "emitted_non_eos_text": "",
                "normalized_eos_token_ids": [99],
                "terminal_eos_token_id": 99,
                "terminal_eos_candidate_position": decode_steps,
                "terminal_eos_text": "",
                "eos_emitted": True,
                "horizon_exhausted": False,
            }
            loaded.append(Stage1PromptInput(prompt, {"eos": eos}, trace))
        repetitions.append(Stage1RepetitionInputs(
            Path(f"rep-{number}.json"),
            str(number) * 64,
            {
                "repetition_number": number,
                "environment": {
                    "python": "test", "transformers": "test", "torch": "test",
                    "device": "cpu", "dtype": "float32", "torch_num_threads": 4,
                },
                "generation": {
                    "strategy": "greedy", "token_selection": "argmax",
                    "do_sample": False, "use_cache": True,
                    "max_decode_input_steps": 16,
                },
            },
            corpus,
            tuple(loaded),
        ))
    return Stage1SetInputs(
        Path("stage1-set-manifest.json"),
        "s" * 64,
        {
            "model_id": "ibm-granite/granite-3.1-1b-a400m-instruct",
            "model_revision": "0da7a48b0276d500ce5922fd2b33944091fc6c09",
            "max_decode_input_steps": 16,
            "torch_num_threads": 4,
        },
        tuple(repetitions),
    )


def _run(*, decode_steps: int = 2):
    inputs = _stage1_inputs(decode_steps=decode_steps)
    with patch(
        "moe_cache_lab.stage1_benchmark.validate_stage1_set", return_value=inputs
    ) as validate:
        result = benchmark_stage1("approved-set.json", "trusted.json")
    validate.assert_called_once_with("approved-set.json", "trusted.json")
    return result


def _aggregate(data, **identity):
    rows = [
        row for row in data["aggregate_rows"]
        if all(row[key] == value for key, value in identity.items())
    ]
    if len(rows) != 1:
        raise AssertionError(f"expected one row for {identity}, got {len(rows)}")
    return rows[0]


class FixedTargetPlanTests(unittest.TestCase):
    def test_plan_is_reused_unchanged_across_controls_and_checks_boundaries(self) -> None:
        inputs = _stage1_inputs()
        prompts = inputs.repetitions[0].prompts
        calibration_prompts = prompts[:4]
        evaluation_prompts = prompts[4:]
        calibration = adapt_prompt_traces(tuple(
            PromptTraceSource(item.prompt.id, item.prompt.order, item.trace)
            for item in calibration_prompts
        ), TOKEN_LAYER_ATOMIC)
        evaluation = adapt_prompt_traces(tuple(
            PromptTraceSource(item.prompt.id, item.prompt.order, item.trace)
            for item in evaluation_prompts
        ), TOKEN_LAYER_ATOMIC)
        grouped_calibration = adapt_prompt_traces(tuple(
            PromptTraceSource(item.prompt.id, item.prompt.order, item.trace)
            for item in calibration_prompts
        ), PREFILL_LAYER_UNION_ATOMIC)
        grouped_evaluation = adapt_prompt_traces(tuple(
            PromptTraceSource(item.prompt.id, item.prompt.order, item.trace)
            for item in evaluation_prompts
        ), PREFILL_LAYER_UNION_ATOMIC)
        first_prompt = tuple(
            bundle for bundle in evaluation
            if bundle.prompt_id == evaluation_prompts[0].prompt.id
        )
        decode = tuple(bundle for bundle in first_prompt if bundle.phase == "generated")
        calibrated = build_fixed_target_plan(
            calibration, evaluation, 16, "calibrated_static_frequency"
        )
        oracle = build_fixed_target_plan(
            evaluation, calibration, 16, "offline_oracle_frequency"
        )
        for plan, policy in (
            (calibrated, "calibrated_static_frequency"),
            (oracle, "offline_oracle_frequency"),
        ):
            whole = simulate_bundles(evaluation, 16, policy, fixed_target_plan=plan)
            prompt = simulate_bundles(first_prompt, 16, policy, fixed_target_plan=plan)
            decode_only = simulate_bundles(decode, 16, policy, fixed_target_plan=plan)
            self.assertEqual(whole.fixed_entries, plan.target_entries)
            self.assertEqual(prompt.fixed_entries, plan.target_entries)
            self.assertEqual(decode_only.fixed_entries, plan.target_entries)
            empty = simulate_bundles((), 16, policy, fixed_target_plan=plan)
            self.assertEqual(empty.combined.requests, 0)
            self.assertEqual(empty.combined.prewarm_loads, len(plan.target_entries))
        with self.assertRaisesRegex(ValueError, "policy/capacity"):
            simulate_bundles(evaluation, 32, "offline_oracle_frequency", fixed_target_plan=oracle)
        with self.assertRaisesRegex(ValueError, "dynamic"):
            simulate_bundles(evaluation, 16, "lru", fixed_target_plan=oracle)
        wrong_view = build_fixed_target_plan(
            grouped_evaluation,
            grouped_calibration,
            32,
            "offline_oracle_frequency",
        )
        with self.assertRaisesRegex(ValueError, "same view"):
            simulate_bundles(evaluation, 32, "offline_oracle_frequency", fixed_target_plan=wrong_view)
        incomplete_oracle = build_fixed_target_plan(
            first_prompt, calibration, 16, "offline_oracle_frequency"
        )
        with self.assertRaisesRegex(ValueError, "exact contiguous subsequence"):
            simulate_bundles(evaluation, 16, "offline_oracle_frequency", fixed_target_plan=incomplete_oracle)
        rogue = adapt_prompt_traces(
            (PromptTraceSource("rogue", 50, _trace("rogue", 0)),),
            TOKEN_LAYER_ATOMIC,
        )
        with self.assertRaisesRegex(ValueError, "exact contiguous subsequence"):
            simulate_bundles(
                rogue, 16, "calibrated_static_frequency",
                fixed_target_plan=calibrated,
            )
        self.assertEqual(
            simulate_bundles(
                first_prompt, 16, "offline_oracle_frequency",
                oracle_bundles=first_prompt,
            ).combined.samples,
            len(first_prompt),
        )
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            simulate_bundles(
                first_prompt, 16, "offline_oracle_frequency",
                oracle_bundles=evaluation,
            )
        first = first_prompt[0]
        shifted_keys = set(first.required_keys)
        shifted_keys.remove(min(shifted_keys))
        shifted_keys.add((first.layer, 99))
        shifted_prompt = (
            replace(first, required_keys=frozenset(shifted_keys)),
            *first_prompt[1:],
        )
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            simulate_bundles(
                shifted_prompt, 16, "offline_oracle_frequency",
                oracle_bundles=first_prompt,
            )
        for plan, policy in (
            (calibrated, "calibrated_static_frequency"),
            (oracle, "offline_oracle_frequency"),
        ):
            with self.subTest(shifted_policy=policy), self.assertRaisesRegex(
                ValueError, "exact contiguous subsequence"
            ):
                simulate_bundles(
                    shifted_prompt, 16, policy, fixed_target_plan=plan
                )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_fixed_target_plan(
                (*calibration, *evaluation), calibration, 16,
                "offline_oracle_frequency",
            )
        with self.assertRaises(TypeError):
            FixedTargetPlan(
                "offline_oracle_frequency", 16, evaluation, calibration,
                target_entries=frozenset({(99, 99)}),
            )

        grouped_oracle = build_fixed_target_plan(
            grouped_evaluation, grouped_calibration, 32,
            "offline_oracle_frequency",
        )
        grouped_first = tuple(
            bundle for bundle in grouped_evaluation
            if bundle.prompt_id == evaluation_prompts[0].prompt.id
        )
        altered_provenance = (
            replace(
                grouped_first[0],
                source_assignment_count=grouped_first[0].source_assignment_count + 1,
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact contiguous subsequence"):
            simulate_bundles(
                altered_provenance, 32, "offline_oracle_frequency",
                fixed_target_plan=grouped_oracle,
            )
        with self.assertRaises(ValueError):
            simulate_bundles(
                (grouped_first[1], grouped_first[0]),
                32,
                "offline_oracle_frequency",
                fixed_target_plan=grouped_oracle,
            )

    def test_plan_rejects_pair_id_or_order_reuse_across_splits(self) -> None:
        trace = _trace("x", 0)
        selected = adapt_prompt_traces(
            (PromptTraceSource("selected", 10, trace),), TOKEN_LAYER_ATOMIC
        )
        opposites = (
            adapt_prompt_traces(
                (PromptTraceSource("selected", 10, trace),), TOKEN_LAYER_ATOMIC
            ),
            adapt_prompt_traces(
                (PromptTraceSource("selected", 11, trace),), TOKEN_LAYER_ATOMIC
            ),
            adapt_prompt_traces(
                (PromptTraceSource("other", 10, trace),), TOKEN_LAYER_ATOMIC
            ),
        )
        for opposite in opposites:
            with self.subTest(opposite=opposite), self.assertRaisesRegex(ValueError, "disjoint"):
                build_fixed_target_plan(
                    selected, opposite, 16, "offline_oracle_frequency"
                )


class Stage1BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = _run()
        cls.data = cls.benchmark.data

    def test_exact_capacities_views_controls_and_global_na(self) -> None:
        self.assertEqual(tuple(self.data["capacities"]), STAGE1_CAPACITIES)
        grouped_8_decode = _aggregate(
            self.data,
            view=PREFILL_LAYER_UNION_ATOMIC,
            capacity=8,
            policy="lfu",
            control="decode_only_cold",
            scope="combined",
        )
        self.assertFalse(grouped_8_decode["feasible"])
        self.assertIn("maximum", grouped_8_decode["n_a_reason"])
        grouped_8_fixed = _aggregate(
            self.data,
            view=PREFILL_LAYER_UNION_ATOMIC,
            capacity=8,
            policy="calibrated_static_frequency",
            control="suite_persistent",
            scope="combined",
        )
        self.assertEqual(grouped_8_fixed["fixed_target_size"], 8)
        self.assertTrue(any(
            plan["view"] == PREFILL_LAYER_UNION_ATOMIC
            and plan["capacity"] == 8
            and plan["policy"] == "calibrated_static_frequency"
            for plan in self.data["fixed_target_plans"]
        ))
        grouped_na_prompts = [
            row for row in self.data["prompt_rows"]
            if row["view"] == PREFILL_LAYER_UNION_ATOMIC
            and row["capacity"] == 8 and row["policy"] == "lfu"
            and row["control"] == "cold_per_prompt"
            and row["scope"] == "combined"
        ]
        self.assertEqual(len(grouped_na_prompts), 8)
        self.assertTrue(all(not row["feasible"] for row in grouped_na_prompts))
        token_8 = _aggregate(
            self.data,
            view=TOKEN_LAYER_ATOMIC,
            capacity=8,
            policy="lfu",
            control="suite_persistent",
            scope="combined",
        )
        self.assertTrue(token_8["feasible"])

    def test_persistent_and_reset_controls_prewarm_and_phase_accounting(self) -> None:
        primary = _aggregate(
            self.data, view=TOKEN_LAYER_ATOMIC, capacity=8,
            policy="calibrated_static_frequency", control="suite_persistent",
            scope="combined",
        )
        cold = _aggregate(
            self.data, view=TOKEN_LAYER_ATOMIC, capacity=8,
            policy="calibrated_static_frequency", control="cold_per_prompt",
            scope="combined",
        )
        self.assertEqual(primary["prewarm_loads"], 8)
        self.assertEqual(cold["prewarm_loads"], 64)
        lru_primary = _aggregate(
            self.data, view=TOKEN_LAYER_ATOMIC, capacity=32, policy="lru",
            control="suite_persistent", scope="combined",
        )
        lru_cold = _aggregate(
            self.data, view=TOKEN_LAYER_ATOMIC, capacity=32, policy="lru",
            control="cold_per_prompt", scope="combined",
        )
        self.assertGreater(lru_primary["hits"], lru_cold["hits"])
        prompt = _aggregate(
            self.data, view=TOKEN_LAYER_ATOMIC, capacity=32, policy="lru",
            control="suite_persistent", scope="prompt/prefill",
        )
        generated = _aggregate(
            self.data, view=TOKEN_LAYER_ATOMIC, capacity=32, policy="lru",
            control="suite_persistent", scope="generated/decode",
        )
        self.assertEqual(lru_primary["requests"], prompt["requests"] + generated["requests"])

    def test_micro_counts_are_not_tripled_and_decode_excludes_prompt(self) -> None:
        primary = _aggregate(
            self.data, view=TOKEN_LAYER_ATOMIC, capacity=32, policy="lfu",
            control="suite_persistent", scope="combined",
        )
        self.assertEqual(primary["requests"], 8 * 12 * 8)
        self.assertEqual(primary["published_repetition_count"], 1)
        self.assertTrue(self.data["repeatability"]["samples_not_tripled"])
        decode = _aggregate(
            self.data, view=TOKEN_LAYER_ATOMIC, capacity=32, policy="lfu",
            control="decode_only_cold", scope="combined",
        )
        self.assertEqual(decode["requests"], 8 * 4 * 8)
        self.assertEqual(decode["source_events"], 8 * 4)

    def test_macros_use_eight_prompts_sample_sd_and_zero_variance(self) -> None:
        stats = sample_statistics(range(8))
        self.assertAlmostEqual(stats["mean"], 3.5)
        self.assertAlmostEqual(stats["sample_standard_deviation"], 2.449489742783178)
        zero = sample_statistics([4] * 8)
        self.assertEqual(zero["sample_standard_deviation"], 0.0)
        macro = next(
            item for item in self.data["cold_per_prompt_macros"]
            if item["view"] == TOKEN_LAYER_ATOMIC
            and item["capacity"] == 8
            and item["policy"] == "calibrated_static_frequency"
            and item["scope"] == "combined"
            and item["metric"] == "prewarm_loads"
        )
        self.assertEqual(macro["statistics"]["n"], 8)
        self.assertEqual(macro["statistics"]["sample_standard_deviation"], 0.0)
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            sample_statistics([1] * 7)

    def test_target_overlap_and_missing_pair(self) -> None:
        self.assertTrue(self.data["fixed_target_overlaps"])
        self.assertTrue(all(row["capacity"] >= 32 for row in self.data["fixed_target_overlaps"]))
        with self.assertRaisesRegex(ValueError, "matched pair"):
            _target_overlap_records({}, {
                TOKEN_LAYER_ATOMIC: (
                    adapt_prompt_traces((PromptTraceSource("a", 0, _trace("a", 0)),), TOKEN_LAYER_ATOMIC),
                    adapt_prompt_traces((PromptTraceSource("b", 1, _trace("b", 0)),), TOKEN_LAYER_ATOMIC),
                ),
                PREFILL_LAYER_UNION_ATOMIC: (
                    adapt_prompt_traces((PromptTraceSource("a", 0, _trace("a", 0)),), PREFILL_LAYER_UNION_ATOMIC),
                    adapt_prompt_traces((PromptTraceSource("b", 1, _trace("b", 0)),), PREFILL_LAYER_UNION_ATOMIC),
                ),
            })

    def test_report_boundaries_oracle_label_reproducibility_and_json(self) -> None:
        report = render_stage1_markdown(self.data)
        for expected in (
            "MEASURED", "SIMULATED", "ESTIMATED", "explicitly non-causal",
            "sample standard deviation", "not tripled samples",
            "not verified Granite runtime residency", "never offloading",
            "| N/A reason |",
            "capacity 8 is below maximum prefill_layer_union_atomic bundle 32",
        ):
            self.assertIn(expected, report)
        with tempfile.TemporaryDirectory() as temporary:
            json_path = write_stage1_json(Path(temporary) / "result.json", self.benchmark)
            markdown_path = write_stage1_markdown(Path(temporary) / "result.md", self.benchmark)
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["format"], "moe-cache-lab.stage1-cache-benchmark")
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), report)
            self.assertNotIn("NaN", json_path.read_text(encoding="utf-8"))

    def test_validation_boundary_prevents_tampered_input_simulation(self) -> None:
        with patch(
            "moe_cache_lab.stage1_benchmark.validate_stage1_set",
            side_effect=ValueError("tampered manifest"),
        ):
            with self.assertRaisesRegex(ValueError, "tampered"):
                benchmark_stage1("bad.json")

    def test_cli_writes_both_structured_and_markdown_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            json_path = Path(temporary) / "stage1.json"
            markdown_path = Path(temporary) / "stage1.md"
            with patch(
                "moe_cache_lab.stage1_benchmark.benchmark_stage1",
                return_value=self.benchmark,
            ), patch("sys.argv", [
                "moe-cache-lab", "benchmark-stage1", "approved.json",
                "--json-output", str(json_path), "--output", str(markdown_path),
            ]):
                main()
            self.assertTrue(json_path.exists())
            self.assertIn("SIMULATED", markdown_path.read_text(encoding="utf-8"))

    def test_all_eos_decode_is_empty_charged_and_fails_stability(self) -> None:
        data = _run(decode_steps=0).data
        decode = _aggregate(
            data, view=TOKEN_LAYER_ATOMIC, capacity=32,
            policy="calibrated_static_frequency", control="decode_only_cold",
            scope="combined",
        )
        self.assertEqual(decode["requests"], 0)
        self.assertEqual(decode["prewarm_loads"], 8 * 32)
        self.assertFalse(data["continuation_rule"]["passed"])
        self.assertTrue(all(
            not prompt["passed"]
            for repetition in data["continuation_rule"]["repetitions"]
            for prompt in repetition["decode_prompt_conditions"]
        ))


class ContinuationRuleTests(unittest.TestCase):
    def test_exact_per_repetition_thresholds_and_no_substitution(self) -> None:
        rows = []
        for number in range(1, 4):
            rows.append({
                "repetition_number": number,
                "lfu_grouped_primary_cap32_transfers": 100,
                "lfu_grouped_primary_cap256_transfers": 90 if number < 3 else 91,
                "lfu_grouped_decode_only_cap256_prompts": [
                    {"prompt_id": f"p{index}", "hits": 1, "requests": 10}
                    for index in range(8)
                ],
            })
        expected = tuple(f"p{index}" for index in range(8))
        result = evaluate_continuation_rule(
            rows,
            repeatability_passed=True,
            expected_evaluation_prompt_ids=expected,
        )
        self.assertTrue(result["repetitions"][0]["passed"])
        self.assertFalse(result["repetitions"][2]["transfer_condition_passed"])
        self.assertFalse(result["passed"])
        rows[2]["lfu_grouped_primary_cap256_transfers"] = 90
        rows[1]["lfu_grouped_decode_only_cap256_prompts"][0] = {
            "prompt_id": "p0", "hits": 0, "requests": 0,
        }
        result = evaluate_continuation_rule(
            rows,
            repeatability_passed=True,
            expected_evaluation_prompt_ids=expected,
        )
        self.assertFalse(result["repetitions"][1]["decode_stability_passed"])
        self.assertFalse(result["passed"])

    def test_prompt_identity_rejects_duplicate_omission_and_order_changes(self) -> None:
        expected = tuple(f"p{index}" for index in range(8))
        base_prompts = [
            {"prompt_id": prompt_id, "hits": 1, "requests": 10}
            for prompt_id in expected
        ]
        for label, changed in (
            ("duplicate", [base_prompts[0], *base_prompts[:-1]]),
            ("omission", base_prompts[:-1]),
            ("order", [base_prompts[1], base_prompts[0], *base_prompts[2:]]),
        ):
            rows = [{
                "repetition_number": number,
                "lfu_grouped_primary_cap32_transfers": 100,
                "lfu_grouped_primary_cap256_transfers": 90,
                "lfu_grouped_decode_only_cap256_prompts": (
                    changed if number == 1 else list(base_prompts)
                ),
            } for number in range(1, 4)]
            with self.subTest(label=label), self.assertRaises(ValueError):
                evaluate_continuation_rule(
                    rows,
                    repeatability_passed=True,
                    expected_evaluation_prompt_ids=expected,
                )
        with self.assertRaisesRegex(ValueError, "unique nonempty"):
            evaluate_continuation_rule(
                [], repeatability_passed=True,
                expected_evaluation_prompt_ids=("p",) * 8,
            )


if __name__ == "__main__":
    unittest.main()
