from dataclasses import replace
from pathlib import Path
import unittest

from moe_cache_lab.cache import simulate, simulate_bundles
from moe_cache_lab.trace import RoutingEvent, RoutingTrace
from moe_cache_lab.views import (
    PREFILL_LAYER_UNION_ATOMIC,
    TOKEN_LAYER_ATOMIC,
    PromptTraceSource,
    SimulationBundle,
    adapt_prompt_traces,
    summarize_bundles,
)
from moe_cache_lab.workflow import load_suite_inputs


TRACKED_MANIFEST = Path("results/v0.2-corpus-v1/manifest-v1.json")


def _source(prompt_id: str, order: int, events: tuple[RoutingEvent, ...]) -> PromptTraceSource:
    return PromptTraceSource(
        prompt_id,
        order,
        RoutingTrace("example/moe", 4, len(events[0].selected_experts), events),
    )


def _bundle(
    view: str,
    phase: str,
    *,
    prompt_id: str = "prompt",
    prompt_order: int = 0,
    layer: int = 0,
    token_position: int | None = 0,
    experts: tuple[int, ...] = (0,),
    source_positions: tuple[int, ...] | None = None,
    source_event_count: int = 1,
    source_assignment_count: int | None = None,
) -> SimulationBundle:
    if source_positions is None:
        source_positions = (token_position,) if token_position is not None else (0,)
    return SimulationBundle(
        phase=phase,
        required_keys=frozenset((layer, expert) for expert in experts),
        view_name=view,
        prompt_id=prompt_id,
        prompt_order=prompt_order,
        layer=layer,
        token_position=token_position,
        source_event_count=source_event_count,
        source_assignment_count=(
            len(experts) if source_assignment_count is None else source_assignment_count
        ),
        source_token_positions=source_positions,
    )


class SimulationViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_suite_inputs(TRACKED_MANIFEST)

    def test_tracked_v02_grouped_counts_and_provenance(self) -> None:
        evaluation = tuple(
            PromptTraceSource(item.prompt.id, item.prompt.order, item.trace)
            for item in self.inputs.evaluation
        )
        calibration = tuple(
            PromptTraceSource(item.prompt.id, item.prompt.order, item.trace)
            for item in self.inputs.calibration
        )
        evaluation_summary = summarize_bundles(adapt_prompt_traces(
            evaluation, PREFILL_LAYER_UNION_ATOMIC
        ))
        self.assertEqual(
            (
                evaluation_summary.combined.bundles,
                evaluation_summary.combined.requests,
                evaluation_summary.combined.source_events,
                evaluation_summary.combined.source_assignments,
                evaluation_summary.max_bundle_size,
            ),
            (576, 8433, 3144, 25152, 32),
        )
        self.assertEqual(
            (
                evaluation_summary.prompt.bundles,
                evaluation_summary.prompt.requests,
                evaluation_summary.prompt.source_events,
                evaluation_summary.prompt.source_assignments,
            ),
            (192, 5361, 2760, 22080),
        )
        self.assertEqual(
            (
                evaluation_summary.generated.bundles,
                evaluation_summary.generated.requests,
                evaluation_summary.generated.source_events,
                evaluation_summary.generated.source_assignments,
            ),
            (384, 3072, 384, 3072),
        )

        calibration_summary = summarize_bundles(adapt_prompt_traces(
            calibration, PREFILL_LAYER_UNION_ATOMIC
        ))
        self.assertEqual(
            (calibration_summary.combined.bundles, calibration_summary.combined.requests),
            (288, 4114),
        )
        self.assertEqual(
            (calibration_summary.prompt.bundles, calibration_summary.prompt.requests),
            (96, 2578),
        )
        self.assertEqual(
            (calibration_summary.generated.bundles, calibration_summary.generated.requests),
            (192, 1536),
        )

    def test_union_identity_chronology_provenance_and_raw_immutability(self) -> None:
        first = PromptTraceSource(
            "first",
            0,
            RoutingTrace(
                "example/moe",
                4,
                2,
                (
                    RoutingEvent("prompt", 0, 0, (0, 1)),
                    RoutingEvent("prompt", 1, 0, (1, 2)),
                    RoutingEvent("prompt", 0, 1, (0, 1)),
                    RoutingEvent("prompt", 1, 1, (1, 2)),
                    RoutingEvent("generated", 2, 0, (2, 3)),
                    RoutingEvent("generated", 2, 1, (0, 3)),
                ),
            ),
        )
        second = PromptTraceSource(
            "second",
            1,
            RoutingTrace(
                "example/moe",
                4,
                2,
                (
                    RoutingEvent("prompt", 0, 0, (2, 3)),
                    RoutingEvent("prompt", 0, 1, (0, 3)),
                    RoutingEvent("generated", 1, 0, (0, 2)),
                    RoutingEvent("generated", 1, 1, (1, 3)),
                ),
            ),
        )
        original_events = (first.trace.events, second.trace.events)
        token = adapt_prompt_traces((first, second), TOKEN_LAYER_ATOMIC)
        grouped = adapt_prompt_traces((first, second), PREFILL_LAYER_UNION_ATOMIC)

        self.assertEqual((first.trace.events, second.trace.events), original_events)
        self.assertEqual(
            [(bundle.prompt_id, bundle.phase, bundle.layer, bundle.token_position) for bundle in grouped],
            [
                ("first", "prompt", 0, None),
                ("first", "prompt", 1, None),
                ("first", "generated", 0, 2),
                ("first", "generated", 1, 2),
                ("second", "prompt", 0, None),
                ("second", "prompt", 1, None),
                ("second", "generated", 0, 1),
                ("second", "generated", 1, 1),
            ],
        )
        self.assertEqual(grouped[0].required_keys, frozenset({(0, 0), (0, 1), (0, 2)}))
        self.assertEqual(grouped[1].required_keys, frozenset({(1, 0), (1, 1), (1, 2)}))
        self.assertEqual(
            (grouped[0].source_event_count, grouped[0].source_assignment_count,
             grouped[0].source_token_positions),
            (2, 4, (0, 1)),
        )
        first_token_generated = [bundle for bundle in token if bundle.prompt_id == "first" and bundle.phase == "generated"]
        first_grouped_generated = [bundle for bundle in grouped if bundle.prompt_id == "first" and bundle.phase == "generated"]
        self.assertEqual(
            first_token_generated,
            [replace(bundle, view_name=TOKEN_LAYER_ATOMIC) for bundle in first_grouped_generated],
        )

    def test_summary_feasibility_and_raw_simulation_reject_small_capacity(self) -> None:
        sources = tuple(
            PromptTraceSource(item.prompt.id, item.prompt.order, item.trace)
            for item in self.inputs.evaluation
        )
        bundles = adapt_prompt_traces(sources, PREFILL_LAYER_UNION_ATOMIC)
        summary = summarize_bundles(bundles)
        self.assertEqual(summary.max_bundle_size, 32)
        for capacity in (8, 16):
            with self.subTest(capacity=capacity):
                self.assertFalse(summary.is_feasible(capacity))
                with self.assertRaisesRegex(
                    ValueError,
                    f"largest prefill_layer_union_atomic bundle 32",
                ):
                    simulate_bundles(bundles, capacity, "lru")
        self.assertTrue(summary.is_feasible(32))
        self.assertEqual(simulate_bundles(bundles, 32, "lru").combined.samples, 576)

    def test_bundle_replay_is_order_invariant_and_phase_attributed(self) -> None:
        source_left = _source("left", 0, (
            RoutingEvent("prompt", 0, 0, (0,)),
            RoutingEvent("prompt", 1, 0, (2,)),
            RoutingEvent("generated", 2, 0, (2,)),
        ))
        source_right = _source("right", 0, (
            RoutingEvent("prompt", 0, 0, (0,)),
            RoutingEvent("prompt", 1, 0, (2,)),
            RoutingEvent("generated", 2, 0, (2,)),
        ))
        left = list(adapt_prompt_traces((source_left,), TOKEN_LAYER_ATOMIC))
        right = list(adapt_prompt_traces((source_right,), TOKEN_LAYER_ATOMIC))
        left[1] = replace(left[1], required_keys=frozenset({(0, 1), (0, 2)}), source_assignment_count=2)
        right[1] = replace(right[1], required_keys=frozenset({(0, 2), (0, 1)}), source_assignment_count=2)
        left_result = simulate_bundles(left, 2, "lru")
        right_result = simulate_bundles(right, 2, "lru")
        self.assertEqual(left_result.combined, right_result.combined)
        self.assertEqual(
            (left_result.prompt.samples, left_result.prompt.requests,
             left_result.generated.samples, left_result.generated.hits),
            (2, 3, 1, 1),
        )

    def test_event_api_remains_exactly_compatible_via_token_view(self) -> None:
        calibration = _source("cal", 0, (
            RoutingEvent("prompt", 0, 0, (0,)),
            RoutingEvent("prompt", 1, 0, (0,)),
        ))
        evaluation = _source("eval", 1, (
            RoutingEvent("prompt", 0, 0, (0,)),
            RoutingEvent("generated", 1, 0, (1,)),
        ))
        evaluation_bundles = adapt_prompt_traces((evaluation,), TOKEN_LAYER_ATOMIC)
        calibration_bundles = adapt_prompt_traces((calibration,), TOKEN_LAYER_ATOMIC)
        for policy in ("lru", "lfu", "offline_oracle_frequency", "calibrated_static_frequency"):
            with self.subTest(policy=policy):
                event_result = simulate(
                    evaluation.trace.events,
                    1,
                    policy,
                    calibration_events=(
                        calibration.trace.events
                        if policy == "calibrated_static_frequency"
                        else None
                    ),
                )
                bundle_result = simulate_bundles(
                    evaluation_bundles,
                    1,
                    policy,
                    calibration_bundles=(
                        calibration_bundles
                        if policy == "calibrated_static_frequency"
                        else None
                    ),
                )
                self.assertEqual(event_result, bundle_result)

    def test_fixed_targets_are_view_specific_without_evaluation_leakage(self) -> None:
        calibration = _source("cal", 0, (
            RoutingEvent("prompt", 0, 0, (0,)),
            RoutingEvent("prompt", 0, 1, (1,)),
            RoutingEvent("prompt", 1, 1, (1,)),
        ))
        evaluation = _source("eval", 1, (
            RoutingEvent("prompt", 0, 0, (2,)),
            RoutingEvent("prompt", 0, 1, (3,)),
            RoutingEvent("prompt", 1, 1, (3,)),
        ))
        targets: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
        for view in (TOKEN_LAYER_ATOMIC, PREFILL_LAYER_UNION_ATOMIC):
            calibration_bundles = adapt_prompt_traces((calibration,), view)
            evaluation_bundles = adapt_prompt_traces((evaluation,), view)
            targets[(view, "calibrated")] = simulate_bundles(
                evaluation_bundles,
                1,
                "calibrated_static_frequency",
                calibration_bundles=calibration_bundles,
            ).fixed_entries
            targets[(view, "oracle")] = simulate_bundles(
                evaluation_bundles, 1, "offline_oracle_frequency"
            ).fixed_entries
        self.assertEqual(targets[(TOKEN_LAYER_ATOMIC, "calibrated")], frozenset({(1, 1)}))
        self.assertEqual(targets[(PREFILL_LAYER_UNION_ATOMIC, "calibrated")], frozenset({(0, 0)}))
        self.assertEqual(targets[(TOKEN_LAYER_ATOMIC, "oracle")], frozenset({(1, 3)}))
        self.assertEqual(targets[(PREFILL_LAYER_UNION_ATOMIC, "oracle")], frozenset({(0, 2)}))
        self.assertTrue(
            targets[(TOKEN_LAYER_ATOMIC, "calibrated")].isdisjoint({(0, 2), (1, 3)})
        )
        self.assertTrue(
            targets[(PREFILL_LAYER_UNION_ATOMIC, "calibrated")].isdisjoint({(0, 2), (1, 3)})
        )
        with self.assertRaisesRegex(ValueError, "same simulation view"):
            simulate_bundles(
                adapt_prompt_traces((evaluation,), PREFILL_LAYER_UNION_ATOMIC),
                1,
                "calibrated_static_frequency",
                calibration_bundles=adapt_prompt_traces(
                    (calibration,), TOKEN_LAYER_ATOMIC
                ),
            )

    def test_grouped_calibrated_static_preserves_hard_capacity_accounting(self) -> None:
        calibration = _source("cal", 0, (
            RoutingEvent("prompt", 0, 0, (0, 1)),
        ))
        evaluation = _source("eval", 1, (
            RoutingEvent("prompt", 0, 0, (2, 3)),
        ))
        result = simulate_bundles(
            adapt_prompt_traces((evaluation,), PREFILL_LAYER_UNION_ATOMIC),
            2,
            "calibrated_static_frequency",
            calibration_bundles=adapt_prompt_traces(
                (calibration,), PREFILL_LAYER_UNION_ATOMIC
            ),
        )
        self.assertEqual(result.fixed_entries, frozenset({(0, 0), (0, 1)}))
        self.assertEqual(
            (result.combined.prewarm_loads, result.combined.demand_loads,
             result.combined.evictions, result.prompt.requests),
            (2, 4, 4, 2),
        )

    def test_calibrated_static_rejects_prompt_provenance_overlap(self) -> None:
        source = _source("shared", 0, (RoutingEvent("prompt", 0, 0, (0,)),))
        bundles = adapt_prompt_traces((source,), TOKEN_LAYER_ATOMIC)
        with self.assertRaisesRegex(ValueError, "prompt provenance must be disjoint"):
            simulate_bundles(
                bundles,
                1,
                "calibrated_static_frequency",
                calibration_bundles=bundles,
            )

        legacy = simulate(
            (RoutingEvent("prompt", 0, 0, (1,)),),
            1,
            "calibrated_static_frequency",
            calibration_events=(RoutingEvent("prompt", 0, 0, (0,)),),
        )
        self.assertEqual(legacy.fixed_entries, frozenset({(0, 0)}))

    def test_empty_evaluation_infers_grouped_calibration_view_and_charges_prewarm(self) -> None:
        calibration = _source(
            "calibration",
            0,
            (RoutingEvent("prompt", 0, 0, (0, 1)),),
        )
        grouped_calibration = adapt_prompt_traces(
            (calibration,), PREFILL_LAYER_UNION_ATOMIC
        )
        result = simulate_bundles(
            (),
            2,
            "calibrated_static_frequency",
            calibration_bundles=grouped_calibration,
        )
        self.assertEqual(result.fixed_entries, frozenset({(0, 0), (0, 1)}))
        self.assertEqual(
            (result.combined.samples, result.combined.requests,
             result.combined.prewarm_loads, result.combined.estimated_expert_transfers),
            (0, 0, 2, 2),
        )

        evaluation = _source(
            "evaluation",
            1,
            (RoutingEvent("prompt", 0, 0, (2, 3)),),
        )
        with self.assertRaisesRegex(ValueError, "same simulation view"):
            simulate_bundles(
                adapt_prompt_traces((evaluation,), TOKEN_LAYER_ATOMIC),
                2,
                "calibrated_static_frequency",
                calibration_bundles=grouped_calibration,
            )

        for policy in ("lru", "lfu", "offline_oracle_frequency"):
            with self.subTest(policy=policy):
                empty = simulate_bundles((), 1, policy)
                self.assertEqual((empty.combined.samples, empty.combined.requests), (0, 0))

    def test_public_bundle_invariants_reject_forged_event_provenance(self) -> None:
        invalid_cases = (
            {
                "view": TOKEN_LAYER_ATOMIC,
                "phase": "prompt",
                "source_event_count": 2,
                "source_positions": (0, 1),
            },
            {
                "view": TOKEN_LAYER_ATOMIC,
                "phase": "prompt",
                "source_assignment_count": 2,
            },
            {
                "view": TOKEN_LAYER_ATOMIC,
                "phase": "prompt",
                "source_positions": (1,),
            },
            {
                "view": PREFILL_LAYER_UNION_ATOMIC,
                "phase": "prompt",
                "token_position": 0,
            },
            {
                "view": PREFILL_LAYER_UNION_ATOMIC,
                "phase": "prompt",
                "token_position": None,
                "source_assignment_count": 2,
            },
            {
                "view": PREFILL_LAYER_UNION_ATOMIC,
                "phase": "prompt",
                "token_position": None,
                "source_event_count": 2,
                "source_positions": (1, 0),
                "source_assignment_count": 2,
            },
            {
                "view": PREFILL_LAYER_UNION_ATOMIC,
                "phase": "generated",
                "token_position": 2,
                "source_event_count": 2,
                "source_positions": (2, 3),
                "source_assignment_count": 2,
            },
        )
        for case in invalid_cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                _bundle(**case)

    def test_token_sequence_rejects_wrong_phase_and_collector_nesting(self) -> None:
        prompt_00 = _bundle(TOKEN_LAYER_ATOMIC, "prompt", layer=0, token_position=0)
        prompt_01 = _bundle(TOKEN_LAYER_ATOMIC, "prompt", layer=0, token_position=1)
        prompt_10 = _bundle(TOKEN_LAYER_ATOMIC, "prompt", layer=1, token_position=0)
        generated_20 = _bundle(TOKEN_LAYER_ATOMIC, "generated", layer=0, token_position=2)
        generated_21 = _bundle(TOKEN_LAYER_ATOMIC, "generated", layer=1, token_position=2)
        generated_30 = _bundle(TOKEN_LAYER_ATOMIC, "generated", layer=0, token_position=3)
        invalid_sequences = (
            ((prompt_00, generated_20, prompt_01), "must precede"),
            ((prompt_00, prompt_10, prompt_01), "layer-major"),
            ((prompt_00, generated_20, generated_30, generated_21), "token-major"),
        )
        for sequence, message in invalid_sequences:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                summarize_bundles(sequence)

    def test_grouped_sequence_rejects_layer_decode_and_duplicate_group_order(self) -> None:
        prompt_0 = _bundle(
            PREFILL_LAYER_UNION_ATOMIC,
            "prompt",
            layer=0,
            token_position=None,
            source_positions=(0, 1),
            source_event_count=2,
            source_assignment_count=2,
        )
        prompt_1 = replace(prompt_0, layer=1, required_keys=frozenset({(1, 0)}))
        generated_20 = _bundle(
            PREFILL_LAYER_UNION_ATOMIC, "generated", layer=0, token_position=2
        )
        generated_21 = _bundle(
            PREFILL_LAYER_UNION_ATOMIC, "generated", layer=1, token_position=2
        )
        generated_30 = _bundle(
            PREFILL_LAYER_UNION_ATOMIC, "generated", layer=0, token_position=3
        )
        invalid_sequences = (
            ((prompt_1, prompt_0), "ascending layer"),
            ((prompt_0, prompt_0), "one prompt bundle"),
            ((prompt_0, generated_20, generated_30, generated_21), "token-major"),
        )
        for sequence, message in invalid_sequences:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                summarize_bundles(sequence)

    def test_sequence_rejects_noncontiguous_duplicate_or_descending_prompts(self) -> None:
        first = _bundle(TOKEN_LAYER_ATOMIC, "prompt", prompt_id="first", prompt_order=0)
        second = _bundle(TOKEN_LAYER_ATOMIC, "prompt", prompt_id="second", prompt_order=1)
        duplicate_order = replace(second, prompt_order=0)
        invalid_sequences = (
            ((first, second, first), "contiguous"),
            ((first, duplicate_order), "unique"),
            ((second, first), "ascending by prompt_order"),
        )
        for sequence, message in invalid_sequences:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                summarize_bundles(sequence)

    def test_sequence_rejects_physical_duplicates_collisions_and_position_rollback(self) -> None:
        prompt_0 = _bundle(TOKEN_LAYER_ATOMIC, "prompt", layer=0, token_position=0)
        prompt_2 = _bundle(TOKEN_LAYER_ATOMIC, "prompt", layer=0, token_position=2)
        generated_collision = _bundle(
            TOKEN_LAYER_ATOMIC, "generated", layer=0, token_position=0
        )
        generated_rollback = _bundle(
            TOKEN_LAYER_ATOMIC, "generated", layer=1, token_position=1
        )
        generated_3 = _bundle(TOKEN_LAYER_ATOMIC, "generated", layer=0, token_position=3)
        invalid_sequences = (
            ((prompt_0, prompt_0), "duplicate physical"),
            ((generated_3, generated_3), "duplicate physical"),
            ((prompt_0, generated_collision), "collide"),
            ((prompt_2, generated_rollback), "must precede"),
        )
        for sequence, message in invalid_sequences:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                summarize_bundles(sequence)

    def test_sequence_validation_accepts_sparse_frozen_view_chronology(self) -> None:
        token = (
            _bundle(TOKEN_LAYER_ATOMIC, "prompt", layer=0, token_position=1),
            _bundle(TOKEN_LAYER_ATOMIC, "prompt", layer=0, token_position=7),
            _bundle(TOKEN_LAYER_ATOMIC, "prompt", layer=3, token_position=2),
            _bundle(TOKEN_LAYER_ATOMIC, "generated", layer=0, token_position=10),
            _bundle(TOKEN_LAYER_ATOMIC, "generated", layer=3, token_position=10),
        )
        grouped = (
            _bundle(
                PREFILL_LAYER_UNION_ATOMIC,
                "prompt",
                layer=0,
                token_position=None,
                source_positions=(1, 7),
                source_event_count=2,
                source_assignment_count=2,
            ),
            _bundle(
                PREFILL_LAYER_UNION_ATOMIC,
                "prompt",
                layer=3,
                token_position=None,
                source_positions=(2,),
            ),
            _bundle(
                PREFILL_LAYER_UNION_ATOMIC, "generated", layer=0, token_position=10
            ),
            _bundle(
                PREFILL_LAYER_UNION_ATOMIC, "generated", layer=3, token_position=10
            ),
        )
        self.assertEqual(summarize_bundles(token).combined.bundles, 5)
        self.assertEqual(summarize_bundles(grouped).combined.bundles, 4)


if __name__ == "__main__":
    unittest.main()
