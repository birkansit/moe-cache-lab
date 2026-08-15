import unittest

from moe_cache_lab.trace import RoutingEvent, RoutingTrace, read_trace, write_trace


class TraceTests(unittest.TestCase):
    def test_trace_round_trip(self) -> None:
        import tempfile
        trace = RoutingTrace(
            model_id="example/moe", num_experts=8, experts_per_token=2,
            events=(RoutingEvent("prompt", 0, 1, (3, 5), 42, (0.7, 0.2)),), source_text="hello",
            model_revision="immutable-test-commit",
        )
        with tempfile.TemporaryDirectory() as directory:
            loaded = read_trace(write_trace(f"{directory}/trace.jsonl", trace))
        self.assertEqual(loaded.model_id, trace.model_id)
        self.assertEqual(loaded.events, trace.events)
        self.assertEqual(loaded.model_revision, "immutable-test-commit")
        self.assertEqual(loaded.expert_requests, ((1, 3), (1, 5)))

    def test_rejects_duplicate_negative_and_out_of_range_experts(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            RoutingEvent("prompt", 0, 0, (1, 1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            RoutingEvent("prompt", 0, 0, (-1,))
        with self.assertRaisesRegex(ValueError, "outside num_experts"):
            RoutingTrace("example/moe", 2, 1, (RoutingEvent("prompt", 0, 0, (2,)),))

    def test_rejects_selection_count_inconsistent_with_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match experts_per_token"):
            RoutingTrace("example/moe", 4, 2, (RoutingEvent("prompt", 0, 0, (1,)),))

    def test_rejects_probability_count_and_range_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            RoutingEvent("prompt", 0, 0, (1, 2), selected_probabilities=(1.0,))
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            RoutingEvent("prompt", 0, 0, (1,), selected_probabilities=(float("nan"),))

    def test_rejects_boolean_or_invalid_numeric_event_fields(self) -> None:
        invalid_events = (
            lambda: RoutingEvent("prompt", True, 0, (1,)),
            lambda: RoutingEvent("prompt", 0, True, (1,)),
            lambda: RoutingEvent("prompt", -1, 0, (1,)),
            lambda: RoutingEvent("prompt", 0, -1, (1,)),
            lambda: RoutingEvent("prompt", 0, 0, (1,), token_id=True),
            lambda: RoutingEvent("prompt", 0, 0, (1,), token_id=-1),
            lambda: RoutingEvent("prompt", 0, 0, (1,), selected_probabilities=(True,)),
            lambda: RoutingEvent("prompt", 0, 0, (1,), selected_probabilities=(float("inf"),)),
        )
        for create_event in invalid_events:
            with self.subTest(create_event=create_event), self.assertRaises(ValueError):
                create_event()

    def test_rejects_duplicate_events_and_phase_regression(self) -> None:
        duplicate = RoutingEvent("prompt", 0, 0, (1,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            RoutingTrace("example/moe", 4, 1, (duplicate, duplicate))
        with self.assertRaisesRegex(ValueError, "cannot follow"):
            RoutingTrace(
                "example/moe",
                4,
                1,
                (RoutingEvent("generated", 1, 0, (1,)), RoutingEvent("prompt", 0, 0, (1,))),
            )

    def test_rejects_physical_prompt_generated_collision(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate physical"):
            RoutingTrace(
                "example/moe",
                4,
                1,
                (RoutingEvent("prompt", 0, 0, (1,)), RoutingEvent("generated", 0, 0, (2,))),
            )

    def test_rejects_prompt_generated_position_rollback(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt token positions must precede"):
            RoutingTrace(
                "example/moe",
                4,
                1,
                (
                    RoutingEvent("prompt", 0, 0, (1,)),
                    RoutingEvent("prompt", 2, 0, (1,)),
                    RoutingEvent("generated", 1, 1, (2,)),
                ),
            )

    def test_rejects_non_increasing_positions_within_phase_layer(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            RoutingTrace(
                "example/moe",
                4,
                1,
                (RoutingEvent("prompt", 2, 0, (1,)), RoutingEvent("prompt", 1, 0, (2,))),
            )

    def test_accepts_layer_major_prompt_and_token_major_generated_order(self) -> None:
        trace = RoutingTrace(
            "example/moe",
            4,
            1,
            (
                RoutingEvent("prompt", 0, 0, (1,)),
                RoutingEvent("prompt", 1, 0, (1,)),
                RoutingEvent("prompt", 0, 1, (2,)),
                RoutingEvent("prompt", 1, 1, (2,)),
                RoutingEvent("generated", 2, 0, (1,)),
                RoutingEvent("generated", 2, 1, (2,)),
                RoutingEvent("generated", 3, 0, (1,)),
                RoutingEvent("generated", 3, 1, (2,)),
            ),
        )
        self.assertEqual(len(trace.events), 8)

    def test_rejects_token_major_prompt_nesting(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer-major collector order"):
            RoutingTrace(
                "example/moe",
                4,
                1,
                (
                    RoutingEvent("prompt", 0, 0, (1,)),
                    RoutingEvent("prompt", 0, 1, (2,)),
                    RoutingEvent("prompt", 1, 0, (1,)),
                    RoutingEvent("prompt", 1, 1, (2,)),
                ),
            )

    def test_rejects_layer_major_generated_nesting(self) -> None:
        with self.assertRaisesRegex(ValueError, "token-major collector order"):
            RoutingTrace(
                "example/moe",
                4,
                1,
                (
                    RoutingEvent("generated", 2, 0, (1,)),
                    RoutingEvent("generated", 3, 0, (1,)),
                    RoutingEvent("generated", 2, 1, (2,)),
                    RoutingEvent("generated", 3, 1, (2,)),
                ),
            )

    def test_reader_rejects_unknown_records_instead_of_silently_dropping_them(self) -> None:
        import json
        import tempfile
        metadata = RoutingTrace(
            "example/moe", 4, 1, (RoutingEvent("prompt", 0, 0, (1,)),)
        ).metadata()
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/bad.jsonl"
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata) + "\n")
                stream.write(json.dumps({"record_type": "ignored"}) + "\n")
            with self.assertRaisesRegex(ValueError, "unexpected record type"):
                read_trace(path)
