from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.cli import main
from moe_cache_lab.trace import TRACE_FORMAT, RoutingEvent, RoutingTrace, write_trace
from moe_cache_lab.trace_v2 import (
    TRACE_SCHEMA_RESOURCE_V2,
    TRACE_VERSION_V2,
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
    load_trace_v2_schema,
    read_trace_v2,
    read_versioned_trace,
    validate_trace_v2_records,
    validate_versioned_trace_records,
    write_trace_v2,
)


def _profile(stage: str, **updates) -> dict:
    record = {
        "routing_stage": stage,
        "num_experts": 8,
        "assigned_experts_per_token": 1,
        "allows_unassigned": True,
    }
    record.update(updates)
    return record


def _metadata(**updates) -> dict:
    record = {
        "record_type": "metadata",
        "format": TRACE_FORMAT,
        "format_version": TRACE_VERSION_V2,
        "model_id": "google/switch-base-8",
        "routing_stages": [_profile("encoder"), _profile("decoder")],
    }
    record.update(updates)
    return record


def _assigned(
    stage: str = "encoder",
    phase: str = "source",
    position: int = 0,
    layer: int = 1,
    experts=(3,),
    **updates,
) -> dict:
    record = {
        "record_type": "routing_selection",
        "routing_stage": stage,
        "phase": phase,
        "token_position": position,
        "layer": layer,
        "assignment_state": "assigned",
        "selected_experts": list(experts),
        "selected_probabilities": [0.75] * len(experts),
    }
    record.update(updates)
    return record


def _unassigned(
    stage: str = "decoder",
    phase: str = "decoder_generated",
    position: int = 1,
    layer: int = 1,
    **updates,
) -> dict:
    record = {
        "record_type": "routing_selection",
        "routing_stage": stage,
        "phase": phase,
        "token_position": position,
        "layer": layer,
        "assignment_state": "unassigned",
        "selected_experts": [],
        "selected_probabilities": [],
        "unassigned_reason": "capacity",
    }
    record.update(updates)
    return record


def _trace() -> RoutingTraceV2:
    return RoutingTraceV2(
        model_id="google/switch-base-8",
        routing_stages=(
            RoutingStageProfile("encoder", 8, 1, True),
            RoutingStageProfile("decoder", 8, 1, True),
        ),
        events=(
            RoutingEventV2(
                "encoder", "source", 0, 1, "assigned", (3,), (0.75,), 10
            ),
            RoutingEventV2(
                "decoder", "decoder_prompt", 0, 1, "assigned", (3,), (0.6,), 0
            ),
            RoutingEventV2(
                "decoder",
                "decoder_generated",
                1,
                1,
                "unassigned",
                (),
                (),
                42,
                "capacity",
            ),
        ),
        capture_method="synthetic post-capacity observation",
        created_at="fixed-test-time",
        transformers_version="5.12.0",
        model_revision="test-revision",
    )


class RoutingTraceV2ContractTests(unittest.TestCase):
    def test_packaged_schema_is_closed_draft_2020_12_v2(self) -> None:
        schema = load_trace_v2_schema()
        self.assertEqual(
            TRACE_SCHEMA_RESOURCE_V2,
            "schemas/routing-trace-v2.schema.json",
        )
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["$id"], "urn:moe-cache-lab:schema:routing-trace:2")
        self.assertFalse(schema["$defs"]["metadata_record"]["additionalProperties"])
        self.assertFalse(
            schema["$defs"]["routing_selection_record"]["additionalProperties"]
        )
        self.assertFalse(
            schema["$defs"]["routing_stage_profile"]["additionalProperties"]
        )
        self.assertEqual(
            schema["$defs"]["metadata_record"]["properties"]["format_version"]["const"],
            2,
        )

    def test_valid_encoder_decoder_and_unassigned_events_preserve_identity(self) -> None:
        trace = validate_trace_v2_records(
            (
                _metadata(),
                _assigned(token_id=10),
                _assigned(
                    "decoder",
                    "decoder_prompt",
                    token_id=0,
                    selected_probabilities=[0.6],
                ),
                _unassigned(token_id=42),
            )
        )
        self.assertEqual(
            trace.expert_requests,
            (("encoder", 1, 3), ("decoder", 1, 3)),
        )
        self.assertEqual((trace.assigned_event_count, trace.unassigned_event_count), (2, 1))
        self.assertEqual(trace.events[-1].expert_requests, ())

    def test_explicit_v2_round_trip_preserves_assigned_and_unassigned(self) -> None:
        trace = _trace()
        with tempfile.TemporaryDirectory() as temporary:
            path = write_trace_v2(Path(temporary) / "trace-v2.jsonl", trace)
            loaded = read_trace_v2(path)
            versioned = read_versioned_trace(path)
        self.assertEqual(loaded, trace)
        self.assertEqual(versioned, trace)

    def test_v2_writer_rejects_v1_object_and_v1_writer_remains_v1(self) -> None:
        v1 = RoutingTrace(
            "example/v1",
            4,
            1,
            (RoutingEvent("prompt", 0, 0, (1,)),),
            created_at="fixed",
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(TypeError, "RoutingTraceV2"):
                write_trace_v2(directory / "bad.jsonl", v1)  # type: ignore[arg-type]
            v1_path = write_trace(directory / "v1.jsonl", v1)
            metadata = json.loads(v1_path.read_text(encoding="utf-8").splitlines()[0])
            loaded = read_versioned_trace(v1_path)
        self.assertEqual(metadata["format_version"], 1)
        self.assertIsInstance(loaded, RoutingTrace)

    def test_missing_invalid_stage_and_stage_phase_mismatch_reject(self) -> None:
        invalid = (
            {key: value for key, value in _assigned().items() if key != "routing_stage"},
            _assigned(routing_stage="other"),
            _assigned("encoder", "decoder_prompt"),
            _assigned("decoder", "source"),
        )
        for event in invalid:
            with self.subTest(event=event), self.assertRaises(ValueError):
                validate_trace_v2_records((_metadata(), event))

    def test_assignment_state_contradictions_reject(self) -> None:
        invalid = (
            _assigned(experts=(), selected_probabilities=[]),
            _assigned(unassigned_reason="capacity"),
            _assigned(experts=(2, 3), selected_probabilities=[0.5]),
            _unassigned(selected_experts=[3]),
            _unassigned(selected_probabilities=[0.75]),
            {key: value for key, value in _unassigned().items() if key != "unassigned_reason"},
            _unassigned(unassigned_reason="other"),
            _assigned(assignment_state="other"),
        )
        for event in invalid:
            with self.subTest(event=event), self.assertRaises(ValueError):
                validate_trace_v2_records((_metadata(), event))

    def test_stage_profile_uniqueness_order_values_and_coverage_reject(self) -> None:
        invalid_metadata = (
            _metadata(routing_stages=[]),
            _metadata(routing_stages=[_profile("encoder"), _profile("encoder")]),
            _metadata(routing_stages=[_profile("decoder"), _profile("encoder")]),
            _metadata(routing_stages=[_profile("encoder", num_experts=0)]),
            _metadata(routing_stages=[_profile("encoder", assigned_experts_per_token=True)]),
            _metadata(routing_stages=[_profile("encoder", allows_unassigned=1)]),
            _metadata(routing_stages=[_profile("other")]),
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_trace_v2_records((metadata, _assigned()))
        with self.assertRaisesRegex(ValueError, "no metadata profile"):
            validate_trace_v2_records(
                (_metadata(routing_stages=[_profile("encoder")]), _assigned("decoder", "decoder_prompt"))
            )

    def test_stage_profile_range_width_and_unassigned_permission_reject(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside stage num_experts"):
            validate_trace_v2_records((_metadata(), _assigned(experts=(8,))))
        with self.assertRaisesRegex(ValueError, "selection count"):
            validate_trace_v2_records(
                (
                    _metadata(
                        routing_stages=[
                            _profile("encoder", assigned_experts_per_token=2)
                        ]
                    ),
                    _assigned(),
                )
            )
        with self.assertRaisesRegex(ValueError, "does not allow"):
            validate_trace_v2_records(
                (
                    _metadata(
                        routing_stages=[
                            _profile("decoder", allows_unassigned=False)
                        ]
                    ),
                    _unassigned(),
                )
            )

    def test_duplicate_physical_event_and_position_boundary_reject(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate routing-stage"):
            validate_trace_v2_records(
                (
                    _metadata(routing_stages=[_profile("decoder")]),
                    _assigned("decoder", "decoder_prompt", 0, 1),
                    _unassigned(position=0, layer=1),
                )
            )
        with self.assertRaisesRegex(ValueError, "prompt positions"):
            validate_trace_v2_records(
                (
                    _metadata(routing_stages=[_profile("decoder")]),
                    _assigned("decoder", "decoder_prompt", 2, 1),
                    _unassigned(position=1, layer=3),
                )
            )

    def test_stage_and_phase_regressions_reject_without_repair(self) -> None:
        with self.assertRaisesRegex(ValueError, "encoder.*follow decoder"):
            validate_trace_v2_records(
                (
                    _metadata(),
                    _assigned("decoder", "decoder_prompt"),
                    _assigned(),
                )
            )
        with self.assertRaisesRegex(ValueError, "decoder_prompt.*cannot follow"):
            validate_trace_v2_records(
                (
                    _metadata(routing_stages=[_profile("decoder")]),
                    _unassigned(position=1),
                    _assigned("decoder", "decoder_prompt", 0),
                )
            )

    def test_wrong_layer_major_and_token_major_orders_reject(self) -> None:
        cases = (
            (
                "encoder source events",
                _metadata(routing_stages=[_profile("encoder")]),
                _assigned(layer=3),
                _assigned(layer=1),
            ),
            (
                "decoder prompt events",
                _metadata(routing_stages=[_profile("decoder")]),
                _assigned("decoder", "decoder_prompt", layer=3),
                _assigned("decoder", "decoder_prompt", layer=1),
            ),
            (
                "decoder generated events",
                _metadata(routing_stages=[_profile("decoder")]),
                _unassigned(position=1, layer=3),
                _unassigned(position=2, layer=1),
                _unassigned(position=1, layer=5),
            ),
        )
        for message, *records in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_trace_v2_records(records)

    def test_non_increasing_position_in_stream_rejects(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_trace_v2_records(
                (
                    _metadata(routing_stages=[_profile("encoder")]),
                    _assigned(position=1, layer=1),
                    _assigned(position=0, layer=1),
                )
            )

    def test_unknown_fields_and_duplicate_json_keys_reject(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown field"):
            validate_trace_v2_records((_metadata(extra=True), _assigned()))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.jsonl"
            path.write_text(
                json.dumps(_metadata())
                + "\n"
                + '{"record_type":"routing_selection","routing_stage":"encoder",'
                '"phase":"source","token_position":0,"layer":1,"layer":3,'
                '"assignment_state":"assigned","selected_experts":[3],'
                '"selected_probabilities":[0.75]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key: layer"):
                read_versioned_trace(path)

    def test_malformed_and_unsupported_versions_do_not_fallback(self) -> None:
        for version, message in ((True, "must be an integer"), (2.0, "must be an integer"), (3, "unsupported")):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, message):
                validate_versioned_trace_records(
                    (_metadata(format_version=version), _assigned())
                )
        with self.assertRaisesRegex(ValueError, "unsupported routing trace format"):
            validate_versioned_trace_records(
                (_metadata(format="other.routing-jsonl"), _assigned())
            )

    def test_v1_and_v2_dispatch_independently(self) -> None:
        v1_records = (
            {
                "record_type": "metadata",
                "format": TRACE_FORMAT,
                "format_version": 1,
                "model_id": "example/v1",
                "num_experts": 4,
                "experts_per_token": 1,
            },
            {
                "record_type": "routing_selection",
                "phase": "prompt",
                "token_position": 0,
                "layer": 0,
                "selected_experts": [1],
            },
        )
        self.assertIsInstance(validate_versioned_trace_records(v1_records), RoutingTrace)
        self.assertIsInstance(
            validate_versioned_trace_records((_metadata(), _assigned())),
            RoutingTraceV2,
        )
        with self.assertRaises(ValueError):
            validate_trace_v2_records(v1_records)

    def test_v1_only_cache_path_rejects_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_trace_v2(Path(temporary) / "trace-v2.jsonl", _trace())
            arguments = [
                "moe-cache-lab",
                "benchmark",
                str(path),
                "--capacity",
                "8",
                "--output",
                str(Path(temporary) / "report.md"),
            ]
            with patch("sys.argv", arguments), redirect_stdout(
                StringIO()
            ), self.assertRaises(ValueError):
                main()


if __name__ == "__main__":
    unittest.main()
