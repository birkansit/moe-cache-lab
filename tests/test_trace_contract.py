from collections import OrderedDict
import json
from pathlib import Path
import tempfile
import unittest

from moe_cache_lab.trace import (
    TRACE_FORMAT,
    TRACE_SCHEMA_RESOURCE,
    TRACE_VERSION,
    load_trace_schema,
    read_trace,
    validate_trace_records,
)
from moe_cache_lab.trace_v2 import read_versioned_trace


ROOT = Path(__file__).resolve().parents[1]


def _metadata(**updates):
    record = {
        "record_type": "metadata",
        "format": TRACE_FORMAT,
        "format_version": TRACE_VERSION,
        "model_id": "external/example-moe",
        "num_experts": 4,
        "experts_per_token": 2,
    }
    record.update(updates)
    return record


def _event(phase="prompt", position=0, layer=0, experts=(1, 3), **updates):
    record = {
        "record_type": "routing_selection",
        "phase": phase,
        "token_position": position,
        "layer": layer,
        "selected_experts": list(experts),
    }
    record.update(updates)
    return record


class CanonicalTraceContractTests(unittest.TestCase):
    def test_packaged_schema_declares_closed_canonical_v1_records(self) -> None:
        schema = load_trace_schema()
        self.assertEqual(TRACE_SCHEMA_RESOURCE, "schemas/routing-trace-v1.schema.json")
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["$id"], "urn:moe-cache-lab:schema:routing-trace:1")
        metadata = schema["$defs"]["metadata_record"]
        event = schema["$defs"]["routing_selection_record"]
        self.assertFalse(metadata["additionalProperties"])
        self.assertFalse(event["additionalProperties"])
        self.assertEqual(metadata["properties"]["format"]["const"], TRACE_FORMAT)
        self.assertEqual(metadata["properties"]["format_version"]["const"], TRACE_VERSION)
        self.assertEqual(event["properties"]["phase"]["enum"], ["prompt", "generated"])

    def test_canonical_valid_trace_preserves_layer_qualified_atomic_event(self) -> None:
        trace = validate_trace_records((_metadata(), _event()))
        self.assertEqual(len(trace.events), 1)
        self.assertEqual(trace.events[0].selected_experts, (1, 3))
        self.assertEqual(trace.expert_requests, ((0, 1), (0, 3)))

        layered = validate_trace_records(
            (
                _metadata(),
                _event(position=0, layer=0, experts=(1, 3)),
                _event(position=0, layer=1, experts=(1, 3)),
            )
        )
        self.assertEqual(
            layered.expert_requests,
            ((0, 1), (0, 3), (1, 1), (1, 3)),
        )

    def test_missing_required_metadata_or_event_field_fails_explicitly(self) -> None:
        for record_index, field in ((0, "model_id"), (1, "selected_experts")):
            records = [_metadata(), _event()]
            del records[record_index][field]
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, rf"missing required field.*{field}"
            ):
                validate_trace_records(records)

    def test_invalid_field_types_values_and_unknown_fields_fail(self) -> None:
        invalid_records = (
            (_metadata(model_id=4), _event()),
            (_metadata(num_experts=True), _event()),
            (_metadata(), _event(position=True)),
            (_metadata(), {**_event(), "selected_experts": 4}),
            (_metadata(), _event(experts=(1, 1))),
            (_metadata(), _event(selected_probabilities=[0.5])),
            (_metadata(), _event(selected_probabilities="not-an-array")),
            (_metadata(), _event(extra="unknown")),
        )
        for records in invalid_records:
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_trace_records(records)

    def test_invalid_chronology_is_rejected_without_repair(self) -> None:
        reversed_prompt = (
            _metadata(experts_per_token=1),
            _event(position=0, layer=1, experts=(1,)),
            _event(position=0, layer=0, experts=(2,)),
        )
        with self.assertRaisesRegex(ValueError, "layer-major collector order"):
            validate_trace_records(reversed_prompt)

        phase_regression = (
            _metadata(experts_per_token=1),
            _event("generated", 1, 0, (1,)),
            _event("prompt", 0, 1, (2,)),
        )
        with self.assertRaisesRegex(ValueError, "cannot follow"):
            validate_trace_records(phase_regression)

    def test_format_schema_version_and_record_type_rejection_are_exact(self) -> None:
        for metadata in (
            _metadata(format="other.routing-jsonl"),
            _metadata(format_version=2),
            _metadata(format_version=True),
        ):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(
                ValueError, "routing trace format|format_version"
            ):
                validate_trace_records((metadata, _event()))
        with self.assertRaisesRegex(ValueError, "unexpected record type"):
            validate_trace_records((_metadata(), {"record_type": "future"}))

    def test_validation_is_independent_of_json_object_key_order(self) -> None:
        metadata = _metadata(
            source_text="input",
            capture_method="external test producer",
            model_revision="revision",
        )
        event = _event(
            token_id=42,
            selected_probabilities=[0.75, 0.25],
        )
        ordered = validate_trace_records((metadata, event))
        reversed_keys = validate_trace_records(
            (
                OrderedDict(reversed(tuple(metadata.items()))),
                OrderedDict(reversed(tuple(event.items()))),
            )
        )
        self.assertEqual(reversed_keys, ordered)

    def test_reader_rejects_duplicate_json_keys(self) -> None:
        metadata = json.dumps(_metadata())
        duplicate_event = (
            '{"record_type":"routing_selection","phase":"prompt",'
            '"token_position":0,"layer":0,"layer":1,"selected_experts":[1,3]}'
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.jsonl"
            path.write_text(metadata + "\n" + duplicate_event + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key: layer"):
                read_trace(path)

    def test_all_tracked_canonical_traces_remain_compatible(self) -> None:
        invalid_fixtures = ROOT / "examples" / "trace-validation-fixtures" / "invalid"
        paths = [
            path
            for path in sorted((ROOT / "examples").rglob("*.jsonl"))
            if invalid_fixtures not in path.parents
        ]
        paths.extend(sorted((ROOT / "results").rglob("*.jsonl")))
        self.assertGreater(len(paths), 1)
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertGreater(len(read_versioned_trace(path).events), 0)


if __name__ == "__main__":
    unittest.main()
