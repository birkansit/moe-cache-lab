from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab.cli import main
from moe_cache_lab.preflight_config import (
    PREFLIGHT_CONFIG_FORMAT,
    PREFLIGHT_CONFIG_LEGACY_VERSION,
    PREFLIGHT_CONFIG_VERSION,
    parse_preflight_config_data,
)
from moe_cache_lab.trace import TRACE_FORMAT, RoutingTrace
from moe_cache_lab.trace_v2 import (
    TRACE_VERSION_V2,
    RoutingTraceV2,
    validate_versioned_trace_records,
)


ROOT = Path(__file__).resolve().parents[1]


def _v1_records(version=1):
    return [
        {
            "record_type": "metadata",
            "format": TRACE_FORMAT,
            "format_version": version,
            "model_id": "synthetic/v1",
            "num_experts": 2,
            "experts_per_token": 1,
        },
        {
            "record_type": "routing_selection",
            "phase": "prompt",
            "token_position": 0,
            "layer": 0,
            "selected_experts": [0],
            "selected_probabilities": [1.0],
        },
    ]


def _v2_records(version=TRACE_VERSION_V2):
    return [
        {
            "record_type": "metadata",
            "format": TRACE_FORMAT,
            "format_version": version,
            "model_id": "synthetic/v2",
            "routing_stages": [
                {
                    "routing_stage": "decoder",
                    "num_experts": 2,
                    "assigned_experts_per_token": 1,
                    "allows_unassigned": True,
                }
            ],
        },
        {
            "record_type": "routing_selection",
            "routing_stage": "decoder",
            "phase": "decoder_prompt",
            "token_position": 0,
            "layer": 0,
            "assignment_state": "assigned",
            "selected_experts": [0],
            "selected_probabilities": [1.0],
        },
    ]


def _preflight_payload(version: int):
    payload = {
        "format": PREFLIGHT_CONFIG_FORMAT,
        "format_version": version,
        "expert_sizes": [
            {"layer_id": 0, "expert_id": 0, "size_bytes": 16},
            {"layer_id": 0, "expert_id": 1, "size_bytes": 16},
        ],
        "capacities_bytes": [16, 32],
        "policies": ["lru"],
        "hardware_profiles": [
            {
                "name": "synthetic",
                "h2d_payload_bandwidth_bytes_per_second": 1000,
                (
                    "setup_latency_ns_per_loaded_expert"
                    if version == PREFLIGHT_CONFIG_LEGACY_VERSION
                    else "setup_latency_ns_per_transfer_operation"
                ): 10,
            }
        ],
    }
    if version == PREFLIGHT_CONFIG_VERSION:
        payload["transfer_operation_plans"] = [
            {"name": "one", "operations_per_logical_load": 1}
        ]
    return payload


class CompatibilityPolicyTests(unittest.TestCase):
    def test_versioned_trace_dispatch_is_exact_and_closed(self) -> None:
        v1 = validate_versioned_trace_records(_v1_records())
        v2 = validate_versioned_trace_records(_v2_records())
        self.assertIsInstance(v1, RoutingTrace)
        self.assertIsInstance(v2, RoutingTraceV2)

        for bad_version in (True, 1.0, 3):
            records = _v1_records(bad_version)
            with self.subTest(version=bad_version), self.assertRaises(ValueError):
                validate_versioned_trace_records(records)

        wrong_family = _v1_records()
        wrong_family[0]["format"] = "other.routing-jsonl"
        with self.assertRaisesRegex(ValueError, "unsupported routing trace format"):
            validate_versioned_trace_records(wrong_family)

    def test_trace_dispatch_does_not_fallback_between_v1_and_v2(self) -> None:
        v1_shape_claiming_v2 = _v1_records(TRACE_VERSION_V2)
        with self.assertRaises(ValueError):
            validate_versioned_trace_records(v1_shape_claiming_v2)

        v2_shape_claiming_v1 = _v2_records(1)
        with self.assertRaises(ValueError):
            validate_versioned_trace_records(v2_shape_claiming_v1)

    def test_preflight_config_has_its_own_strict_version_namespace(self) -> None:
        legacy = parse_preflight_config_data(
            _preflight_payload(PREFLIGHT_CONFIG_LEGACY_VERSION)
        )
        current = parse_preflight_config_data(
            _preflight_payload(PREFLIGHT_CONFIG_VERSION)
        )
        self.assertEqual(legacy.format_version, 1)
        self.assertEqual(current.format_version, 2)
        self.assertEqual(set(current.expert_size_map()), {(0, 0), (0, 1)})

        for bad_version in (True, 1.0, 4):
            payload = _preflight_payload(PREFLIGHT_CONFIG_VERSION)
            payload["format_version"] = bad_version
            with self.subTest(version=bad_version), self.assertRaisesRegex(
                ValueError, "unsupported preflight config format_version"
            ):
                parse_preflight_config_data(payload)

        self.assertEqual(PREFLIGHT_CONFIG_VERSION, 2)
        self.assertNotIn(
            "routing_stage",
            _preflight_payload(PREFLIGHT_CONFIG_VERSION)["expert_sizes"][0],
        )

    def test_equal_trace_and_config_version_numbers_do_not_imply_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path = root / "trace-v2.jsonl"
            config_path = root / "preflight-v2.json"
            trace_path.write_text(
                "\n".join(json.dumps(record, sort_keys=True) for record in _v2_records())
                + "\n",
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(_preflight_payload(PREFLIGHT_CONFIG_VERSION), sort_keys=True),
                encoding="utf-8",
            )
            stderr = StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "moe-cache-lab",
                    "analyze",
                    str(trace_path),
                    "--preflight-config",
                    str(config_path),
                ],
            ), redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                main()

        self.assertEqual(caught.exception.code, 2)
        self.assertIn(
            "requires stage-qualified preflight config format_version 3",
            stderr.getvalue(),
        )

    def test_canonical_policy_states_required_cross_format_rules(self) -> None:
        text = (ROOT / "COMPATIBILITY.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        required = (
            "Independent version namespaces",
            "Numbers in different namespaces have no implied relationship",
            "new model family, checkpoint, collector, or inference runtime is **not** by itself",
            "does not retry another validator",
            "never reinterpreted as v2",
            "never down-converted to v1",
            "This config-language change does not create a routing-trace v3",
            "equal version numbers do not establish compatibility",
            "v2 | config v3 | supported stage-qualified single-trace pre-flight workflow",
            "v1 compatibility workflow and rejects config v3",
            "not required to depend on collector Python classes",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, normalized)

    def test_policy_document_is_linked_and_packaged(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        trace_format = (ROOT / "TRACE_FORMAT.md").read_text(encoding="utf-8")
        preflight = (ROOT / "PREFLIGHT.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        distribution_audit = (ROOT / "scripts" / "audit_distributions.py").read_text(
            encoding="utf-8"
        )
        for text in (readme, trace_format, preflight):
            self.assertIn("COMPATIBILITY.md", text)
        self.assertIn('"COMPATIBILITY.md"', pyproject)
        self.assertIn("include COMPATIBILITY.md", manifest)
        self.assertIn('"COMPATIBILITY.md"', distribution_audit)


if __name__ == "__main__":
    unittest.main()
