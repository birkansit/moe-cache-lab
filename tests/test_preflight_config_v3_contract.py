import copy
import json
from pathlib import Path
import tempfile
import unittest

from moe_cache_lab.preflight_config import (
    PREFLIGHT_CONFIG_FORMAT,
    parse_preflight_config_data,
    preflight_config_data,
    read_preflight_config,
)


def _payload_v3() -> dict:
    return {
        "format": PREFLIGHT_CONFIG_FORMAT,
        "format_version": 3,
        "expert_sizes": [
            {
                "routing_stage": "decoder",
                "layer_id": 3,
                "expert_id": 5,
                "size_bytes": 20,
            },
            {
                "routing_stage": "encoder",
                "layer_id": 3,
                "expert_id": 5,
                "size_bytes": 10,
            },
            {
                "routing_stage": "encoder",
                "layer_id": 1,
                "expert_id": 7,
                "size_bytes": 30,
            },
        ],
        "capacities_bytes": [64, 32, 64],
        "policies": ["lfu", "lru", "lfu"],
        "hardware_profiles": [
            {
                "name": "z",
                "h2d_payload_bandwidth_bytes_per_second": 100,
                "setup_latency_ns_per_transfer_operation": 5,
            },
            {
                "name": "a",
                "h2d_payload_bandwidth_bytes_per_second": 200,
                "setup_latency_ns_per_transfer_operation": 0,
            },
        ],
        "transfer_operation_plans": [
            {"name": "two", "operations_per_logical_load": 2},
            {"name": "one", "operations_per_logical_load": 1},
        ],
    }


def _expert_rows(config) -> list[tuple[str, int, int, int]]:
    return [
        (
            item.routing_stage,
            item.layer_id,
            item.expert_id,
            item.size_bytes,
        )
        for item in config.expert_sizes
    ]


class PreflightConfigV3ContractTests(unittest.TestCase):
    def test_accepts_stage_qualified_identity_and_normalizes_deterministically(self) -> None:
        config = parse_preflight_config_data(_payload_v3())

        self.assertEqual(config.format_version, 3)
        self.assertEqual(
            _expert_rows(config),
            [
                ("encoder", 1, 7, 30),
                ("encoder", 3, 5, 10),
                ("decoder", 3, 5, 20),
            ],
        )
        self.assertEqual(config.capacities_bytes, (32, 64))
        self.assertEqual(config.policies, ("lru", "lfu"))
        self.assertEqual([item.name for item in config.hardware_profiles], ["a", "z"])
        self.assertEqual(
            [item.name for item in config.transfer_operation_plans],
            ["one", "two"],
        )
        self.assertEqual(
            config.expert_size_map(),
            {
                ("encoder", 1, 7): 30,
                ("encoder", 3, 5): 10,
                ("decoder", 3, 5): 20,
            },
        )

    def test_encoder_and_decoder_same_numeric_ids_remain_distinct(self) -> None:
        payload = _payload_v3()
        payload["expert_sizes"] = [
            {
                "routing_stage": "encoder",
                "layer_id": 3,
                "expert_id": 5,
                "size_bytes": 10,
            },
            {
                "routing_stage": "decoder",
                "layer_id": 3,
                "expert_id": 5,
                "size_bytes": 20,
            },
        ]
        config = parse_preflight_config_data(payload)
        size_map = config.expert_size_map()

        self.assertEqual(len(size_map), 2)
        self.assertEqual(size_map[("encoder", 3, 5)], 10)
        self.assertEqual(size_map[("decoder", 3, 5)], 20)

    def test_accepts_decoder_only_and_empty_expert_size_arrays(self) -> None:
        decoder_only = _payload_v3()
        decoder_only["expert_sizes"] = [
            {
                "routing_stage": "decoder",
                "layer_id": 0,
                "expert_id": 1,
                "size_bytes": 16,
            }
        ]
        self.assertEqual(
            _expert_rows(parse_preflight_config_data(decoder_only)),
            [("decoder", 0, 1, 16)],
        )

        all_unassigned_compatible = _payload_v3()
        all_unassigned_compatible["expert_sizes"] = []
        config = parse_preflight_config_data(all_unassigned_compatible)
        self.assertEqual(tuple(config.expert_sizes), ())
        self.assertEqual(config.expert_size_map(), {})

    def test_round_trip_is_exact_and_file_reader_matches_data_parser(self) -> None:
        config = parse_preflight_config_data(_payload_v3())
        serialized = preflight_config_data(config)
        reparsed = parse_preflight_config_data(serialized)
        self.assertEqual(reparsed, config)
        self.assertEqual(preflight_config_data(reparsed), serialized)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight-v3.json"
            path.write_text(
                json.dumps(serialized, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            self.assertEqual(read_preflight_config(path), config)

    def test_rejects_missing_unknown_or_legacy_expert_identity_fields(self) -> None:
        cases: list[tuple[str, dict]] = []

        missing_stage = _payload_v3()
        del missing_stage["expert_sizes"][0]["routing_stage"]
        cases.append(("missing-stage", missing_stage))

        unknown_stage = _payload_v3()
        unknown_stage["expert_sizes"][0]["routing_stage"] = "router"
        cases.append(("unknown-stage", unknown_stage))

        unknown_record_field = _payload_v3()
        unknown_record_field["expert_sizes"][0]["unexpected"] = 1
        cases.append(("unknown-record-field", unknown_record_field))

        legacy_record = _payload_v3()
        legacy_record["expert_sizes"] = [
            {"layer_id": 0, "expert_id": 0, "size_bytes": 16}
        ]
        cases.append(("legacy-record", legacy_record))

        unknown_top_level = _payload_v3()
        unknown_top_level["unexpected"] = 1
        cases.append(("unknown-top-level", unknown_top_level))

        missing_top_level = _payload_v3()
        del missing_top_level["transfer_operation_plans"]
        cases.append(("missing-top-level", missing_top_level))

        for label, payload in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_preflight_config_data(payload)

    def test_rejects_duplicate_stage_qualified_identity(self) -> None:
        payload = _payload_v3()
        payload["expert_sizes"].append(copy.deepcopy(payload["expert_sizes"][1]))
        with self.assertRaises(ValueError):
            parse_preflight_config_data(payload)

    def test_rejects_invalid_v3_numeric_domains(self) -> None:
        for field, invalid_values in (
            ("layer_id", (True, -1, 1.5)),
            ("expert_id", (True, -1, 1.5)),
            ("size_bytes", (True, 0, -1, 1.5)),
        ):
            for value in invalid_values:
                payload = _payload_v3()
                payload["expert_sizes"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    parse_preflight_config_data(payload)

    def test_rejects_invalid_or_unsupported_versions(self) -> None:
        for version in (True, 3.0, 0, 4, -1):
            payload = _payload_v3()
            payload["format_version"] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                parse_preflight_config_data(payload)

    def test_v3_retains_config_v2_capacity_policy_profile_and_plan_rules(self) -> None:
        invalid_payloads: list[dict] = []

        empty_capacities = _payload_v3()
        empty_capacities["capacities_bytes"] = []
        invalid_payloads.append(empty_capacities)

        invalid_capacity = _payload_v3()
        invalid_capacity["capacities_bytes"] = [0]
        invalid_payloads.append(invalid_capacity)

        invalid_policy = _payload_v3()
        invalid_policy["policies"] = ["random"]
        invalid_payloads.append(invalid_policy)

        empty_profiles = _payload_v3()
        empty_profiles["hardware_profiles"] = []
        invalid_payloads.append(empty_profiles)

        duplicate_profile = _payload_v3()
        duplicate_profile["hardware_profiles"][1]["name"] = "z"
        invalid_payloads.append(duplicate_profile)

        empty_plans = _payload_v3()
        empty_plans["transfer_operation_plans"] = []
        invalid_payloads.append(empty_plans)

        duplicate_plan = _payload_v3()
        duplicate_plan["transfer_operation_plans"][1]["name"] = "two"
        invalid_payloads.append(duplicate_plan)

        for index, payload in enumerate(invalid_payloads):
            with self.subTest(case=index), self.assertRaises(ValueError):
                parse_preflight_config_data(payload)


if __name__ == "__main__":
    unittest.main()
