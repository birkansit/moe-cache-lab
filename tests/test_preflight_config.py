import json
from pathlib import Path
import tempfile
import unittest

from moe_cache_lab.preflight_config import (
    parse_preflight_config_data,
    preflight_config_data,
    read_preflight_config,
)


def _config_payload() -> dict:
    return {
        "format": "moe-cache-lab.preflight-config",
        "format_version": 1,
        "expert_sizes": [
            {"layer_id": 1, "expert_id": 2, "size_bytes": 30},
            {"layer_id": 0, "expert_id": 1, "size_bytes": 20},
        ],
        "capacities_bytes": [64, 32, 64],
        "policies": ["lfu", "lru", "lfu"],
        "hardware_profiles": [
            {
                "name": "z",
                "h2d_payload_bandwidth_bytes_per_second": 100,
                "setup_latency_ns_per_loaded_expert": 5,
            },
            {
                "name": "a",
                "h2d_payload_bandwidth_bytes_per_second": 200,
                "setup_latency_ns_per_loaded_expert": 0,
            },
        ],
    }


class PreflightConfigTests(unittest.TestCase):
    def test_normalizes_semantically_equivalent_input_deterministically(self) -> None:
        config = parse_preflight_config_data(_config_payload())
        self.assertEqual(
            [(item.layer_id, item.expert_id, item.size_bytes) for item in config.expert_sizes],
            [(0, 1, 20), (1, 2, 30)],
        )
        self.assertEqual(config.capacities_bytes, (32, 64))
        self.assertEqual(config.policies, ("lru", "lfu"))
        self.assertEqual([profile.name for profile in config.hardware_profiles], ["a", "z"])
        self.assertEqual(config.expert_size_map(), {(0, 1): 20, (1, 2): 30})

        reordered = _config_payload()
        reordered["expert_sizes"] = list(reversed(reordered["expert_sizes"]))
        reordered["capacities_bytes"] = [32, 64]
        reordered["policies"] = ["lru", "lfu"]
        reordered["hardware_profiles"] = list(reversed(reordered["hardware_profiles"]))
        self.assertEqual(config, parse_preflight_config_data(reordered))
        self.assertEqual(
            preflight_config_data(config),
            preflight_config_data(parse_preflight_config_data(reordered)),
        )

    def test_duplicate_layer_qualified_expert_size_rejects(self) -> None:
        payload = _config_payload()
        payload["expert_sizes"].append(
            {"layer_id": 0, "expert_id": 1, "size_bytes": 99}
        )
        with self.assertRaisesRegex(ValueError, "duplicate expert-size identity"):
            parse_preflight_config_data(payload)

    def test_duplicate_hardware_profile_name_rejects(self) -> None:
        payload = _config_payload()
        payload["hardware_profiles"][1]["name"] = "z"
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            parse_preflight_config_data(payload)

    def test_requires_strict_top_level_format_and_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "top-level JSON object"):
            parse_preflight_config_data([])
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            parse_preflight_config_data({"format": "moe-cache-lab.preflight-config"})

        payload = _config_payload()
        payload["format"] = "other"
        with self.assertRaisesRegex(ValueError, "unsupported preflight config format"):
            parse_preflight_config_data(payload)

        for invalid_version in (2, True, 1.0):
            payload = _config_payload()
            payload["format_version"] = invalid_version
            with self.subTest(format_version=invalid_version), self.assertRaisesRegex(
                ValueError, "unsupported preflight config format_version"
            ):
                parse_preflight_config_data(payload)

        payload = _config_payload()
        payload["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            parse_preflight_config_data(payload)

    def test_rejects_invalid_capacity_policy_and_expert_numbers(self) -> None:
        for capacity in (0, -1, 1.5, True):
            payload = _config_payload()
            payload["capacities_bytes"] = [capacity]
            with self.subTest(capacity=capacity), self.assertRaisesRegex(
                ValueError, "positive integers"
            ):
                parse_preflight_config_data(payload)

        payload = _config_payload()
        payload["policies"] = ["offline_oracle_frequency"]
        with self.assertRaisesRegex(ValueError, "unsupported preflight cache policy"):
            parse_preflight_config_data(payload)

        for field, value in (("layer_id", True), ("expert_id", -1), ("size_bytes", 0)):
            payload = _config_payload()
            payload["expert_sizes"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_preflight_config_data(payload)

    def test_hardware_profile_validation_is_reused(self) -> None:
        payload = _config_payload()
        payload["hardware_profiles"][0]["h2d_payload_bandwidth_bytes_per_second"] = True
        with self.assertRaisesRegex(ValueError, "H2D payload bandwidth"):
            parse_preflight_config_data(payload)

        payload = _config_payload()
        payload["hardware_profiles"][0]["setup_latency_ns_per_loaded_expert"] = -1
        with self.assertRaisesRegex(ValueError, "setup latency"):
            parse_preflight_config_data(payload)

    def test_read_preflight_config_parses_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(_config_payload()), encoding="utf-8")
            self.assertEqual(
                read_preflight_config(path),
                parse_preflight_config_data(_config_payload()),
            )


if __name__ == "__main__":
    unittest.main()
