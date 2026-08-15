import ast
from dataclasses import fields
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import torch

from moe_cache_lab import runtime_copy_executor as executor
from moe_cache_lab import runtime_copy_validation as validation
from moe_cache_lab.trace_v2 import RoutingEventV2


def _event(stage, phase, position, expert=None):
    return RoutingEventV2(
        routing_stage=stage,
        phase=phase,
        token_position=position,
        layer=0,
        assignment_state="assigned" if expert is not None else "unassigned",
        selected_experts=(expert,) if expert is not None else (),
        selected_probabilities=(1.0,) if expert is not None else (),
        token_id=position,
        unassigned_reason=None if expert is not None else "capacity",
    )


def _payload(keys):
    specs = {}
    sources = {}
    for offset, key in enumerate(keys):
        wi = torch.tensor([float(offset + 1)], dtype=torch.float32)
        wo = torch.tensor([float(offset + 11)], dtype=torch.float32)
        components = (
            executor.ComponentSpec("wi.weight", (1,), "torch.float32", "cpu", True, 1, 4, 4),
            executor.ComponentSpec("wo.weight", (1,), "torch.float32", "cpu", True, 1, 4, 4),
        )
        specs[key] = executor.ExpertPayloadSpec(key, components, 8)
        sources[key] = {"wi.weight": wi, "wo.weight": wo}
    return specs, sources


def _matching_artifacts():
    common = {
        "model_id": validation.MODEL_ID,
        "revision": validation.MODEL_REVISION,
        "torch_version": "2.12.0+cpu",
        "transformers_version": "5.12.0",
        "device": "cpu",
        "dtype": "float32",
        "policy": "lru",
        "capacity_bytes": 16,
        "trace_sha256": "a" * 64,
        "payload_manifest_sha256": "b" * 64,
        "event_count": 4,
        "expert_request_count": 4,
        "hits": 1,
        "misses": 3,
        "eviction_count": 1,
        "final_resident_bytes": 16,
        "final_resident_keys": [["encoder", 0, 0], ["decoder", 0, 1]],
    }
    prediction = {
        "format": validation.PREDICTION_FORMAT,
        **common,
        "simulated_demand_load_bytes": 24,
    }
    observation = {
        "format": executor.OBSERVATION_FORMAT,
        **common,
        "completed_physical_logical_loads": 3,
        "actual_component_copy_count": 6,
        "profiler_aten_copy_count": 6,
        "actual_source_operand_logical_bytes": 24,
        "all_copy_content_equal": True,
        "copy_records": [
            {
                "key": ["encoder", 0, index // 2],
                "component_name": "wi.weight" if index % 2 == 0 else "wo.weight",
                "source_shape": [1],
                "destination_shape": [1],
                "source_dtype": "torch.float32",
                "destination_dtype": "torch.float32",
                "source_device": "cpu",
                "destination_device": "cpu",
                "source_logical_bytes": 4,
                "destination_logical_bytes": 4,
                "content_equal": True,
            }
            for index in range(6)
        ],
        "loaded_key_bytes": [
            {"key": ["encoder", 0, index], "component_count": 2, "copied_source_bytes": 8}
            for index in range(3)
        ],
    }
    payload_manifest = {
        "experts": [
            {
                "key": ["encoder", 0, index],
                "expert_payload_bytes": 8,
                "components": [
                    {
                        "name": name,
                        "shape": [1],
                        "dtype": "torch.float32",
                        "device": "cpu",
                        "logical_payload_bytes": 4,
                    }
                    for name in ("wi.weight", "wo.weight")
                ],
            }
            for index in range(3)
        ]
    }
    return prediction, observation, payload_manifest


class RuntimeCopyValidationTests(unittest.TestCase):
    def test_executor_api_and_imports_exclude_prediction_and_project_replay(self):
        names = {field.name for field in fields(executor.ExecutorInput)}
        forbidden_fields = {
            "prediction_path",
            "hits",
            "misses",
            "miss_sequence",
            "demand_load_bytes",
            "eviction_decisions",
            "final_resident_keys",
        }
        self.assertTrue(names.isdisjoint(forbidden_fields))

        source = inspect.getsource(executor)
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        self.assertFalse(any("byte_cache" in name or "hardware_cost" in name for name in imports))
        self.assertNotIn("simulate_versioned_byte_cache", source)
        self.assertNotIn("_ByteDynamicCache", source)

    def test_independent_lru_is_stage_qualified_and_unassigned_is_inert(self):
        keys = (
            ("encoder", 0, 0),
            ("decoder", 0, 0),
            ("encoder", 0, 1),
        )
        specs, sources = _payload(keys)
        events = (
            _event("encoder", "source", 0, 0),
            _event("decoder", "decoder_prompt", 0, 0),
            _event("encoder", "source", 1, None),
            _event("encoder", "source", 2, 0),
            _event("encoder", "source", 3, 1),
        )
        result = executor.execute_independent_replay(
            events,
            specs,
            sources,
            capacity_bytes=16,
            slot_count=2,
        )

        self.assertEqual((result["hits"], result["misses"], result["eviction_count"]), (1, 3, 1))
        self.assertEqual(result["completed_physical_logical_loads"], 3)
        self.assertEqual(result["actual_component_copy_count"], 6)
        self.assertEqual(result["profiler_aten_copy_count"], 6)
        self.assertEqual(result["actual_source_operand_logical_bytes"], 24)
        self.assertTrue(result["all_copy_content_equal"])
        self.assertEqual(
            result["final_resident_keys"],
            [["encoder", 0, 0], ["encoder", 0, 1]],
        )

    def test_resident_hit_performs_zero_copies_and_unassigned_does_not_evict(self):
        key = ("encoder", 0, 0)
        specs, sources = _payload((key,))
        result = executor.execute_independent_replay(
            (
                _event("encoder", "source", 0, 0),
                _event("encoder", "source", 1, None),
                _event("encoder", "source", 2, 0),
            ),
            specs,
            sources,
            capacity_bytes=8,
            slot_count=1,
        )
        self.assertEqual((result["hits"], result["misses"]), (1, 1))
        self.assertEqual(result["actual_component_copy_count"], 2)
        self.assertEqual(result["profiler_aten_copy_count"], 2)

    def test_invalid_payload_and_oversized_slots_fail_explicitly(self):
        key = ("encoder", 0, 0)
        specs, sources = _payload((key,))
        with self.assertRaisesRegex(ValueError, "slots exceed byte capacity"):
            executor.execute_independent_replay(
                (_event("encoder", "source", 0, 0),),
                specs,
                sources,
                capacity_bytes=8,
                slot_count=2,
                profile_copies=False,
            )
        with self.assertRaisesRegex(ValueError, "source keys"):
            executor.execute_independent_replay(
                (_event("encoder", "source", 0, 0),),
                specs,
                {},
                capacity_bytes=8,
                slot_count=1,
                profile_copies=False,
            )

    def test_comparer_uses_exact_equality_and_detects_every_frozen_perturbation(self):
        prediction, observation, payload_manifest = _matching_artifacts()
        self.assertEqual(
            validation.compare_prediction_observation(
                prediction, observation, payload_manifest
            )["outcome"],
            "AGREEMENT",
        )
        mutations = {
            "hits": 2,
            "misses": 4,
            "actual_source_operand_logical_bytes": 25,
            "profiler_aten_copy_count": 5,
            "final_resident_keys": [["encoder", 0, 99]],
            "all_copy_content_equal": False,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = dict(observation)
                changed[field] = value
                comparison = validation.compare_prediction_observation(
                    prediction, changed, payload_manifest
                )
                self.assertEqual(comparison["outcome"], "DISAGREEMENT")
                self.assertTrue(any(not check["passed"] for check in comparison["checks"]))

        changed = json.loads(json.dumps(observation))
        changed["copy_records"][0]["destination_logical_bytes"] = 5
        comparison = validation.compare_prediction_observation(
            prediction, changed, payload_manifest
        )
        self.assertEqual(comparison["outcome"], "DISAGREEMENT")

        changed = json.loads(json.dumps(observation))
        changed["loaded_key_bytes"][0]["copied_source_bytes"] = 9
        comparison = validation.compare_prediction_observation(
            prediction, changed, payload_manifest
        )
        self.assertEqual(comparison["outcome"], "DISAGREEMENT")

    def test_serialization_and_hashes_are_deterministic(self):
        left = {"z": [3, 2, 1], "a": {"value": "stable"}}
        right = {"a": {"value": "stable"}, "z": [3, 2, 1]}
        self.assertEqual(executor.canonical_json_bytes(left), executor.canonical_json_bytes(right))
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            executor.write_canonical_json(first, left)
            executor.write_canonical_json(second, right)
            self.assertEqual(executor.sha256_file(first), executor.sha256_file(second))
            self.assertEqual(json.loads(first.read_text()), left)

    def test_predictor_contains_exactly_one_canonical_simulation_call(self):
        source = inspect.getsource(validation.run_predictor)
        self.assertEqual(source.count("simulate_versioned_byte_cache("), 1)


if __name__ == "__main__":
    unittest.main()
