import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.audit_distributions import (
    EXTERNAL_PRODUCER_EXAMPLE_FILES,
    REQUIRED_SDIST_PATHS,
)
from moe_cache_lab.preflight_config import PreflightConfigV3, read_preflight_config
from moe_cache_lab.trace_v2 import RoutingTraceV2, read_versioned_trace


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "external-producer-no-model"
V04 = (
    ROOT
    / "results"
    / "v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980"
)
EXPECTED_OLD_HASHES = {
    "v1-markdown": "219751d3a63b97177f6d3133dd2e7b69fc941eed2416e72568125f4e2936542d",
    "v1-json": "464638a63f66e2c8797f4e7c59efea226f30050032ede5138255f675d0ff4a42",
    "b5-markdown": "9e9543cc7f126138e68c0b7ab3b072511215097c3d70e42745d23290bfa74ba3",
    "b5-json": "387bdb273cd90a38577e63b7f82c1791988f10778d6ea0d18c16ca995bfb4599",
    "v04-result": "ed857ef581f80fbbe8e694bc869e9265e785e5c48bd6e00118b20db7d85e7b72",
    "v04-report": "d9b5399a125d162c3c7cdb3f847f30c196b56fd31ace9ee714f9dd4d5b81ceaf",
}


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _run(arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _cli(arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return _run([sys.executable, "-m", "moe_cache_lab.cli", *arguments], cwd=cwd)


def _expected_hashes() -> dict[str, str]:
    return {
        filename: digest
        for digest, filename in (
            line.split(maxsplit=1)
            for line in (EXAMPLE / "expected.sha256")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }


class ExternalProducerWorkflowTests(unittest.TestCase):
    def _produce(self, root: Path, name: str = "trace.jsonl") -> Path:
        output = root / "artifacts" / name
        completed = _run(
            [sys.executable, str(EXAMPLE / "producer.py"), "--output", str(output)],
            cwd=root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        return output

    def _generate_workflow_artifacts(self, root: Path) -> dict[str, Path]:
        artifacts = root / "artifacts"
        trace = self._produce(root)

        human = _cli(["validate-trace", str(trace)], cwd=root)
        self.assertEqual(human.returncode, 0, human.stderr.decode())
        self.assertEqual(human.stderr, b"")
        human_lines = human.stdout.decode("utf-8").splitlines()
        self.assertIn("Trace validation: VALID", human_lines)
        self.assertIn("Trace format version: 2", human_lines)

        machine = _cli(["validate-trace", str(trace), "--json"], cwd=root)
        self.assertEqual(machine.returncode, 0, machine.stderr.decode())
        self.assertEqual(machine.stderr, b"")
        validation = artifacts / "validation.json"
        validation.write_bytes(machine.stdout.replace(b"\r\n", b"\n"))

        routing_markdown = artifacts / "routing-analysis.md"
        routing_json = artifacts / "routing-analysis.json"
        descriptive = _cli(
            [
                "analyze",
                str(trace),
                "--workload-id",
                "external-producer-demo",
                "--output",
                str(routing_markdown),
                "--json-output",
                str(routing_json),
            ],
            cwd=root,
        )
        self.assertEqual(descriptive.returncode, 0, descriptive.stderr.decode())

        preflight_markdown = artifacts / "preflight-report.md"
        preflight_json = artifacts / "preflight-report.json"
        preflight = _cli(
            [
                "analyze",
                str(trace),
                "--preflight-config",
                str(EXAMPLE / "preflight-config.json"),
                "--workload-id",
                "external-producer-demo",
                "--output",
                str(preflight_markdown),
                "--json-output",
                str(preflight_json),
            ],
            cwd=root,
        )
        self.assertEqual(preflight.returncode, 0, preflight.stderr.decode())
        return {
            "artifacts/trace.jsonl": trace,
            "artifacts/validation.json": validation,
            "artifacts/routing-analysis.md": routing_markdown,
            "artifacts/routing-analysis.json": routing_json,
            "artifacts/preflight-report.md": preflight_markdown,
            "artifacts/preflight-report.json": preflight_json,
        }

    def test_producer_is_standalone_stdlib_only_and_byte_deterministic(self) -> None:
        source = (EXAMPLE / "producer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.add((node.module or "").split(".", 1)[0])
        self.assertEqual(imported, {"argparse", "json", "pathlib"})
        self.assertTrue(imported.isdisjoint({"moe_cache_lab", "torch", "transformers"}))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._produce(root, "first.jsonl").read_bytes()
            second = self._produce(root, "second.jsonl").read_bytes()
        self.assertEqual(first, second)
        self.assertNotIn(str(ROOT).encode(), first)

    def test_generated_trace_is_stage_safe_and_preserves_unassigned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = read_versioned_trace(self._produce(Path(temporary)))

        self.assertIsInstance(trace, RoutingTraceV2)
        self.assertEqual(
            tuple(profile.routing_stage for profile in trace.routing_stages),
            ("encoder", "decoder"),
        )
        self.assertIn(("encoder", 0, 1), trace.expert_requests)
        self.assertIn(("decoder", 0, 1), trace.expert_requests)
        unassigned = next(
            event for event in trace.events if event.assignment_state == "unassigned"
        )
        self.assertEqual(unassigned.selected_experts, ())
        self.assertEqual(unassigned.selected_probabilities, ())
        self.assertEqual(unassigned.expert_requests, ())

    def test_complete_offline_workflow_reproduces_hashes_and_boundaries(self) -> None:
        expected = _expected_hashes()
        self.assertEqual(
            set(expected),
            {
                "artifacts/trace.jsonl",
                "artifacts/validation.json",
                "artifacts/routing-analysis.md",
                "artifacts/routing-analysis.json",
                "artifacts/preflight-report.md",
                "artifacts/preflight-report.json",
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = self._generate_workflow_artifacts(root)
            for name, path in outputs.items():
                with self.subTest(name=name):
                    self.assertEqual(
                        hashlib.sha256(path.read_bytes()).hexdigest(), expected[name]
                    )

            validation = json.loads(outputs["artifacts/validation.json"].read_bytes())
            routing = json.loads(outputs["artifacts/routing-analysis.json"].read_bytes())
            preflight = json.loads(outputs["artifacts/preflight-report.json"].read_bytes())

            self.assertTrue(validation["valid"])
            self.assertEqual(validation["trace_format_version"], 2)
            self.assertEqual(validation["summary"]["routing_stages"], ["encoder", "decoder"])
            self.assertEqual(
                (validation["summary"]["assigned_event_count"], validation["summary"]["unassigned_event_count"]),
                (4, 1),
            )
            self.assertEqual(routing["workload_id"], "external-producer-demo")
            self.assertIsNone(routing["locality"])
            self.assertEqual(routing["claim_boundary"]["cache"], "No cache simulation is run by this v2 report.")
            self.assertEqual(
                preflight["workload_byte_context"]["referenced_expert_keys"],
                [
                    {"routing_stage": "encoder", "layer_id": 0, "expert_id": 1},
                    {"routing_stage": "decoder", "layer_id": 0, "expert_id": 1},
                    {"routing_stage": "decoder", "layer_id": 0, "expert_id": 2},
                ],
            )
            self.assertEqual(preflight["claim_boundary"]["cache_outcomes"], "SIMULATED")
            self.assertEqual(preflight["claim_boundary"]["transfer_service"], "ESTIMATED")
            self.assertEqual(
                preflight["claim_boundary"]["runtime_performance_and_residency"],
                "NOT ESTABLISHED",
            )
            cells = {
                (row["capacity_bytes"], row["policy"]): (row["hits"], row["misses"])
                for row in preflight["simulated_byte_cache_sensitivity"]
            }
            self.assertEqual(cells[(5, "lru")], (0, 4))
            self.assertEqual(cells[(9, "lru")], (1, 3))

            bundle = root / "artifacts" / "bundle"
            created = _cli(
                [
                    "bundle-create",
                    "--experiment-id",
                    "external-producer-no-model",
                    "--config",
                    str(EXAMPLE / "preflight-config.json"),
                    "--report-json",
                    str(outputs["artifacts/preflight-report.json"]),
                    "--report-markdown",
                    str(outputs["artifacts/preflight-report.md"]),
                    "--embed-trace",
                    "synthetic-external",
                    str(outputs["artifacts/trace.jsonl"]),
                    "--output-dir",
                    str(bundle),
                ],
                cwd=root,
            )
            self.assertEqual(created.returncode, 0, created.stderr.decode())
            verified = _cli(["bundle-verify", str(bundle)], cwd=root)
            self.assertEqual(verified.returncode, 0, verified.stderr.decode())
            self.assertEqual(json.loads(created.stdout), json.loads(verified.stdout))
            self.assertEqual(json.loads(created.stdout)["artifact_count"], 5)
            self.assertEqual(json.loads(created.stdout)["trace_count"], 1)

            for path in bundle.rglob("*"):
                if path.is_file():
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn(str(ROOT), text)
                    self.assertNotIn("moe-cache-lab" + "-v05-dev", text)

    def test_config_docs_and_distribution_contract_preserve_claim_boundaries(self) -> None:
        config = read_preflight_config(EXAMPLE / "preflight-config.json")
        self.assertIsInstance(config, PreflightConfigV3)
        self.assertEqual(config.expert_size_map()[("encoder", 0, 1)], 3)
        self.assertEqual(config.expert_size_map()[("decoder", 0, 1)], 5)
        self.assertEqual(config.capacities_bytes, (5, 9))
        self.assertEqual(config.policies, ("lru", "lfu"))

        readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
        for required in (
            "SYNTHETIC demonstration",
            "canonical-file validity only",
            "SIMULATED",
            "ESTIMATED",
            "NOT ESTABLISHED",
            "does not claim one universal",
            "does not need `ProducerResult`",
        ):
            self.assertIn(required, readme)

        self.assertEqual(
            set(EXTERNAL_PRODUCER_EXAMPLE_FILES),
            {"README.md", "expected.sha256", "preflight-config.json", "producer.py"},
        )
        for name in EXTERNAL_PRODUCER_EXAMPLE_FILES:
            self.assertIn(
                f"examples/external-producer-no-model/{name}",
                REQUIRED_SDIST_PATHS,
            )

    def test_six_previous_hash_anchors_remain_exact(self) -> None:
        v1 = {
            name: digest
            for digest, name in (
                line.split(maxsplit=1)
                for line in (ROOT / "examples/no-download-preflight/expected.sha256")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        b5 = {
            name: digest
            for digest, name in (
                line.split(maxsplit=1)
                for line in (
                    ROOT / "examples/no-download-stage-qualified-preflight/expected.sha256"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        self.assertEqual(v1["preflight-report.md"], EXPECTED_OLD_HASHES["v1-markdown"])
        self.assertEqual(v1["preflight-report.json"], EXPECTED_OLD_HASHES["v1-json"])
        self.assertEqual(
            b5["stage-qualified-preflight-report.md"],
            EXPECTED_OLD_HASHES["b5-markdown"],
        )
        self.assertEqual(
            b5["stage-qualified-preflight-report.json"],
            EXPECTED_OLD_HASHES["b5-json"],
        )
        self.assertEqual(
            hashlib.sha256((V04 / "v04-result.json").read_bytes()).hexdigest(),
            EXPECTED_OLD_HASHES["v04-result"],
        )
        self.assertEqual(
            hashlib.sha256((V04 / "v04-report.md").read_bytes()).hexdigest(),
            EXPECTED_OLD_HASHES["v04-report"],
        )

    def test_core_only_ci_runs_the_packaged_installed_wheel_workflow(self) -> None:
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        audit = (ROOT / "scripts/audit_core_workflow.py").read_text(encoding="utf-8")
        combined = ci + "\n" + audit
        self.assertIn("scripts/audit_core_workflow.py", ci)
        for required in (
            '"external-producer-no-model"',
            '"producer.py", "--output", "artifacts/trace.jsonl"',
            '"validate-trace", "artifacts/trace.jsonl", "--json"',
            '"external-producer-demo"',
            '"bundle-verify", "artifacts/bundle"',
            "find_spec('torch') is None",
            "find_spec('transformers') is None",
        ):
            with self.subTest(required=required):
                self.assertIn(required, combined)


if __name__ == "__main__":
    unittest.main()
