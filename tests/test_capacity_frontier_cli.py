import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from moe_cache_lab import cli


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "examples" / "external-producer-no-model"
V1_TRACE = ROOT / "examples" / "no-download-preflight" / "trace.jsonl"
V1_CONFIG = (
    ROOT / "examples" / "no-download-preflight" / "preflight-config.json"
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _cli(arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "moe_cache_lab.cli", *arguments],
        cwd=cwd,
        env=_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class CapacityFrontierCliTests(unittest.TestCase):
    def _produce_v2(self, root: Path) -> Path:
        trace = root / "trace.jsonl"
        completed = subprocess.run(
            [sys.executable, str(EXTERNAL / "producer.py"), "--output", str(trace)],
            cwd=root,
            env=_environment(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        return trace

    def test_help_exposes_complete_command_surface(self) -> None:
        completed = _cli(["capacity-frontier", "--help"], cwd=ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        help_text = completed.stdout.decode("utf-8")
        self.assertIn("trace", help_text)
        self.assertIn("--preflight-config", help_text)
        self.assertIn("--output", help_text)
        self.assertIn("--json-output", help_text)

    def test_documented_external_producer_command_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._produce_v2(root)
            first_md = root / "frontier-1.md"
            first_json = root / "frontier-1.json"
            second_md = root / "frontier-2.md"
            second_json = root / "frontier-2.json"
            for markdown, machine in (
                (first_md, first_json),
                (second_md, second_json),
            ):
                completed = _cli(
                    [
                        "capacity-frontier",
                        str(trace),
                        "--preflight-config",
                        str(EXTERNAL / "preflight-config.json"),
                        "--output",
                        str(markdown),
                        "--json-output",
                        str(machine),
                    ],
                    cwd=root,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr.decode())
                self.assertEqual(completed.stdout, b"")
                self.assertEqual(completed.stderr, b"")

            self.assertEqual(first_md.read_bytes(), second_md.read_bytes())
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            data = json.loads(first_json.read_text(encoding="utf-8"))
            self.assertEqual(data["format"], "moe-cache-lab.capacity-frontier")
            self.assertEqual(data["format_version"], 1)
            self.assertEqual(data["trace"]["format_version"], 2)
            self.assertIsNotNone(data["byte_frontier"])
            self.assertIn("SIMULATED", first_md.read_text(encoding="utf-8"))
            self.assertEqual(
                hashlib.sha256(first_json.read_bytes()).hexdigest(),
                hashlib.sha256(second_json.read_bytes()).hexdigest(),
            )

    def test_stdout_is_human_report_and_count_mode_needs_no_config(self) -> None:
        completed = _cli(["capacity-frontier", str(V1_TRACE)], cwd=ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stderr, b"")
        report = completed.stdout.decode("utf-8")
        self.assertIn("# Exact event-atomic LRU capacity frontier", report)
        self.assertIn("Count-capacity frontier", report)
        self.assertNotIn("Byte-capacity frontier", report)

    def test_missing_malformed_and_invalid_chronology_are_clean_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.jsonl"
            malformed.write_text("{not json}\n", encoding="utf-8")
            invalid = root / "invalid-chronology.jsonl"
            records = V1_TRACE.read_text(encoding="utf-8").splitlines()
            invalid.write_text(
                "\n".join((records[0], records[2], records[1], *records[3:])) + "\n",
                encoding="utf-8",
            )
            for trace in (root / "missing.jsonl", malformed, invalid):
                with self.subTest(trace=trace.name):
                    completed = _cli(["capacity-frontier", str(trace)], cwd=root)
                    self.assertEqual(completed.returncode, 2)
                    stderr = completed.stderr.decode("utf-8")
                    self.assertIn("trace input error", stderr)
                    self.assertNotIn("Traceback", stderr)

    def test_incompatible_configs_fail_before_creating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v2_trace = self._produce_v2(root)
            cases = (
                (V1_TRACE, EXTERNAL / "preflight-config.json", "stage-qualified"),
                (v2_trace, V1_CONFIG, "requires stage-qualified"),
            )
            for index, (trace, config, expected) in enumerate(cases):
                markdown = root / f"failure-{index}.md"
                machine = root / f"failure-{index}.json"
                completed = _cli(
                    [
                        "capacity-frontier",
                        str(trace),
                        "--preflight-config",
                        str(config),
                        "--output",
                        str(markdown),
                        "--json-output",
                        str(machine),
                    ],
                    cwd=root,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(expected, completed.stderr.decode("utf-8"))
                self.assertNotIn("Traceback", completed.stderr.decode("utf-8"))
                self.assertFalse(markdown.exists())
                self.assertFalse(machine.exists())

    def test_missing_referenced_size_is_clean_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._produce_v2(root)
            payload = json.loads(
                (EXTERNAL / "preflight-config.json").read_text(encoding="utf-8")
            )
            payload["expert_sizes"] = []
            config = root / "missing-sizes.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            markdown = root / "failure.md"
            machine = root / "failure.json"
            completed = _cli(
                [
                    "capacity-frontier",
                    str(trace),
                    "--preflight-config",
                    str(config),
                    "--output",
                    str(markdown),
                    "--json-output",
                    str(machine),
                ],
                cwd=root,
            )
            self.assertEqual(completed.returncode, 2)
            stderr = completed.stderr.decode("utf-8")
            self.assertIn("missing referenced canonical keys", stderr)
            self.assertNotIn("Traceback", stderr)
            self.assertFalse(markdown.exists())
            self.assertFalse(machine.exists())

    def test_invalid_byte_size_is_a_clean_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._produce_v2(root)
            payload = json.loads(
                (EXTERNAL / "preflight-config.json").read_text(encoding="utf-8")
            )
            payload["expert_sizes"][0]["size_bytes"] = 0
            config = root / "invalid-size.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "failure.md"
            completed = _cli(
                [
                    "capacity-frontier",
                    str(trace),
                    "--preflight-config",
                    str(config),
                    "--output",
                    str(output),
                ],
                cwd=root,
            )
            self.assertEqual(completed.returncode, 2)
            stderr = completed.stderr.decode("utf-8")
            self.assertIn("preflight config input error", stderr)
            self.assertIn("positive integer", stderr)
            self.assertNotIn("Traceback", stderr)
            self.assertFalse(output.exists())

    def test_unexpected_programmer_failure_is_not_blanket_caught(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["moe-cache-lab", "capacity-frontier", str(V1_TRACE)],
        ), mock.patch(
            "moe_cache_lab.capacity_frontier.analyze_capacity_frontier",
            side_effect=AssertionError("programmer defect"),
        ), mock.patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaisesRegex(AssertionError, "programmer defect"):
                cli.main()


if __name__ == "__main__":
    unittest.main()
