from contextlib import redirect_stdout
from io import StringIO
import importlib.util
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

from moe_cache_lab import __version__
from moe_cache_lab.cli import main
from moe_cache_lab.preflight_output import PREFLIGHT_LIFECYCLE_VERSION


ROOT = Path(__file__).resolve().parents[1]


def _load_distribution_audit():
    path = ROOT / "scripts" / "audit_distributions.py"
    spec = importlib.util.spec_from_file_location("distribution_audit", path)
    if spec is None or spec.loader is None:
        raise AssertionError("distribution audit script cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V06PublicReleaseTests(unittest.TestCase):
    def test_version_release_notes_and_lifecycle_schema_are_consistent(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        notes = (ROOT / "V06_RELEASE_NOTES.md").read_text(encoding="utf-8")
        self.assertEqual(__version__, "0.6.0")
        self.assertIn('version = "0.6.0"', pyproject)
        self.assertIn("Current package version: **0.6.0**", notes)
        self.assertEqual(PREFLIGHT_LIFECYCLE_VERSION, 3)

        for implemented in (
            "Modeled transfer-operation sensitivity",
            "Cache lifecycle simulation",
            "Descriptive sensitivity summaries",
            "Lifecycle transfer-service matrix",
            "Compatibility",
        ):
            self.assertIn(implemented, notes)
        for boundary in (
            "MEASURED only where trace provenance",
            "SIMULATED",
            "ESTIMATED",
            "modeled assumptions",
            "physical transfer granularity",
            "physical expert residency",
            "end-to-end latency",
            "throughput",
            "tokens/sec",
            "speedup",
            "not an inference accelerator",
        ):
            self.assertIn(boundary, notes)

    def test_public_cli_adds_lifecycle_without_private_commands(self) -> None:
        output = StringIO()
        with patch.object(sys, "argv", ["moe-cache-lab", "--help"]):
            with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("analyze-lifecycle", help_text)
        for forbidden in (
            "collect-stage1",
            "validate-stage1",
            "benchmark-stage1",
            "_collect-stage1-child",
            "collect-v04",
            "validate-v04",
            "_collect-v04-child",
        ):
            self.assertNotIn(forbidden, help_text)

    def test_generic_distribution_audit_and_ci_cover_public_v06(self) -> None:
        audit = _load_distribution_audit()
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertEqual(audit._validate_version("0.6.0"), "0.6.0")
        self.assertEqual(audit._validate_version("0.6.0rc1"), "0.6.0rc1")
        for module in (
            "byte_cache.py",
            "hardware_cost.py",
            "preflight.py",
            "preflight_output.py",
            "sensitivity_summary.py",
        ):
            self.assertTrue(
                any(path.endswith(module) for path in audit.REQUIRED_PACKAGE_MODULES)
            )
        for document in ("V05_VALIDATION.md", "V06_RELEASE_NOTES.md"):
            self.assertIn(document, audit.REQUIRED_SDIST_PATHS)
            self.assertIn(f"include {document}", manifest)
        self.assertIn("scripts/audit_distributions.py", manifest)
        self.assertIn("moe_cache_lab-0.6.0-py3-none-any.whl", ci)
        self.assertIn("moe_cache_lab-0.6.0.tar.gz", ci)
        self.assertIn("scripts/audit_distributions.py", ci)
        self.assertNotIn("audit_v05_distributions.py", ci)

    def test_public_release_surface_has_no_private_development_leakage(self) -> None:
        release_facing = (
            ROOT / "README.md",
            ROOT / "PREFLIGHT.md",
            ROOT / "V06_RELEASE_NOTES.md",
            ROOT / "examples" / "no-download-preflight" / "README.md",
            ROOT / "examples" / "no-download-preflight" / "preflight-config.json",
            ROOT / "examples" / "no-download-preflight" / "trace.jsonl",
        )
        email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        forbidden_patterns = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\b[A-Za-z]:[\\/]",
                r"/(?:home|Users|tmp)/",
            )
        )
        for path in release_facing:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIsNone(email_pattern.search(text))
                for pattern in forbidden_patterns:
                    self.assertIsNone(pattern.search(text))


if __name__ == "__main__":
    unittest.main()
