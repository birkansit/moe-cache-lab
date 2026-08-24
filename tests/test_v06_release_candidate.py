import importlib.util
import re
from pathlib import Path
import unittest

from moe_cache_lab import __version__
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


class V06ReleaseTests(unittest.TestCase):
    def test_version_release_notes_and_lifecycle_schema_are_consistent(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        notes = (ROOT / "V06_RELEASE_NOTES.md").read_text(encoding="utf-8")
        self.assertEqual(__version__, "0.8.0")
        self.assertIn('version = "0.8.0"', pyproject)
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
        self.assertNotIn("independent peer review", notes.lower())
        self.assertNotIn("independently " + "reviewed", notes.lower())

    def test_release_surface_preserves_v1_v2_and_separates_evidence_classes(self) -> None:
        preflight = (ROOT / "PREFLIGHT.md").read_text(encoding="utf-8")
        output = (ROOT / "src" / "moe_cache_lab" / "preflight_output.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Established pre-flight analysis/config formats 1 and 2", preflight)
        self.assertIn('"cache_lifecycle_scenarios": "SIMULATED"', output)
        self.assertIn('"descriptive_sensitivity_summaries": "SIMULATED"', output)
        self.assertIn('"serialized_h2d_transfer_service": "ESTIMATED"', output)
        self.assertIn("serialized/no-overlap", preflight)
        self.assertIn("caller-supplied", preflight)

    def test_distribution_audit_and_ci_cover_v06_artifacts_and_modules(self) -> None:
        audit = _load_distribution_audit()
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertEqual(audit._validate_version("0.6.0"), "0.6.0")
        self.assertEqual(audit._validate_version("0.6.0rc1"), "0.6.0rc1")
        with self.assertRaisesRegex(ValueError, "canonical release or rc version"):
            audit._validate_version("v0.6")
        for module in (
            "byte_cache.py",
            "hardware_cost.py",
            "preflight.py",
            "preflight_output.py",
            "sensitivity_summary.py",
            "trace.py",
            "routing-trace-v1.schema.json",
        ):
            self.assertTrue(any(path.endswith(module) for path in audit.REQUIRED_PACKAGE_MODULES))
        self.assertIn("V06_RELEASE_NOTES.md", audit.REQUIRED_RELEASE_DOCS)
        self.assertIn("TRACE_FORMAT.md", audit.REQUIRED_RELEASE_DOCS)
        self.assertIn("include TRACE_FORMAT.md", manifest)
        self.assertIn("include V06_RELEASE_NOTES.md", manifest)
        self.assertIn("schemas *.json", manifest)
        self.assertIn("scripts/audit_distributions.py", manifest)
        self.assertIn("moe_cache_lab-0.8.0-py3-none-any.whl", ci)
        self.assertIn("moe_cache_lab-0.8.0.tar.gz", ci)
        self.assertIn("scripts/audit_distributions.py", ci)

    def test_v06_release_facing_files_have_no_private_process_or_identity_leakage(self) -> None:
        release_facing = (
            ROOT / "README.md",
            ROOT / "PREFLIGHT.md",
            ROOT / "V06_RELEASE_NOTES.md",
            ROOT / "examples" / "no-download-preflight" / "README.md",
            ROOT / "examples" / "no-download-preflight" / "preflight-config.json",
            ROOT / "examples" / "no-download-preflight" / "trace.jsonl",
        )
        email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        local_path_patterns = (
            re.compile(r"\b[A-Za-z]:[\\/]"),
            re.compile(r"/(?:home|Users|tmp)/"),
        )
        forbidden_patterns = tuple(
            re.compile(pattern)
            for pattern in (
                r"\bdirector\b",
                r"\bworker(?:_impl)?\b",
                r"\breviewer " + r"agent\b",
                r"\bchain of " + r"thought\b",
                r"\bsystem " + r"prompt\b",
                r"\bdeveloper " + r"prompt\b",
                r"\borchestration " + r"transcript\b",
                r"\bindependent(?:ly)? peer review(?:ed)?\b",
            )
        )
        secret_markers = ("api_key", "access_token", "secret_token", "bearer ")

        for path in release_facing:
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            with self.subTest(path=path.name):
                self.assertIsNone(email_pattern.search(text))
                for pattern in local_path_patterns + forbidden_patterns:
                    self.assertIsNone(pattern.search(lowered))
                for marker in secret_markers:
                    self.assertNotIn(marker, lowered)


if __name__ == "__main__":
    unittest.main()
