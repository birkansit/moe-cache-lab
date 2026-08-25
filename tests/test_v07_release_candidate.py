import re
from pathlib import Path
import unittest

from moe_cache_lab import __version__


ROOT = Path(__file__).resolve().parents[1]


class V07ReleaseCandidateTests(unittest.TestCase):
    def test_current_version_and_release_notes_are_consistent(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        notes = (ROOT / "V07_RELEASE_NOTES.md").read_text(encoding="utf-8")
        notes_normalized = " ".join(notes.split())

        self.assertEqual(__version__, "0.9.0")
        self.assertIn('version = "0.9.0"', pyproject)
        self.assertIn("Current package version: **0.9.0**", readme)
        self.assertIn("Current package version: **0.7.0**", notes)
        self.assertIn("V07_RELEASE_NOTES.md", readme)

        for required in (
            "Canonical trace v1 and v2 coexistence",
            "`(layer, expert_id)`",
            "`(routing_stage, layer, expert_id)`",
            "reject invalid order without sorting or repair",
            "base package has no required PyTorch or Transformers dependency",
            "`[granite]`",
            "`[switch]`",
            "SwitchTransformers",
            "configured-universe normalized entropy",
            "caller-explicit cumulative top-k selection shares",
            "representativeness score",
            "MEASURED test-replay",
            "Experiment bundles",
            "canonical public CLI",
            "WALKTHROUGH.md",
            "remains v1-only",
            "SIMULATED",
            "ESTIMATED",
            "not an inference accelerator",
        ):
            with self.subTest(required=required):
                self.assertIn(required, notes_normalized)

        for forbidden in (
            "moe-cache-lab" + "-v05-dev",
            "[" + "READY]",
            "[" + "ACTIVE]",
            "[" + "COMPLETED]",
            "worker" + "_impl",
            "independently " + "reviewed",
            "has been publicly released",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden.lower(), notes.lower())
        self.assertIsNone(re.search(r"\b[A-Za-z]:[\\/]", notes))

    def test_release_notes_preserve_switch_and_bundle_boundaries(self) -> None:
        notes = (ROOT / "V07_RELEASE_NOTES.md").read_text(encoding="utf-8")
        notes_normalized = " ".join(notes.split())

        for required in (
            "public Granite collection command uses the `[granite]` extra",
            "`[switch]` extra isolates dependencies for the narrow reviewed "
            "Switch research collector/path",
            "no public Switch collection CLI is provided",
            "structurally and observationally exercised on one pinned "
            "Transformers SwitchTransformers family/path",
            "bounded config, tool, and environment provenance",
            "SHA-256 integrity anchors",
            "trace may be explicitly embedded",
            "external reference plus SHA-256",
            "External references are metadata-only and are never fetched by "
            "bundle creation or verification",
            "Bundle integrity does not upgrade scientific evidence quality or "
            "change an artifact's evidence class",
            "collect with a validated collector or import a canonical trace -> "
            "analyze offline -> compare/report only where that trace/config "
            "contract supports it -> optionally create/verify an experiment bundle",
        ):
            with self.subTest(required=required):
                self.assertIn(required, notes_normalized)

    def test_v07_release_document_is_in_both_distributions(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        audit = (ROOT / "scripts" / "audit_distributions.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"V07_RELEASE_NOTES.md",', pyproject)
        self.assertIn("include V07_RELEASE_NOTES.md", manifest)
        self.assertGreaterEqual(audit.count('"V07_RELEASE_NOTES.md",'), 2)

    def test_ci_pins_current_artifacts_and_keeps_v07_core_contract(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        audit = (ROOT / "scripts" / "audit_core_workflow.py").read_text(
            encoding="utf-8"
        )
        combined = ci + "\n" + audit
        for required in (
            "moe_cache_lab-0.9.0-py3-none-any.whl",
            "moe_cache_lab-0.9.0.tar.gz",
            "--version 0.9.0",
            "find_spec('torch') is None",
            "find_spec('transformers') is None",
            "load_trace_schema",
            "load_trace_v2_schema",
            "artifacts/trace.jsonl",
            "no-download-stage-qualified-preflight",
            "stage-qualified-preflight-report.md",
            "stage-qualified-preflight-report.json",
            "bundle-create",
            "bundle-verify",
            "V07_RELEASE_NOTES.md",
            "python -m pip check",
        ):
            with self.subTest(required=required):
                self.assertIn(required, combined)


if __name__ == "__main__":
    unittest.main()
