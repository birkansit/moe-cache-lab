import importlib.util
from pathlib import Path
import unittest

from moe_cache_lab import __version__


ROOT = Path(__file__).resolve().parents[1]
V07_HISTORICAL_RECORDS = tuple(sorted(ROOT.glob("V07_*.md")))


def _load_distribution_audit():
    path = ROOT / "scripts" / "audit_distributions.py"
    spec = importlib.util.spec_from_file_location("distribution_audit_c4", path)
    if spec is None or spec.loader is None:
        raise AssertionError("distribution audit script cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_publication_hygiene_audit():
    path = ROOT / "scripts" / "audit_publication_hygiene.py"
    spec = importlib.util.spec_from_file_location("publication_hygiene_release_surface", path)
    if spec is None or spec.loader is None:
        raise AssertionError("publication hygiene audit script cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseSurfaceHygieneTests(unittest.TestCase):
    def test_release_docs_exclude_private_process_and_local_identity_leakage(self) -> None:
        audit = _load_publication_hygiene_audit()
        result = audit.audit_repository(ROOT)
        self.assertEqual(result.violations, (), audit.format_violations(result.violations))

    def test_contributor_and_historical_docs_have_current_release_framing(self) -> None:
        contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        brief = (ROOT / "PROJECT_BRIEF.md").read_text(encoding="utf-8")
        decisions = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")

        self.assertNotIn("Release `0.4.0` pins", contributing)
        for required in (
            "current package version is `0.8.0`",
            "base package requires neither PyTorch nor Transformers",
            "built-in public Granite collection CLI",
            "does not provide a public Switch collection CLI",
            "no broad model-family compatibility claim",
            "installed-package core/offline matrix",
            "Python 3.10, 3.11, and 3.12",
            "Torch and Transformers deliberately absent",
            "`scripts/audit_core_workflow.py`",
            "focused portable core contracts",
            "full source-tree regression command",
            "Python 3.10 ML/full-regression",
            "Torch 2.12.0 CPU and Transformers 5.12.0",
        ):
            with self.subTest(document="CONTRIBUTING.md", required=required):
                self.assertIn(required, contributing)
        self.assertNotIn("source-tree test command used by the portable CI", contributing)

        self.assertNotIn("Release `0.4.0` is", brief)
        for required in (
            "current package version is **0.8.0**",
            "[`README.md`](README.md)",
            "[`V07_RELEASE_NOTES.md`](V07_RELEASE_NOTES.md)",
            "[`V08_RELEASE_NOTES.md`](V08_RELEASE_NOTES.md)",
            "## Current v0.8 contract",
            "## Historical progression: V0.1 through V0.4",
            "There is no public Switch collection CLI",
            "**MEASURED**",
            "**SIMULATED**",
            "**ESTIMATED**",
        ):
            with self.subTest(document="PROJECT_BRIEF.md", required=required):
                self.assertIn(required, brief)

        for required in (
            "historical entries remain useful provenance",
            "current source capability and claim boundaries",
            "historical statements here",
            "do not override the current source contract",
            "[`README.md`](README.md)",
            "[`V07_RELEASE_NOTES.md`](V07_RELEASE_NOTES.md)",
            "[`V08_RELEASE_NOTES.md`](V08_RELEASE_NOTES.md)",
        ):
            with self.subTest(document="DECISIONS.md", required=required):
                self.assertIn(required, decisions)

    def test_release_docs_are_self_contained_and_version_neutral(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        walkthrough = (ROOT / "WALKTHROUGH.md").read_text(encoding="utf-8")
        self.assertIn("Current package version: **0.8.0**", readme)
        self.assertIn("The repository and package include:", readme)
        self.assertIn("Release validation runs source compilation", readme)
        self.assertIn("The project does not provide:", readme)
        self.assertNotIn("current development tree", readme.lower())
        self.assertNotIn("later release gate", readme.lower())
        self.assertIn("Bounded constructed CPU copy validation", readme)
        self.assertIn("bounded constructed CPU copy validation", walkthrough)
        self.assertIn("**MEASURED test-replay observations**", walkthrough)

    def test_trace_family_and_preflight_boundary_are_explicit(self) -> None:
        compatibility = (ROOT / "COMPATIBILITY.md").read_text(encoding="utf-8")
        compatibility_normalized = " ".join(compatibility.split())
        trace_format = (ROOT / "TRACE_FORMAT.md").read_text(encoding="utf-8")
        trace_format_normalized = " ".join(trace_format.split())
        preflight = (ROOT / "PREFLIGHT.md").read_text(encoding="utf-8")

        for required in (
            "routing-trace-v1.schema.json",
            "routing-trace-v2.schema.json",
            "`(layer, expert_id)`",
            "`(routing_stage, layer, expert_id)`",
            "`assignment_state: \"assigned\"`",
            "`assignment_state: \"unassigned\"`",
            "creates zero expert-cache requests",
            "Invalid chronology is rejected, never sorted or repaired",
            "all encoder events precede all decoder events",
            "decoder-prompt position is smaller than every decoder-generated",
        ):
            with self.subTest(required=required):
                self.assertIn(required, trace_format_normalized)

        self.assertIn("public single-trace workflow supports two exact identity pairings", preflight)
        self.assertIn("canonical trace v1 with config v1/v2", preflight)
        self.assertIn("canonical trace v2 with config v3", preflight)
        self.assertIn("never flattened", preflight)
        self.assertIn(
            "Pre-flight-config v3 does not, by itself, imply routing trace v3",
            compatibility_normalized,
        )
        self.assertNotIn("future pre-flight-config v3", compatibility.lower())

    def test_v07_history_and_current_v2_preflight_are_chronologically_distinct(self) -> None:
        release_notes = (ROOT / "V07_RELEASE_NOTES.md").read_text(encoding="utf-8")
        brief = (ROOT / "PROJECT_BRIEF.md").read_text(encoding="utf-8")
        preflight = (ROOT / "PREFLIGHT.md").read_text(encoding="utf-8")
        walkthrough = (ROOT / "WALKTHROUGH.md").read_text(encoding="utf-8")

        self.assertIn("`simulate_versioned_byte_cache()`", release_notes)
        self.assertIn("`(routing_stage, layer, expert_id)`", release_notes)
        self.assertIn("`(routing_stage, layer_id, expert_id)`", preflight)
        for text in (release_notes, preflight):
            self.assertIn("unassigned", text)
            self.assertIn("zero", text)

        for text in (release_notes, brief, preflight, walkthrough):
            self.assertIn("stage-qualified", text)
            self.assertIn("v2", text)
            self.assertIn("pre-flight", text)

        self.assertIn("remains v1-only", release_notes)
        self.assertIn("Single-trace v2 pre-flight uses stage-qualified config v3", brief)
        self.assertIn(
            "Published v0.7 history remains",
            (ROOT / "DECISIONS.md").read_text(encoding="utf-8"),
        )
        self.assertIn("canonical trace v2 with config v3", preflight)
        self.assertIn("config v3 is rejected for trace v1", walkthrough)

        self.assertIn("public v2 pre-flight workflow", " ".join(release_notes.split()))
        self.assertIn("stage-qualified fixture", " ".join(preflight.split()))
        self.assertNotIn("does not add v2 cache simulation", release_notes)
        self.assertNotIn("without adding a v2 cache", brief)
        self.assertNotIn("a v2 cache/preflight pipeline", walkthrough)

    def test_v07_records_remove_internal_coordination_but_preserve_findings(self) -> None:
        preserved = {
            "V07_SECOND_MODEL_GATE.md": (
                "**NO-GO for a second collector",
                "92fe2d22b024d9937146fe097ba3d3a7ba146e1b",
            ),
            "V07_TRACE_V2_DESIGN.md": (
                "`(routing_stage, layer, expert_id)`",
                "No unknown permits fabrication, omission, stage flattening, or chronology",
            ),
            "V07_SWITCH_SMOKE.md": (
                "Status: **PASS",
                "**SIMULATED** result",
            ),
            "V07_RUNTIME_VALIDATION_GATE.md": (
                "Status: **GO",
                "**NO-GO** for cache/transfer validation",
            ),
            "V07_RUNTIME_VALIDATION_RESULT.md": (
                "Status: **AGREEMENT**",
                "**MEASURED test-replay copy observations**",
            ),
        }
        for name, phrases in preserved.items():
            text = (ROOT / name).read_text(encoding="utf-8")
            normalized = " ".join(text.split())
            for phrase in phrases:
                with self.subTest(path=name, phrase=phrase):
                    self.assertIn(phrase, normalized)

    def test_distribution_audit_requires_both_schemas_and_release_docs(self) -> None:
        audit = _load_distribution_audit()
        for schema in (
            "moe_cache_lab/schemas/routing-trace-v1.schema.json",
            "moe_cache_lab/schemas/routing-trace-v2.schema.json",
        ):
            self.assertIn(schema, audit.REQUIRED_PACKAGE_MODULES)
            self.assertIn(f"src/{schema}", audit.REQUIRED_SDIST_PATHS)
        for document in (
            "README.md",
            "WALKTHROUGH.md",
            "PREFLIGHT.md",
            "TRACE_FORMAT.md",
            "V07_RELEASE_NOTES.md",
        ):
            self.assertIn(document, audit.REQUIRED_RELEASE_DOCS)
            self.assertIn(document, audit.REQUIRED_SDIST_PATHS)

        audit_source = (ROOT / "scripts" / "audit_distributions.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("wheel is missing required package modules", audit_source)
        self.assertNotIn("wheel is missing required v0.6 modules", audit_source)
        self.assertIn("scripts/audit_publication_hygiene.py", audit.REQUIRED_SDIST_PATHS)
        self.assertIn(
            "include scripts/audit_publication_hygiene.py",
            (ROOT / "MANIFEST.in").read_text(encoding="utf-8"),
        )
        self.assertEqual(__version__, "0.8.0")
        self.assertIn('version = "0.8.0"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
