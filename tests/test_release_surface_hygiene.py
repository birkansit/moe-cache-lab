import importlib.util
from pathlib import Path
import re
import unittest

from moe_cache_lab import __version__


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DOCS = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "PROJECT_BRIEF.md",
    ROOT / "DECISIONS.md",
    ROOT / "WALKTHROUGH.md",
    ROOT / "PREFLIGHT.md",
    ROOT / "TRACE_FORMAT.md",
    ROOT / "V06_RELEASE_NOTES.md",
    ROOT / "V07_RELEASE_NOTES.md",
    ROOT / "examples" / "no-download-preflight" / "README.md",
)


def _load_distribution_audit():
    path = ROOT / "scripts" / "audit_distributions.py"
    spec = importlib.util.spec_from_file_location("distribution_audit_c4", path)
    if spec is None or spec.loader is None:
        raise AssertionError("distribution audit script cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseSurfaceHygieneTests(unittest.TestCase):
    def test_release_docs_exclude_private_process_and_local_identity_leakage(self) -> None:
        email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        forbidden_patterns = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"moe-cache-lab-v05-dev",
                r"\b[A-Za-z]:[\\/](?:Users|Projects)[\\/]",
                r"/(?:home|Users)/",
                r"\[(?:READY|ACTIVE|COMPLETED)\]",
                r"\bworker(?:_impl)?\b",
                r"\bdirector\b",
                r"\bCodex\b",
                r"\breviewer agent\b",
                r"\borchestration transcript\b",
                r"\bindependently reviewed\b",
                r"\bIssue #54\b",
                r"\baccess_token\b",
                r"\bapi_key\b",
                r"\bbearer\s+[A-Za-z0-9._~-]+",
                r"\bghp_[A-Za-z0-9]+\b",
                r"\bgithub_pat_[A-Za-z0-9_]+\b",
                r"\bAKIA[0-9A-Z]{16}\b",
                r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY",
            )
        )

        for path in RELEASE_DOCS:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                self.assertIsNone(email_pattern.search(text))
                for pattern in forbidden_patterns:
                    self.assertIsNone(pattern.search(text))

    def test_contributor_and_historical_docs_have_current_v07_framing(self) -> None:
        contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        brief = (ROOT / "PROJECT_BRIEF.md").read_text(encoding="utf-8")
        decisions = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")

        self.assertNotIn("Release `0.4.0` pins", contributing)
        for required in (
            "current package version is `0.7.0`",
            "base package requires neither PyTorch nor Transformers",
            "built-in public Granite collection CLI",
            "does not provide a public Switch collection CLI",
            "no broad model-family compatibility claim",
        ):
            with self.subTest(document="CONTRIBUTING.md", required=required):
                self.assertIn(required, contributing)

        self.assertNotIn("Release `0.4.0` is", brief)
        for required in (
            "current package version is **0.7.0**",
            "[`README.md`](README.md)",
            "[`V07_RELEASE_NOTES.md`](V07_RELEASE_NOTES.md)",
            "## Current v0.7 contract",
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
            "current v0.7 user-facing capability and claim boundaries",
            "historical statements here do not override that current contract",
            "[`README.md`](README.md)",
            "[`V07_RELEASE_NOTES.md`](V07_RELEASE_NOTES.md)",
        ):
            with self.subTest(document="DECISIONS.md", required=required):
                self.assertIn(required, decisions)

    def test_release_docs_are_self_contained_and_version_neutral(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        walkthrough = (ROOT / "WALKTHROUGH.md").read_text(encoding="utf-8")
        self.assertIn("Current package version: **0.7.0**", readme)
        self.assertIn("The repository and package include:", readme)
        self.assertIn("Release validation runs source compilation", readme)
        self.assertIn("The project does not provide:", readme)
        self.assertNotIn("current development tree", readme.lower())
        self.assertNotIn("later release gate", readme.lower())
        self.assertIn("Bounded constructed CPU copy validation", readme)
        self.assertIn("bounded constructed CPU copy validation", walkthrough)
        self.assertIn("**MEASURED test-replay observations**", walkthrough)

    def test_trace_family_and_preflight_boundary_are_explicit(self) -> None:
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

        self.assertIn("pre-flight configuration and report pipeline is for", preflight)
        self.assertIn("canonical routing trace **v1 only**", preflight)
        self.assertIn("`analyze --preflight-config` intentionally rejects v2", preflight)
        self.assertIn("must never be flattened together", preflight)

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
        self.assertEqual(__version__, "0.7.0")
        self.assertIn('version = "0.7.0"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
