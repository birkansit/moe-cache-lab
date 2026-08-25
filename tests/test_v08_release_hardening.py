import importlib.util
from pathlib import Path
import unittest

from moe_cache_lab import __version__


ROOT = Path(__file__).resolve().parents[1]


def _load_distribution_audit():
    path = ROOT / "scripts" / "audit_distributions.py"
    spec = importlib.util.spec_from_file_location("distribution_audit_v08", path)
    if spec is None or spec.loader is None:
        raise AssertionError("distribution audit cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V08ReleaseHardeningTests(unittest.TestCase):
    def test_release_version_metadata_and_python_support_are_frozen(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertEqual(__version__, "0.9.0")
        self.assertIn('version = "0.9.0"', pyproject)
        self.assertIn(
            'description = "Trace-driven pre-flight analysis for MoE routing, '
            'expert caching, and offloading research."',
            pyproject,
        )
        for version in ("3.10", "3.11", "3.12"):
            self.assertIn(f'"Programming Language :: Python :: {version}"', pyproject)
        self.assertIn('license = "Apache-2.0"', pyproject)
        self.assertIn('Repository = "https://github.com/birkansit/moe-cache-lab"', pyproject)
        self.assertIn('Issues = "https://github.com/birkansit/moe-cache-lab/issues"', pyproject)

    def test_release_notes_preserve_all_evidence_classes_and_bounded_d_result(self) -> None:
        notes = (ROOT / "V08_RELEASE_NOTES.md").read_text(encoding="utf-8")
        normalized = " ".join(notes.split())
        for required in (
            "# v0.8 release notes",
            "## Release scope",
            "**Stage-qualified v2 pre-flight.**",
            "canonical-file validity only",
            "Producer semantic mapping and non-interference are separate claims",
            "**SYNTHETIC**",
            "**SOURCE-READ**",
            "**MEASURED** only when trace provenance",
            "**SIMULATED**",
            "**ESTIMATED**",
            "**NOT ESTABLISHED**",
            "not a universal runtime rejection",
            "No runtime adapter or integration was added",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized)
        self.assertNotIn("has been publicly released", notes.lower())
        self.assertNotIn("compatible with all", notes.lower())

    def test_optional_torch_advisory_boundary_is_explicit(self) -> None:
        notes = (ROOT / "V08_RELEASE_NOTES.md").read_text(encoding="utf-8")
        normalized = " ".join(notes.split())
        for required in (
            "PyTorch 2.12.0 with Transformers 5.12.0",
            "GHSA-rrmf-rvhw-rf47",
            "CVE-2025-3000",
            "PYSEC-2025-194",
            "`torch.jit.script`",
            "does not call that JIT API or an equivalent indirect path",
            "not a claim that PyTorch 2.12.0 is generally safe",
            "full-model producer and non-interference evidence is version-frozen",
            "has not been re-established on PyTorch 2.13.0",
            "base installation remains Torch-free",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized)

    def test_current_docs_share_one_external_workflow_and_identity_boundary(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        walkthrough = (ROOT / "WALKTHROUGH.md").read_text(encoding="utf-8")
        combined = " ".join((readme + "\n" + walkthrough).split())
        for required in (
            "produce/import canonical JSONL",
            "validate the canonical v1/v2 file offline",
            "analyze descriptive routing evidence offline",
            "run pre-flight only for a compatible trace/config pair",
            "create and verify a reproducibility bundle",
            "Canonical JSONL is the external interoperability boundary",
            "does not need the internal `ProducerResult` type",
            "canonical-file validity only",
            "producer semantic mapping and non-interference require separate evidence",
            "trace v2 with config v3",
            "V2 lifecycle analysis remains deferred",
        ):
            with self.subTest(required=required):
                self.assertIn(required, combined)
        self.assertIn("`(routing_stage, layer)` identity", combined)
        self.assertIn("never flatten", combined)

    def test_release_surface_ships_substantive_release_docs(self) -> None:
        audit = _load_distribution_audit()
        required_docs = {
            "README.md",
            "COMPATIBILITY.md",
            "CONTRIBUTING.md",
            "EXTERNAL_RUNTIME_INTEROP.md",
            "PREFLIGHT.md",
            "PRODUCER_CONFORMANCE.md",
            "TRACE_FORMAT.md",
            "V08_RELEASE_NOTES.md",
            "WALKTHROUGH.md",
        }
        self.assertTrue(required_docs <= set(audit.REQUIRED_RELEASE_DOCS))
        self.assertTrue(required_docs <= set(audit.REQUIRED_SDIST_PATHS))
        source_modules = {
            path.relative_to(ROOT / "src").as_posix()
            for path in (ROOT / "src" / "moe_cache_lab").glob("*.py")
        }
        self.assertTrue(source_modules <= set(audit.REQUIRED_PACKAGE_MODULES))
        self.assertTrue(
            {f"src/{path}" for path in source_modules}
            <= set(audit.REQUIRED_SDIST_PATHS)
        )
        self.assertIn("scripts/audit_core_workflow.py", audit.REQUIRED_SDIST_PATHS)
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("include scripts/audit_core_workflow.py", manifest)
        self.assertIn("global-exclude *.py[cod]", manifest)
        self.assertIn("global-exclude __pycache__", manifest)

    def test_distribution_audit_rejects_generated_model_and_runtime_trees(self) -> None:
        audit = _load_distribution_audit()
        names = {
            "moe_cache_lab-0.8.0/src/moe_cache_lab/__init__.py",
            "moe_cache_lab-0.8.0/build/generated.txt",
            "moe_cache_lab-0.8.0/external/vllm/source.py",
            "moe_cache_lab-0.8.0/weights/model.safetensors",
        }
        rejected = audit._unexpected_archive_paths(names)
        self.assertEqual(
            rejected,
            [
                "moe_cache_lab-0.8.0/build/generated.txt",
                "moe_cache_lab-0.8.0/external/vllm/source.py",
                "moe_cache_lab-0.8.0/weights/model.safetensors",
            ],
        )

    def test_ci_has_real_core_matrix_and_separate_pinned_ml_lane(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        core = ci.split("  core-offline:", 1)[1].split("  ml-full-regression:", 1)[0]
        ml = ci.split("  ml-full-regression:", 1)[1].split("  distributions:", 1)[0]
        self.assertIn('python-version: ["3.10", "3.11", "3.12"]', core)
        self.assertIn("scripts/audit_core_workflow.py", core)
        self.assertIn("tests.test_v04_portability", core)
        self.assertIn("tests.test_external_producer_workflow", core)
        self.assertIn("find_spec('torch') is None", core)
        self.assertIn("find_spec('transformers') is None", core)
        self.assertNotIn("torch==", core)
        self.assertNotIn("transformers==", core)
        self.assertIn('python-version: "3.10"', ml)
        self.assertIn('"torch==2.12.0"', ml)
        self.assertIn('"transformers==5.12.0"', ml)
        self.assertIn("python -m unittest discover -s tests -v", ml)
        self.assertIn('HF_HUB_OFFLINE: "1"', ci)
        self.assertIn('TRANSFORMERS_OFFLINE: "1"', ci)
        self.assertEqual(ci.count("Audit tracked release surface"), 2)
        self.assertNotIn("Audit tracked publication candidate", ci)

    def test_internal_release_checklist_is_absent_from_publishable_contracts(self) -> None:
        checklist = "V08_RELEASE_CHECKLIST.md"
        audit = _load_distribution_audit()
        self.assertFalse((ROOT / checklist).exists())
        self.assertNotIn(checklist, audit.REQUIRED_RELEASE_DOCS)
        self.assertNotIn(checklist, audit.REQUIRED_SDIST_PATHS)
        for relative in (
            "README.md",
            "MANIFEST.in",
            "pyproject.toml",
            "scripts/audit_core_workflow.py",
            "scripts/audit_distributions.py",
        ):
            with self.subTest(relative=relative):
                self.assertNotIn(
                    checklist,
                    (ROOT / relative).read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
