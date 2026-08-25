from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from moe_cache_lab import __version__


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V09ReleaseHardeningTests(unittest.TestCase):
    def test_release_candidate_keeps_current_package_identity(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertEqual(__version__, "0.9.0")
        self.assertIn('version = "0.9.0"', pyproject)

    def test_release_notes_describe_scope_and_exact_frontier_contract(self) -> None:
        notes = (ROOT / "V09_RELEASE_NOTES.md").read_text(encoding="utf-8")
        normalized = " ".join(notes.split())
        for required in (
            "# v0.9 release notes",
            "Exact event-atomic LRU Capacity Frontier",
            "count-capacity frontier",
            "heterogeneous byte-capacity frontier",
            "25%/50%/75%/90%",
            "compulsory first-use miss fraction",
            "maximum reachable cold-start hit fraction",
            "does not sweep every integer capacity",
            "B1 count-reuse",
            "B2 heterogeneous byte-reuse",
            "not stable public APIs",
            "external evidence foundation",
            "Bash/Linux installed-wheel path",
            "Expected input failures remain concise command-line errors",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized)

    def test_release_notes_preserve_scientific_boundaries(self) -> None:
        notes = (ROOT / "V09_RELEASE_NOTES.md").read_text(encoding="utf-8")
        normalized = " ".join(notes.split())
        for required in (
            "SOURCE-SUPPORTED INFERENCE — NOT CAPTURE-PROVEN",
            "Producer non-interference remains **NOT ESTABLISHED**",
            "canonical-file validity, producer semantic mapping, and producer non-interference",
            "Neither result alone establishes portable byte reproducibility",
            "**SIMULATED** event-atomic LRU cache abstraction",
            "Transfer-service quantities are **ESTIMATED**",
            "Routing observations are **MEASURED** only when their provenance establishes measurement",
            "Physical residency, H2D or DRAM movement, latency, throughput, tokens/sec, speedup",
            "no generic cacheability score, model ranking, policy ranking, capacity recommendation",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized)
        self.assertIn(
            "`unattainable` does not mean that caching or offloading is impossible",
            normalized,
        )
        self.assertNotIn("birkansit/moe-cache-lab-" + "v05-dev", notes)
        self.assertNotIn("[" + "ACTIVE]", notes)
        self.assertNotIn("[" + "READY]", notes)

    def test_v09_notes_ship_in_all_release_document_surfaces(self) -> None:
        distribution = _load_module(
            "distribution_audit_v09", "scripts/audit_distributions.py"
        )
        core = _load_module("core_audit_v09", "scripts/audit_core_workflow.py")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertTrue((ROOT / "V09_RELEASE_NOTES.md").is_file())
        self.assertIn("V09_RELEASE_NOTES.md", distribution.REQUIRED_RELEASE_DOCS)
        self.assertIn("V09_RELEASE_NOTES.md", distribution.REQUIRED_SDIST_PATHS)
        self.assertIn("V09_RELEASE_NOTES.md", core.REQUIRED_DOCS)
        self.assertIn('"V09_RELEASE_NOTES.md"', pyproject)
        self.assertIn("include V09_RELEASE_NOTES.md", manifest)

    def test_canonical_docs_include_capacity_frontier(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        external = (
            ROOT / "examples" / "external-producer-no-model" / "README.md"
        ).read_text(encoding="utf-8")
        for document in (readme, external):
            self.assertIn("moe-cache-lab capacity-frontier", document)
            self.assertIn("25%/50%/75%/90%", document)
            self.assertIn("SIMULATED", document)
        self.assertIn("artifacts/capacity-frontier.json", external)

    def test_ubuntu_installed_wheel_golden_path_executes_frontier(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        block = ci.split(
            "- name: Execute Bash onboarding golden path from installed wheel", 1
        )[1].split("- name: Exercise fresh core-only installed wheel", 1)[0]
        for required in (
            'source "$ONBOARDING_ROOT/.venv/bin/activate"',
            "python -m pip install --no-deps",
            'PackagedExample="$VIRTUAL_ENV/share/moe-cache-lab/examples/external-producer-no-model"',
            "moe-cache-lab capacity-frontier artifacts/trace.jsonl",
            "--preflight-config preflight-config.json",
            "--output artifacts/capacity-frontier.md",
            "--json-output artifacts/capacity-frontier.json",
            "moe-cache-lab.capacity-frontier",
            "Caller-supplied cache-model inputs",
        ):
            with self.subTest(required=required):
                self.assertIn(required, block)
        self.assertNotIn("PYTHONPATH", block)
        self.assertEqual(block.count("sha256sum -c expected.sha256"), 1)
        self.assertNotIn("capacity-frontier", block.split("bundle-create", 1)[1])
        descriptive_analyze = block.index(
            "moe-cache-lab analyze artifacts/trace.jsonl \\\n"
            "            --workload-id external-producer-demo"
        )
        frontier = block.index("moe-cache-lab capacity-frontier artifacts/trace.jsonl")
        frontier_boundary = block.index("moe-cache-lab.capacity-frontier")
        configured_preflight = block.index(
            "moe-cache-lab analyze artifacts/trace.jsonl \\\n"
            "            --preflight-config preflight-config.json"
        )
        hashes = block.index("sha256sum -c expected.sha256")
        bundle = block.index("moe-cache-lab bundle-create")
        self.assertLess(
            descriptive_analyze,
            frontier,
        )
        self.assertLess(frontier, frontier_boundary)
        self.assertLess(frontier_boundary, configured_preflight)
        self.assertLess(configured_preflight, hashes)
        self.assertLess(hashes, bundle)

    def test_installed_core_audit_covers_both_frontier_modes(self) -> None:
        audit = (ROOT / "scripts" / "audit_core_workflow.py").read_text(
            encoding="utf-8"
        )
        for required in (
            '"v1-capacity-frontier.json"',
            'v1_frontier_data["byte_frontier"] is None',
            'v2_frontier_data["byte_frontier"] is not None',
            'v1_frontier_data["trace"]["format_version"] == 1',
            'v2_frontier_data["trace"]["format_version"] == 2',
            '"Caller-supplied cache-model inputs"',
            'find_spec("torch") is None',
            'find_spec("transformers") is None',
        ):
            with self.subTest(required=required):
                self.assertIn(required, audit)

    def test_ci_retains_portable_core_matrix_and_pinned_ml_lane(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        core = ci.split("  core-offline:", 1)[1].split("  ml-full-regression:", 1)[0]
        ml = ci.split("  ml-full-regression:", 1)[1].split("  distributions:", 1)[0]
        self.assertIn('python-version: ["3.10", "3.11", "3.12"]', core)
        self.assertIn("scripts/audit_core_workflow.py", core)
        self.assertIn("torch==2.12.0", ml)
        self.assertIn("transformers==5.12.0", ml)
        self.assertIn("python -m unittest discover -s tests -v", ml)

    def test_frontier_modules_remain_distribution_requirements(self) -> None:
        distribution = _load_module(
            "distribution_modules_v09", "scripts/audit_distributions.py"
        )
        self.assertIn(
            "moe_cache_lab/capacity_frontier.py",
            distribution.REQUIRED_PACKAGE_MODULES,
        )
        self.assertIn(
            "moe_cache_lab/capacity_frontier_output.py",
            distribution.REQUIRED_PACKAGE_MODULES,
        )


if __name__ == "__main__":
    unittest.main()
