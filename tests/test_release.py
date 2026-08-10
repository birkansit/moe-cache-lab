from pathlib import Path
import re
import unittest

import moe_cache_lab


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_version_and_metadata_match(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "0.5.0")
        self.assertEqual(moe_cache_lab.__version__, "0.5.0")

    def test_current_public_docs_exist(self) -> None:
        for name in (
            "README.md",
            "PREFLIGHT.md",
            "V05_RELEASE_NOTES.md",
            "V05_VALIDATION.md",
            "CONTRIBUTING.md",
            "LICENSE",
        ):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_archived_experiment_material_is_not_in_clean_tree(self) -> None:
        for name in (
            "results",
            "PROJECT_BRIEF.md",
            "DECISIONS.md",
            "PLAN.md",
            "RELEASE_CANDIDATE.md",
            "STAGE1_RESULTS.md",
            "STAGE2_FEASIBILITY.md",
            "V04_EXPERIMENT.md",
            "V04_RESULTS.md",
        ):
            self.assertFalse((ROOT / name).exists(), name)

    def test_validation_claim_boundary_is_explicit(self) -> None:
        text = (ROOT / "V05_VALIDATION.md").read_text(encoding="utf-8")
        for required in (
            "MEASURED routing",
            "SIMULATED cache behavior",
            "MEASURED raw transfer diagnostic",
            "ESTIMATED transfer service",
            "not PyTorch GPU inference",
            "not measured Granite runtime latency",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
