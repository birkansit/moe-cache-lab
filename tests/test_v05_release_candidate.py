import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class V05ReleaseTests(unittest.TestCase):
    def test_current_docs_state_preflight_identity_and_claim_boundaries(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        preflight = (ROOT / "PREFLIGHT.md").read_text(encoding="utf-8")
        notes = (ROOT / "V05_RELEASE_NOTES.md").read_text(encoding="utf-8")
        demo = (ROOT / "examples" / "no-download-preflight" / "README.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("trace-driven MoE pre-flight analysis toolkit", readme)
        self.assertIn("not an inference accelerator", readme)
        self.assertIn("Granite/Transformers-specific", readme)
        self.assertIn("Who this is for", readme)
        self.assertIn("When not to use it", readme)
        self.assertIn("examples/no-download-preflight/README.md", readme)
        self.assertIn("PREFLIGHT.md", readme)
        self.assertIn("Current package version: **0.7.0**", readme)
        self.assertIn("Current package version: **0.5.0**", notes)

        for document in (readme, preflight, notes):
            self.assertIn("SIMULATED", document)
            self.assertIn("ESTIMATED", document)
            self.assertNotIn("0.5.0rc1", document)

        self.assertIn("runtime speedup", readme)
        self.assertIn("runtime speedup", notes)
        self.assertIn("speedup", preflight)
        self.assertIn("MEASURED only", preflight)
        self.assertIn("synthetic", demo.lower())
        self.assertIn("fictional", demo.lower())
        self.assertIn("does not implement expert swapping/offloading", notes)

    def test_release_facing_files_exclude_process_terms_secrets_and_local_identity(self) -> None:
        release_facing = (
            ROOT / "README.md",
            ROOT / "PREFLIGHT.md",
            ROOT / "V05_RELEASE_NOTES.md",
            ROOT / "examples" / "no-download-preflight" / "README.md",
            ROOT / "examples" / "no-download-preflight" / "preflight-config.json",
            ROOT / "examples" / "no-download-preflight" / "trace.jsonl",
        )
        email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        local_path_patterns = (
            re.compile(r"\b[A-Za-z]:[\\/]"),
            re.compile(r"/(?:home|Users|tmp)/"),
        )
        process_term_patterns = (
            re.compile(r"\bdirector\b"),
            re.compile(r"\bworker\b"),
        )
        nonfinal_state_patterns = (
            re.compile(r"\bcurrent private development candidate\b"),
            re.compile(r"\bcurrent private candidate\b"),
            re.compile(r"\bprivate release candidate\b"),
            re.compile(r"\bcurrent release candidate\b"),
            re.compile(r"\bv0\.5 release candidate\b"),
            re.compile(r"\brelease-candidate gate\b"),
        )
        secret_markers = ("api_key", "access_token", "secret_token", "bearer ")

        for path in release_facing:
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            with self.subTest(path=path.name):
                for pattern in process_term_patterns + nonfinal_state_patterns:
                    self.assertIsNone(pattern.search(lowered))
                self.assertIsNone(email_pattern.search(text))
                for pattern in local_path_patterns:
                    self.assertIsNone(pattern.search(text))
                for marker in secret_markers:
                    self.assertNotIn(marker, lowered)

    def test_release_notes_only_describe_implemented_v05_scope(self) -> None:
        notes = (ROOT / "V05_RELEASE_NOTES.md").read_text(encoding="utf-8")
        for required in (
            "routing-locality analysis",
            "phase-layer summaries",
            "observed-support Gini",
            "byte-aware",
            "LRU/LFU",
            "serialized/no-overlap transfer-service sensitivity",
            "offline `analyze` workflow",
            "tracked synthetic no-download demo",
        ):
            self.assertIn(required, notes)

        self.assertIn("No second model-family collector", notes)
        self.assertIn("does not claim that a Git tag, GitHub Release, PyPI publication", notes)


if __name__ == "__main__":
    unittest.main()
