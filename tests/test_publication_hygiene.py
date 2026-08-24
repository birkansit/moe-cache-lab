import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_audit():
    path = ROOT / "scripts" / "audit_publication_hygiene.py"
    spec = importlib.util.spec_from_file_location("publication_hygiene_audit", path)
    if spec is None or spec.loader is None:
        raise AssertionError("publication hygiene audit script cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit()


class PublicationHygieneTests(unittest.TestCase):
    def _audit_text(self, text: str, relative: str = "note.md"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return AUDIT.audit_paths(root, (relative,))

    def _audit_bytes(self, payload: bytes, relative: str):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return AUDIT.audit_paths(root, (relative,))

    def test_required_content_leaks_are_rejected_with_rule_and_location(self) -> None:
        private_repo = "moe-cache-lab" + "-v05-dev"
        ready = "[" + "READY]"
        active = "[" + "ACTIVE]"
        completed = "[" + "COMPLETED]"
        project_role_case = "Implementation " + "work" + "er waits for the direc" + "tor review."
        implementation_role = "worker" + "_impl"
        director_checkpoint = "Direc" + "tor checkpoint approves the branch handoff."
        codex_task = "Co" + "dex task orchestration assigns an agent review."
        reviewer_agent = "reviewer " + "agent"
        developer_prompt = "developer " + "prompt contents"
        chain_of_thought = "chain of " + "thought transcript"
        private_issue = "Continue " + "Iss" + "ue #" + "76 after internal approval."
        windows_path = "C:" + r"\Users\alice\project\trace.jsonl"
        windows_project = "D:" + r"\Projects\sample\trace.jsonl"
        local_app_data = "%" + r"LOCALAPPDATA%\Programs\tool\config.json"
        unix_path = "/" + "home/alice/project/trace.jsonl"
        mac_path = "/" + "Users/alice/project/trace.jsonl"
        github_pat = "gh" + "p_" + ("A" * 36)
        api_key = "api" + "_key = " + ("B" * 24)
        email = "alice" + "@example.com"
        private_key = "-----BEGIN " + "PRIVATE KEY-----"
        independent = "independent" + "ly approved for publication"
        publication = "This note authoriz" + "es public repository release."
        cases = (
            (private_repo, "private-repository"),
            (ready, "task-state-marker"),
            (active, "task-state-marker"),
            (completed, "task-state-marker"),
            (project_role_case, "project-orchestration"),
            (implementation_role, "project-orchestration"),
            (director_checkpoint, "project-orchestration"),
            (codex_task, "project-orchestration"),
            (reviewer_agent, "project-orchestration"),
            (developer_prompt, "project-orchestration"),
            (chain_of_thought, "project-orchestration"),
            (private_issue, "private-issue-coordination"),
            (windows_path, "local-personal-path"),
            (windows_project, "local-personal-path"),
            (local_app_data, "local-personal-path"),
            (unix_path, "local-personal-path"),
            (mac_path, "local-personal-path"),
            (github_pat, "github-credential"),
            (api_key, "generic-credential"),
            (email, "email-address"),
            (private_key, "private-key-header"),
            (independent, "unsupported-independent-review"),
            (publication, "publication-process"),
        )
        for text, expected_rule in cases:
            with self.subTest(expected_rule=expected_rule):
                result = self._audit_text(text)
                rule_ids = {violation.rule_id for violation in result.violations}
                self.assertIn(expected_rule, rule_ids)
                diagnostic = AUDIT.format_violations(result.violations)
                self.assertIn("note.md:1", diagnostic)
                self.assertIn(f"[{expected_rule}]", diagnostic)

    def test_aws_bearer_and_github_long_token_variants_are_rejected(self) -> None:
        cases = (
            ("AS" + "IA" + ("A" * 16), "aws-access-key"),
            ("Bearer " + ("x" * 24), "generic-credential"),
            ("github" + "_pat_" + ("A" * 24), "github-credential"),
        )
        for text, expected_rule in cases:
            with self.subTest(expected_rule=expected_rule):
                result = self._audit_text(text)
                self.assertIn(expected_rule, {item.rule_id for item in result.violations})

    def test_prefixed_environment_credentials_are_rejected_in_env_files(self) -> None:
        cases = (
            (".env", "OPENAI_" + "API_KEY=" + ("a" * 32)),
            (".env.local", "SERVICE_" + "ACCESS_TOKEN=" + ("b" * 32)),
            ("settings.env", "VENDOR_" + "SECRET_KEY=" + ("c" * 32)),
            ("config.local", "SYSTEM_" + "SECRET_TOKEN=" + ("d" * 32)),
        )
        for relative, text in cases:
            with self.subTest(relative=relative):
                result = self._audit_text(text, relative)
                self.assertIn("generic-credential", {item.rule_id for item in result.violations})
                self.assertEqual(result.text_path_count, 1)
                self.assertEqual(result.binary_path_count, 0)

    def test_pem_and_key_files_are_content_scanned(self) -> None:
        header = "-----BEGIN " + "PRIVATE KEY-----"
        for relative in ("secret.pem", "private.key"):
            with self.subTest(relative=relative):
                result = self._audit_text(header, relative)
                self.assertIn("private-key-header", {item.rule_id for item in result.violations})
                self.assertEqual(result.text_path_count, 1)
                self.assertEqual(result.binary_path_count, 0)

    def test_extensionless_and_unknown_utf8_text_are_scanned(self) -> None:
        private_repo = "moe-cache-lab" + "-v05-dev"
        for relative in ("Dockerfile", "ContainerSpec"):
            with self.subTest(relative=relative):
                result = self._audit_text(f"FROM scratch # {private_repo}", relative)
                self.assertIn("private-repository", {item.rule_id for item in result.violations})
                self.assertEqual(result.text_path_count, 1)
                self.assertEqual(result.binary_path_count, 0)

    def test_unknown_binary_is_deliberately_skipped(self) -> None:
        result = self._audit_bytes(b"binary\x00payload\xff\xfe", "artifact.dat")
        self.assertEqual(result.violations, ())
        self.assertEqual(result.text_path_count, 0)
        self.assertEqual(result.binary_path_count, 1)

    def test_known_text_invalid_utf8_is_a_violation(self) -> None:
        result = self._audit_bytes(b"not-text\xff\xfe", "README.md")
        self.assertIn("non-utf8-text", {item.rule_id for item in result.violations})
        self.assertEqual(result.text_path_count, 1)
        self.assertEqual(result.binary_path_count, 0)

    def test_unknown_invalid_utf8_is_classified_as_binary(self) -> None:
        result = self._audit_bytes(b"opaque\xff\xfe", "opaque.data")
        self.assertEqual(result.violations, ())
        self.assertEqual(result.text_path_count, 0)
        self.assertEqual(result.binary_path_count, 1)

    def test_tracked_generated_and_model_artifact_paths_are_rejected(self) -> None:
        cases = (
            (".venv/pyvenv.cfg", "tracked-generated-artifact"),
            ("build/output.txt", "tracked-generated-artifact"),
            ("pkg/__pycache__/module.pyc", "tracked-generated-artifact"),
            ("weights/model.safetensors", "tracked-model-artifact"),
            ("assets/tokenizer.json", "tracked-model-artifact"),
        )
        for relative, expected_rule in cases:
            with self.subTest(relative=relative):
                result = self._audit_text("fixture", relative)
                self.assertIn(expected_rule, {item.rule_id for item in result.violations})

    def test_legitimate_historical_and_technical_phrases_pass(self) -> None:
        text = "\n".join(
            (
                "Raw traces and prompts may be private or sensitive.",
                "PrivateUsage is a Windows process-memory metric.",
                "A worker thread processes copy operations in a child process.",
                "See upstream Issue #33869: https://github.com/vllm-project/vllm/issues/33869",
                "Pinned revision 92fe2d22b024d9937146fe097ba3d3a7ba146e1b was inspected on 2026-08-13.",
                "Passing authorizes Stage 2 feasibility inspection only; never offloading.",
            )
        )
        result = self._audit_text(text)
        self.assertEqual(result.violations, ())

    def test_upstream_issue_allowance_requires_matching_issue_number(self) -> None:
        text = "Internal " + "Iss" + "ue #" + "76; see https://github.com/example/project/issues/75"
        result = self._audit_text(text)
        self.assertIn("private-issue-coordination", {item.rule_id for item in result.violations})

    def test_exact_bundle_path_fixture_allowlist_is_narrow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "tests" / "test_experiment_bundle.py"
            allowed.parent.mkdir(parents=True)
            allowed.write_text("/" + "home/user/private.jsonl", encoding="utf-8")
            other = root / "tests" / "other_test.py"
            other.write_text("/" + "home/user/private.jsonl", encoding="utf-8")
            allowed_result = AUDIT.audit_paths(root, ("tests/test_experiment_bundle.py",))
            rejected_result = AUDIT.audit_paths(root, ("tests/other_test.py",))
        self.assertEqual(allowed_result.violations, ())
        self.assertIn("local-personal-path", {item.rule_id for item in rejected_result.violations})

    def test_frozen_v04_review_phrase_allowlist_is_exact(self) -> None:
        phrase = "independent" + "ly approved and frozen for bounded implementation plus"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "V04_EXPERIMENT.md"
            frozen.write_text(phrase, encoding="utf-8")
            other = root / "OTHER.md"
            other.write_text(phrase, encoding="utf-8")
            frozen_result = AUDIT.audit_paths(root, ("V04_EXPERIMENT.md",))
            other_result = AUDIT.audit_paths(root, ("OTHER.md",))
        self.assertEqual(frozen_result.violations, ())
        self.assertIn("unsupported-independent-review", {item.rule_id for item in other_result.violations})

    def test_results_and_diagnostics_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "z.md").write_text("[" + "ACTIVE]", encoding="utf-8")
            (root / "a.md").write_text("C:" + r"\Projects\sample\x", encoding="utf-8")
            first = AUDIT.audit_paths(root, ("z.md", "a.md"))
            second = AUDIT.audit_paths(root, ("a.md", "z.md"))
        self.assertEqual(first, second)
        self.assertEqual([item.path for item in first.violations], ["a.md", "z.md"])
        self.assertEqual(AUDIT.format_violations(first.violations), AUDIT.format_violations(second.violations))

    def test_real_tracked_repository_passes_shared_audit(self) -> None:
        result = AUDIT.audit_repository(ROOT)
        self.assertEqual(result.violations, (), AUDIT.format_violations(result.violations))
        self.assertGreater(result.tracked_path_count, 100)
        self.assertGreater(result.text_path_count, 100)
        self.assertGreaterEqual(result.binary_path_count, 0)
        self.assertEqual(
            result.tracked_path_count,
            result.text_path_count + result.binary_path_count,
        )


if __name__ == "__main__":
    unittest.main()
