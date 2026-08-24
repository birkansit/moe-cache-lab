import hashlib
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "EXTERNAL_RUNTIME_INTEROP.md"
VLLM_SHA = "b26039b09fc97aa00f095a99eda503b7dad594ec"
LLAMA_CPP_SHA = "8144f3192e5a3131cd043f284525e6ceebf82d0f"

FROZEN_MANIFEST_HASHES = {
    "preflight-report.md":
        "219751d3a63b97177f6d3133dd2e7b69fc941eed2416e72568125f4e2936542d",
    "preflight-report.json":
        "464638a63f66e2c8797f4e7c59efea226f30050032ede5138255f675d0ff4a42",
    "stage-qualified-preflight-report.md":
        "9e9543cc7f126138e68c0b7ab3b072511215097c3d70e42745d23290bfa74ba3",
    "stage-qualified-preflight-report.json":
        "387bdb273cd90a38577e63b7f82c1791988f10778d6ea0d18c16ca995bfb4599",
    "artifacts/trace.jsonl":
        "f0bb67c204714b5189416fa2c7bfa76dd7d93770fc092ee660501b854ac63839",
    "artifacts/validation.json":
        "2910d4b23030bcaaf25de63c5b1edaf0c8ce1a726340c36bd063c40819171740",
    "artifacts/routing-analysis.md":
        "a8d32c66e028a73f25abc8eb7678a7acfeccbc729e4695d8459055d5c216cf9f",
    "artifacts/routing-analysis.json":
        "9505a06c213714287273d25a15ae3a91be0e3bd94240e2dcae8120714cec51de",
    "artifacts/preflight-report.md":
        "4ab2398ad81a47a7abdf8e1fb5e8d27d0f2588ffc0fcc51f1ef9ad1debec5805",
    "artifacts/preflight-report.json":
        "2b76fc90d65dfdfbaaa5195e62eb43200a7e867d8e832de1bb09b84ee7f8feee",
}
FROZEN_FILE_HASHES = {
    ROOT / "results/v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980/v04-result.json":
        "ed857ef581f80fbbe8e694bc869e9265e785e5c48bd6e00118b20db7d85e7b72",
    ROOT / "results/v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980/v04-report.md":
        "d9b5399a125d162c3c7cdb3f847f30c196b56fd31ace9ee714f9dd4d5b81ceaf",
}


def _read_hash_manifest(path: Path) -> dict[str, str]:
    return {
        filename: digest
        for digest, filename in (
            line.split(maxsplit=1)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


class ExternalRuntimeInteropPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = DOCUMENT.read_text(encoding="utf-8")

    def test_immutable_revisions_licenses_and_official_sources_are_recorded(self) -> None:
        for required in (
            VLLM_SHA,
            LLAMA_CPP_SHA,
            "https://github.com/vllm-project/vllm",
            "https://github.com/ggml-org/llama.cpp",
            "Apache License 2.0",
            "MIT License",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_all_twelve_frozen_pass_criteria_and_no_go_rules_are_present(self) -> None:
        criteria_section = self.text.split("### Mandatory PASS criteria", 1)[1].split(
            "### Frozen NO-GO rules", 1
        )[0]
        numbers = re.findall(r"(?m)^(\d+)\. ", criteria_section)
        self.assertEqual(numbers, [str(number) for number in range(1, 13)])
        normalized = " ".join(self.text.split())
        for required in (
            "Failure to establish any mandatory item prevents `PASS`.",
            "`NO-GO` is bounded negative evidence",
            "No new routing trace version",
        ):
            self.assertIn(required, normalized)

    def test_source_evidence_names_exact_native_symbols(self) -> None:
        for required in (
            "BaseRouter._select_experts",
            "RoutedExpertsCapturer",
            "RoutedExpertsManager.store_batch",
            "MoERunner._apply_quant_method",
            "llm_graph_context::build_moe_ffn",
            "ffn_moe_topk",
            "llama_context::graph_get_cb",
            "ggml_backend_sched_eval_callback",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_both_candidate_verdicts_and_final_verdict_are_bounded(self) -> None:
        self.assertEqual(self.text.count("`NO-GO` at revision"), 2)
        self.assertEqual(self.text.count("**FINAL D VERDICT: NO-GO**"), 1)
        self.assertIn("exact pinned revisions and local no-model/no-system-change boundary", self.text)
        self.assertNotIn("can never support", self.text)
        self.assertNotIn("compatible with all", self.text.lower())

    def test_source_only_evidence_is_not_mislabeled_as_runtime_measurement(self) -> None:
        self.assertIn("**Execution performed: NO.**", self.text)
        self.assertIn("Source inspection is **SOURCE-READ evidence", self.text)
        self.assertNotIn("**MEASURED**", self.text)
        self.assertNotIn("MEASURED:", self.text)
        self.assertNotIn("routing non-interference was validated", self.text.lower())

    def test_contract_does_not_expand_or_leak_private_context(self) -> None:
        forbidden = (
            "trace " + "v3",
            "C:\\",
            "C:/",
            "AppData",
            "moe-cache-lab" + "-v05-dev",
            "private coordination",
            "director handoff",
        )
        for value in forbidden:
            with self.subTest(value=value):
                self.assertNotIn(value, self.text)

    def test_no_external_runtime_checkout_is_tracked(self) -> None:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        )
        tracked = completed.stdout.decode("utf-8").split("\0")
        forbidden_roots = ("vllm/", "llama.cpp/", "external/vllm/", "external/llama.cpp/")
        self.assertFalse(
            [path for path in tracked if path.startswith(forbidden_roots)],
            "external runtime source must remain outside the repository",
        )

    def test_all_twelve_frozen_artifact_hashes_remain_exact(self) -> None:
        manifests = {}
        for path in (
            ROOT / "examples/no-download-preflight/expected.sha256",
            ROOT / "examples/no-download-stage-qualified-preflight/expected.sha256",
            ROOT / "examples/external-producer-no-model/expected.sha256",
        ):
            manifests.update(_read_hash_manifest(path))
        self.assertEqual(manifests, FROZEN_MANIFEST_HASHES)
        self.assertEqual(len(FROZEN_MANIFEST_HASHES) + len(FROZEN_FILE_HASHES), 12)
        for path, expected in FROZEN_FILE_HASHES.items():
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
