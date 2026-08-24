from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "PRODUCER_CONFORMANCE.md"


class ProducerConformancePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = POLICY.read_text(encoding="utf-8")
        cls.normalized = " ".join(cls.text.split())
        cls.lowered = cls.normalized.lower()

    def test_policy_exists_and_is_linked_from_external_contract_docs(self) -> None:
        self.assertTrue(POLICY.is_file())
        for name in ("README.md", "TRACE_FORMAT.md", "COMPATIBILITY.md"):
            with self.subTest(name=name):
                self.assertIn(
                    "PRODUCER_CONFORMANCE.md",
                    (ROOT / name).read_text(encoding="utf-8"),
                )

    def test_three_claims_are_explicitly_separate(self) -> None:
        for required in (
            "Canonical trace valid",
            "Producer path semantically validated",
            "Producer path non-interference validated",
            "These claims are independent and must be reported separately",
            "may still be canonical trace valid",
            "may be described specifically as **semantically validated**",
            "must not be called generically validated, non-interfering, or conforming",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.normalized)

    def test_native_boundary_chronology_identity_and_unassigned_rules_are_frozen(self) -> None:
        for required in (
            "native routing and actual dispatch behavior is authoritative",
            "pre-capacity routing preference or selection",
            "post-capacity actual dispatch or assignment",
            "must not sort events to make validation pass",
            "silently repair chronology regressions",
            "`(layer, expert_id)`",
            "`(routing_stage, layer, expert_id)`",
            "must not flatten encoder and decoder",
            "zero assigned expert requests",
            "must not fabricate a preferred or likely expert",
            "Missing data alone is not evidence of a capacity drop",
        ):
            with self.subTest(required=required):
                self.assertIn(required.lower(), self.lowered)

    def test_provenance_non_interference_and_no_go_are_bounded(self) -> None:
        for required in (
            "model identifier and immutable revision when available",
            "capture method and exact observation boundary",
            "frozen input/workload",
            "defines no universal score",
            "no universal numeric tolerance",
            "identity agreement must be reported directly",
            "only aggregate expert counts are available",
            "chronology would have to be guessed, sorted, repaired, or deduplicated",
            "a capacity-drop or unassigned reason would have to be inferred",
            "provenance cannot be bounded sufficiently",
        ):
            with self.subTest(required=required):
                self.assertIn(required.lower(), self.lowered)

    def test_external_jsonl_and_version_boundaries_do_not_claim_a_framework(self) -> None:
        for required in (
            "canonical jsonl contract remains the primary external interoperability boundary",
            "does not define plugin discovery",
            "external producers are not required to depend on internal python collector classes",
            "new model, checkpoint, producer, or runtime alone",
            "justify routing trace v3",
            "separate schema/version design decision",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.lowered)

    def test_granite_and_switch_grounding_is_non_ranking_and_marks_limits(self) -> None:
        for required in (
            "Granite/Transformers v1",
            "SwitchTransformers v2",
            "maximum selected-probability delta `0.0`",
            "returned post-capacity assignment tensor",
            "maximum absolute difference `0.0`",
            "Neither row ranks the models or producers",
            "not established",
        ):
            with self.subTest(required=required):
                self.assertIn(required.lower(), self.lowered)

    def test_policy_does_not_upgrade_scientific_claims(self) -> None:
        for required in (
            "does not establish representative workload behavior",
            "physical cpu/gpu residency",
            "actual transfer traffic",
            "latency, throughput, tokens per second, speedup",
            "an optimal cache policy/capacity",
            "broad model/runtime compatibility",
            "cache results derived later remain **simulated**",
            "transfer-service results remain **estimated**",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.lowered)

    def test_policy_is_present_in_wheel_and_sdist_contracts(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        audit = (ROOT / "scripts" / "audit_distributions.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"PRODUCER_CONFORMANCE.md",', pyproject)
        self.assertIn("include PRODUCER_CONFORMANCE.md", manifest)
        self.assertGreaterEqual(audit.count('"PRODUCER_CONFORMANCE.md",'), 2)


if __name__ == "__main__":
    unittest.main()
