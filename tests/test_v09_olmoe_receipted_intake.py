from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "V09_OLMOE_RECEIPTED_INTAKE.md"
LIVE_TEST = ROOT / "tests" / "test_v09_olmoe_receipted_live_reproduction.py"


class V09OlmoeReceiptedIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = INTAKE.read_text(encoding="utf-8")
        cls.normalized = " ".join(cls.text.split())
        cls.lowered = cls.normalized.lower()

    def test_artifact_and_reproduction_results_are_exact(self) -> None:
        self.assertTrue(INTAKE.is_file())
        for required in (
            "793bf319dee34cc0ed9863d57ebfd5c9d1a33811",
            "323cdbb3a8a52a9f2240a914ba7c28690058e6e292ba502f3ddc999b5d5067e5",
            "d02c69e4185dbba70305fa9f3e0dd75e23d9bc27410d3ba2aa4703240b0b0e4a",
            "0da662cc23d0f4d18ff3ee80824f40a539c200f53e002257c08fc8da7a4ac1d0",
            "1495fac92ee9ff34e7ed22657921835008292161a900605538dad901dd3d3a65",
            "40aca8ef5176d900353e0cc68ea8807a95db91e445cca1f20fa265b3d127b015",
            "Artifact SHA verification | `AGREEMENT`",
            "Mechanical canonical reproduction | `AGREEMENT`",
            "Historical-converter regression | `PASS`",
            "Raw-to-canonical preservation audit | `PASS`",
            "Canonical validity | `PASS`",
            "4,784",
            "38,272",
            "`0..15`",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

        self.assertIn(
            "83168fb10391646a97bb0c00c159094784a0ef72dc7962d18055020701f1d9b4",
            self.text,
        )

    def test_new_capture_does_not_backfill_historical_provenance(self) -> None:
        for required in (
            "NEW, separately receipted OLMoE capture",
            "does not recover, repair, or backfill provenance",
            "historical OLMoE raw fixture",
            "It does not upgrade any historical provenance field",
        ):
            with self.subTest(required=required):
                self.assertIn(required.lower(), self.lowered)

    def test_semantic_mapping_pass_is_bounded_and_rank_is_not_preserved(self) -> None:
        for required in (
            "Producer semantic mapping | **`BOUNDED PASS — selected routed-expert membership on the reviewed pinned path`**",
            "Producer path semantically validated: `BOUNDED PASS — selected routed-expert membership on the reviewed pinned path`",
            "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "Transformers 5.15.1",
            "4,800",
            "native tuple `topk_index` branch",
            "Native selection rank/order | `NOT PRESERVED`",
            "Producer non-interference | `NOT ESTABLISHED`",
            "Producer path non-interference validated: `NOT ESTABLISHED`",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

        self.assertIn("The path must not be called generically conforming", self.text)
        self.assertNotIn("producer path is conforming", self.lowered)
        self.assertNotIn("producer is conforming", self.lowered)

    def test_source_mapping_chain_names_the_native_execution_path(self) -> None:
        for required in (
            "OlmoeTopKRouter.forward()",
            "router_indices",
            "OlmoeSparseMoeBlock.forward()",
            "top_k_index",
            "self.experts(hidden_states, top_k_index, top_k_weights)",
            "OlmoeExperts.forward()",
            "expert mask",
            "does not expose a separate shared-expert execution path analogous to the reviewed Kimi path",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_superseded_byte_target_and_key_order_boundary_are_explicit(self) -> None:
        for required in (
            "28725ebb5ebcd97bb8f8d8a165e342a1a3cc5777d28befb4337b8bb45f984b97",
            "withdrew/superseded that byte target",
            "does **not** independently re-compare those old bytes record-for-record",
            "canonical v1 reader/validator is independent of JSON object key order",
            "exact byte/hash reproducibility is a separate requirement from semantic reader validity",
            "No trace-format rule requiring sorted JSON keys",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_recorded_provenance_is_not_mislabeled_as_generic_measurement(self) -> None:
        self.assertIn(
            "These environment/revision fields are **recorded execution provenance**",
            self.text,
        )
        self.assertIn("They are not grouped under a generic `MEASURED` label", self.text)
        self.assertNotIn("MEASURED: resolved revision", self.text)
        self.assertNotIn("MEASURED: environment", self.text)
        self.assertNotIn("/Users/", self.text)
        self.assertNotIn(".hermes", self.text)

    def test_public_v08_validation_source_identity_is_frozen(self) -> None:
        for blob in (
            "eb409fbe60bb475acd5d402d8c621875b784e633",
            "04711b433b5a3616c2d5709d69b6926ea5516360",
            "bb465665930037f31766f7c6c8329fd767a777be",
            "8405f854484081e08450a6c642eb320743fda494",
        ):
            with self.subTest(blob=blob):
                self.assertIn(blob, self.text)
        self.assertIn(
            "byte-identical to the public v0.8.0 validation source",
            self.text,
        )

    def test_final_surface_is_offline_and_claims_no_runtime_effect(self) -> None:
        self.assertFalse(LIVE_TEST.exists())
        for required in (
            "final tracked validation surface for this record is offline",
            "downstream cache outcomes remain **SIMULATED**",
            "downstream transfer-service quantities remain **ESTIMATED**",
            "does not support runtime-performance or broad-compatibility claims",
        ):
            with self.subTest(required=required):
                self.assertIn(required.lower(), self.lowered)

        for forbidden in (
            "speedup established",
            "latency improvement",
            "throughput improvement",
            "optimal cache",
            "broad olmoe compatibility",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.lowered)


if __name__ == "__main__":
    unittest.main()
