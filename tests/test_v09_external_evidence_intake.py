from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "V09_EXTERNAL_EVIDENCE_INTAKE.md"


class V09ExternalEvidenceIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = INTAKE.read_text(encoding="utf-8")
        cls.normalized = " ".join(cls.text.split())
        cls.lowered = cls.normalized.lower()
        cls.kimi = cls.text.split("## Kimi-Linear-48B historical intake", 1)[1].split(
            "## Historical OLMoE-1B-7B-0125 intake", 1
        )[0]
        cls.olmoe = cls.text.split("## Historical OLMoE-1B-7B-0125 intake", 1)[1].split(
            "## Separate later OLMoE capture", 1
        )[0]
        cls.kimi_normalized = " ".join(cls.kimi.split())
        cls.olmoe_normalized = " ".join(cls.olmoe.split())

    def test_intake_keeps_three_producer_claims_separate(self) -> None:
        self.assertTrue(INTAKE.is_file())
        for required in (
            "Canonical trace valid",
            "Producer path semantically validated",
            "Producer path non-interference validated",
            "Artifact intake acceptance below is a separate evidence-governance decision",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.normalized)

        self.assertNotIn("Producer path semantically validated: `PASS`", self.text)
        self.assertNotIn("Producer path non-interference validated: `PASS`", self.text)

    def test_kimi_identity_and_membership_boundaries_are_bounded(self) -> None:
        for required in (
            "06ad026c928d1c05eb59f0b1db55e1317c1899b1d3a687741ab5c49fdc014965",
            "3a2367a43e6d8d3bd3aefc91ef9e294428d1444b0822e5a86dcc777c21e418ac",
            "7f08cae05115f97b881a529177c0d345f8dbd0fb87195702aab26442a04fa756",
            "Bounded `SOURCE-READ` support for selected routed-expert membership mapping",
            "SOURCE-SUPPORTED INFERENCE — NOT CAPTURE-PROVEN",
            "Capture-specific immutable model/checkpoint revision: **`NOT RECORDED`**",
            "Capture-specific Python version: **`NOT RECORDED`**",
            "Capture-specific Torch version: **`NOT RECORDED`**",
            "Capture-specific device/dtype: **`NOT RECORDED`**",
            "shared-expert execution path is outside this trace",
            "caller-supplied analysis assumption",
            "ACCEPTED AS BOUNDED EXTERNAL INTAKE EVIDENCE",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.kimi_normalized)

        self.assertIn("Producer path semantically validated: `NOT ESTABLISHED`", self.kimi_normalized)
        self.assertIn("Producer path non-interference validated: `NOT ESTABLISHED`", self.kimi_normalized)

    def test_historical_olmoe_keeps_missing_run_provenance_missing(self) -> None:
        for required in (
            "7d8e7f63551aa24ddab98134b267b6bdf1b4f55d72a6fa7589042f54a4a4383b",
            "83168fb10391646a97bb0c00c159094784a0ef72dc7962d18055020701f1d9b4",
            "Capture-specific immutable model revision: **`NOT RECORDED`**",
            "Capture-specific Python version: **`NOT RECORDED`**",
            "Capture-specific Torch version: **`NOT RECORDED`**",
            "Capture-specific Transformers version: **`NOT RECORDED`**",
            "Capture-specific device/dtype: **`NOT RECORDED`**",
            "Capture-specific exact invocation: **`NOT RECORDED`**",
            "Capture-specific active gate-output branch: **`NOT RECORDED`**",
            "does **not** claim that a separate OLMoE shared-expert path was omitted",
            "ACCEPTED AS BOUNDED HISTORICAL EXTERNAL INTAKE EVIDENCE WITH PROVENANCE GAPS",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.olmoe_normalized)

        self.assertIn("Producer path semantically validated: `NOT ESTABLISHED`", self.olmoe_normalized)
        self.assertIn("Producer path non-interference validated: `NOT ESTABLISHED`", self.olmoe_normalized)
        self.assertIn(
            "A later OLMoE recapture with additional receipt information is a separate artifact",
            self.olmoe_normalized,
        )
        self.assertIn("not used to backfill", self.olmoe_normalized.lower())

    def test_rank_redistribution_and_downstream_evidence_classes_stay_bounded(self) -> None:
        self.assertIn("`NOT PRESERVED`", self.kimi_normalized)
        self.assertIn("`NOT PRESERVED`", self.olmoe_normalized)
        for section in (self.kimi_normalized, self.olmoe_normalized):
            with self.subTest(section=section[:30]):
                self.assertIn("metadata-only external reference", section)
                self.assertIn(
                    "Project-controlled copying/redistribution of raw or canonical trace bytes: **`NOT ESTABLISHED`**",
                    section,
                )
        self.assertIn("downstream cache outcomes must remain labeled **SIMULATED**", self.normalized)
        self.assertIn("transfer-service quantities must remain labeled **ESTIMATED**", self.normalized)

    def test_intake_contains_no_generic_producer_conformance_upgrade(self) -> None:
        for forbidden in (
            "Kimi producer path is conforming",
            "OLMoE producer path is conforming",
            "producer path is semantically validated",
            "producer path is non-interfering",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden.lower(), self.lowered)


if __name__ == "__main__":
    unittest.main()
