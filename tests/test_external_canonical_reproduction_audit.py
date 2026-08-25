import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_external_canonical_reproduction.py"
SPEC = importlib.util.spec_from_file_location("external_repro_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _kimi_raw_rows() -> list[dict]:
    rows = []
    for token in range(300):
        layers = []
        for ordinal in range(26):
            base = (token + ordinal * 11) % 240
            layers.append([base + offset for offset in range(8)])
        rows.append({"tok": token, "layers": layers})
    return rows


def _kimi_canonical_rows(raw: list[dict]) -> list[dict]:
    rows = [
        {
            "format": "moe-cache-lab.routing-jsonl",
            "format_version": 2,
            "model_id": "synthetic/kimi-audit",
            "record_type": "metadata",
            "routing_stages": [
                {
                    "allows_unassigned": False,
                    "assigned_experts_per_token": 8,
                    "num_experts": 256,
                    "routing_stage": "decoder",
                }
            ],
        }
    ]
    for raw_row in raw:
        for ordinal, experts in enumerate(raw_row["layers"]):
            rows.append(
                {
                    "assignment_state": "assigned",
                    "layer": ordinal + 1,
                    "phase": "decoder_generated",
                    "record_type": "routing_selection",
                    "routing_stage": "decoder",
                    "selected_experts": experts,
                    "selected_probabilities": [],
                    "token_position": raw_row["tok"],
                }
            )
    return rows


def _olmoe_raw_rows() -> list[dict]:
    rows = []
    for token in range(1, 300):
        layers = []
        for layer in range(16):
            base = (token + layer * 3) % 56
            layers.append([base + offset for offset in range(8)])
        rows.append({"tok": token, "layers": layers})
    return rows


def _olmoe_canonical_rows(raw: list[dict]) -> list[dict]:
    rows = [
        {
            "format": "moe-cache-lab.routing-jsonl",
            "format_version": 1,
            "model_id": "synthetic/olmoe-audit",
            "record_type": "metadata",
            "num_experts": 64,
            "experts_per_token": 8,
        }
    ]
    for raw_row in raw:
        for layer, experts in enumerate(raw_row["layers"]):
            rows.append(
                {
                    "layer": layer,
                    "phase": "generated",
                    "record_type": "routing_selection",
                    "selected_experts": experts,
                    "token_position": raw_row["tok"],
                }
            )
    return rows


class ExternalCanonicalReproductionAuditTests(unittest.TestCase):
    def _kimi_fixture(self, directory: Path) -> tuple[Path, Path, list[dict], list[dict]]:
        raw_rows = _kimi_raw_rows()
        canonical_rows = _kimi_canonical_rows(raw_rows)
        raw = directory / "kimi-raw.jsonl"
        canonical = directory / "kimi-canonical.jsonl"
        _write_jsonl(raw, raw_rows)
        _write_jsonl(canonical, canonical_rows)
        return raw, canonical, raw_rows, canonical_rows

    def test_kimi_full_preservation_audit_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw, canonical, _, _ = self._kimi_fixture(Path(temporary))
            result = AUDIT.audit_kimi(raw, canonical)
        self.assertEqual(result["preservation_audit"], "PASS")
        self.assertEqual(result["canonical_events"], 7_800)
        self.assertEqual(result["expert_requests"], 62_400)
        self.assertEqual((result["layer_min"], result["layer_max"]), (1, 26))

    def test_kimi_wrong_zero_based_layer_mapping_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw, canonical, _, canonical_rows = self._kimi_fixture(directory)
            canonical_rows[1]["layer"] = 0
            _write_jsonl(canonical, canonical_rows)
            with self.assertRaisesRegex(AUDIT.AuditFailure, "layer mapping mismatch"):
                AUDIT.audit_kimi(raw, canonical)

    def test_kimi_changed_membership_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw, canonical, _, canonical_rows = self._kimi_fixture(directory)
            canonical_rows[51]["selected_experts"] = [240, 241, 242, 243, 244, 245, 246, 247]
            _write_jsonl(canonical, canonical_rows)
            with self.assertRaisesRegex(AUDIT.AuditFailure, "expert membership/list changed"):
                AUDIT.audit_kimi(raw, canonical)

    def test_kimi_dropped_event_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw, canonical, _, canonical_rows = self._kimi_fixture(directory)
            del canonical_rows[100]
            _write_jsonl(canonical, canonical_rows)
            with self.assertRaisesRegex(AUDIT.AuditFailure, "expected 7800 canonical events"):
                AUDIT.audit_kimi(raw, canonical)

    def test_kimi_reordered_events_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw, canonical, _, canonical_rows = self._kimi_fixture(directory)
            canonical_rows[1], canonical_rows[2] = canonical_rows[2], canonical_rows[1]
            _write_jsonl(canonical, canonical_rows)
            with self.assertRaisesRegex(AUDIT.AuditFailure, "layer mapping mismatch"):
                AUDIT.audit_kimi(raw, canonical)

    def test_kimi_wrong_raw_routed_layer_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw, canonical, raw_rows, canonical_rows = self._kimi_fixture(directory)
            raw_rows[0]["layers"] = raw_rows[0]["layers"][:-1]
            _write_jsonl(raw, raw_rows)
            _write_jsonl(canonical, canonical_rows)
            with self.assertRaisesRegex(AUDIT.AuditFailure, "expected 26 routed layers"):
                AUDIT.audit_kimi(raw, canonical)

    def test_olmoe_full_preservation_audit_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw_rows = _olmoe_raw_rows()
            canonical_rows = _olmoe_canonical_rows(raw_rows)
            raw = directory / "olmoe-raw.jsonl"
            canonical = directory / "olmoe-canonical.jsonl"
            _write_jsonl(raw, raw_rows)
            _write_jsonl(canonical, canonical_rows)
            result = AUDIT.audit_olmoe(raw, canonical)
        self.assertEqual(result["preservation_audit"], "PASS")
        self.assertEqual(result["canonical_events"], 4_784)
        self.assertEqual(result["expert_requests"], 38_272)
        self.assertEqual((result["layer_min"], result["layer_max"]), (0, 15))


if __name__ == "__main__":
    unittest.main()
