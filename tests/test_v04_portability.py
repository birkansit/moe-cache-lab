from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from moe_cache_lab.v04_aggregation import aggregate_children


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "results" / "v0.4-cpu-profiler-attempt-b247eb85-eb21-4e3a-886e-6ba250cbb980"
RESULT = ATTEMPT / "v04-result.json"
REPORT = ATTEMPT / "v04-report.md"
RESULT_SHA256 = "ed857ef581f80fbbe8e694bc869e9265e785e5c48bd6e00118b20db7d85e7b72"
REPORT_SHA256 = "d9b5399a125d162c3c7cdb3f847f30c196b56fd31ace9ee714f9dd4d5b81ceaf"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate_input() -> list[dict]:
    manifest = json.loads((ATTEMPT / "v04-set.json").read_text(encoding="utf-8"))
    children = [
        json.loads((ATTEMPT / reference["path"]).read_text(encoding="utf-8"))
        for reference in manifest["children"]
    ]
    return [
        {
            "child_number": child["child_number"],
            "evaluation": [
                {
                    name: prompt[name]
                    for name in (
                        "prompt_id",
                        "mode_order",
                        "reference_total_ns",
                        "profiled_total_ns",
                        "reference_working_set_delta",
                        "profiled_working_set_delta",
                        "reference_private_delta",
                        "profiled_private_delta",
                    )
                }
                for prompt in child["evaluation"]
            ],
        }
        for child in children
    ]


class V04HistoricalPortabilityTests(unittest.TestCase):
    def test_aggregate_is_exactly_the_frozen_python_310_result(self) -> None:
        regenerated = aggregate_children(_aggregate_input())
        recorded = json.loads(RESULT.read_text(encoding="utf-8"))

        self.assertEqual(regenerated, recorded)
        payload = (
            json.dumps(regenerated, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        self.assertEqual(payload, RESULT.read_bytes())

        # These fields cover the historical sample-SD and CV paths, including
        # the one-ULP cv_sum_reference divergence seen with statistics.stdev on
        # newer CPython versions.
        self.assertEqual(regenerated["child_ratio_stats"]["sample_sd"], 0.006958304661225125)
        self.assertEqual(regenerated["child_delta_stats"]["sample_sd"], 94762385.0678457)
        self.assertEqual(regenerated["prompt_macro"]["sample_sd"], 0.006049666403458691)
        self.assertEqual(regenerated["cv_sum_reference"], 0.024856668659024556)
        self.assertEqual(regenerated["cv_sum_profiled"], 0.02629763658060313)
        self.assertEqual(regenerated["cv_paired_ratio"], 0.006815956024537018)

        self.assertEqual(_sha256(RESULT), RESULT_SHA256)
        self.assertEqual(_sha256(REPORT), REPORT_SHA256)


if __name__ == "__main__":
    unittest.main()
