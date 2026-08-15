from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from moe_cache_lab.v04 import (
    V04_CHILD_FORMAT,
    V04_RESULT_FORMAT,
    aggregate_children,
    load_v04_json,
    render_markdown,
    validate_v04,
)


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_UUID = "b247eb85-eb21-4e3a-886e-6ba250cbb980"
ATTEMPT = ROOT / "results" / f"v0.4-cpu-profiler-attempt-{ATTEMPT_UUID}"
DEFERRED_UUID = "d6f7cac2-d67a-46e8-8267-d65c861ea009"
HASHES = {
    "admission-1.json": "e7b10ae3b54ab54c7f9e303243ca5424769629809d9bbec9c44d565461a06fc2",
    "admission-2.json": "9141fdd218e99f59270478dbbd709aad7971a4448f0f7fc06f95c8516307cbac",
    "admission-3.json": "2a601e9b72cfb3ccddce13f359bc362a957552ce971a685ded351479a99e5815",
    "admission-4.json": "6072af8271fa7348a880fa47f50c4aab87a840ded4e66dba724bd7b119d839e8",
    "child-1/v04-child.json": "34e78cffa893db40f6e6e777c10d919d4bd910d501e0ae94c17151b24eb29a10",
    "child-2/v04-child.json": "140bfc12a8d57b93c16d3fd9d1170e36e65a5f9ce3b1ac92434024492d6c533f",
    "child-3/v04-child.json": "e24f9f0046057f1a9a6dc0568f122a103eadfd64df03d61815241458549f08dd",
    "child-4/v04-child.json": "322645b181646cb2f566ac929fcc5fc4d318de5200e95278b1ebd6542debc3e7",
    "v04-report.md": "d9b5399a125d162c3c7cdb3f847f30c196b56fd31ace9ee714f9dd4d5b81ceaf",
    "v04-result.json": "ed857ef581f80fbbe8e694bc869e9265e785e5c48bd6e00118b20db7d85e7b72",
    "v04-set.json": "1f87491f3d0512912e593422a2cc0b012ebed343291726131fa51dc2e103fe60",
}
RATIOS = (
    1.011750743135379,
    1.023570952041221,
    1.0199959335756976,
    1.028220842549842,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profiled_runs(child: dict) -> list[dict]:
    return [run for run in child["calibration"] if run["mode"] == "profiled_exact"] + [
        run
        for prompt in child["evaluation"]
        for run in prompt["runs"]
        if run["mode"] == "profiled_exact"
    ]


class V04ValidEvidenceTests(unittest.TestCase):
    def test_valid_stable_attempt_is_pinned_reproducible_and_bounded(self):
        self.assertTrue(ATTEMPT.is_dir())
        self.assertFalse(ATTEMPT.is_symlink())
        entries = list(ATTEMPT.rglob("*"))
        self.assertTrue(all(not path.is_symlink() for path in entries))
        files = {path.relative_to(ATTEMPT).as_posix(): path for path in entries if path.is_file()}
        directories = {path.relative_to(ATTEMPT).as_posix() for path in entries if path.is_dir()}
        self.assertEqual(directories, {"child-1", "child-2", "child-3", "child-4"})
        self.assertEqual(set(files), set(HASHES))
        self.assertEqual({name: _sha256(files[name]) for name in sorted(files)}, HASHES)

        manifest_path = ATTEMPT / "v04-set.json"
        self.assertEqual(validate_v04(manifest_path), manifest_path.resolve())
        manifest = load_v04_json(manifest_path)
        self.assertEqual(manifest["attempt_uuid"], ATTEMPT_UUID)
        self.assertEqual(manifest["state"], "valid_stable")
        self.assertEqual(manifest["retry_of"], DEFERRED_UUID)
        self.assertEqual(
            manifest["retry_source_path"],
            f"v0.4-cpu-profiler-attempt-{DEFERRED_UUID}/v04-set.json",
        )
        self.assertEqual(manifest["retry_source_state"], "deferred")
        self.assertEqual(manifest["retry_source_reason"], "host busy or memory gate failed")
        self.assertEqual(
            manifest["retry_source_sha256"],
            "5bb411929ebaf0dd7b149781ac3f1a7808e3050df62b11ce9be290c8a5ec2a31",
        )
        self.assertEqual(len(manifest["admissions"]), 4)
        self.assertEqual(len(manifest["children"]), 4)
        self.assertEqual(manifest["failures"], [])
        self.assertTrue(all(load_v04_json(ATTEMPT / ref["path"])["admitted"] for ref in manifest["admissions"]))

        children = [
            load_v04_json(ATTEMPT / ref["path"], expected_format=V04_CHILD_FORMAT)
            for ref in manifest["children"]
        ]
        self.assertEqual([child["child_number"] for child in children], [1, 2, 3, 4])
        baseline_events = None
        for child in children:
            self.assertEqual(len(child["calibration"]), 8)
            self.assertEqual(len(child["evaluation"]), 8)
            calibration = [run for run in child["calibration"] if run["mode"] == "profiled_exact"]
            evaluation = [
                run for prompt in child["evaluation"] for run in prompt["runs"]
                if run["mode"] == "profiled_exact"
            ]
            self.assertEqual(sum(run["event_count"] for run in calibration), 2664)
            self.assertEqual(sum(run["assignment_count"] for run in calibration), 21312)
            self.assertEqual(sum(run["event_count"] for run in evaluation), 5832)
            self.assertEqual(sum(run["assignment_count"] for run in evaluation), 46656)
            self.assertTrue(all(run["semantic"]["actual_decode_input_steps"] == 16 for run in _profiled_runs(child)))
            self.assertTrue(all(run["semantic"]["horizon_exhausted"] for run in _profiled_runs(child)))
            events = [event for run in _profiled_runs(child) for event in run["events"]]
            if baseline_events is None:
                baseline_events = events
            else:
                self.assertEqual(len(events), len(baseline_events))
                maximum_probability_delta = max(
                    abs(actual - expected)
                    for left, right in zip(events, baseline_events)
                    for actual, expected in zip(
                        left["selected_probabilities"], right["selected_probabilities"]
                    )
                )
                self.assertEqual(maximum_probability_delta, 0.0)
                self.assertTrue(all(
                    left["selected_experts"] == right["selected_experts"]
                    and left["token_id"] == right["token_id"]
                    for left, right in zip(events, baseline_events)
                ))

        aggregate_input = [{
            "child_number": child["child_number"],
            "evaluation": [{name: prompt[name] for name in (
                "prompt_id", "mode_order", "reference_total_ns", "profiled_total_ns",
                "reference_working_set_delta", "profiled_working_set_delta",
                "reference_private_delta", "profiled_private_delta",
            )} for prompt in child["evaluation"]],
        } for child in children]
        regenerated = aggregate_children(aggregate_input)
        recorded = load_v04_json(ATTEMPT / "v04-result.json", expected_format=V04_RESULT_FORMAT)
        self.assertEqual(regenerated, recorded)
        regenerated_json = (
            json.dumps(regenerated, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        self.assertEqual(regenerated_json, (ATTEMPT / "v04-result.json").read_bytes())
        self.assertEqual(render_markdown(regenerated), (ATTEMPT / "v04-report.md").read_text(encoding="utf-8"))

        self.assertEqual(recorded["decision"], "valid_stable")
        self.assertEqual(tuple(row["ratio"] for row in recorded["children"]), RATIOS)
        self.assertLessEqual(recorded["cv_sum_reference"], 0.10)
        self.assertLessEqual(recorded["cv_sum_profiled"], 0.10)
        self.assertLessEqual(recorded["cv_paired_ratio"], 0.10)
        self.assertLessEqual(recorded["paired_ratio_span"], 0.15)

        report = (ATTEMPT / "v04-report.md").read_text(encoding="utf-8")
        for boundary in (
            "Measured CPU wall-time",
            "process-memory snapshots are descriptive and order-conditioned",
            "No GPU, offload, cache speedup, or production-throughput claim",
            "paired repeatability checks, not independent workload samples",
            "no per-mode peak overhead",
        ):
            self.assertIn(boundary, report)
        for overclaim in ("achieved speedup", "production acceleration", "GPU result"):
            self.assertNotIn(overclaim, report)


if __name__ == "__main__":
    unittest.main()
