import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_cache_lab import __version__
from moe_cache_lab.experiment_bundle import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    EmbeddedTrace,
    ExternalTrace,
    RuntimeProvenance,
    verify_experiment_bundle,
    write_experiment_bundle,
)
from moe_cache_lab.trace import RoutingEvent, RoutingTrace, write_trace
from moe_cache_lab.trace_v2 import (
    RoutingEventV2,
    RoutingStageProfile,
    RoutingTraceV2,
    write_trace_v2,
)


def _v1(path: Path) -> Path:
    return write_trace(
        path,
        RoutingTrace(
            model_id="synthetic/v1",
            num_experts=4,
            experts_per_token=1,
            events=(
                RoutingEvent("prompt", 0, 0, (1,), token_id=10),
                RoutingEvent("generated", 1, 0, (2,), token_id=11),
            ),
            capture_method="synthetic",
            created_at="fixed",
        ),
    )


def _v2(path: Path) -> Path:
    return write_trace_v2(
        path,
        RoutingTraceV2(
            model_id="synthetic/v2",
            routing_stages=(
                RoutingStageProfile("encoder", 4, 1, True),
                RoutingStageProfile("decoder", 8, 1, True),
            ),
            events=(
                RoutingEventV2(
                    "encoder", "source", 0, 1, "assigned", (3,), (0.75,), 10
                ),
                RoutingEventV2(
                    "decoder",
                    "decoder_prompt",
                    0,
                    1,
                    "unassigned",
                    (),
                    (),
                    0,
                    "capacity",
                ),
            ),
            capture_method="synthetic",
            created_at="fixed",
        ),
    )


def _write_bundle(root: Path, traces=(), **updates):
    arguments = {
        "experiment_id": "experiment-01",
        "config": {"z": [3, 2, 1], "a": {"enabled": True}},
        "report_json": b'{"result":"stable","value":1}\n',
        "report_markdown": b"# Stable report\n\nExact bytes.\n",
        "traces": traces,
        "runtime_provenance": RuntimeProvenance(),
    }
    arguments.update(updates)
    return write_experiment_bundle(root, **arguments)


def _manifest(root: Path):
    return json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))


def _rewrite_manifest(root: Path, mutate) -> None:
    path = root / "bundle-manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (root / "bundle-manifest.sha256").write_text(
        f"{digest}  bundle-manifest.json\n",
        encoding="ascii",
        newline="\n",
    )


def _inventory(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class ExperimentBundleTests(unittest.TestCase):
    def test_same_inputs_in_different_parents_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source_a = _v1(parent / "source-a" / "trace.jsonl")
            source_b = parent / "other-source" / "renamed.jsonl"
            source_b.parent.mkdir()
            source_b.write_bytes(source_a.read_bytes())
            first = parent / "parent-a" / "bundle"
            second = parent / "parent-b" / "bundle"

            result_a = _write_bundle(first, (EmbeddedTrace("workload", source_a),))
            result_b = _write_bundle(second, (EmbeddedTrace("workload", source_b),))

            self.assertEqual(_inventory(first), _inventory(second))
            self.assertEqual(result_a.manifest_sha256, result_b.manifest_sha256)
            serialized = b"".join(_inventory(first).values())
            self.assertNotIn(str(source_a).encode(), serialized)
            self.assertNotIn(str(source_b).encode(), serialized)

    def test_tool_environment_and_optional_runtime_provenance_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            _write_bundle(
                root,
                runtime_provenance=RuntimeProvenance(
                    torch_version="2.12.0+cpu",
                    transformers_version="5.12.0",
                    device="cpu",
                    dtype="float32",
                ),
            )
            manifest = _manifest(root)
            environment = json.loads((root / "environment.json").read_text())

        self.assertEqual(manifest["moe_cache_lab_version"], __version__)
        self.assertEqual(environment["moe_cache_lab_version"], __version__)
        self.assertEqual(
            environment["runtime"],
            {
                "torch_version": "2.12.0+cpu",
                "transformers_version": "5.12.0",
                "device": "cpu",
                "dtype": "float32",
            },
        )
        keys = set(environment) | set(environment["python"]) | set(environment["platform"])
        self.assertTrue(
            {"executable", "home", "hostname", "username", "cwd"}.isdisjoint(keys)
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            _write_bundle(root)
            unavailable = json.loads((root / "environment.json").read_text())["runtime"]
        self.assertTrue(all(value is None for value in unavailable.values()))

    def test_config_is_canonical_and_rejects_non_json_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            _write_bundle(root, config={"z": 1, "a": [True, None]})
            self.assertEqual(
                (root / "bundle-config.json").read_bytes(),
                b'{"a":[true,null],"z":1}\n',
            )

        invalid = (
            {"value": float("nan")},
            {"value": float("inf")},
            {"value": float("-inf")},
            {"value": {1, 2}},
            {1: "non-string-key"},
            {"value": (1, 2)},
        )
        for index, config in enumerate(invalid):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    _write_bundle(Path(temporary) / f"bundle-{index}", config=config)

        circular = {}
        circular["self"] = circular
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            ValueError, "circular"
        ):
            _write_bundle(Path(temporary) / "bundle-circular", config=circular)

    def test_report_bytes_are_exact_and_strict_json_is_required(self) -> None:
        report_json = b'{"b":2,"a":1}\r\n'
        report_markdown = "# Report\r\nopaque\r\n".encode()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            _write_bundle(
                root, report_json=report_json, report_markdown=report_markdown
            )
            manifest = _manifest(root)
            records = {item["role"]: item for item in manifest["artifacts"]}
            self.assertEqual((root / "report.json").read_bytes(), report_json)
            self.assertEqual((root / "report.md").read_bytes(), report_markdown)
            self.assertEqual(records["report_json"]["size_bytes"], len(report_json))
            self.assertEqual(
                records["report_markdown"]["sha256"],
                hashlib.sha256(report_markdown).hexdigest(),
            )

        invalid = (
            b"{",
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":1e999}',
            b'{"x":1,"x":2}',
        )
        for index, report in enumerate(invalid):
            with self.subTest(report=report), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    _write_bundle(
                        Path(temporary) / f"bundle-{index}", report_json=report
                    )

    def test_report_tampering_and_extra_inventory_fail_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            _write_bundle(root)
            before = (root / "bundle-manifest.json").read_bytes()
            report_path = root / "report.md"
            original = report_path.read_bytes()
            report_path.write_bytes(b"!" + original[1:])
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_experiment_bundle(root)
            self.assertEqual((root / "bundle-manifest.json").read_bytes(), before)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            _write_bundle(root)
            (root / "unlisted.txt").write_text("extra")
            with self.assertRaisesRegex(ValueError, "inventory"):
                verify_experiment_bundle(root)

    def test_verifier_rejects_rehashed_nonstandard_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            _write_bundle(root)
            report = b'{"value":1e999}\n'
            (root / "report.json").write_bytes(report)

            def update(value):
                record = next(
                    item for item in value["artifacts"] if item["role"] == "report_json"
                )
                record["size_bytes"] = len(report)
                record["sha256"] = hashlib.sha256(report).hexdigest()

            _rewrite_manifest(root, update)
            with self.assertRaisesRegex(ValueError, "strict JSON"):
                verify_experiment_bundle(root)

    def test_embedded_v1_and_v2_copy_exact_bytes_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            v1 = _v1(parent / "inputs" / "v1.jsonl")
            v2 = _v2(parent / "inputs" / "v2.jsonl")
            root = parent / "bundle"
            result = _write_bundle(
                root,
                (EmbeddedTrace("granite", v1), EmbeddedTrace("switch", v2)),
            )

            self.assertEqual(
                (root / "traces" / "granite.jsonl").read_bytes(), v1.read_bytes()
            )
            self.assertEqual(
                (root / "traces" / "switch.jsonl").read_bytes(), v2.read_bytes()
            )
            records = _manifest(root)["traces"]
            self.assertEqual(
                [(item["trace_id"], item["trace_format_version"]) for item in records],
                [("granite", 1), ("switch", 2)],
            )
            self.assertEqual((result.artifact_count, result.trace_count), (6, 2))
            self.assertEqual(verify_experiment_bundle(root).manifest_sha256, result.manifest_sha256)

    def test_chronology_invalid_embedded_trace_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            valid = write_trace(
                parent / "valid.jsonl",
                RoutingTrace(
                    "synthetic/v1",
                    4,
                    1,
                    (
                        RoutingEvent("prompt", 0, 0, (0,)),
                        RoutingEvent("prompt", 0, 1, (1,)),
                    ),
                    created_at="fixed",
                ),
            )
            lines = valid.read_bytes().splitlines(keepends=True)
            invalid = parent / "invalid.jsonl"
            invalid.write_bytes(lines[0] + lines[2] + lines[1])
            root = parent / "bundle"
            with self.assertRaisesRegex(ValueError, "layer-major"):
                _write_bundle(root, (EmbeddedTrace("broken", invalid),))
            self.assertFalse(root.exists())

    def test_external_reference_is_metadata_only_and_never_fetched(self) -> None:
        reference = ExternalTrace(
            "private-trace",
            "https://example.invalid/private/trace.jsonl",
            "a" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("external trace must never be fetched"),
        ):
            root = Path(temporary) / "bundle"
            _write_bundle(root, (reference,))
            result = verify_experiment_bundle(root)
            manifest = _manifest(root)
            self.assertEqual(result.trace_count, 1)
            self.assertEqual(
                manifest["traces"],
                [
                    {
                        "trace_id": "private-trace",
                        "mode": "external",
                        "reference": reference.reference,
                        "sha256": "a" * 64,
                    }
                ],
            )
            self.assertFalse((root / "traces").exists())

    def test_external_local_paths_and_invalid_hashes_reject(self) -> None:
        references = (
            "/home/user/private.jsonl",
            r"C:\Users\user\private.jsonl",
            r"\\server\share\private.jsonl",
            "file:///home/user/private.jsonl",
            "FILE://C:/private.jsonl",
            "../private.jsonl",
        )
        for reference in references:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                ExternalTrace("private", reference, "a" * 64)
        with self.assertRaises(ValueError):
            ExternalTrace("private", "logical:trace", "A" * 64)

    def test_duplicate_and_unsafe_ids_reject(self) -> None:
        duplicate = (
            ExternalTrace("same", "logical:first", "a" * 64),
            ExternalTrace("same", "logical:second", "b" * 64),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unique"):
                _write_bundle(Path(temporary) / "bundle", duplicate)
        for invalid in ("../escape", "a/b", "", ".", "name with spaces"):
            with self.subTest(value=invalid), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    _write_bundle(Path(temporary) / "bundle", experiment_id=invalid)

    def test_missing_and_tampered_embedded_trace_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = _v1(parent / "source.jsonl")
            root = parent / "bundle"
            _write_bundle(root, (EmbeddedTrace("trace", source),))
            embedded = root / "traces" / "trace.jsonl"
            embedded.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_experiment_bundle(root)

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = _v1(parent / "source.jsonl")
            root = parent / "bundle"
            _write_bundle(root, (EmbeddedTrace("trace", source),))
            embedded = root / "traces" / "trace.jsonl"
            embedded.write_bytes(embedded.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_experiment_bundle(root)

    def test_embedded_trace_manifest_version_must_match_exact_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = _v1(parent / "source.jsonl")
            root = parent / "bundle"
            _write_bundle(root, (EmbeddedTrace("trace", source),))
            _rewrite_manifest(
                root,
                lambda value: value["traces"][0].update(trace_format_version=2),
            )
            with self.assertRaisesRegex(ValueError, "format/version metadata mismatch"):
                verify_experiment_bundle(root)

    def test_manifest_anchor_version_unknown_fields_traversal_and_collision_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            _write_bundle(root)
            manifest_path = root / "bundle-manifest.json"
            manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "sidecar mismatch"):
                verify_experiment_bundle(root)

        mutations = (
            lambda value: value.update(format_version=2),
            lambda value: value.update(format_version=1.0),
            lambda value: value.update(unknown="forbidden"),
            lambda value: value["artifacts"][0].update(path="../escape.json"),
            lambda value: value["artifacts"][1].update(path="bundle-config.json"),
        )
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "bundle"
                _write_bundle(root)
                _rewrite_manifest(root, mutation)
                with self.assertRaises(ValueError):
                    verify_experiment_bundle(root)

    def test_manifest_is_closed_versioned_and_contains_no_generated_local_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            result = _write_bundle(root)
            manifest = _manifest(root)
            serialized = b"".join(_inventory(root).values()).decode("utf-8")

        self.assertEqual(manifest["format"], BUNDLE_FORMAT)
        self.assertEqual(manifest["format_version"], BUNDLE_VERSION)
        self.assertEqual(
            set(manifest),
            {
                "format",
                "format_version",
                "experiment_id",
                "moe_cache_lab_version",
                "artifacts",
                "traces",
            },
        )
        self.assertEqual(len(result.manifest_sha256), 64)
        lowered = serialized.lower()
        for forbidden in (
            "timestamp",
            "created_at",
            "uuid",
            "temporary",
            "sys.executable",
            "hostname",
            "username",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
