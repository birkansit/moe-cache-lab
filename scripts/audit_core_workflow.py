"""Exercise the installed core-only release workflow without ML dependencies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile


EXPECTED_VERSION = "0.8.0"
REQUIRED_DOCS = (
    "README.md",
    "COMPATIBILITY.md",
    "CONTRIBUTING.md",
    "EXTERNAL_RUNTIME_INTEROP.md",
    "PREFLIGHT.md",
    "PRODUCER_CONFORMANCE.md",
    "TRACE_FORMAT.md",
    "V05_RELEASE_NOTES.md",
    "V06_RELEASE_NOTES.md",
    "V07_RELEASE_NOTES.md",
    "V08_RELEASE_NOTES.md",
    "WALKTHROUGH.md",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    expected_returncode: int = 0,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != expected_returncode:
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected_returncode}: "
            f"{arguments!r}\nstdout={completed.stdout.decode(errors='replace')}\n"
            f"stderr={completed.stderr.decode(errors='replace')}"
        )
    return completed


def _cli(arguments: list[str], *, cwd: Path, expected_returncode: int = 0):
    return _run(
        [sys.executable, "-m", "moe_cache_lab.cli", *arguments],
        cwd=cwd,
        expected_returncode=expected_returncode,
    )


def _read_hash_manifest(path: Path) -> dict[str, str]:
    return {
        name: digest
        for digest, name in (
            line.split(maxsplit=1)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def _verify_hashes(root: Path, manifest: Path) -> None:
    for relative, expected in _read_hash_manifest(manifest).items():
        path = root / relative
        _require(path.is_file(), f"expected artifact is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == expected, f"SHA-256 mismatch for {relative}: {actual}")


def audit(source_root: Path, expected_version: str) -> None:
    source_root = source_root.resolve()
    _require((source_root / "pyproject.toml").is_file(), "source root is invalid")
    _require(importlib.util.find_spec("torch") is None, "Torch must be absent")
    _require(
        importlib.util.find_spec("transformers") is None,
        "Transformers must be absent",
    )

    import moe_cache_lab
    from moe_cache_lab.trace import load_trace_schema
    from moe_cache_lab.trace_v2 import load_trace_v2_schema

    package_path = Path(moe_cache_lab.__file__).resolve()
    _require(
        "site-packages" in package_path.as_posix(),
        f"audit must use an installed package, got {package_path}",
    )
    _require(moe_cache_lab.__version__ == expected_version, "package version mismatch")
    _require(
        load_trace_schema()["$id"] == "urn:moe-cache-lab:schema:routing-trace:1",
        "trace-v1 schema is unavailable",
    )
    _require(
        load_trace_v2_schema()["$id"] == "urn:moe-cache-lab:schema:routing-trace:2",
        "trace-v2 schema is unavailable",
    )

    console = Path(sysconfig.get_path("scripts")) / (
        "moe-cache-lab.exe" if os.name == "nt" else "moe-cache-lab"
    )
    _require(console.is_file(), "installed moe-cache-lab console script is missing")

    doc_root = Path(sys.prefix) / "share" / "doc" / "moe-cache-lab"
    for name in REQUIRED_DOCS:
        _require((doc_root / name).is_file(), f"installed documentation is missing {name}")

    installed_example = (
        Path(sys.prefix)
        / "share"
        / "moe-cache-lab"
        / "examples"
        / "external-producer-no-model"
    )
    for name in ("README.md", "expected.sha256", "preflight-config.json", "producer.py"):
        _require((installed_example / name).is_file(), f"installed example is missing {name}")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _run([str(console), "--help"], cwd=root)

        fixtures = source_root / "examples" / "trace-validation-fixtures"
        valid_v1 = fixtures / "valid" / "v1-minimal.jsonl"
        invalid = fixtures / "invalid" / "chronology-regression.jsonl"
        valid = _cli(["validate-trace", str(valid_v1)], cwd=root)
        _require(b"Trace validation: VALID" in valid.stdout, "valid trace was not accepted")
        rejected = _cli(
            ["validate-trace", str(invalid)],
            cwd=root,
            expected_returncode=2,
        )
        _require(not rejected.stdout, "invalid human validation wrote stdout")
        _require(
            b"Error code: chronology_regression" in rejected.stderr,
            "invalid chronology did not retain its stable error category",
        )

        v1_example = source_root / "examples" / "no-download-preflight"
        _cli(
            [
                "analyze",
                str(v1_example / "trace.jsonl"),
                "--output",
                "v1-routing.md",
                "--json-output",
                "v1-routing.json",
            ],
            cwd=root,
        )
        _cli(
            [
                "analyze",
                str(v1_example / "trace.jsonl"),
                "--preflight-config",
                str(v1_example / "preflight-config.json"),
                "--output",
                "preflight-report.md",
                "--json-output",
                "preflight-report.json",
            ],
            cwd=root,
        )
        _verify_hashes(root, v1_example / "expected.sha256")

        v2_example = source_root / "examples" / "no-download-stage-qualified-preflight"
        _cli(
            [
                "analyze",
                str(v2_example / "trace.jsonl"),
                "--preflight-config",
                str(v2_example / "preflight-config.json"),
                "--workload-id",
                "synthetic-stage-qualified-demo",
                "--output",
                "stage-qualified-preflight-report.md",
                "--json-output",
                "stage-qualified-preflight-report.json",
            ],
            cwd=root,
        )
        _verify_hashes(root, v2_example / "expected.sha256")

        workflow = root / "external-producer-no-model"
        shutil.copytree(installed_example, workflow)
        artifacts = workflow / "artifacts"
        artifacts.mkdir()
        _run(
            [sys.executable, "producer.py", "--output", "artifacts/trace.jsonl"],
            cwd=workflow,
        )
        _cli(["validate-trace", "artifacts/trace.jsonl"], cwd=workflow)
        validation = _cli(
            ["validate-trace", "artifacts/trace.jsonl", "--json"],
            cwd=workflow,
        )
        (artifacts / "validation.json").write_bytes(
            validation.stdout.replace(b"\r\n", b"\n")
        )
        _cli(
            [
                "analyze",
                "artifacts/trace.jsonl",
                "--workload-id",
                "external-producer-demo",
                "--output",
                "artifacts/routing-analysis.md",
                "--json-output",
                "artifacts/routing-analysis.json",
            ],
            cwd=workflow,
        )
        _cli(
            [
                "analyze",
                "artifacts/trace.jsonl",
                "--preflight-config",
                "preflight-config.json",
                "--workload-id",
                "external-producer-demo",
                "--output",
                "artifacts/preflight-report.md",
                "--json-output",
                "artifacts/preflight-report.json",
            ],
            cwd=workflow,
        )
        incompatible = _cli(
            [
                "analyze",
                "artifacts/trace.jsonl",
                "--preflight-config",
                str(v1_example / "preflight-config.json"),
                "--output",
                "artifacts/incompatible.md",
                "--json-output",
                "artifacts/incompatible.json",
            ],
            cwd=workflow,
            expected_returncode=2,
        )
        _require(
            b"requires stage-qualified preflight config format_version 3"
            in incompatible.stderr,
            "trace-v2/config-v1 rejection boundary changed",
        )
        _require(
            not (artifacts / "incompatible.md").exists()
            and not (artifacts / "incompatible.json").exists(),
            "incompatible trace/config pair created output",
        )
        _verify_hashes(workflow, workflow / "expected.sha256")
        _cli(
            [
                "bundle-create",
                "--experiment-id",
                "external-producer-no-model",
                "--config",
                "preflight-config.json",
                "--report-json",
                "artifacts/preflight-report.json",
                "--report-markdown",
                "artifacts/preflight-report.md",
                "--embed-trace",
                "synthetic-external",
                "artifacts/trace.jsonl",
                "--output-dir",
                "artifacts/bundle",
            ],
            cwd=workflow,
        )
        _cli(["bundle-verify", "artifacts/bundle"], cwd=workflow)

        validation_data = json.loads((artifacts / "validation.json").read_text())
        preflight_data = json.loads(
            (artifacts / "preflight-report.json").read_text(encoding="utf-8")
        )
        _require(validation_data["trace_format_version"] == 2, "C7 trace is not v2")
        _require(
            preflight_data["claim_boundary"]["cache_outcomes"] == "SIMULATED",
            "C7 cache evidence boundary changed",
        )
        _require(
            preflight_data["claim_boundary"]["transfer_service"] == "ESTIMATED",
            "C7 transfer evidence boundary changed",
        )
        _require(
            preflight_data["claim_boundary"]["runtime_performance_and_residency"]
            == "NOT ESTABLISHED",
            "C7 runtime evidence boundary changed",
        )

    _require(importlib.util.find_spec("torch") is None, "Torch appeared during audit")
    _require(
        importlib.util.find_spec("transformers") is None,
        "Transformers appeared during audit",
    )
    print(
        "installed core workflow audit: OK "
        f"(Python {sys.version_info.major}.{sys.version_info.minor}, package {expected_version})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-version", default=EXPECTED_VERSION)
    args = parser.parse_args(argv)
    audit(args.source_root, args.expected_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
