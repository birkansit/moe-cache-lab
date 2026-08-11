"""Deterministically audit one moe-cache-lab wheel/sdist version pair."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import re
import tarfile
import zipfile


EXPECTED_NAME = "moe-cache-lab"
EXPECTED_LICENSE_EXPRESSION = "Apache-2.0"
REQUIRED_PACKAGE_MODULES = (
    "moe_cache_lab/__init__.py",
    "moe_cache_lab/byte_cache.py",
    "moe_cache_lab/hardware_cost.py",
    "moe_cache_lab/preflight.py",
    "moe_cache_lab/preflight_output.py",
    "moe_cache_lab/sensitivity_summary.py",
)
REQUIRED_RELEASE_DOCS = (
    "README.md",
    "PREFLIGHT.md",
    "V06_RELEASE_NOTES.md",
)
REQUIRED_SDIST_PATHS = (
    "LICENSE",
    "MANIFEST.in",
    "PREFLIGHT.md",
    "README.md",
    "V05_RELEASE_NOTES.md",
    "V05_VALIDATION.md",
    "V06_RELEASE_NOTES.md",
    "examples/no-download-preflight/README.md",
    "examples/no-download-preflight/expected.sha256",
    "examples/no-download-preflight/preflight-config.json",
    "examples/no-download-preflight/trace.jsonl",
    "pyproject.toml",
    "scripts/audit_distributions.py",
    "setup.py",
    "src/moe_cache_lab/__init__.py",
    "src/moe_cache_lab/byte_cache.py",
    "src/moe_cache_lab/hardware_cost.py",
    "src/moe_cache_lab/preflight.py",
    "src/moe_cache_lab/preflight_output.py",
    "src/moe_cache_lab/sensitivity_summary.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_version(version: str) -> str:
    _require(
        bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}(?:rc[0-9]+)?", version)),
        "expected version must be a canonical release or rc version",
    )
    return version


def audit_wheel(path: Path, version: str) -> None:
    _require(path.is_file(), f"wheel does not exist: {path}")
    _require(
        path.name == f"moe_cache_lab-{version}-py3-none-any.whl",
        f"unexpected wheel filename: {path.name}",
    )

    with zipfile.ZipFile(path) as archive:
        names = tuple(archive.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        _require(len(metadata_names) == 1, "wheel must contain exactly one METADATA file")
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))

        _require(metadata.get("Name") == EXPECTED_NAME, "wheel Name metadata mismatch")
        _require(metadata.get("Version") == version, "wheel Version metadata mismatch")
        _require(
            metadata.get("License-Expression") == EXPECTED_LICENSE_EXPRESSION,
            "wheel License-Expression metadata mismatch",
        )
        _require(metadata.get("License") is None, "legacy wheel License metadata must be absent")
        _require(
            "LICENSE" in metadata.get_all("License-File", []),
            "wheel License-File metadata must include LICENSE",
        )
        _require(
            not any(value.startswith("License ::") for value in metadata.get_all("Classifier", [])),
            "legacy license classifier must be absent from wheel metadata",
        )
        _require(
            any(name.endswith(".dist-info/licenses/LICENSE") for name in names),
            "wheel must contain .dist-info/licenses/LICENSE",
        )
        missing_modules = [name for name in REQUIRED_PACKAGE_MODULES if name not in names]
        _require(
            not missing_modules,
            "wheel is missing required v0.6 modules: " + ", ".join(missing_modules),
        )
        docs_root = (
            f"moe_cache_lab-{version}.data/data/share/doc/moe-cache-lab"
        )
        missing_docs = [
            name for name in REQUIRED_RELEASE_DOCS
            if f"{docs_root}/{name}" not in names
        ]
        _require(
            not missing_docs,
            "wheel is missing release-facing docs: " + ", ".join(missing_docs),
        )


def audit_sdist(path: Path, version: str) -> None:
    _require(path.is_file(), f"sdist does not exist: {path}")
    _require(
        path.name == f"moe_cache_lab-{version}.tar.gz",
        f"unexpected sdist filename: {path.name}",
    )
    expected_root = f"moe_cache_lab-{version}"

    with tarfile.open(path, mode="r:gz") as archive:
        file_names = {member.name for member in archive.getmembers() if member.isfile()}

    roots = {name.split("/", 1)[0] for name in file_names}
    _require(roots == {expected_root}, f"unexpected sdist archive roots: {sorted(roots)}")
    missing = [
        relative
        for relative in REQUIRED_SDIST_PATHS
        if f"{expected_root}/{relative}" not in file_names
    ]
    _require(not missing, f"sdist is missing required files: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=_validate_version)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--sdist", required=True, type=Path)
    args = parser.parse_args(argv)

    audit_wheel(args.wheel, args.version)
    audit_sdist(args.sdist, args.version)
    print(f"wheel audit: OK ({args.wheel.name})")
    print(f"sdist audit: OK ({args.sdist.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
