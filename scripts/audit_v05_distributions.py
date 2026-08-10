"""Deterministic v0.5 wheel/sdist audit."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import tarfile
import zipfile


EXPECTED_NAME = "moe-cache-lab"
EXPECTED_VERSION = "0.5.0"
EXPECTED_LICENSE_EXPRESSION = "Apache-2.0"
EXPECTED_SDIST_ROOT = "moe_cache_lab-0.5.0"
REQUIRED_SDIST_PATHS = (
    "LICENSE",
    "MANIFEST.in",
    "PREFLIGHT.md",
    "README.md",
    "V05_RELEASE_NOTES.md",
    "V05_VALIDATION.md",
    "examples/no-download-preflight/README.md",
    "examples/no-download-preflight/expected.sha256",
    "examples/no-download-preflight/preflight-config.json",
    "examples/no-download-preflight/trace.jsonl",
    "pyproject.toml",
    "setup.py",
    "src/moe_cache_lab/__init__.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_wheel(path: Path) -> None:
    _require(path.is_file(), f"wheel does not exist: {path}")
    _require(
        path.name == "moe_cache_lab-0.5.0-py3-none-any.whl",
        f"unexpected wheel filename: {path.name}",
    )

    with zipfile.ZipFile(path) as archive:
        names = tuple(archive.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        _require(len(metadata_names) == 1, "wheel must contain exactly one METADATA file")
        metadata_text = archive.read(metadata_names[0]).decode("utf-8")
        metadata = Parser().parsestr(metadata_text)

        _require(metadata.get("Name") == EXPECTED_NAME, "wheel Name metadata mismatch")
        _require(metadata.get("Version") == EXPECTED_VERSION, "wheel Version metadata mismatch")
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


def audit_sdist(path: Path) -> None:
    _require(path.is_file(), f"sdist does not exist: {path}")
    _require(path.name == "moe_cache_lab-0.5.0.tar.gz", f"unexpected sdist filename: {path.name}")

    with tarfile.open(path, mode="r:gz") as archive:
        file_names = {member.name for member in archive.getmembers() if member.isfile()}

    roots = {name.split("/", 1)[0] for name in file_names}
    _require(roots == {EXPECTED_SDIST_ROOT}, f"unexpected sdist archive roots: {sorted(roots)}")

    missing = [
        relative
        for relative in REQUIRED_SDIST_PATHS
        if f"{EXPECTED_SDIST_ROOT}/{relative}" not in file_names
    ]
    _require(not missing, f"sdist is missing required files: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--sdist", required=True, type=Path)
    args = parser.parse_args(argv)

    audit_wheel(args.wheel)
    audit_sdist(args.sdist)
    print(f"wheel audit: OK ({args.wheel.name})")
    print(f"sdist audit: OK ({args.sdist.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
