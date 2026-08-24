"""Audit the tracked public-export candidate for publication hygiene.

The audit considers only paths returned by ``git ls-files --cached``. It never
scans Git history, GitHub state, or untracked files, and it never rewrites the
tree. Path rules apply to every tracked path.

Content scanning is deliberately not limited to a fixed filename/suffix list.
Known text paths must decode as UTF-8. Unknown regular files are inspected
deterministically: files containing NUL bytes or failing UTF-8 decoding are
classified as binary and skipped; other valid UTF-8 files are scanned even when
their name or suffix is unfamiliar. This catches extensionless/config/key text
without attempting to decode arbitrary binary blobs.

Each content allowlist is rule-, path-, and phrase-specific. One permits the
experiment-bundle test's synthetic absolute-path rejection fixtures. The other
preserves a hash-bound frozen V0.4 protocol phrase. Neither weakens its rule for
any other content.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Iterable, NamedTuple, Pattern


KNOWN_TEXT_SUFFIXES = frozenset(
    {
        ".cfg",
        ".cmd",
        ".env",
        ".in",
        ".ini",
        ".json",
        ".jsonl",
        ".key",
        ".md",
        ".pem",
        ".ps1",
        ".py",
        ".pyi",
        ".rst",
        ".sha256",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
KNOWN_TEXT_NAMES = frozenset(
    {
        ".dockerignore",
        ".env",
        ".gitattributes",
        ".gitignore",
        "Dockerfile",
        "LICENSE",
        "Makefile",
    }
)

GENERATED_DIRECTORY_NAMES = frozenset(
    {
        ".eggs",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "htmlcov",
        "venv",
    }
)
GENERATED_FILE_SUFFIXES = (".pyc", ".pyo")
MODEL_FILE_NAMES = frozenset(
    {
        "flax_model.msgpack",
        "generation_config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "spiece.model",
        "tf_model.h5",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
)
MODEL_FILE_SUFFIXES = (".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors")


class ContentRule(NamedTuple):
    rule_id: str
    description: str
    pattern: Pattern[str]


class Violation(NamedTuple):
    path: str
    rule_id: str
    description: str
    line: int | None = None


class AuditResult(NamedTuple):
    tracked_path_count: int
    text_path_count: int
    binary_path_count: int
    violations: tuple[Violation, ...]


_PRIVATE_REPOSITORY = "moe-cache-lab" + "-v05-dev"
_DIRECTOR = "direc" + "tor"
_WORKER = "work" + "er"
_CODEX = "Co" + "dex"
_INDEPENDENTLY = "independent" + "ly"
_REVIEWER = "review" + "er"
_AGENT = "ag" + "ent"
_ORCHESTRATION = "orchestra" + "tion"
_TRANSCRIPT = "trans" + "cript"
_SYSTEM = "sys" + "tem"
_DEVELOPER = "develop" + "er"
_PROMPT = "pro" + "mpt"
_CHAIN = "cha" + "in"
_THOUGHT = "thou" + "ght"

CONTENT_RULES = (
    ContentRule(
        "private-repository",
        "private development repository identity",
        re.compile(re.escape(_PRIVATE_REPOSITORY), re.IGNORECASE),
    ),
    ContentRule(
        "task-state-marker",
        "private task-state marker",
        re.compile(r"\[(?:READY|ACTIVE|COMPLETED)\]", re.IGNORECASE),
    ),
    ContentRule(
        "project-orchestration",
        "private project-coordination wording",
        re.compile(
            rf"(?:\b{_WORKER}_impl\b|"
            rf"\b(?:implementation\s+)?{_WORKER}(?:_impl)?\b.{{0,120}}"
            rf"\b(?:{_DIRECTOR}|task|issue|handoff|review)\b|"
            rf"\b{_DIRECTOR}\b.{{0,120}}\b(?:decision|checkpoint|review|approval|"
            rf"authoriz(?:e|ed|ation)|{_WORKER})\b|"
            rf"\b{_CODEX}\b.{{0,120}}\b(?:task|agent|{_DIRECTOR}|{_WORKER}|"
            rf"{_ORCHESTRATION}|review)\b|"
            rf"\b{_REVIEWER}\s+{_AGENT}\b|"
            rf"\b{_ORCHESTRATION}\s+{_TRANSCRIPT}\b|"
            rf"\b(?:{_SYSTEM}|{_DEVELOPER})\s+{_PROMPT}\b|"
            rf"\b{_CHAIN}\s+of\s+{_THOUGHT}\b)",
            re.IGNORECASE,
        ),
    ),
    ContentRule(
        "private-issue-coordination",
        "internal Issue-number coordination without an upstream public issue URL",
        re.compile(r"\bIssue\s+#\d+\b", re.IGNORECASE),
    ),
    ContentRule(
        "local-personal-path",
        "absolute local user/project or LOCALAPPDATA path",
        re.compile(
            r"(?:\b[A-Za-z]:[\\/](?:Users|Projects)[\\/][^\s'\"`]+|"
            r"%LOCALAPPDATA%[\\/][^\s'\"`]+|"
            r"/(?:home|Users)/[A-Za-z0-9._-]+(?:/[^\s'\"`]+)?)",
            re.IGNORECASE,
        ),
    ),
    ContentRule(
        "github-credential",
        "GitHub personal-access-token pattern",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{20,255})\b"),
    ),
    ContentRule(
        "generic-credential",
        "API key, access token, secret key, or bearer credential assignment",
        re.compile(
            r"(?:\b(?:[A-Za-z][A-Za-z0-9]*_)*(?:api[_-]?key|access[_-]?token|"
            r"secret[_-]?(?:key|token))\s*[:=]\s*['\"]?[A-Za-z0-9_./+~=:\-]{20,}|"
            r"\bBearer\s+[A-Za-z0-9._~+/=\-]{20,})",
            re.IGNORECASE,
        ),
    ),
    ContentRule(
        "email-address",
        "personal email address in the tracked candidate",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    ContentRule(
        "aws-access-key",
        "AWS access-key identifier",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    ContentRule(
        "private-key-header",
        "PEM/OpenSSH private-key header",
        re.compile(r"-----BEGIN\s+(?:(?:RSA|EC|OPENSSH)\s+)?PRIVATE KEY-----"),
    ),
    ContentRule(
        "unsupported-independent-review",
        "unsupported distinct-review approval wording",
        re.compile(rf"\b{_INDEPENDENTLY}\s+(?:reviewed|approved)\b", re.IGNORECASE),
    ),
    ContentRule(
        "publication-process",
        "private release-process wording",
        re.compile(
            r"(?:\bauthoriz(?:e|es|ed|ation)\b.{0,120}"
            r"\b(?:public repository|publication|release|tag|PyPI|push|PR)\b|"
            r"\b(?:public repository|publication|release|tag|PyPI|push|PR)\b"
            r".{0,120}\bauthoriz(?:e|es|ed|ation)\b)",
            re.IGNORECASE,
        ),
    ),
)

# Exact negative-fixture and frozen-protocol contexts. Each exemption applies
# only to its named rule, file, and phrase.
_BUNDLE_PATH_FIXTURES = (
    "/" + "home/user/private.jsonl",
    "C:" + r"\Users\user\private.jsonl",
)
_FROZEN_V04_REVIEW_PHRASE = (
    "independent" + "ly approved and frozen for bounded implementation plus"
)
CONTENT_ALLOWLIST = {
    ("local-personal-path", "tests/test_experiment_bundle.py"): _BUNDLE_PATH_FIXTURES,
    ("unsupported-independent-review", "V04_EXPERIMENT.md"): (
        _FROZEN_V04_REVIEW_PHRASE,
    ),
}


def _sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def _normalize_relative_path(value: str | Path) -> str:
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"tracked path must be a safe repository-relative path: {value}")
    return path.as_posix()


def tracked_paths(root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot enumerate tracked files: {detail}")
    decoded = completed.stdout.decode("utf-8")
    paths = tuple(part for part in decoded.split("\0") if part)
    normalized = tuple(_normalize_relative_path(path) for path in paths)
    if len(set(normalized)) != len(normalized):
        raise ValueError("tracked path enumeration contains duplicates")
    return tuple(sorted(normalized, key=_sort_key))


def _is_known_text_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return path.name in KNOWN_TEXT_NAMES or path.suffix.lower() in KNOWN_TEXT_SUFFIXES


def _path_violations(relative: str) -> list[Violation]:
    path = PurePosixPath(relative)
    lowered_parts = tuple(part.casefold() for part in path.parts)
    name = path.name.casefold()
    violations: list[Violation] = []
    if any(part in GENERATED_DIRECTORY_NAMES or part.endswith(".egg-info") for part in lowered_parts):
        violations.append(
            Violation(relative, "tracked-generated-artifact", "tracked virtualenv/cache/build artifact")
        )
    if name.endswith(GENERATED_FILE_SUFFIXES):
        violations.append(
            Violation(relative, "tracked-generated-artifact", "tracked generated Python bytecode")
        )
    if name in MODEL_FILE_NAMES or name.endswith(MODEL_FILE_SUFFIXES):
        violations.append(
            Violation(relative, "tracked-model-artifact", "tracked model/tokenizer/weight artifact")
        )
    return violations


def _line_is_allowlisted(rule: ContentRule, relative: str, line: str) -> bool:
    allowed_fragments = CONTENT_ALLOWLIST.get((rule.rule_id, relative), ())
    if any(fragment in line for fragment in allowed_fragments):
        return True
    if rule.rule_id == "private-issue-coordination":
        references = set(re.findall(r"\bIssue\s+#(\d+)\b", line, re.IGNORECASE))
        linked = set(
            re.findall(r"https://github\.com/[^\s)]+/issues/(\d+)", line, re.IGNORECASE)
        )
        return bool(references) and references <= linked
    return False


def _decode_for_content_scan(relative: str, payload: bytes) -> tuple[str | None, Violation | None]:
    known_text = _is_known_text_path(relative)
    if b"\x00" in payload:
        if known_text:
            return None, Violation(relative, "non-utf8-text", "known text file contains binary NUL bytes")
        return None, None
    try:
        return payload.decode("utf-8"), None
    except UnicodeDecodeError:
        if known_text:
            return None, Violation(relative, "non-utf8-text", "known text file is not valid UTF-8")
        return None, None


def audit_paths(root: Path, relative_paths: Iterable[str | Path]) -> AuditResult:
    root = root.resolve()
    normalized = tuple(_normalize_relative_path(path) for path in relative_paths)
    if len(set(normalized)) != len(normalized):
        raise ValueError("audit path input contains duplicates")
    ordered = tuple(sorted(normalized, key=_sort_key))
    violations: list[Violation] = []
    text_path_count = 0
    binary_path_count = 0

    for relative in ordered:
        violations.extend(_path_violations(relative))
        absolute = root.joinpath(*PurePosixPath(relative).parts)
        if absolute.is_symlink():
            violations.append(Violation(relative, "tracked-symlink", "tracked symlink is not audited"))
            continue
        if not absolute.is_file():
            violations.append(Violation(relative, "tracked-file-missing", "tracked path is not a regular file"))
            continue

        text, decode_violation = _decode_for_content_scan(relative, absolute.read_bytes())
        if decode_violation is not None:
            violations.append(decode_violation)
            text_path_count += 1
            continue
        if text is None:
            binary_path_count += 1
            continue

        text_path_count += 1
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule in CONTENT_RULES:
                if rule.pattern.search(line) and not _line_is_allowlisted(rule, relative, line):
                    violations.append(Violation(relative, rule.rule_id, rule.description, line_number))

    ordered_violations = tuple(
        sorted(
            violations,
            key=lambda item: (_sort_key(item.path), item.line or 0, item.rule_id),
        )
    )
    return AuditResult(len(ordered), text_path_count, binary_path_count, ordered_violations)


def audit_repository(root: Path) -> AuditResult:
    return audit_paths(root, tracked_paths(root))


def format_violations(violations: Iterable[Violation]) -> str:
    lines = []
    for violation in violations:
        location = violation.path
        if violation.line is not None:
            location += f":{violation.line}"
        lines.append(f"{location}: [{violation.rule_id}] {violation.description}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Git worktree root (defaults to the repository containing this script)",
    )
    args = parser.parse_args(argv)
    result = audit_repository(args.root)
    if result.violations:
        print(format_violations(result.violations))
        return 1
    print(
        "publication hygiene audit: OK "
        f"({result.tracked_path_count} tracked paths, {result.text_path_count} UTF-8 text files, "
        f"{result.binary_path_count} binary files skipped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
