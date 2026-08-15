"""Run the private, bounded Issue #54 Switch CPU copy validation experiment."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from moe_cache_lab.runtime_copy_executor import (
    EXECUTOR_INPUT_FORMAT,
    canonical_json_bytes,
    sha256_file,
    write_canonical_json,
)
from moe_cache_lab.runtime_copy_validation import (
    CAPACITY_BYTES,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_SHA256,
    SLOT_COUNT,
    compare_attempt,
    verify_prompt,
    write_artifact_manifest,
)


EXPECTED_ASSETS = {
    "config.json": (1_860, "6a1e1426873221034f8488e514ac127218c833e7d357b1e12dc002889c98fa53"),
    "generation_config.json": (147, "f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a"),
    "pytorch_model.bin": (1_238_895_063, "ff91705b718f692fa0c994a49d094154583db40b536e80f20f52e05498ff6856"),
    "special_tokens_map.json": (2_201, "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f"),
    "spiece.model": (791_656, "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"),
    "tokenizer.json": (2_422_095, "5f0ed8ab5b8cfa9812bb73752f1d80c292e52bcf5a87a144dc9ab2d251056cbb"),
    "tokenizer_config.json": (2_349, "4969f8d76ef05a16553bd2b07b3501673ae8d36972aea88a0f78ad31a3ff2de9"),
}


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_physical_memory() -> tuple[int, int]:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    function = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    function.argtypes = [ctypes.POINTER(_MemoryStatus)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(status.ullAvailPhys), int(status.ullTotalPhys)


def _snapshot_dir() -> Path:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if cache_root.name != "hub":
        cache_root = cache_root / "hub"
    return (
        cache_root
        / "models--google--switch-base-8"
        / "snapshots"
        / MODEL_REVISION
    )


def verify_cached_assets() -> list[dict[str, object]]:
    snapshot = _snapshot_dir()
    records = []
    for name, (expected_size, expected_hash) in EXPECTED_ASSETS.items():
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"required cached asset is unavailable: {name}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != expected_size or actual_hash != expected_hash:
            raise RuntimeError(f"cached asset does not match frozen identity: {name}")
        records.append({"name": name, "size_bytes": actual_size, "sha256": actual_hash})
    return records


def _run_child(command: list[str], *, cwd: Path, environment: dict[str, str], log_name: str) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    write_canonical_json(
        cwd / log_name,
        {"command_role": log_name.removesuffix(".json"), "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
    )
    if completed.returncode:
        raise RuntimeError(f"{log_name} process failed with exit code {completed.returncode}")


def run(attempt_dir: Path, prompt_file: Path) -> dict[str, object]:
    if attempt_dir.exists():
        raise FileExistsError(f"attempt directory already exists: {attempt_dir}")
    attempt_dir.mkdir(parents=True)
    prompt = prompt_file.read_text(encoding="utf-8")
    prompt_hash = verify_prompt(prompt)
    (attempt_dir / "frozen-prompt.txt").write_text(prompt, encoding="utf-8", newline="")
    assets = verify_cached_assets()

    import torch
    import transformers

    if torch.__version__ != "2.12.0+cpu" or transformers.__version__ != "5.12.0":
        raise RuntimeError("runtime versions differ from the frozen protocol")
    available, total = available_physical_memory()
    checkpoint_size = EXPECTED_ASSETS["pytorch_model.bin"][0]
    # Credible transient-load allowance: two checkpoint payloads, all staging
    # slots, and one GiB for interpreter/model workspace.
    required_available = 2 * checkpoint_size + CAPACITY_BYTES + 1024**3
    admission = {
        "format": "moe-cache-lab.runtime-copy-environment/v1",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "cpu",
        "dtype": "float32",
        "prompt_sha256": prompt_hash,
        "cached_assets": assets,
        "available_physical_memory_bytes": available,
        "total_physical_memory_bytes": total,
        "admission_required_available_bytes": required_available,
        "admission_passed": available >= required_available,
        "admission_formula": "2 * checkpoint bytes + staging capacity + 1 GiB workspace",
    }
    write_canonical_json(attempt_dir / "environment.json", admission)
    if not admission["admission_passed"]:
        raise RuntimeError("available physical memory failed the frozen-run safety admission")

    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
    )
    source_root = str((Path(__file__).resolve().parents[1] / "src").resolve())
    environment["PYTHONPATH"] = source_root
    predictor = [
        sys.executable,
        "-m",
        "moe_cache_lab.runtime_copy_validation",
        "--attempt-dir",
        str(attempt_dir),
        "--prompt-file",
        str(attempt_dir / "frozen-prompt.txt"),
    ]
    _run_child(predictor, cwd=attempt_dir, environment=environment, log_name="predictor-process.json")

    predictor_result = json.loads((attempt_dir / "predictor-result.json").read_text(encoding="utf-8"))
    prediction_path = attempt_dir / predictor_result["prediction_path"]
    if sha256_file(prediction_path) != predictor_result["prediction_sha256"]:
        raise RuntimeError("frozen prediction was not durably sealed")
    executor_input = {
        "format": EXECUTOR_INPUT_FORMAT,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "cpu",
        "dtype": "float32",
        "policy": "lru",
        "capacity_bytes": CAPACITY_BYTES,
        "slot_count": SLOT_COUNT,
        "trace_path": predictor_result["trace_path"],
        "trace_sha256": predictor_result["trace_sha256"],
        "payload_manifest_path": predictor_result["payload_manifest_path"],
        "payload_manifest_sha256": predictor_result["payload_manifest_sha256"],
        "output_path": "runtime-observation.json",
    }
    input_path = write_canonical_json(attempt_dir / "executor-input.json", executor_input)
    # The independent process receives only executor-input.json.  The frozen
    # prediction path and counters are intentionally absent from that schema.
    executor = [
        sys.executable,
        "-m",
        "moe_cache_lab.runtime_copy_executor",
        "--input",
        input_path.name,
    ]
    _run_child(executor, cwd=attempt_dir, environment=environment, log_name="executor-process.json")

    comparison = compare_attempt(attempt_dir)
    manifest_path, manifest_hash = write_artifact_manifest(attempt_dir)
    result = {
        "outcome": comparison["outcome"],
        "artifact_manifest_path": manifest_path.name,
        "artifact_manifest_sha256": manifest_hash,
        "attempt_uuid": attempt_dir.name.removeprefix("v07-runtime-copy-validation-"),
    }
    write_canonical_json(attempt_dir / "run-result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--prompt-file", type=Path, required=True)
    arguments = parser.parse_args(argv)
    attempt = arguments.attempt_dir or Path("artifacts") / f"v07-runtime-copy-validation-{uuid.uuid4()}"
    result = run(attempt.resolve(), arguments.prompt_file.resolve())
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0 if result["outcome"] == "AGREEMENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
