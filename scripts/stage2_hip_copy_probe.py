"""Measure raw HIP host/device copies without loading a model.

This diagnostic is intentionally independent of PyTorch.  It proves only that
the selected HIP runtime can allocate and copy bytes.  It does not establish
framework, model, expert-cache, latency, or acceleration support.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


FORMAT = "moe-cache-lab.stage2-hip-copy-probe/v1"
DEFAULT_DLL = Path(r"C:\Program Files\AMD\ROCm\6.2\bin\amdhip64_6.dll")
MIB = 1024 * 1024
HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2


class HipRuntime:
    """Typed, checked subset of the HIP runtime API used by this probe."""

    def __init__(self, dll_path: Path) -> None:
        self.dll_path = dll_path.resolve(strict=True)
        self.dll = ctypes.CDLL(str(self.dll_path))
        self._declare_api()
        self.call_count = 0

    def _declare_api(self) -> None:
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)
        declarations = {
            "hipInit": ([ctypes.c_uint], ctypes.c_int),
            "hipSetDevice": ([ctypes.c_int], ctypes.c_int),
            "hipMalloc": ([c_void_pp, ctypes.c_size_t], ctypes.c_int),
            "hipHostMalloc": ([c_void_pp, ctypes.c_size_t, ctypes.c_uint], ctypes.c_int),
            "hipMemcpy": (
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
                ctypes.c_int,
            ),
            "hipDeviceSynchronize": ([], ctypes.c_int),
            "hipFree": ([ctypes.c_void_p], ctypes.c_int),
            "hipHostFree": ([ctypes.c_void_p], ctypes.c_int),
            "hipRuntimeGetVersion": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "hipDriverGetVersion": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
        }
        for name, (argtypes, restype) in declarations.items():
            function = getattr(self.dll, name)
            function.argtypes = argtypes
            function.restype = restype

    def checked(self, name: str, *args: object) -> None:
        result = int(getattr(self.dll, name)(*args))
        self.call_count += 1
        if result != 0:
            raise RuntimeError(f"{name} failed with HIP error {result}")

    def version(self, name: str) -> int:
        value = ctypes.c_int()
        self.checked(name, ctypes.byref(value))
        return int(value.value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_fill_sha256(size: int, value: int) -> str:
    digest = hashlib.sha256()
    block = bytes([value]) * min(MIB, size)
    remaining = size
    while remaining:
        part = block[: min(len(block), remaining)]
        digest.update(part)
        remaining -= len(part)
    return digest.hexdigest()


def _validate_round_trip(runtime: HipRuntime, host: int, device: int, size: int) -> str:
    fill = 0xA5
    ctypes.memset(host, fill, size)
    runtime.checked(
        "hipMemcpy", device, host, size, HIP_MEMCPY_HOST_TO_DEVICE
    )
    ctypes.memset(host, 0, size)
    runtime.checked(
        "hipMemcpy", host, device, size, HIP_MEMCPY_DEVICE_TO_HOST
    )
    runtime.checked("hipDeviceSynchronize")
    actual = hashlib.sha256(ctypes.string_at(host, size)).hexdigest()
    expected = _expected_fill_sha256(size, fill)
    if actual != expected:
        raise RuntimeError("HIP H2D/D2H validation hash mismatch")
    return actual


def _allocate(runtime: HipRuntime, size: int) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
    device = ctypes.c_void_p()
    host = ctypes.c_void_p()
    runtime.checked("hipMalloc", ctypes.byref(device), size)
    try:
        runtime.checked("hipHostMalloc", ctypes.byref(host), size, 0)
    except BaseException:
        runtime.checked("hipFree", device)
        raise
    return host, device


def _free(runtime: HipRuntime, host: ctypes.c_void_p, device: ctypes.c_void_p) -> None:
    runtime.checked("hipFree", device)
    runtime.checked("hipHostFree", host)


def _copy_parts(
    runtime: HipRuntime,
    host: int,
    device: int,
    parts: tuple[int, ...],
    direction: int,
) -> None:
    offset = 0
    for size in parts:
        if direction == HIP_MEMCPY_HOST_TO_DEVICE:
            destination, source = device + offset, host + offset
        else:
            destination, source = host + offset, device + offset
        runtime.checked("hipMemcpy", destination, source, size, direction)
        offset += size


def _measure_case(
    runtime: HipRuntime,
    *,
    name: str,
    allocation_size: int,
    parts: tuple[int, ...],
    direction: int,
    warmups: int,
    trials: int,
    repetitions: int,
) -> dict[str, object]:
    if min(allocation_size, warmups, trials, repetitions, *parts) <= 0:
        raise ValueError("measurement sizes and counts must be positive")
    if sum(parts) > allocation_size:
        raise ValueError("copy parts exceed allocation")

    host_pointer, device_pointer = _allocate(runtime, allocation_size)
    host = int(host_pointer.value or 0)
    device = int(device_pointer.value or 0)
    if not host or not device:
        raise RuntimeError("HIP returned a null allocation")
    try:
        validation_sha256 = _validate_round_trip(
            runtime, host, device, allocation_size
        )
        for _ in range(warmups):
            _copy_parts(runtime, host, device, parts, direction)
        runtime.checked("hipDeviceSynchronize")

        bytes_per_repetition = sum(parts)
        bytes_per_trial = bytes_per_repetition * repetitions
        raw_trials: list[dict[str, int | float]] = []
        for trial_number in range(1, trials + 1):
            start_ns = time.perf_counter_ns()
            for _ in range(repetitions):
                _copy_parts(runtime, host, device, parts, direction)
            runtime.checked("hipDeviceSynchronize")
            duration_ns = time.perf_counter_ns() - start_ns
            seconds = duration_ns / 1_000_000_000
            raw_trials.append(
                {
                    "trial": trial_number,
                    "duration_ns": duration_ns,
                    "bytes": bytes_per_trial,
                    "decimal_gb_per_second": bytes_per_trial / seconds / 1e9,
                    "microseconds_per_repetition": duration_ns / repetitions / 1000,
                }
            )
    finally:
        _free(runtime, host_pointer, device_pointer)

    rates = [float(trial["decimal_gb_per_second"]) for trial in raw_trials]
    times = [float(trial["microseconds_per_repetition"]) for trial in raw_trials]
    return {
        "name": name,
        "allocation_bytes": allocation_size,
        "copy_part_bytes": list(parts),
        "direction": "host_to_device"
        if direction == HIP_MEMCPY_HOST_TO_DEVICE
        else "device_to_host",
        "warmup_repetitions": warmups,
        "timed_trials": trials,
        "copies_per_trial": repetitions,
        "timer": "time.perf_counter_ns",
        "timing_boundary": (
            "start immediately before the first synchronous hipMemcpy; "
            "end after the final hipDeviceSynchronize returns"
        ),
        "allocation_reused_across_warmups_and_trials": True,
        "round_trip_validation_sha256": validation_sha256,
        "trials": raw_trials,
        "summary": {
            "decimal_gb_per_second": {
                "min": min(rates),
                "median": statistics.median(rates),
                "max": max(rates),
            },
            "microseconds_per_repetition": {
                "min": min(times),
                "median": statistics.median(times),
                "max": max(times),
            },
        },
    }


def run_probe(dll_path: Path) -> dict[str, object]:
    runtime = HipRuntime(dll_path)
    script_path = Path(__file__).resolve(strict=True)
    started_at = datetime.now(timezone.utc).isoformat()
    runtime.checked("hipInit", 0)
    runtime.checked("hipSetDevice", 0)
    runtime_version = runtime.version("hipRuntimeGetVersion")
    driver_version = runtime.version("hipDriverGetVersion")

    cases = [
        _measure_case(
            runtime,
            name="bulk_h2d_64mib",
            allocation_size=64 * MIB,
            parts=(64 * MIB,),
            direction=HIP_MEMCPY_HOST_TO_DEVICE,
            warmups=4,
            trials=7,
            repetitions=32,
        ),
        _measure_case(
            runtime,
            name="bulk_d2h_64mib",
            allocation_size=64 * MIB,
            parts=(64 * MIB,),
            direction=HIP_MEMCPY_DEVICE_TO_HOST,
            warmups=4,
            trials=7,
            repetitions=32,
        ),
        _measure_case(
            runtime,
            name="expert_h2d_contiguous_3mib",
            allocation_size=3 * MIB,
            parts=(3 * MIB,),
            direction=HIP_MEMCPY_HOST_TO_DEVICE,
            warmups=8,
            trials=7,
            repetitions=2048,
        ),
        _measure_case(
            runtime,
            name="expert_h2d_physical_2mib_plus_1mib",
            allocation_size=3 * MIB,
            parts=(2 * MIB, MIB),
            direction=HIP_MEMCPY_HOST_TO_DEVICE,
            warmups=8,
            trials=7,
            repetitions=2048,
        ),
    ]
    result = {
        "format": FORMAT,
        "created_at": started_at,
        "claim_boundary": (
            "raw HIP byte-copy diagnostic only; not PyTorch, model, cache, "
            "latency, or acceleration evidence"
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "process_id": os.getpid(),
            "probe_script_path": str(script_path),
            "probe_script_sha256": _sha256_file(script_path),
            "dll_path": str(runtime.dll_path),
            "dll_size_bytes": runtime.dll_path.stat().st_size,
            "dll_sha256": _sha256_file(runtime.dll_path),
            "hip_runtime_version_integer": runtime_version,
            "hip_driver_version_integer": driver_version,
            "device_index": 0,
        },
        "method": {
            "api": "ctypes.CDLL with declared argtypes/restype",
            "host_allocation": "hipHostMalloc flags=0",
            "device_allocation": "hipMalloc",
            "copy": "synchronous hipMemcpy plus hipDeviceSynchronize",
            "validation": "0xA5 H2D/D2H round trip with full-buffer SHA-256",
            "all_nonzero_api_results_raise": True,
            "allocations_freed_in_finally": True,
        },
        "all_api_calls_returned_zero": True,
        "api_call_count": runtime.call_count,
        "cases": cases,
    }
    validate_result(result)
    return result


def validate_result(result: dict[str, object]) -> None:
    """Validate a retained probe result without requiring HIP hardware."""

    if result.get("format") != FORMAT or result.get("all_api_calls_returned_zero") is not True:
        raise ValueError("invalid HIP probe format or API status")
    environment = result.get("environment")
    method = result.get("method")
    cases = result.get("cases")
    if not isinstance(environment, dict) or not isinstance(method, dict) or not isinstance(cases, list):
        raise ValueError("invalid HIP probe structure")
    if not isinstance(result.get("api_call_count"), int) or result["api_call_count"] <= 0:
        raise ValueError("invalid HIP API call count")
    if not isinstance(environment.get("dll_sha256"), str) or len(environment["dll_sha256"]) != 64:
        raise ValueError("invalid HIP DLL hash")
    if not isinstance(environment.get("probe_script_sha256"), str) or len(environment["probe_script_sha256"]) != 64:
        raise ValueError("invalid HIP probe script hash")
    if method.get("all_nonzero_api_results_raise") is not True or method.get("allocations_freed_in_finally") is not True:
        raise ValueError("probe method does not guarantee checked calls and cleanup")

    expected_names = {
        "bulk_h2d_64mib",
        "bulk_d2h_64mib",
        "expert_h2d_contiguous_3mib",
        "expert_h2d_physical_2mib_plus_1mib",
    }
    if {case.get("name") for case in cases if isinstance(case, dict)} != expected_names:
        raise ValueError("missing or duplicate HIP probe case")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid HIP probe case")
        parts = case.get("copy_part_bytes")
        trials = case.get("trials")
        repetitions = case.get("copies_per_trial")
        allocation = case.get("allocation_bytes")
        if (
            not isinstance(parts, list)
            or not parts
            or any(not isinstance(value, int) or value <= 0 for value in parts)
            or not isinstance(trials, list)
            or len(trials) != case.get("timed_trials")
            or not isinstance(repetitions, int)
            or repetitions <= 0
            or not isinstance(allocation, int)
            or sum(parts) > allocation
        ):
            raise ValueError("invalid HIP probe case dimensions")
        expected_validation = _expected_fill_sha256(allocation, 0xA5)
        if case.get("round_trip_validation_sha256") != expected_validation:
            raise ValueError("invalid HIP probe validation hash")

        rates: list[float] = []
        times: list[float] = []
        expected_bytes = sum(parts) * repetitions
        for index, trial in enumerate(trials, start=1):
            if not isinstance(trial, dict) or trial.get("trial") != index:
                raise ValueError("invalid HIP probe trial order")
            duration_ns = trial.get("duration_ns")
            byte_count = trial.get("bytes")
            rate = trial.get("decimal_gb_per_second")
            microseconds = trial.get("microseconds_per_repetition")
            if (
                not isinstance(duration_ns, int)
                or duration_ns <= 0
                or byte_count != expected_bytes
                or not isinstance(rate, (int, float))
                or not isinstance(microseconds, (int, float))
                or not math.isfinite(rate)
                or not math.isfinite(microseconds)
                or rate <= 0
                or microseconds <= 0
            ):
                raise ValueError("invalid HIP probe trial values")
            expected_rate = expected_bytes / (duration_ns / 1_000_000_000) / 1e9
            expected_microseconds = duration_ns / repetitions / 1000
            if not math.isclose(rate, expected_rate, rel_tol=1e-12) or not math.isclose(
                microseconds, expected_microseconds, rel_tol=1e-12
            ):
                raise ValueError("HIP probe derived trial metric mismatch")
            rates.append(float(rate))
            times.append(float(microseconds))

        summary = case.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("invalid HIP probe summary")
        for name, values in (
            ("decimal_gb_per_second", rates),
            ("microseconds_per_repetition", times),
        ):
            record = summary.get(name)
            expected = {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
            if not isinstance(record, dict) or record != expected:
                raise ValueError("HIP probe summary mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON here; omit to print JSON on stdout",
    )
    args = parser.parse_args()
    result = run_probe(args.dll)
    payload = json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8", newline="\n")
        print(args.output)


if __name__ == "__main__":
    main()
