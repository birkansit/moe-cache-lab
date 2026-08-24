"""Pure, frozen aggregation for the historical V0.4 evidence contract.

V0.4 was recorded with CPython 3.10. Its ``statistics.stdev`` implementation
accumulated float inputs exactly, converted the exact sample variance to a
binary64 float, and then called ``math.sqrt``. Later CPython versions instead
compute a correctly rounded square root directly from the exact fraction. The
two algorithms can differ by one binary64 ULP.

The helpers below freeze the historical boundary explicitly:

1. convert every finite binary64 input to its exact rational value;
2. accumulate the mean and sample variance with exact rational arithmetic;
3. convert the exact mean or variance once to binary64; and
4. apply ``math.sqrt`` to the converted variance.

This deliberately reproduces the Python-3.10-era V0.4 result. It is not
decimal rounding, a tolerance, or a general replacement for ``statistics``.
"""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Any, Iterable


V04_CHILD_COUNT = 4
V04_FORMAT_VERSION = 1
V04_RESULT_FORMAT = "moe-cache-lab.v04-result"
REFERENCE_MODE = "reference_no_observer"
PROFILED_MODE = "profiled_exact"
EVALUATION_IDS = (
    "eval-factual-01", "eval-factual-02", "eval-coding-01", "eval-coding-02",
    "eval-math-01", "eval-summary-01", "eval-reasoning-01",
    "eval-conversation-01",
)


def _child_number(value: int) -> int:
    if isinstance(value, bool) or value not in range(1, V04_CHILD_COUNT + 1):
        raise ValueError("V0.4 child number must be 1 through 4")
    return value


def evaluation_schedule(child_number: int) -> tuple[tuple[str, tuple[str, str]], ...]:
    """Return the exact manifest-order evaluation counterbalance."""
    _child_number(child_number)
    odd_reference_first = child_number in (1, 3)
    result = []
    for index, prompt_id in enumerate(EVALUATION_IDS):
        reference_first = odd_reference_first if index % 2 == 0 else not odd_reference_first
        modes = (REFERENCE_MODE, PROFILED_MODE) if reference_first else (PROFILED_MODE, REFERENCE_MODE)
        result.append((prompt_id, modes))
    return tuple(result)


def aggregate_children(children: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Strict hierarchical n=4/n=8 aggregation from validated child records."""
    records = tuple(children)
    if len(records) != V04_CHILD_COUNT or tuple(record.get("child_number") for record in records) != (1, 2, 3, 4):
        raise ValueError("V0.4 aggregation requires children 1 through 4 exactly")
    child_rows = []
    by_prompt: dict[str, list[float]] = {prompt_id: [] for prompt_id in EVALUATION_IDS}
    memory_by_prompt: dict[str, dict[str, list[float]]] = {
        prompt_id: {name: [] for name in (
            "reference_working_set_delta", "profiled_working_set_delta",
            "reference_private_delta", "profiled_private_delta",
        )} for prompt_id in EVALUATION_IDS
    }
    for child in records:
        prompts = child.get("evaluation")
        if not isinstance(prompts, list) or tuple(row.get("prompt_id") for row in prompts) != EVALUATION_IDS:
            raise ValueError("V0.4 child evaluation prompt order is invalid")
        expected = dict(evaluation_schedule(child["child_number"]))
        sum_r = sum_p = 0
        ratios = []
        child_memory = {name: 0 for name in next(iter(memory_by_prompt.values()))}
        for row in prompts:
            if tuple(row.get("mode_order", ())) != expected[row["prompt_id"]]:
                raise ValueError("V0.4 child mode counterbalance is invalid")
            r = _positive_int(row.get("reference_total_ns"), "reference elapsed")
            p = _positive_int(row.get("profiled_total_ns"), "profiled elapsed")
            ratio = p / r
            sum_r += r
            sum_p += p
            ratios.append(ratio)
            by_prompt[row["prompt_id"]].append(ratio)
            for name in child_memory:
                value = row.get(name)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("V0.4 descriptive memory deltas must be integers")
                child_memory[name] += value
                memory_by_prompt[row["prompt_id"]][name].append(float(value))
        child_rows.append({
            "child_number": child["child_number"], "sum_reference_ns": sum_r,
            "sum_profiled_ns": sum_p, "delta_ns": sum_p - sum_r,
            "ratio": sum_p / sum_r, "prompt_ratio_stats": _stats(ratios),
            "descriptive_memory_delta_sums": child_memory,
        })
    ratios = [row["ratio"] for row in child_rows]
    deltas = [float(row["delta_ns"]) for row in child_rows]
    sum_r_values = [float(row["sum_reference_ns"]) for row in child_rows]
    sum_p_values = [float(row["sum_profiled_ns"]) for row in child_rows]
    cv_r, cv_p, cv_ratio = _cv(sum_r_values), _cv(sum_p_values), _cv(ratios)
    ratio_span = (max(ratios) - min(ratios)) / min(ratios)
    stable = cv_r <= 0.10 and cv_p <= 0.10 and cv_ratio <= 0.10 and ratio_span <= 0.15
    prompt_rows = [{"prompt_id": prompt_id, **_stats(values)} for prompt_id, values in by_prompt.items()]
    memory_prompt_rows = [
        {"prompt_id": prompt_id, **{name: _stats(values) for name, values in fields.items()}}
        for prompt_id, fields in memory_by_prompt.items()
    ]
    return {
        "format": V04_RESULT_FORMAT, "format_version": V04_FORMAT_VERSION,
        "child_count": 4, "workload_prompt_count": 8,
        "children": child_rows, "child_ratio_stats": _stats(ratios),
        "child_delta_stats": _stats(deltas),
        "cv_sum_reference": cv_r, "cv_sum_profiled": cv_p,
        "cv_paired_ratio": cv_ratio, "paired_ratio_span": ratio_span,
        "prompt_rows": prompt_rows,
        "prompt_macro": _stats([row["mean"] for row in prompt_rows]),
        "descriptive_memory": {
            "claim": "before/after signed deltas only; order-conditioned; no per-mode peak overhead",
            "prompt_rows": memory_prompt_rows,
            "prompt_macro": {
                name: _stats([row[name]["mean"] for row in memory_prompt_rows])
                for name in next(iter(memory_by_prompt.values()))
            },
        },
        "decision": "valid_stable" if stable else "valid_inconclusive",
    }


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V0.4 {label} must be a positive integer")
    return value


def _exact_mean(data: tuple[float, ...]) -> float:
    total = sum((Fraction.from_float(value) for value in data), Fraction())
    return float(total / len(data))


def _historical_sample_sd(data: tuple[float, ...]) -> float:
    if len(data) < 2:
        return 0.0
    exact = tuple(Fraction.from_float(value) for value in data)
    mean = sum(exact, Fraction()) / len(exact)
    variance = sum(((value - mean) ** 2 for value in exact), Fraction()) / (len(exact) - 1)
    return math.sqrt(float(variance))


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    data = tuple(float(value) for value in values)
    if not data or any(not math.isfinite(value) for value in data):
        raise ValueError("V0.4 statistics require finite nonempty values")
    return {
        "n": len(data),
        "mean": _exact_mean(data),
        "sample_sd": _historical_sample_sd(data),
        "min": min(data),
        "max": max(data),
    }


def _cv(values: Iterable[float]) -> float:
    data = tuple(float(value) for value in values)
    if not data or any(not math.isfinite(value) for value in data):
        raise ValueError("V0.4 CV requires finite nonempty values")
    mean = _exact_mean(data)
    if mean <= 0:
        raise ValueError("V0.4 CV mean must be positive")
    return _historical_sample_sd(data) / mean
