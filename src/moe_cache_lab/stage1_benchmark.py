"""Auditable Stage 1 cache simulations over validated repetition sets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from .cache import (
    CacheResult,
    CacheSimulation,
    FixedTargetPlan,
    build_fixed_target_plan,
    simulate_bundles,
)
from .stage1 import Stage1PromptInput, Stage1SetInputs, validate_stage1_set
from .views import (
    PREFILL_LAYER_UNION_ATOMIC,
    TOKEN_LAYER_ATOMIC,
    PromptTraceSource,
    SimulationBundle,
    adapt_prompt_traces,
    summarize_bundles,
)
from .workflow import SUITE_POLICIES

STAGE1_BENCHMARK_FORMAT = "moe-cache-lab.stage1-cache-benchmark"
STAGE1_BENCHMARK_VERSION = 1
STAGE1_CAPACITIES = (8, 16, 32, 64, 128, 192, 256, 384, 512, 768)
STAGE1_VIEWS = (TOKEN_LAYER_ATOMIC, PREFILL_LAYER_UNION_ATOMIC)
STAGE1_CONTROLS = (
    "suite_persistent",
    "cold_per_prompt",
    "decode_only_cold",
)
FIXED_POLICIES = frozenset({
    "calibrated_static_frequency",
    "offline_oracle_frequency",
})


@dataclass(frozen=True)
class Stage1Benchmark:
    """Validated inputs plus deterministic structured benchmark output."""

    inputs: Stage1SetInputs
    data: dict[str, Any]


def benchmark_stage1(
    set_manifest_path: str | Path,
    trusted_v02_manifest: str | Path = Path("results/v0.2-corpus-v1/manifest-v1.json"),
) -> Stage1Benchmark:
    """Validate the complete Stage 1 set before performing any simulation."""

    inputs = validate_stage1_set(set_manifest_path, trusted_v02_manifest)
    return Stage1Benchmark(inputs, _benchmark_validated_set(inputs))


def write_stage1_json(path: str | Path, benchmark: Stage1Benchmark) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            benchmark.data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def write_stage1_markdown(path: str | Path, benchmark: Stage1Benchmark) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_stage1_markdown(benchmark.data), encoding="utf-8", newline="\n"
    )
    return destination


def sample_statistics(values: Iterable[int | float]) -> dict[str, float]:
    """Frozen n=8 workload macro definition with sample standard deviation."""

    sequence = tuple(float(value) for value in values)
    if len(sequence) != 8 or any(not math.isfinite(value) for value in sequence):
        raise ValueError("workload macro statistics require exactly 8 finite prompt values")
    return {
        "n": 8,
        "mean": mean(sequence),
        "sample_standard_deviation": stdev(sequence),
        "minimum": min(sequence),
        "maximum": max(sequence),
    }


def evaluate_continuation_rule(
    repetitions: Iterable[dict[str, Any]],
    *,
    repeatability_passed: bool,
    expected_evaluation_prompt_ids: Iterable[str],
) -> dict[str, Any]:
    """Evaluate the frozen integer transfer and per-prompt decode thresholds."""

    records = tuple(repetitions)
    expected_prompt_ids = tuple(expected_evaluation_prompt_ids)
    if (
        len(expected_prompt_ids) != 8
        or len(set(expected_prompt_ids)) != 8
        or any(not isinstance(prompt_id, str) or not prompt_id for prompt_id in expected_prompt_ids)
    ):
        raise ValueError("continuation evaluation requires 8 unique nonempty expected prompt IDs")
    if len(records) != 3:
        raise ValueError("continuation evaluation requires exactly three repetitions")
    output: list[dict[str, Any]] = []
    for expected_number, record in enumerate(records, start=1):
        if record.get("repetition_number") != expected_number:
            raise ValueError("continuation repetitions must be ordered 1, 2, 3")
        transfer_32 = record.get("lfu_grouped_primary_cap32_transfers")
        transfer_256 = record.get("lfu_grouped_primary_cap256_transfers")
        prompt_rows = record.get("lfu_grouped_decode_only_cap256_prompts")
        if (
            isinstance(transfer_32, bool)
            or not isinstance(transfer_32, int)
            or transfer_32 < 0
            or isinstance(transfer_256, bool)
            or not isinstance(transfer_256, int)
            or transfer_256 < 0
            or not isinstance(prompt_rows, list)
            or len(prompt_rows) != 8
        ):
            raise ValueError("continuation counters are incomplete")
        prompt_ids = tuple(prompt.get("prompt_id") for prompt in prompt_rows)
        if prompt_ids != expected_prompt_ids:
            raise ValueError("continuation prompt rows must match the exact evaluation prompt ID order")
        transfer_passed = transfer_256 * 10 <= transfer_32 * 9
        decode_rows = []
        for prompt in prompt_rows:
            hits = prompt.get("hits")
            requests = prompt.get("requests")
            if (
                isinstance(hits, bool)
                or not isinstance(hits, int)
                or hits < 0
                or isinstance(requests, bool)
                or not isinstance(requests, int)
                or requests < 0
                or hits > requests
            ):
                raise ValueError("continuation prompt counters are invalid")
            passed = requests > 0 and hits * 100 >= requests * 10
            decode_rows.append({
                "prompt_id": prompt.get("prompt_id"),
                "hits": hits,
                "requests": requests,
                "hit_rate": hits / requests if requests else 0.0,
                "passed": passed,
            })
        decode_passed = all(row["passed"] for row in decode_rows)
        output.append({
            "repetition_number": expected_number,
            "cap32_estimated_transfers": transfer_32,
            "cap256_estimated_transfers": transfer_256,
            "transfer_condition_passed": transfer_passed,
            "decode_prompt_conditions": decode_rows,
            "decode_stability_passed": decode_passed,
            "passed": repeatability_passed and transfer_passed and decode_passed,
        })
    return {
        "rule": (
            "prefill_layer_union_atomic lfu suite-persistent cap256 transfers "
            "<= 90% of cap32 in each repetition, and every decode-only cold "
            "evaluation prompt at lfu cap256 has >=10% hits in each repetition"
        ),
        "repeatability_passed": repeatability_passed,
        "repetitions": output,
        "passed": repeatability_passed and all(item["passed"] for item in output),
        "authorization_if_passed": "Stage 2 feasibility inspection only; never offloading",
    }


def render_stage1_markdown(data: dict[str, Any]) -> str:
    """Render Markdown exclusively from the deterministic structured artifact."""

    if data.get("format") != STAGE1_BENCHMARK_FORMAT:
        raise ValueError("unsupported Stage 1 benchmark data")
    measured = data["measured"]
    reproducibility = data["reproducibility"]
    environment = reproducibility["environment"]
    generation = reproducibility["generation"]
    lines = [
        "# MoE cache-lab Stage 1 simulated cache benchmark", "",
        "## MEASURED routing and workload metadata", "",
        "Routing events and assignments are measured from validated immutable traces. "
        "Prompt categories describe workload diversity only and do not identify expert semantics.", "",
        f"- Model: `{reproducibility['model_id']}` at `{reproducibility['model_revision']}`",
        f"- Repetitions validated: {data['repeatability']['validated_repetitions']}; results publish one denominator set, not tripled samples.",
        f"- Calibration measured events/requests: {measured['calibration_events']} / {measured['calibration_requests']}",
        f"- Evaluation measured events/requests: {measured['evaluation_events']} / {measured['evaluation_requests']}",
        f"- EOS-shortened prompts: calibration {measured['eos_shortened_calibration_prompts']} of 4; evaluation {measured['eos_shortened_evaluation_prompts']} of 8", "",
        "`prefill_layer_union_atomic` is a conservative simultaneous-active-set sensitivity view, not verified Granite runtime residency, scheduling, or transfer behavior.", "",
        "## Reproducibility", "",
        f"- Stage 1 set manifest SHA-256: `{reproducibility['stage1_set_manifest_sha256']}`",
        f"- Trusted V0.2 manifest SHA-256: `{reproducibility['trusted_v02_manifest_sha256']}`",
        f"- Corpus: `{reproducibility['corpus_version']}` / `{reproducibility['corpus_sha256']}`",
        f"- Environment: Python `{environment['python']}`, Transformers `{environment['transformers']}`, PyTorch `{environment['torch']}`, device `{environment['device']}`, dtype `{environment['dtype']}`, threads `{reproducibility['torch_num_threads']}`",
        f"- Generation: `{generation['strategy']}` / `{generation['token_selection']}`, do_sample `{str(generation['do_sample']).lower()}`, use_cache `{str(generation['use_cache']).lower()}`, max decode-input steps `{reproducibility['max_decode_input_steps']}`",
        f"- Calibration order: {', '.join(reproducibility['calibration_prompt_order'])}",
        f"- Evaluation order: {', '.join(reproducibility['evaluation_prompt_order'])}",
        f"- Capacities: {', '.join(map(str, data['capacities']))}",
        f"- Views: {', '.join(f'`{value}`' for value in data['views'])}",
        f"- Policies: {', '.join(f'`{value}`' for value in data['policies'])}",
        f"- Controls: {', '.join(f'`{value}`' for value in data['controls'])}", "",
        "`offline_oracle_frequency` is explicitly non-causal and selects its immutable target from the full evaluation view. `calibrated_static_frequency` selects only from the full disjoint calibration view. Targets are selected once per view/capacity and reused unchanged across every control.", "",
        "## SIMULATED cache outcomes and ESTIMATED transfers", "",
        "Cache outcomes are simulated. Transfers are estimates equal to prewarm plus demand loads; they are not measured data movement, latency, speedup, runtime scheduling, or actual offloading.", "",
        "| view | cap | policy | control | scope | N/A reason | workloads | fixed target | bundles | source events | source assignments | requests | hits | misses | hit rate | prewarm | demand | evictions | ESTIMATED transfers |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in data["aggregate_rows"]:
        if not row["feasible"]:
            lines.append(
                f"| {row['view']} | {row['capacity']} | {row['policy']} | {row['control']} | {row['scope']} | {row['n_a_reason']} | {row['workload_count']} | {row['fixed_target_size'] if row['fixed_target_size'] is not None else '-'} | N/A | {row['source_events']} | {row['source_assignments']} | {row['requests']} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        lines.append(
            f"| {row['view']} | {row['capacity']} | {row['policy']} | {row['control']} | {row['scope']} | - | {row['workload_count']} | {row['fixed_target_size'] if row['fixed_target_size'] is not None else '-'} | {row['bundles']} | {row['source_events']} | {row['source_assignments']} | {row['requests']} | {row['hits']} | {row['misses']} | {row['hit_rate']:.2%} | {row['prewarm_loads']} | {row['demand_loads']} | {row['evictions']} | {row['estimated_transfers']} |"
        )
    lines.extend(["", "## Exact per-prompt control rows", "",
        "Cold-per-prompt and decode-only controls reset cache state per prompt. Fixed prewarm is charged once per prompt; decode-only rows exclude every prompt access.", "",
        "| view | cap | policy | control | scope | prompt | fixed target | bundles | requests | hits | misses | hit rate | prewarm | demand | evictions | ESTIMATED transfers |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in data["prompt_rows"]:
        if not row["feasible"]:
            lines.append(
                f"| {row['view']} | {row['capacity']} | {row['policy']} | {row['control']} | {row['scope']} | {row['prompt_id']} | {row['fixed_target_size'] if row['fixed_target_size'] is not None else '-'} | N/A | {row['requests']} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        lines.append(
            f"| {row['view']} | {row['capacity']} | {row['policy']} | {row['control']} | {row['scope']} | {row['prompt_id']} | {row['fixed_target_size'] if row['fixed_target_size'] is not None else '-'} | {row['bundles']} | {row['requests']} | {row['hits']} | {row['misses']} | {row['hit_rate']:.2%} | {row['prewarm_loads']} | {row['demand_loads']} | {row['evictions']} | {row['estimated_transfers']} |"
        )
    lines.extend(["", "## Cold-per-prompt macro statistics", "",
        "Macro values use exactly n=8 prompt controls: arithmetic mean, sample standard deviation (n-1), minimum, and maximum. Suite-persistent and decode-only controls have micro totals only.", "",
        "| view | cap | policy | scope | metric | n | mean | sample SD | min | max |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for macro in data["cold_per_prompt_macros"]:
        stats = macro["statistics"]
        lines.append(
            f"| {macro['view']} | {macro['capacity']} | {macro['policy']} | {macro['scope']} | {macro['metric']} | {stats['n']} | {stats['mean']:.6g} | {stats['sample_standard_deviation']:.6g} | {stats['minimum']:.6g} | {stats['maximum']:.6g} |"
        )
    lines.extend(["", "## Fixed-target overlap across views", "",
        "Only capacities feasible in both views are compared.", "",
        "| capacity | policy | token target size | grouped target size | overlap |",
        "| ---: | --- | ---: | ---: | ---: |",
    ])
    for row in data["fixed_target_overlaps"]:
        lines.append(
            f"| {row['capacity']} | {row['policy']} | {row['token_target_size']} | {row['grouped_target_size']} | {row['overlap_size']} |"
        )
    lines.extend(["", "## EOS and decode accounting", "",
        "| split | prompt | requested steps | actual routed steps | EOS emitted | horizon exhausted |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ])
    for row in data["eos_accounting"]:
        lines.append(
            f"| {row['split']} | {row['prompt_id']} | {row['requested_steps']} | {row['actual_steps']} | {str(row['eos_emitted']).lower()} | {str(row['horizon_exhausted']).lower()} |"
        )
    continuation = data["continuation_rule"]
    lines.extend(["", "## Prospectively frozen continuation rule", "",
        "This confirmation rule was chosen after exploratory V0.2 max-step-2 diagnostics; it is not blind preregistration and cannot be changed after Stage 1 results are visible.", "",
        f"- Repeatability gate: {'PASS' if continuation['repeatability_passed'] else 'FAIL'}",
    ])
    for row in continuation["repetitions"]:
        lines.append(
            f"- Repetition {row['repetition_number']}: transfer {'PASS' if row['transfer_condition_passed'] else 'FAIL'} ({row['cap256_estimated_transfers']} vs cap32 {row['cap32_estimated_transfers']}); decode stability {'PASS' if row['decode_stability_passed'] else 'FAIL'}"
        )
    lines.extend([f"- Overall confirmation: {'PASS' if continuation['passed'] else 'FAIL'}", "",
        "Passing authorizes Stage 2 feasibility inspection only and never offloading. Negative and null results are preserved without interpretation tuning.",
    ])
    return "\n".join(lines) + "\n"


def _benchmark_validated_set(inputs: Stage1SetInputs) -> dict[str, Any]:
    first = inputs.repetitions[0]
    first_calibration = tuple(item for item in first.prompts if item.prompt.split == "calibration")
    first_evaluation = tuple(item for item in first.prompts if item.prompt.split == "evaluation")
    if len(first_calibration) != 4 or len(first_evaluation) != 8:
        raise ValueError("Stage 1 benchmark requires exactly 4 calibration and 8 evaluation prompts")

    aggregate_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    macros: list[dict[str, Any]] = []
    plans: dict[tuple[str, int, str], FixedTargetPlan] = {}
    view_inputs: dict[str, tuple[tuple[SimulationBundle, ...], tuple[SimulationBundle, ...]]] = {}

    for view in STAGE1_VIEWS:
        calibration = _adapt(first_calibration, view)
        evaluation = _adapt(first_evaluation, view)
        view_inputs[view] = calibration, evaluation
        summary = summarize_bundles(evaluation)
        per_prompt = _bundles_by_prompt(evaluation, first_evaluation)
        for capacity in STAGE1_CAPACITIES:
            feasible = capacity >= summary.max_bundle_size
            for policy in SUITE_POLICIES:
                plan = None
                if policy in FIXED_POLICIES:
                    selection = calibration if policy == "calibrated_static_frequency" else evaluation
                    opposite = evaluation if policy == "calibrated_static_frequency" else calibration
                    plan = build_fixed_target_plan(selection, opposite, capacity, policy)
                    plans[(view, capacity, policy)] = plan
                if not feasible:
                    for control in STAGE1_CONTROLS:
                        bundles = evaluation if control != "decode_only_cold" else tuple(
                            bundle for bundle in evaluation if bundle.phase == "generated"
                        )
                        for scope in ("combined", "prompt/prefill", "generated/decode"):
                            aggregate_rows.append(_infeasible_row(
                                view, capacity, policy, control, scope, bundles,
                                f"capacity {capacity} is below maximum {view} bundle {summary.max_bundle_size}",
                                len(plan.target_entries) if plan is not None else None,
                            ))
                    for prompt in first_evaluation:
                        prompt_bundles = per_prompt[prompt.prompt.id]
                        for control in ("cold_per_prompt", "decode_only_cold"):
                            bundles = (
                                prompt_bundles if control == "cold_per_prompt"
                                else tuple(
                                    bundle for bundle in prompt_bundles
                                    if bundle.phase == "generated"
                                )
                            )
                            for scope in ("combined", "prompt/prefill", "generated/decode"):
                                prompt_rows.append(_infeasible_row(
                                    view, capacity, policy, control, scope, bundles,
                                    f"capacity {capacity} is below maximum {view} bundle {summary.max_bundle_size}",
                                    len(plan.target_entries) if plan is not None else None,
                                    prompt_id=prompt.prompt.id,
                                    workload_count=1,
                                ))
                    continue

                primary = _simulate(evaluation, capacity, policy, plan)
                aggregate_rows.extend(_simulation_records(
                    primary, evaluation, view, "suite_persistent", 8
                ))

                cold_simulations = []
                decode_simulations = []
                for prompt in first_evaluation:
                    prompt_bundles = per_prompt[prompt.prompt.id]
                    cold = _simulate(prompt_bundles, capacity, policy, plan)
                    decode_bundles = tuple(
                        bundle for bundle in prompt_bundles if bundle.phase == "generated"
                    )
                    decode = _simulate(decode_bundles, capacity, policy, plan)
                    cold_simulations.append((prompt.prompt.id, prompt_bundles, cold))
                    decode_simulations.append((prompt.prompt.id, decode_bundles, decode))
                    prompt_rows.extend(_simulation_records(
                        cold, prompt_bundles, view, "cold_per_prompt", 1,
                        prompt_id=prompt.prompt.id,
                    ))
                    prompt_rows.extend(_simulation_records(
                        decode, decode_bundles, view, "decode_only_cold", 1,
                        prompt_id=prompt.prompt.id,
                    ))
                aggregate_rows.extend(_aggregate_simulations(
                    cold_simulations, view, capacity, policy, "cold_per_prompt"
                ))
                aggregate_rows.extend(_aggregate_simulations(
                    decode_simulations, view, capacity, policy, "decode_only_cold"
                ))
                macros.extend(_macro_records(
                    prompt_rows, view=view, capacity=capacity, policy=policy
                ))

    _assert_repetition_results_equal(inputs)
    continuation_inputs = [
        _continuation_record(repetition) for repetition in inputs.repetitions
    ]
    continuation = evaluate_continuation_rule(
        continuation_inputs,
        repeatability_passed=True,
        expected_evaluation_prompt_ids=(
            item.prompt.id for item in first_evaluation
        ),
    )
    measured = _measured_record(first_calibration, first_evaluation, view_inputs)
    overlaps = _target_overlap_records(plans, view_inputs)
    eos = [_eos_row(item) for item in (*first_calibration, *first_evaluation)]
    return {
        "format": STAGE1_BENCHMARK_FORMAT,
        "format_version": STAGE1_BENCHMARK_VERSION,
        "capacities": list(STAGE1_CAPACITIES),
        "views": list(STAGE1_VIEWS),
        "policies": list(SUITE_POLICIES),
        "controls": list(STAGE1_CONTROLS),
        "reproducibility": {
            "stage1_set_manifest": inputs.manifest_path.name,
            "stage1_set_manifest_sha256": inputs.manifest_sha256,
            "trusted_v02_manifest_sha256": "d5ca0013b1dfee96b353bb964666eaf9f6335494b0279362f536b4e8bc359e30",
            "model_id": inputs.manifest["model_id"],
            "model_revision": inputs.manifest["model_revision"],
            "corpus_version": first.corpus.version,
            "corpus_sha256": first.corpus.sha256,
            "max_decode_input_steps": inputs.manifest["max_decode_input_steps"],
            "torch_num_threads": inputs.manifest["torch_num_threads"],
            "environment": first.manifest["environment"],
            "generation": first.manifest["generation"],
            "repetition_manifest_sha256": [
                repetition.manifest_sha256 for repetition in inputs.repetitions
            ],
            "calibration_prompt_order": [item.prompt.id for item in first_calibration],
            "evaluation_prompt_order": [item.prompt.id for item in first_evaluation],
        },
        "repeatability": {
            "validated_repetitions": 3,
            "semantic_comparison": "PASS",
            "probability_max_abs_tolerance": 1e-6,
            "published_result_repetition": 1,
            "samples_not_tripled": True,
        },
        "measured": measured,
        "eos_accounting": eos,
        "aggregate_rows": aggregate_rows,
        "prompt_rows": prompt_rows,
        "cold_per_prompt_macros": macros,
        "fixed_target_plans": _plan_records(plans),
        "fixed_target_overlaps": overlaps,
        "continuation_rule": continuation,
        "claim_boundary": {
            "routing": "MEASURED",
            "cache_outcomes": "SIMULATED",
            "transfers": "ESTIMATED",
            "runtime_speedup_claimed": False,
            "expert_semantics_inferred": False,
            "grouped_view_verified_runtime_residency": False,
        },
    }


def _adapt(
    prompts: tuple[Stage1PromptInput, ...], view: str
) -> tuple[SimulationBundle, ...]:
    return adapt_prompt_traces(tuple(
        PromptTraceSource(item.prompt.id, item.prompt.order, item.trace)
        for item in prompts
    ), view)


def _bundles_by_prompt(
    bundles: tuple[SimulationBundle, ...], prompts: tuple[Stage1PromptInput, ...]
) -> dict[str, tuple[SimulationBundle, ...]]:
    return {
        prompt.prompt.id: tuple(
            bundle for bundle in bundles if bundle.prompt_id == prompt.prompt.id
        )
        for prompt in prompts
    }


def _simulate(
    bundles: tuple[SimulationBundle, ...],
    capacity: int,
    policy: str,
    plan: FixedTargetPlan | None,
) -> CacheSimulation:
    return simulate_bundles(
        bundles, capacity, policy, fixed_target_plan=plan
    )


def _scope_bundles(
    bundles: tuple[SimulationBundle, ...], scope: str
) -> tuple[SimulationBundle, ...]:
    if scope == "combined":
        return bundles
    phase = "prompt" if scope == "prompt/prefill" else "generated"
    return tuple(bundle for bundle in bundles if bundle.phase == phase)


def _result_record(
    result: CacheResult,
    bundles: tuple[SimulationBundle, ...],
) -> dict[str, Any]:
    return {
        "bundles": result.samples,
        "source_events": sum(bundle.source_event_count for bundle in bundles),
        "source_assignments": sum(bundle.source_assignment_count for bundle in bundles),
        "requests": result.requests,
        "hits": result.hits,
        "misses": result.misses,
        "hit_rate": result.hit_rate,
        "prewarm_loads": result.prewarm_loads,
        "demand_loads": result.demand_loads,
        "evictions": result.evictions,
        "estimated_transfers": result.estimated_expert_transfers,
    }


def _simulation_records(
    simulation: CacheSimulation,
    bundles: tuple[SimulationBundle, ...],
    view: str,
    control: str,
    workload_count: int,
    *,
    prompt_id: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for result in simulation.rows():
        record = {
            "view": view,
            "capacity": simulation.capacity,
            "policy": simulation.policy,
            "control": control,
            "scope": result.scope,
            "prompt_id": prompt_id,
            "workload_count": workload_count,
            "repetition_count": 3,
            "published_repetition_count": 1,
            "feasible": True,
            "n_a_reason": None,
            "fixed_target_size": len(simulation.fixed_entries) if simulation.policy in FIXED_POLICIES else None,
        }
        record.update(_result_record(result, _scope_bundles(bundles, result.scope)))
        rows.append(record)
    return rows


def _aggregate_simulations(
    simulations: list[tuple[str, tuple[SimulationBundle, ...], CacheSimulation]],
    view: str,
    capacity: int,
    policy: str,
    control: str,
) -> list[dict[str, Any]]:
    fixed_sets = {simulation.fixed_entries for _, _, simulation in simulations}
    if len(fixed_sets) != 1:
        raise ValueError("fixed targets changed across cold prompt controls")
    fixed_target_size = (
        len(next(iter(fixed_sets))) if policy in FIXED_POLICIES else None
    )
    rows = []
    for scope in ("combined", "prompt/prefill", "generated/decode"):
        results = [next(row for row in simulation.rows() if row.scope == scope)
                   for _, _, simulation in simulations]
        scope_bundles = tuple(
            bundle
            for _, bundles, _ in simulations
            for bundle in _scope_bundles(bundles, scope)
        )
        requests = sum(result.requests for result in results)
        row = {
            "view": view, "capacity": capacity, "policy": policy,
            "control": control, "scope": scope, "prompt_id": None,
            "workload_count": 8, "repetition_count": 3,
            "published_repetition_count": 1, "feasible": True,
            "n_a_reason": None,
            "fixed_target_size": fixed_target_size,
            "bundles": sum(result.samples for result in results),
            "source_events": sum(bundle.source_event_count for bundle in scope_bundles),
            "source_assignments": sum(bundle.source_assignment_count for bundle in scope_bundles),
            "requests": requests,
            "hits": sum(result.hits for result in results),
            "misses": sum(result.misses for result in results),
            "prewarm_loads": sum(result.prewarm_loads for result in results),
            "demand_loads": sum(result.demand_loads for result in results),
            "evictions": sum(result.evictions for result in results),
            "estimated_transfers": sum(result.estimated_expert_transfers for result in results),
        }
        row["hit_rate"] = row["hits"] / requests if requests else 0.0
        rows.append(row)
    return rows


def _infeasible_row(
    view: str, capacity: int, policy: str, control: str, scope: str,
    bundles: tuple[SimulationBundle, ...], reason: str,
    fixed_target_size: int | None,
    *,
    prompt_id: str | None = None,
    workload_count: int = 8,
) -> dict[str, Any]:
    scoped = _scope_bundles(bundles, scope)
    return {
        "view": view, "capacity": capacity, "policy": policy,
        "control": control, "scope": scope, "prompt_id": prompt_id,
        "workload_count": workload_count, "repetition_count": 3,
        "published_repetition_count": 1, "feasible": False,
        "n_a_reason": reason, "bundles": None,
        "fixed_target_size": fixed_target_size,
        "source_events": sum(bundle.source_event_count for bundle in scoped),
        "source_assignments": sum(bundle.source_assignment_count for bundle in scoped),
        "requests": sum(len(bundle.required_keys) for bundle in scoped),
        "hits": None, "misses": None, "hit_rate": None,
        "prewarm_loads": None, "demand_loads": None, "evictions": None,
        "estimated_transfers": None,
    }


def _macro_records(
    prompt_rows: list[dict[str, Any]], *, view: str, capacity: int, policy: str
) -> list[dict[str, Any]]:
    relevant = [
        row for row in prompt_rows
        if row["view"] == view and row["capacity"] == capacity
        and row["policy"] == policy and row["control"] == "cold_per_prompt"
    ]
    metrics = (
        "bundles", "source_events", "source_assignments", "requests", "hits",
        "misses", "hit_rate", "prewarm_loads", "demand_loads", "evictions",
        "estimated_transfers",
    )
    output = []
    for scope in ("combined", "prompt/prefill", "generated/decode"):
        scope_rows = [row for row in relevant if row["scope"] == scope]
        if len(scope_rows) != 8:
            raise ValueError("cold-per-prompt macros require all eight prompt/control pairs")
        for metric in metrics:
            output.append({
                "view": view, "capacity": capacity, "policy": policy,
                "scope": scope, "metric": metric,
                "statistics": sample_statistics(row[metric] for row in scope_rows),
            })
    return output


def _target_overlap_records(
    plans: dict[tuple[str, int, str], FixedTargetPlan],
    view_inputs: dict[str, tuple[tuple[SimulationBundle, ...], tuple[SimulationBundle, ...]]],
) -> list[dict[str, Any]]:
    grouped_max = summarize_bundles(view_inputs[PREFILL_LAYER_UNION_ATOMIC][1]).max_bundle_size
    output = []
    for capacity in STAGE1_CAPACITIES:
        if capacity < grouped_max:
            continue
        for policy in sorted(FIXED_POLICIES):
            token = plans.get((TOKEN_LAYER_ATOMIC, capacity, policy))
            grouped = plans.get((PREFILL_LAYER_UNION_ATOMIC, capacity, policy))
            if token is None or grouped is None:
                raise ValueError("fixed-target overlap requires a matched pair of feasible views")
            output.append({
                "capacity": capacity, "policy": policy,
                "token_target_size": len(token.target_entries),
                "grouped_target_size": len(grouped.target_entries),
                "overlap_size": len(token.target_entries.intersection(grouped.target_entries)),
            })
    return output


def _plan_records(
    plans: dict[tuple[str, int, str], FixedTargetPlan]
) -> list[dict[str, Any]]:
    return [
        {
            "view": view,
            "capacity": capacity,
            "policy": policy,
            "selection_scope": plan.selection_scope,
            "target_size": len(plan.target_entries),
            "target_entries": [list(key) for key in sorted(plan.target_entries)],
            "selected_prompt_provenance": [
                {"prompt_order": order, "prompt_id": prompt_id}
                for order, prompt_id in sorted(plan.selected_prompt_provenance)
            ],
            "opposite_prompt_provenance": [
                {"prompt_order": order, "prompt_id": prompt_id}
                for order, prompt_id in sorted(plan.opposite_prompt_provenance)
            ],
        }
        for (view, capacity, policy), plan in sorted(plans.items())
    ]


def _measured_record(
    calibration: tuple[Stage1PromptInput, ...],
    evaluation: tuple[Stage1PromptInput, ...],
    views: dict[str, tuple[tuple[SimulationBundle, ...], tuple[SimulationBundle, ...]]],
) -> dict[str, Any]:
    def measured(prompts: tuple[Stage1PromptInput, ...]) -> tuple[int, int]:
        return (
            sum(len(item.trace.events) for item in prompts),
            sum(len(item.trace.expert_requests) for item in prompts),
        )
    calibration_events, calibration_requests = measured(calibration)
    evaluation_events, evaluation_requests = measured(evaluation)
    view_records = {}
    for name, (calibration_bundles, evaluation_bundles) in views.items():
        view_records[name] = {
            "calibration": _summary_dict(calibration_bundles),
            "evaluation": _summary_dict(evaluation_bundles),
        }
    return {
        "calibration_prompt_count": 4,
        "evaluation_prompt_count": 8,
        "calibration_events": calibration_events,
        "calibration_requests": calibration_requests,
        "evaluation_events": evaluation_events,
        "evaluation_requests": evaluation_requests,
        "eos_shortened_calibration_prompts": sum(
            item.record["eos"]["actual_decode_input_steps_routed"] < 16
            for item in calibration
        ),
        "eos_shortened_evaluation_prompts": sum(
            item.record["eos"]["actual_decode_input_steps_routed"] < 16
            for item in evaluation
        ),
        "views": view_records,
    }


def _summary_dict(bundles: tuple[SimulationBundle, ...]) -> dict[str, Any]:
    summary = summarize_bundles(bundles)
    return {
        "prompt_count": summary.prompt_count,
        "combined": vars(summary.combined),
        "prompt": vars(summary.prompt),
        "generated": vars(summary.generated),
    }


def _eos_row(item: Stage1PromptInput) -> dict[str, Any]:
    eos = item.record["eos"]
    return {
        "split": item.prompt.split,
        "prompt_id": item.prompt.id,
        "requested_steps": eos["max_decode_input_steps_requested"],
        "actual_steps": eos["actual_decode_input_steps_routed"],
        "eos_emitted": eos["eos_emitted"],
        "horizon_exhausted": eos["horizon_exhausted"],
    }


def _continuation_record(repetition: Any) -> dict[str, Any]:
    evaluation = tuple(item for item in repetition.prompts if item.prompt.split == "evaluation")
    bundles = _adapt(evaluation, PREFILL_LAYER_UNION_ATOMIC)
    per_prompt = _bundles_by_prompt(bundles, evaluation)
    primary_32 = simulate_bundles(bundles, 32, "lfu")
    primary_256 = simulate_bundles(bundles, 256, "lfu")
    prompts = []
    for item in evaluation:
        decode = tuple(
            bundle for bundle in per_prompt[item.prompt.id] if bundle.phase == "generated"
        )
        result = simulate_bundles(decode, 256, "lfu").combined
        prompts.append({
            "prompt_id": item.prompt.id,
            "hits": result.hits,
            "requests": result.requests,
        })
    return {
        "repetition_number": repetition.manifest["repetition_number"],
        "lfu_grouped_primary_cap32_transfers": primary_32.combined.estimated_expert_transfers,
        "lfu_grouped_primary_cap256_transfers": primary_256.combined.estimated_expert_transfers,
        "lfu_grouped_decode_only_cap256_prompts": prompts,
    }


def _assert_repetition_results_equal(inputs: Stage1SetInputs) -> None:
    baseline = None
    for repetition in inputs.repetitions:
        calibration_prompts = tuple(
            item for item in repetition.prompts if item.prompt.split == "calibration"
        )
        evaluation = tuple(item for item in repetition.prompts if item.prompt.split == "evaluation")
        fingerprints = []
        for view in STAGE1_VIEWS:
            calibration_bundles = _adapt(calibration_prompts, view)
            bundles = _adapt(evaluation, view)
            per_prompt = _bundles_by_prompt(bundles, evaluation)
            maximum = summarize_bundles(bundles).max_bundle_size
            for capacity in STAGE1_CAPACITIES:
                if capacity < maximum:
                    continue
                for policy in SUITE_POLICIES:
                    plan = None
                    if policy in FIXED_POLICIES:
                        selection = (
                            calibration_bundles
                            if policy == "calibrated_static_frequency" else bundles
                        )
                        opposite = (
                            bundles
                            if policy == "calibrated_static_frequency"
                            else calibration_bundles
                        )
                        plan = build_fixed_target_plan(
                            selection, opposite, capacity, policy
                        )
                    primary = _simulate(bundles, capacity, policy, plan)
                    fingerprints.append((
                        view, capacity, policy, "suite_persistent", None,
                        _simulation_fingerprint(primary),
                    ))
                    for prompt in evaluation:
                        prompt_bundles = per_prompt[prompt.prompt.id]
                        cold = _simulate(prompt_bundles, capacity, policy, plan)
                        decode = _simulate(tuple(
                            bundle for bundle in prompt_bundles
                            if bundle.phase == "generated"
                        ), capacity, policy, plan)
                        fingerprints.extend((
                            (
                                view, capacity, policy, "cold_per_prompt",
                                prompt.prompt.id, _simulation_fingerprint(cold),
                            ),
                            (
                                view, capacity, policy, "decode_only_cold",
                                prompt.prompt.id, _simulation_fingerprint(decode),
                            ),
                        ))
        current = tuple(fingerprints)
        if baseline is None:
            baseline = current
        elif current != baseline:
            raise ValueError("Stage 1 simulated repetition results do not match")


def _simulation_fingerprint(simulation: CacheSimulation) -> tuple[Any, ...]:
    return (
        tuple(tuple(vars(row).values()) for row in simulation.rows()),
        tuple(sorted(simulation.fixed_entries)),
    )
