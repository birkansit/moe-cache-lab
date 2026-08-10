"""Human-readable reports for measured traces and simulated cache results."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from .cache import CacheSimulation
from .trace import RoutingTrace
from .workflow import LoadedPromptTrace, SUITE_POLICIES, SuiteBenchmark


def render_report(trace: RoutingTrace, simulations: Iterable[CacheSimulation]) -> str:
    simulations = tuple(simulations)
    frequencies = Counter(trace.expert_requests)
    phase_counts = Counter(event.phase for event in trace.events)
    lines = [
        "# MoE cache-lab benchmark report", "",
        "## Scope and interpretation", "",
        "- Routing selections are **measured** from the capture method named below.",
        "- Cache outcomes are **simulated** by atomically replaying routing-event bundles through one chronological shared cache.",
        "- Expert identity is the layer-qualified `(layer_id, expert_id)` weight object; capacity is the total number resident across the model.",
        "- Dynamic caches start empty. Their demand loads and **estimated** transfers both equal misses.",
        "- `offline_oracle_frequency` is explicitly non-causal: its fixed target set uses the full evaluation trace and is restored after every event. Misses temporarily displace non-required targets when necessary; all post-prewarm loads, including restorations, are demand loads.",
        "- Every resident removal is an eviction. With a full fixed target set, each miss causes two demand loads and two evictions: temporary requirement plus target restoration.",
        "- Phase rows are attributed from that same shared-cache run, not cold reruns. One-time oracle prewarm appears only in the combined row.",
        "- Generated/decode samples are routing events for emitted tokens when those tokens are used as decode inputs.",
        "- Estimated transfers are not measured data movement, latency, throughput, runtime performance, or actual expert offloading.",
        "- Expert IDs are identifiers only; this report makes no semantic claims about them.", "",
        "## Trace", "",
        f"- Model: `{trace.model_id}`",
        f"- Experts per routed layer: {trace.num_experts}; selected per token/layer: {trace.experts_per_token}",
        f"- Routing events: {len(trace.events)} (prompt/prefill {phase_counts['prompt']}, generated/decode {phase_counts['generated']})",
        f"- Layer-qualified expert requests: {len(trace.expert_requests)}; distinct layer-qualified experts: {len(frequencies)}",
        f"- Capture method: {trace.capture_method}", "",
        "## Simulated cache policies", "",
        "| policy | scope | capacity | samples | requests | hits | misses | hit rate | prewarm loads | demand loads | evictions | estimated transfers |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for simulation in simulations:
        for result in simulation.rows():
            lines.append(
                f"| {simulation.policy} | {result.scope} | {simulation.capacity} | {result.samples} | "
                f"{result.requests} | {result.hits} | {result.misses} | {result.hit_rate:.2%} | "
                f"{result.prewarm_loads} | {result.demand_loads} | {result.evictions} | "
                f"{result.estimated_expert_transfers} |"
            )
    if frequencies:
        top = ", ".join(
            f"layer {layer} / expert {expert} ({count})"
            for (layer, expert), count in frequencies.most_common(10)
        )
        lines.extend(["", "## Most requested layer-qualified experts", "", top])
    return "\n".join(lines) + "\n"


def write_report(path: str | Path, report: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8", newline="\n")
    return destination


def render_suite_report(benchmark: SuiteBenchmark) -> str:
    """Render an auditable V0.2 corpus benchmark with capacity curves."""
    inputs = benchmark.inputs
    manifest = inputs.manifest
    environment = manifest["environment"]
    generation = manifest["generation"]
    evaluation_events = tuple(
        event for item in inputs.evaluation for event in item.trace.events
    )
    phase_counts = Counter(event.phase for event in evaluation_events)
    request_count = sum(len(event.selected_experts) for event in evaluation_events)
    calibration_events = tuple(
        event for item in inputs.calibration for event in item.trace.events
    )
    calibration_requests = sum(len(event.selected_experts) for event in calibration_events)
    prompt_assignments, active_group_objects, group_minimum, group_maximum = prefill_grouping_metrics(
        inputs.evaluation
    )
    calibration_order = ", ".join(item.prompt.id for item in inputs.calibration)
    evaluation_order = ", ".join(item.prompt.id for item in inputs.evaluation)
    lines = [
        "# MoE cache-lab V0.2 corpus benchmark", "",
        "## MEASURED routing and workload metadata", "",
        "Routing events come from the recorded model traces. Corpus categories describe workload diversity only; they do not identify expert meaning, and no expert semantics may be inferred from IDs, frequency, prompts, or categories.", "",
        f"- Model: `{manifest['model_id']}`",
        f"- Routing config: {manifest['routing']['experts_per_layer']} experts per routed layer; {manifest['routing']['selected_experts_per_token']} selected per token; {manifest['routing']['routed_layer_count']} routed layers `{manifest['routing']['routed_layer_ids']}`",
        f"- Corpus: `{inputs.corpus.version}`; canonical JSON SHA-256 `{inputs.corpus.sha256}`",
        f"- Calibration prompts: {len(inputs.calibration)}; order: {calibration_order}",
        f"- Calibration measured routing samples/requests: {len(calibration_events)} / {calibration_requests}",
        f"- Evaluation prompts: {len(inputs.evaluation)}; order: {evaluation_order}",
        f"- Reported evaluation routing samples: {len(evaluation_events)} (prompt/prefill {phase_counts['prompt']}, generated/decode {phase_counts['generated']})",
        f"- Reported evaluation layer-qualified requests: {request_count}", "",
        f"Evaluation prefill grouping metadata: {prompt_assignments} prompt token-expert assignments; {active_group_objects} unique active `(prompt, layer, expert)` objects across prompt/layer groups; min/max unique expert working set per prompt/layer group: {group_minimum}/{group_maximum}.", "",
        "Granite prefill dispatch groups assignments from all prompt tokens in a layer by expert. The simulator intentionally retains per-token/layer atomic events, so simulated per-token prompt hits do not demonstrate cache reuse between actual grouped dispatch units. Capacities below grouped prefill working sets have no validated runtime scheduling interpretation.", "",
        "Each prompt was an independent model context during collection. Cache simulation preserves one chronological cache across the evaluation prompt order above.", "",
        "## Reproducibility", "",
        f"- Manifest: `{inputs.manifest_path.name}`; file SHA-256 `{inputs.manifest_sha256}`",
        f"- Requested model revision: `{manifest['model_revision']['requested'] or 'repository default'}`; resolved model/tokenizer commit: `{manifest['model_revision']['resolved']}`",
        f"- Corpus path recorded by manifest: `{manifest['corpus']['path']}`",
        f"- moe-cache-lab `{environment['moe_cache_lab']}`; Python `{environment['python']}`; Transformers `{environment['transformers']}`; PyTorch `{environment['torch']}`",
        f"- Collection: device `{environment['device']}`, dtype `{environment['dtype']}`, strategy `{generation['strategy']}`, token selection `{generation['token_selection']}`, do_sample `{str(generation['do_sample']).lower()}`, max_new_tokens `{generation['max_new_tokens']}`, use_cache `{str(generation['use_cache']).lower()}`",
        f"- Capacities: {', '.join(str(value) for value in benchmark.capacities)} total simultaneously resident layer-qualified expert objects",
        f"- Policies: {', '.join(f'`{name}`' for name in SUITE_POLICIES)}", "",
        "Policy definitions: `lru` is event-atomic least-recently-used; `lfu` is online resident-frequency LFU; `calibrated_static_frequency` selects a fixed target only from the four calibration traces; `offline_oracle_frequency` is non-causal and selects a fixed target from the full evaluation trace. Both fixed policies obey hard capacity and restore their targets after each event. Calibration traces never contribute reported outcomes.", "",
        "## SIMULATED cache outcomes and ESTIMATED transfers", "",
        "All rows are simulation results. Transfer counts are estimates derived from prewarm and demand loads, not observed model data movement or runtime performance. Negative results are retained unchanged.", "",
        "| policy | capacity | scope | samples | requests | hits | misses | hit rate | prewarm loads | demand loads | evictions | ESTIMATED transfers |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for simulation in benchmark.simulations:
        for result in simulation.rows():
            lines.append(
                f"| {simulation.policy} | {simulation.capacity} | {result.scope} | {result.samples} | "
                f"{result.requests} | {result.hits} | {result.misses} | {result.hit_rate:.2%} | "
                f"{result.prewarm_loads} | {result.demand_loads} | {result.evictions} | "
                f"{result.estimated_expert_transfers} |"
            )
    return "\n".join(lines) + "\n"


def prefill_grouping_metrics(
    traces: Iterable[LoadedPromptTrace],
) -> tuple[int, int, int, int]:
    """Return assignments, active objects, and min/max prompt-layer working sets."""
    groups: dict[tuple[str, int], set[int]] = {}
    assignments = 0
    for item in traces:
        for event in item.trace.events:
            if event.phase != "prompt":
                continue
            assignments += len(event.selected_experts)
            groups.setdefault((item.prompt.id, event.layer), set()).update(event.selected_experts)
    sizes = [len(experts) for experts in groups.values()]
    return assignments, sum(sizes), min(sizes, default=0), max(sizes, default=0)
