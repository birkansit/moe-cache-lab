"""Command line interface for collection, simulation, and reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .cache import simulate
from .granite_dependencies import (
    DEFAULT_MODEL,
    GraniteDependencyError,
    load_granite_modules,
)
from .report import render_report, render_suite_report, write_report
from .trace import read_trace, write_trace
from .trace_v2 import RoutingTraceV2, read_versioned_trace
from .trace_validation import (
    render_trace_validation_human,
    render_trace_validation_json,
    validate_trace_file,
)
from .workflow import DEFAULT_CAPACITIES, benchmark_suite, collect_corpus


def inspect_model(model_id: str = DEFAULT_MODEL, revision: str | None = None):
    """Lazy compatibility wrapper for the optional Granite inspector."""
    return load_granite_modules()[2].inspect_model(model_id, revision)


def collect_trace(
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    max_new_tokens: int = 0,
    *,
    revision: str | None = None,
):
    """Lazy compatibility wrapper for optional Granite trace collection."""
    return load_granite_modules()[2].collect_trace(
        prompt,
        model_id,
        max_new_tokens,
        revision=revision,
    )


def _require_granite_dependencies(parser: argparse.ArgumentParser) -> None:
    try:
        load_granite_modules()
    except GraniteDependencyError as error:
        parser.error(str(error))


def _read_bundle_config(path: Path) -> dict:
    """Read one strict JSON object without normalizing duplicate keys."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard numeric constant {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"bundle config is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("bundle config must be a JSON object")
    return value


def _bundle_summary(result) -> str:
    return json.dumps(
        {
            "artifact_count": result.artifact_count,
            "experiment_id": result.experiment_id,
            "manifest_sha256": result.manifest_sha256,
            "trace_count": result.trace_count,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_analyze_input(
    parser: argparse.ArgumentParser,
    path: Path,
    reader,
    *,
    label: str,
):
    """Read one user-controlled analyze input through its strict reader."""
    try:
        return reader(path)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(f"{label}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="moe-cache-lab",
        description=(
            "Offline routing-evidence analysis for strict canonical v1/v2 traces, "
            "with optional model-specific collection and reproducibility bundles."
        ),
        epilog=(
            "Canonical path: collect with a validated optional collector or import "
            "a canonical trace -> validate -> analyze offline -> compare/report where the "
            "contract applies -> optionally bundle-create and bundle-verify."
        ),
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="COMMAND"
    )
    inspect = commands.add_parser(
        "inspect",
        help="inspect the built-in Granite model config (requires the granite extra)",
    )
    inspect.add_argument("--model", default=DEFAULT_MODEL)
    inspect.add_argument("--revision")
    collect = commands.add_parser(
        "collect",
        help="collect with the built-in Granite path (requires the granite extra)",
    )
    collect.add_argument("--prompt", required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--model", default=DEFAULT_MODEL)
    collect.add_argument("--revision")
    collect.add_argument("--max-new-tokens", type=int, default=0)
    validate_trace = commands.add_parser(
        "validate-trace",
        help="strictly validate one canonical v1/v2 trace without analysis",
    )
    validate_trace.add_argument("trace", type=Path)
    validate_trace.add_argument(
        "--json",
        action="store_true",
        dest="json_mode",
        help="emit one deterministic machine-readable validation object",
    )
    analyze = commands.add_parser(
        "analyze",
        help="analyze a strict canonical v1/v2 trace offline without ML dependencies",
    )
    analyze.add_argument("trace", type=Path)
    analyze.add_argument(
        "--output", type=Path, help="write deterministic Markdown instead of stdout"
    )
    analyze.add_argument(
        "--json-output", type=Path, help="also write deterministic structured JSON"
    )
    analyze.add_argument(
        "--preflight-config",
        type=Path,
        help=(
            "run compatible cache/transfer preflight: trace v1 with config v1/v2 "
            "or trace v2 with stage-qualified config v3"
        ),
    )
    analyze.add_argument(
        "--workload-id",
        default="trace",
        help="stable path-independent workload ID for v2 evidence (default: trace)",
    )
    analyze.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        help="explicit v2 locality ranks in caller order; no ranks are assumed",
    )
    capacity_frontier = commands.add_parser(
        "capacity-frontier",
        help=(
            "derive the exact event-atomic LRU count frontier and optional "
            "caller-sized byte frontier"
        ),
    )
    capacity_frontier.add_argument("trace", type=Path)
    capacity_frontier.add_argument(
        "--preflight-config",
        type=Path,
        help=(
            "derive the optional byte frontier from a compatible existing "
            "preflight config"
        ),
    )
    capacity_frontier.add_argument(
        "--output",
        type=Path,
        help="write bounded deterministic Markdown instead of stdout",
    )
    capacity_frontier.add_argument(
        "--json-output",
        type=Path,
        help="also write the complete deterministic machine frontier",
    )
    bundle_create = commands.add_parser(
        "bundle-create",
        help="create a deterministic offline experiment-integrity bundle",
    )
    bundle_create.add_argument("--experiment-id", required=True)
    bundle_create.add_argument("--config", type=Path, required=True)
    bundle_create.add_argument("--report-json", type=Path, required=True)
    bundle_create.add_argument("--report-markdown", type=Path, required=True)
    bundle_create.add_argument("--output-dir", type=Path, required=True)
    bundle_create.add_argument(
        "--embed-trace",
        nargs=2,
        action="append",
        default=[],
        metavar=("TRACE_ID", "PATH"),
        help="embed an existing strict canonical trace; repeat as needed",
    )
    bundle_create.add_argument(
        "--external-trace",
        nargs=3,
        action="append",
        default=[],
        metavar=("TRACE_ID", "REFERENCE", "SHA256"),
        help="record an immutable metadata-only trace reference; never fetched",
    )
    bundle_create.add_argument("--torch-version")
    bundle_create.add_argument("--transformers-version")
    bundle_create.add_argument("--device")
    bundle_create.add_argument("--dtype")
    bundle_verify = commands.add_parser(
        "bundle-verify",
        help="strictly verify an offline experiment bundle without repair",
    )
    bundle_verify.add_argument("bundle", type=Path)
    lifecycle = commands.add_parser(
        "analyze-lifecycle",
        help="historical compatibility: compare v1 cold/persistent cache scenarios",
    )
    lifecycle.add_argument("manifest", type=Path)
    lifecycle.add_argument("--preflight-config", type=Path, required=True)
    lifecycle.add_argument("--output", type=Path, required=True)
    lifecycle.add_argument("--json-output", type=Path, required=True)
    benchmark = commands.add_parser(
        "benchmark",
        help="historical compatibility: simulate v1 count-capacity policies",
    )
    benchmark.add_argument("trace", type=Path)
    benchmark.add_argument("--capacity", type=int, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    collect_suite = commands.add_parser(
        "collect-corpus",
        help="historical reproduction: collect the Granite corpus (requires extra)",
    )
    collect_suite.add_argument("--corpus", type=Path, default=Path("benchmarks/corpus-v1.json"))
    collect_suite.add_argument("--output-dir", type=Path, required=True)
    collect_suite.add_argument("--model", default=DEFAULT_MODEL)
    collect_suite.add_argument("--revision")
    collect_suite.add_argument("--max-new-tokens", type=int, default=2)
    suite = commands.add_parser(
        "benchmark-suite",
        help="historical reproduction: validate a corpus and simulate curves",
    )
    suite.add_argument("manifest", type=Path)
    suite.add_argument("--capacities", type=int, nargs="+", default=list(DEFAULT_CAPACITIES))
    suite.add_argument("--output", type=Path, required=True)
    stage1_collect = commands.add_parser(
        "collect-stage1",
        help="historical reproduction: collect the frozen Stage 1 trace set",
    )
    stage1_collect.add_argument(
        "--corpus", type=Path, default=Path("benchmarks/corpus-v1.json")
    )
    stage1_collect.add_argument("--output-dir", type=Path, required=True)
    stage1_validate = commands.add_parser(
        "validate-stage1",
        help="historical reproduction: validate a Stage 1 set and V0.2 prefix",
    )
    stage1_validate.add_argument("manifest", type=Path)
    stage1_validate.add_argument(
        "--trusted-v02-manifest",
        type=Path,
        default=Path("results/v0.2-corpus-v1/manifest-v1.json"),
    )
    stage1_benchmark = commands.add_parser(
        "benchmark-stage1",
        help="historical reproduction: simulate frozen Stage 1 views/controls",
    )
    stage1_benchmark.add_argument("manifest", type=Path)
    stage1_benchmark.add_argument("--json-output", type=Path, required=True)
    stage1_benchmark.add_argument("--output", type=Path, required=True)
    stage1_benchmark.add_argument(
        "--trusted-v02-manifest",
        type=Path,
        default=Path("results/v0.2-corpus-v1/manifest-v1.json"),
    )
    stage1_child = commands.add_parser("_collect-stage1-child", help=argparse.SUPPRESS)
    stage1_child.add_argument("--corpus", type=Path, required=True)
    stage1_child.add_argument("--output-dir", type=Path, required=True)
    stage1_child.add_argument("--repetition-number", type=int, required=True)
    stage1_child.add_argument("--process-uuid", required=True)
    v04_collect = commands.add_parser(
        "collect-v04",
        help="historical reproduction: collect the reviewed V0.4 CPU experiment",
    )
    v04_collect.add_argument("--corpus", type=Path, default=Path("benchmarks/corpus-v1.json"))
    v04_collect.add_argument("--output-root", type=Path, default=Path("results"))
    v04_collect.add_argument("--rerun-of", type=Path)
    v04_collect.add_argument(
        "--retry-of", type=Path,
        help="authorized recovery lineage for one deferred, invalid, or blocked attempt",
    )
    v04_validate = commands.add_parser(
        "validate-v04",
        help="historical reproduction: validate a V0.4 attempt manifest",
    )
    v04_validate.add_argument("manifest", type=Path)
    v04_child = commands.add_parser("_collect-v04-child", help=argparse.SUPPRESS)
    v04_child.add_argument("--corpus", type=Path, required=True)
    v04_child.add_argument("--output-dir", type=Path, required=True)
    v04_child.add_argument("--child-number", type=int, required=True)
    v04_child.add_argument("--process-uuid", required=True)
    commands._choices_actions = [
        action
        for action in commands._choices_actions
        if action.help != argparse.SUPPRESS
    ]
    args = parser.parse_args()
    if args.command == "inspect":
        try:
            result = inspect_model(args.model, args.revision)
        except GraniteDependencyError as error:
            parser.error(str(error))
        print(result)
    elif args.command == "collect":
        try:
            trace = collect_trace(
                args.prompt, args.model, args.max_new_tokens, revision=args.revision
            )
        except GraniteDependencyError as error:
            parser.error(str(error))
        print(write_trace(args.output, trace))
    elif args.command == "validate-trace":
        validation = validate_trace_file(args.trace)
        if args.json_mode:
            print(render_trace_validation_json(validation), end="")
        elif validation.valid:
            print(render_trace_validation_human(validation), end="")
        else:
            print(render_trace_validation_human(validation), end="", file=sys.stderr)
        if not validation.valid:
            raise SystemExit(2)
    elif args.command == "capacity-frontier":
        from .capacity_frontier import (
            CapacityFrontierInputError,
            analyze_capacity_frontier,
        )
        from .capacity_frontier_output import (
            render_capacity_frontier_json,
            render_capacity_frontier_report,
        )
        from .preflight_config import PreflightConfigV3, read_preflight_config

        trace = _read_analyze_input(
            parser,
            args.trace,
            read_versioned_trace,
            label="trace input error",
        )
        sizes = None
        if args.preflight_config is not None:
            config = _read_analyze_input(
                parser,
                args.preflight_config,
                read_preflight_config,
                label="preflight config input error",
            )
            if isinstance(trace, RoutingTraceV2):
                if not isinstance(config, PreflightConfigV3):
                    parser.error(
                        "routing trace v2 requires stage-qualified preflight config "
                        "format_version 3"
                    )
            elif isinstance(config, PreflightConfigV3):
                parser.error(
                    "preflight config format_version 3 is stage-qualified and "
                    "cannot be used with trace v1"
                )
            sizes = config.expert_size_map()
        try:
            result = analyze_capacity_frontier(trace, sizes)
        except CapacityFrontierInputError as error:
            parser.error(str(error))
        report = render_capacity_frontier_report(result)
        if args.output is None:
            print(report, end="")
        else:
            write_report(args.output, report)
        if args.json_output is not None:
            write_report(args.json_output, render_capacity_frontier_json(result))
    elif args.command == "analyze":
        trace = _read_analyze_input(
            parser,
            args.trace,
            read_versioned_trace,
            label="trace input error",
        )
        if isinstance(trace, RoutingTraceV2):
            if args.preflight_config is not None:
                if args.top_k is not None:
                    parser.error(
                        "--top-k cannot be combined with trace v2 preflight analysis"
                    )
                from .preflight import run_stage_qualified_preflight_analysis
                from .preflight_config import PreflightConfigV3, read_preflight_config
                from .preflight_output import (
                    render_stage_qualified_preflight_json,
                    render_stage_qualified_preflight_report,
                )

                config = _read_analyze_input(
                    parser,
                    args.preflight_config,
                    read_preflight_config,
                    label="preflight config input error",
                )
                if not isinstance(config, PreflightConfigV3):
                    parser.error(
                        "routing trace v2 requires stage-qualified preflight config "
                        "format_version 3"
                    )
                result = run_stage_qualified_preflight_analysis(
                    trace, config, workload_id=args.workload_id
                )
                report = render_stage_qualified_preflight_report(result)
                json_report = (
                    None
                    if args.json_output is None
                    else render_stage_qualified_preflight_json(result)
                )
            else:
                from .v2_analysis_output import (
                    analyze_v2_trace,
                    render_v2_analysis_json,
                    render_v2_analysis_report,
                )

                try:
                    result = analyze_v2_trace(
                        trace,
                        workload_id=args.workload_id,
                        top_k=None if args.top_k is None else tuple(args.top_k),
                    )
                except (TypeError, ValueError) as error:
                    parser.error(str(error))
                report = render_v2_analysis_report(result)
                json_report = (
                    None
                    if args.json_output is None
                    else render_v2_analysis_json(result)
                )
        elif args.top_k is not None:
            parser.error("--top-k is available only for trace v2 analysis")
        elif args.preflight_config is None:
            from .analysis import analyze_routing
            from .analysis_output import render_analysis_json, render_analysis_report

            summary = analyze_routing(trace)
            report = render_analysis_report(summary)
            json_report = (
                None
                if args.json_output is None
                else render_analysis_json(summary)
            )
        else:
            from .preflight import run_preflight_analysis
            from .preflight_config import PreflightConfigV3, read_preflight_config
            from .preflight_output import render_preflight_json, render_preflight_report

            config = _read_analyze_input(
                parser,
                args.preflight_config,
                read_preflight_config,
                label="preflight config input error",
            )
            if isinstance(config, PreflightConfigV3):
                parser.error(
                    "preflight config format_version 3 is stage-qualified and "
                    "cannot be used with trace v1"
                )
            result = run_preflight_analysis(
                trace,
                config,
            )
            report = render_preflight_report(result)
            json_report = (
                None
                if args.json_output is None
                else render_preflight_json(result)
            )

        if args.output is None:
            print(report, end="")
        else:
            write_report(args.output, report)
        if args.json_output is not None:
            write_report(args.json_output, json_report)
    elif args.command == "bundle-create":
        from .experiment_bundle import (
            EmbeddedTrace,
            ExternalTrace,
            RuntimeProvenance,
            write_experiment_bundle,
        )

        try:
            traces = tuple(
                EmbeddedTrace(trace_id, Path(path))
                for trace_id, path in args.embed_trace
            ) + tuple(
                ExternalTrace(trace_id, reference, sha256)
                for trace_id, reference, sha256 in args.external_trace
            )
            result = write_experiment_bundle(
                args.output_dir,
                experiment_id=args.experiment_id,
                config=_read_bundle_config(args.config),
                report_json=args.report_json.read_bytes(),
                report_markdown=args.report_markdown.read_bytes(),
                traces=traces,
                runtime_provenance=RuntimeProvenance(
                    torch_version=args.torch_version,
                    transformers_version=args.transformers_version,
                    device=args.device,
                    dtype=args.dtype,
                ),
            )
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        print(_bundle_summary(result))
    elif args.command == "bundle-verify":
        from .experiment_bundle import verify_experiment_bundle

        try:
            result = verify_experiment_bundle(args.bundle)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        print(_bundle_summary(result))
    elif args.command == "benchmark":
        trace = read_trace(args.trace)
        results = [
            simulate(trace.events, args.capacity, name)
            for name in ("lru", "lfu", "offline_oracle_frequency")
        ]
        print(write_report(args.output, render_report(trace, results)))
    elif args.command == "analyze-lifecycle":
        from .preflight import run_preflight_lifecycle_analysis
        from .preflight_config import PreflightConfigV3, read_preflight_config
        from .preflight_output import (
            render_preflight_lifecycle_json,
            render_preflight_lifecycle_report,
        )

        config = read_preflight_config(args.preflight_config)
        if isinstance(config, PreflightConfigV3):
            parser.error(
                "preflight config format_version 3 is stage-qualified and "
                "cannot be used with v1 lifecycle analysis"
            )
        lifecycle_result = run_preflight_lifecycle_analysis(args.manifest, config)
        print(write_report(
            args.output,
            render_preflight_lifecycle_report(lifecycle_result),
        ))
        print(write_report(
            args.json_output,
            render_preflight_lifecycle_json(lifecycle_result),
        ))
    elif args.command == "collect-corpus":
        try:
            print(collect_corpus(
                args.corpus,
                args.output_dir,
                model_id=args.model,
                revision=args.revision,
                max_new_tokens=args.max_new_tokens,
            ))
        except GraniteDependencyError as error:
            parser.error(str(error))
    elif args.command == "benchmark-suite":
        benchmark_result = benchmark_suite(args.manifest, args.capacities)
        print(write_report(args.output, render_suite_report(benchmark_result)))
    elif args.command == "collect-stage1":
        _require_granite_dependencies(parser)
        from .stage1 import collect_stage1_set
        print(collect_stage1_set(args.corpus, args.output_dir))
    elif args.command == "validate-stage1":
        from .stage1 import validate_stage1_set
        print(validate_stage1_set(
            args.manifest, args.trusted_v02_manifest
        ).manifest_path)
    elif args.command == "benchmark-stage1":
        from .stage1_benchmark import (
            benchmark_stage1,
            write_stage1_json,
            write_stage1_markdown,
        )
        benchmark_result = benchmark_stage1(
            args.manifest, args.trusted_v02_manifest
        )
        print(write_stage1_json(args.json_output, benchmark_result))
        print(write_stage1_markdown(args.output, benchmark_result))
    elif args.command == "_collect-stage1-child":
        _require_granite_dependencies(parser)
        from .stage1 import collect_stage1_repetition
        print(collect_stage1_repetition(
            args.corpus,
            args.output_dir,
            args.repetition_number,
            args.process_uuid,
        ))
    elif args.command == "collect-v04":
        _require_granite_dependencies(parser)
        from .v04 import collect_v04
        print(collect_v04(
            args.corpus, args.output_root, rerun_of=args.rerun_of,
            retry_of=args.retry_of,
        ))
    elif args.command == "validate-v04":
        from .v04 import validate_v04
        print(validate_v04(args.manifest))
    elif args.command == "_collect-v04-child":
        _require_granite_dependencies(parser)
        from .v04 import collect_v04_child
        print(collect_v04_child(
            args.corpus, args.output_dir, args.child_number, args.process_uuid
        ))


if __name__ == "__main__":
    main()
