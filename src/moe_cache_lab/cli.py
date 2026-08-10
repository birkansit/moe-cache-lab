"""Command line interface for trace collection, analysis, simulation, and reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cache import simulate
from .collector import DEFAULT_MODEL, collect_trace, inspect_model
from .report import render_report, render_suite_report, write_report
from .trace import read_trace, write_trace
from .workflow import DEFAULT_CAPACITIES, benchmark_suite, collect_corpus


def main() -> None:
    parser = argparse.ArgumentParser(prog="moe-cache-lab")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect",
        help="inspect model config without loading weights",
    )
    inspect.add_argument("--model", default=DEFAULT_MODEL)
    inspect.add_argument("--revision")

    collect = commands.add_parser(
        "collect",
        help="collect a measured Granite routing JSONL trace",
    )
    collect.add_argument("--prompt", required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--model", default=DEFAULT_MODEL)
    collect.add_argument("--revision")
    collect.add_argument("--max-new-tokens", type=int, default=0)

    analyze = commands.add_parser(
        "analyze",
        help="analyze an existing routing trace without model execution",
    )
    analyze.add_argument("trace", type=Path)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--json-output", type=Path)
    analyze.add_argument("--preflight-config", type=Path)

    benchmark = commands.add_parser(
        "benchmark",
        help="simulate count-capacity policies and write a Markdown report",
    )
    benchmark.add_argument("trace", type=Path)
    benchmark.add_argument("--capacity", type=int, required=True)
    benchmark.add_argument("--output", type=Path, required=True)

    collect_suite = commands.add_parser(
        "collect-corpus",
        help="collect the versioned local corpus with one model load",
    )
    collect_suite.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/corpus-v1.json"),
    )
    collect_suite.add_argument("--output-dir", type=Path, required=True)
    collect_suite.add_argument("--model", default=DEFAULT_MODEL)
    collect_suite.add_argument("--revision")
    collect_suite.add_argument("--max-new-tokens", type=int, default=2)

    suite = commands.add_parser(
        "benchmark-suite",
        help="validate a corpus manifest and simulate capacity curves",
    )
    suite.add_argument("manifest", type=Path)
    suite.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=list(DEFAULT_CAPACITIES),
    )
    suite.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "inspect":
        print(inspect_model(args.model, args.revision))

    elif args.command == "collect":
        trace = collect_trace(
            args.prompt,
            args.model,
            args.max_new_tokens,
            revision=args.revision,
        )
        print(write_trace(args.output, trace))

    elif args.command == "analyze":
        trace = read_trace(args.trace)

        if args.preflight_config is None:
            from .analysis import analyze_routing
            from .analysis_output import render_analysis_json, render_analysis_report

            summary = analyze_routing(trace)
            report = render_analysis_report(summary)
            json_report = (
                None if args.json_output is None else render_analysis_json(summary)
            )
        else:
            from .preflight import run_preflight_analysis
            from .preflight_config import read_preflight_config
            from .preflight_output import render_preflight_json, render_preflight_report

            result = run_preflight_analysis(
                trace,
                read_preflight_config(args.preflight_config),
            )
            report = render_preflight_report(result)
            json_report = (
                None if args.json_output is None else render_preflight_json(result)
            )

        if args.output is None:
            print(report, end="")
        else:
            write_report(args.output, report)

        if args.json_output is not None:
            write_report(args.json_output, json_report)

    elif args.command == "benchmark":
        trace = read_trace(args.trace)
        results = [
            simulate(trace.events, args.capacity, name)
            for name in ("lru", "lfu", "offline_oracle_frequency")
        ]
        print(write_report(args.output, render_report(trace, results)))

    elif args.command == "collect-corpus":
        print(
            collect_corpus(
                args.corpus,
                args.output_dir,
                model_id=args.model,
                revision=args.revision,
                max_new_tokens=args.max_new_tokens,
            )
        )

    elif args.command == "benchmark-suite":
        benchmark_result = benchmark_suite(args.manifest, args.capacities)
        print(write_report(args.output, render_suite_report(benchmark_result)))


if __name__ == "__main__":
    main()
