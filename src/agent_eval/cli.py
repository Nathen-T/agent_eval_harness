from __future__ import annotations

import argparse
from pathlib import Path

from agent_eval.compare import compare_runs, latest_runs
from agent_eval.runner import run_eval, save_run
from agent_eval.scorers import ExactMatchScorer, KeywordScorer
from agent_eval.systems import mock_system, real_llm_adapter
from agent_eval.task import load_tasks


DEFAULT_TESTSET = Path("data/testset.jsonl")
DEFAULT_RUNS_DIR = Path("runs")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Run and compare small LLM/agent evaluations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an eval and save a JSON result.")
    run_parser.add_argument("--testset", default=DEFAULT_TESTSET, type=Path)
    run_parser.add_argument("--out", default=DEFAULT_RUNS_DIR, type=Path)
    run_parser.add_argument("--system", choices=["mock", "real"], default="mock")
    run_parser.set_defaults(func=_run_command)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two saved runs, or the two latest runs by default."
    )
    compare_parser.add_argument("run_a", nargs="?", type=Path)
    compare_parser.add_argument("run_b", nargs="?", type=Path)
    compare_parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, type=Path)
    compare_parser.set_defaults(func=_compare_command)

    return parser


def _run_command(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.testset)
    scorers = [ExactMatchScorer(), KeywordScorer()]
    system_fn = mock_system if args.system == "mock" else real_llm_adapter

    result = run_eval(system_fn, tasks, scorers)
    path = save_run(result, args.out)

    print(f"Saved run: {path}")
    print(f"System: {result['system']}")
    print(f"Tasks: {result['task_count']}")
    print()
    print("Aggregates")
    for metric, score in sorted(result["aggregates"].items()):
        print(f"  {metric}: {score:.3f}")

    return 0


def _compare_command(args: argparse.Namespace) -> int:
    run_a = args.run_a
    run_b = args.run_b

    if run_a is None and run_b is None:
        recent_runs = latest_runs(args.runs_dir, count=2)
        if len(recent_runs) < 2:
            raise SystemExit(
                f"Need at least two run files in {args.runs_dir} to compare by default."
            )
        run_a, run_b = recent_runs
    elif run_a is None or run_b is None:
        raise SystemExit("Provide both RUN_A and RUN_B, or neither to compare latest runs.")

    regressions = compare_runs(run_a, run_b)
    return 1 if regressions else 0
