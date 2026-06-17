from __future__ import annotations

import argparse
from pathlib import Path

from example_rag.demo import run_demo
from example_rag.generators import MockGenerator
from example_rag.retrievers import build_retriever
from rag_eval.compare import compare_runs, latest_runs
from rag_eval.data import load_squad_v2
from rag_eval.pipeline import RAGPipeline
from rag_eval.runner import run_eval, save_run
from rag_eval.scorers import (
    AnswerCorrectnessScorer,
    GroundednessScorer,
    RetrievalHitRateScorer,
)

DEFAULT_RUNS_DIR = Path("runs")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Run and compare offline-first RAG QA evaluations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a RAG eval and save a JSON result.")
    run_parser.add_argument("--k", type=int, default=5)
    run_parser.add_argument("--retriever", choices=["bm25", "tfidf"], default="bm25")
    run_parser.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR)
    run_parser.set_defaults(func=_run_command)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two saved runs, or the two latest runs by default."
    )
    compare_parser.add_argument("run_a", nargs="?", type=Path)
    compare_parser.add_argument("run_b", nargs="?", type=Path)
    compare_parser.add_argument("--threshold", type=float, default=0.01)
    compare_parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    compare_parser.set_defaults(func=_compare_command)

    demo_parser = subparsers.add_parser(
        "demo", help="Run k=5 and k=1 squad_v2 configs, then compare them."
    )
    demo_parser.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR)
    demo_parser.add_argument("--retriever", choices=["bm25", "tfidf"], default="bm25")
    demo_parser.add_argument("--threshold", type=float, default=0.01)
    demo_parser.set_defaults(func=_demo_command)

    return parser


def _run_command(args: argparse.Namespace) -> int:
    tasks, corpus = _load_tasks_and_corpus()
    pipeline = build_mock_pipeline(corpus, retriever_name=args.retriever, k=args.k)
    result = run_eval(pipeline, tasks, default_scorers(), testset_name="squad_v2")
    path = save_run(result, args.out)

    print(f"Saved run: {path}")
    print(f"System: {result['system']}")
    print(f"Testset: {result['testset']}")
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

    regressions = compare_runs(run_a, run_b, threshold=args.threshold)
    return 1 if regressions else 0


def _demo_command(args: argparse.Namespace) -> int:
    run_demo(runs_dir=args.out, retriever_name=args.retriever, threshold=args.threshold)
    return 0


def _load_tasks_and_corpus():
    return load_squad_v2()


def build_mock_pipeline(
    corpus,
    retriever_name: str = "bm25",
    k: int = 5,
) -> RAGPipeline:
    retriever = build_retriever(retriever_name)
    generator = MockGenerator()
    return RAGPipeline(
        retriever,
        generator,
        corpus=corpus,
        k=k,
        name=f"mock-{retriever_name}-k{k}",
    )


def default_scorers():
    return [
        AnswerCorrectnessScorer(),
        GroundednessScorer(),
        RetrievalHitRateScorer(),
    ]
