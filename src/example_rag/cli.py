from __future__ import annotations

import argparse
from pathlib import Path
import re

from example_rag.demo import run_demo
from example_rag.generators import DEFAULT_EXTRACTIVE_QA_MODEL, build_generator
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
    try:
        return args.func(args)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Run and compare offline-first RAG QA evaluations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a RAG eval and save a JSON result.")
    run_parser.add_argument("--k", type=int, default=5)
    run_parser.add_argument("--retriever", choices=["bm25", "tfidf"], default="bm25")
    _add_generator_args(run_parser)
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
    _add_generator_args(demo_parser)
    demo_parser.add_argument("--threshold", type=float, default=0.01)
    demo_parser.set_defaults(func=_demo_command)

    return parser


def _add_generator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generator",
        choices=["mock", "local", "api", "hf"],
        default="mock",
        help=(
            "Generator tier to use. Defaults to the offline deterministic mock. "
            "'hf' runs a local HuggingFace extractive QA model on CPU (no server, "
            "no token)."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Model id. For local/api this is the OpenAI-compatible model name; "
            "for hf it is a HuggingFace QA model (default "
            "distilbert-base-cased-distilled-squad)."
        ),
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL. Defaults to LM Studio locally.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for hosted providers; local servers may use any placeholder.",
    )


def _run_command(args: argparse.Namespace) -> int:
    tasks, corpus = _load_tasks_and_corpus()
    pipeline = build_pipeline(
        corpus,
        retriever_name=args.retriever,
        generator_name=args.generator,
        k=args.k,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
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
    run_demo(
        runs_dir=args.out,
        retriever_name=args.retriever,
        generator_name=args.generator,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        threshold=args.threshold,
    )
    return 0


def _load_tasks_and_corpus():
    return load_squad_v2()


def build_pipeline(
    corpus,
    retriever_name: str = "bm25",
    generator_name: str = "mock",
    k: int = 5,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> RAGPipeline:
    retriever = build_retriever(retriever_name)
    generator = build_generator(
        generator_name,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    return RAGPipeline(
        retriever,
        generator,
        corpus=corpus,
        k=k,
        name=system_label(generator_name, retriever_name, k, model),
    )


def build_mock_pipeline(
    corpus,
    retriever_name: str = "bm25",
    k: int = 5,
) -> RAGPipeline:
    return build_pipeline(corpus, retriever_name=retriever_name, k=k)


def system_label(
    generator_name: str,
    retriever_name: str,
    k: int,
    model: str | None = None,
) -> str:
    if generator_name == "mock":
        return f"mock-{retriever_name}-k{k}"

    if generator_name == "hf":
        model_label = _sanitize_label(model or DEFAULT_EXTRACTIVE_QA_MODEL)
        return f"hf-{model_label}-k{k}"

    model_label = _sanitize_label(model or "model")
    prefix = "lmstudio" if generator_name == "local" else "api"
    return f"{prefix}-{model_label}-k{k}"


def _sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return sanitized or "model"


def default_scorers():
    return [
        AnswerCorrectnessScorer(),
        GroundednessScorer(),
        RetrievalHitRateScorer(),
    ]
