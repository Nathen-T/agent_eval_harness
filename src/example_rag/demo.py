from __future__ import annotations

from pathlib import Path
import re

from example_rag.generators import build_generator
from example_rag.retrievers import build_retriever
from rag_eval.compare import compare_runs
from rag_eval.data import load_squad_v2
from rag_eval.pipeline import RAGPipeline
from rag_eval.runner import run_eval, save_run
from rag_eval.scorers import (
    AnswerCorrectnessScorer,
    GroundednessScorer,
    RetrievalHitRateScorer,
)

HEALTHY_K = 5
STARVED_K = 1


def run_demo(
    runs_dir: str | Path = "runs",
    retriever_name: str = "bm25",
    generator_name: str = "mock",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    threshold: float = 0.01,
) -> tuple[Path, Path]:
    """Run the same eval with healthy (k=5) and starved (k=1) retrieval, then compare.

    Only ``k`` changes between the two runs, so any metric drop is attributable to
    retrieving less context.
    """

    tasks, corpus = load_squad_v2()
    scorers = [AnswerCorrectnessScorer(), GroundednessScorer(), RetrievalHitRateScorer()]

    healthy_result = run_eval(
        _pipeline(
            corpus,
            retriever_name,
            generator_name,
            HEALTHY_K,
            model,
            base_url,
            api_key,
        ),
        tasks,
        scorers,
        "squad_v2",
    )
    starved_result = run_eval(
        _pipeline(
            corpus,
            retriever_name,
            generator_name,
            STARVED_K,
            model,
            base_url,
            api_key,
        ),
        tasks,
        scorers,
        "squad_v2",
    )

    healthy_run = save_run(healthy_result, runs_dir=runs_dir)
    starved_run = save_run(starved_result, runs_dir=runs_dir)

    print(f"Healthy run (k={HEALTHY_K}): {healthy_run}")
    print(f"Starved run (k={STARVED_K}): {starved_run}")
    print()
    _print_headline(healthy_result, starved_result)
    print()
    compare_runs(healthy_run, starved_run, threshold=threshold)
    return healthy_run, starved_run


def _pipeline(
    corpus,
    retriever_name: str,
    generator_name: str,
    k: int,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
) -> RAGPipeline:
    return RAGPipeline(
        build_retriever(retriever_name),
        build_generator(
            generator_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
        ),
        corpus=corpus,
        k=k,
        name=_system_label(generator_name, retriever_name, k, model),
    )


def _system_label(
    generator_name: str,
    retriever_name: str,
    k: int,
    model: str | None,
) -> str:
    if generator_name == "mock":
        return f"mock-{retriever_name}-k{k}"

    model_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", (model or "model").strip()).strip("-")
    prefix = "lmstudio" if generator_name == "local" else "api"
    return f"{prefix}-{model_label or 'model'}-k{k}"


def _print_headline(healthy_result, starved_result) -> None:
    healthy = healthy_result["aggregates"]
    starved = starved_result["aggregates"]

    print("Headline")
    for metric in ("retrieval_hit", "groundedness", "f1"):
        before = healthy.get(metric, 0.0)
        after = starved.get(metric, 0.0)
        print(f"  {metric}: k={HEALTHY_K} {before:.3f} -> k={STARVED_K} {after:.3f}")
