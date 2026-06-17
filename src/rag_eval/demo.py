from __future__ import annotations

from pathlib import Path

from rag_eval.compare import compare_runs
from rag_eval.data import build_corpus, load_squad_v2_subset
from rag_eval.pipeline import FixedOrderRetriever, MockGenerator, RAGPipeline, build_retriever
from rag_eval.runner import run_eval, save_run
from rag_eval.scorers import (
    AnswerCorrectnessScorer,
    GroundednessScorer,
    RetrievalHitRateScorer,
)


def run_demo(
    runs_dir: str | Path = "runs",
    retriever_name: str = "bm25",
    threshold: float = 0.01,
) -> tuple[Path, Path]:
    """Run healthy and starved retrieval configs, then compare them."""

    tasks = load_squad_v2_subset()
    corpus, _ = build_corpus(tasks)
    scorers = [AnswerCorrectnessScorer(), GroundednessScorer(), RetrievalHitRateScorer()]

    healthy_pipeline = RAGPipeline(
        build_retriever(retriever_name),
        MockGenerator(),
        corpus=corpus,
        k=5,
        name=f"mock-{retriever_name}-k5",
    )
    weakened_pipeline = RAGPipeline(
        FixedOrderRetriever(),
        MockGenerator(),
        corpus=corpus,
        k=1,
        name="mock-fixed-order-k1",
    )

    healthy_result = run_eval(healthy_pipeline, tasks, scorers, testset_name="squad_v2")
    weakened_result = run_eval(weakened_pipeline, tasks, scorers, testset_name="squad_v2")

    healthy_run = save_run(
        healthy_result,
        runs_dir=runs_dir,
    )
    weakened_run = save_run(
        weakened_result,
        runs_dir=runs_dir,
    )

    print(f"Healthy run (k=5): {healthy_run}")
    print(f"Weakened run (fixed-order k=1): {weakened_run}")
    print()
    _print_headline(healthy_result, weakened_result)
    print()
    compare_runs(healthy_run, weakened_run, threshold=threshold)
    return healthy_run, weakened_run


def _print_headline(healthy_result, starved_result) -> None:
    healthy_groundedness = healthy_result["aggregates"].get("groundedness", 0.0)
    starved_groundedness = starved_result["aggregates"].get("groundedness", 0.0)
    healthy_hit = healthy_result["aggregates"].get("retrieval_hit", 0.0)
    starved_hit = starved_result["aggregates"].get("retrieval_hit", 0.0)

    print("Headline")
    print(
        "  Hallucination proxy "
        f"(1 - groundedness): k=5 {(1 - healthy_groundedness):.3f} "
        f"-> k=1 {(1 - starved_groundedness):.3f}"
    )
    print(f"  Retrieval hit-rate: k=5 {healthy_hit:.3f} -> k=1 {starved_hit:.3f}")
