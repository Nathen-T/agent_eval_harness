from __future__ import annotations

from pathlib import Path

from rag_eval.compare import compare_runs
from rag_eval.data import DEFAULT_CORPUS_DIR, build_corpus, load_custom_testset
from rag_eval.pipeline import MockGenerator, MockGeneratorConfig, MockRetriever, RAGPipeline
from rag_eval.runner import run_eval, save_run
from rag_eval.scorers import (
    AnswerCorrectnessScorer,
    GroundednessScorer,
    RetrievalHitRateScorer,
)


def run_demo(runs_dir: str | Path = "runs", threshold: float = 0.01) -> tuple[Path, Path]:
    """Run good and intentionally worse mock configs, then compare them."""

    tasks = load_custom_testset()
    corpus = build_corpus(tasks, corpus_dir=DEFAULT_CORPUS_DIR)
    scorers = [AnswerCorrectnessScorer(), GroundednessScorer(), RetrievalHitRateScorer()]

    good_pipeline = RAGPipeline(
        MockRetriever(corpus, k=3),
        MockGenerator(MockGeneratorConfig(use_context=True)),
        name="mock-good-k3",
    )
    bad_pipeline = RAGPipeline(
        MockRetriever(corpus, k=3),
        MockGenerator(MockGeneratorConfig(use_context=False)),
        name="mock-bad-k3",
    )

    good_run = save_run(
        run_eval(good_pipeline, tasks, scorers, testset_name="custom"),
        runs_dir=runs_dir,
    )
    bad_run = save_run(
        run_eval(bad_pipeline, tasks, scorers, testset_name="custom"),
        runs_dir=runs_dir,
    )

    print(f"Good run: {good_run}")
    print(f"Bad run:  {bad_run}")
    print()
    compare_runs(good_run, bad_run, threshold=threshold)
    return good_run, bad_run
