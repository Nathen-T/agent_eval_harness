from __future__ import annotations

from pathlib import Path

from rag_eval.compare import compare_runs
from rag_eval.data import build_corpus, load_custom_testset
from rag_eval.pipeline import MockGenerator, MockGeneratorConfig, MockRetriever, RAGPipeline
from rag_eval.runner import run_eval, save_run
from rag_eval.scorers import (
    AnswerCorrectnessScorer,
    GroundednessScorer,
    RetrievalHitRateScorer,
)


def test_mock_rag_eval_runs_end_to_end() -> None:
    tasks = load_custom_testset()
    corpus = build_corpus(tasks, corpus_dir=Path("data/corpus"))
    pipeline = _pipeline(corpus, use_context=True)
    scorers = _scorers()

    result = run_eval(pipeline, tasks, scorers, testset_name="custom")

    assert result["task_count"] == len(tasks) == 8
    assert len(result["tasks"]) == len(tasks)
    assert set(result["aggregates"]) == {"em", "f1", "groundedness", "retrieval_hit"}
    assert result["aggregates"]["em"] >= 0.9
    assert result["aggregates"]["f1"] >= 0.9
    assert result["aggregates"]["groundedness"] >= 0.9
    assert result["aggregates"]["retrieval_hit"] >= 0.9

    for task_result in result["tasks"]:
        assert task_result["retrieved_doc_ids"]
        for score in task_result["scores"].values():
            assert 0.0 <= score <= 1.0


def test_compare_runs_flags_mock_regressions(tmp_path: Path) -> None:
    tasks = load_custom_testset()
    corpus = build_corpus(tasks, corpus_dir=Path("data/corpus"))
    scorers = _scorers()

    good_result = run_eval(_pipeline(corpus, use_context=True), tasks, scorers, "custom")
    bad_result = run_eval(_pipeline(corpus, use_context=False), tasks, scorers, "custom")

    good_run = save_run(good_result, tmp_path)
    bad_run = save_run(bad_result, tmp_path)

    assert compare_runs(good_run, good_run) == []

    regressions = compare_runs(good_run, bad_run, threshold=0.01)
    assert "groundedness" in regressions
    assert "em" in regressions
    assert "f1" in regressions
    assert "retrieval_hit" not in regressions


def _pipeline(corpus, use_context: bool) -> RAGPipeline:
    return RAGPipeline(
        MockRetriever(corpus, k=3),
        MockGenerator(MockGeneratorConfig(use_context=use_context)),
        name=f"test-mock-{use_context}",
    )


def _scorers():
    return [AnswerCorrectnessScorer(), GroundednessScorer(), RetrievalHitRateScorer()]
