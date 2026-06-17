from __future__ import annotations

import json
from pathlib import Path

from rag_eval.compare import compare_runs
from rag_eval.data import build_corpus
from rag_eval.pipeline import BM25Retriever, MockGenerator, RAGPipeline
from rag_eval.runner import run_eval, save_run
from rag_eval.scorers import (
    AnswerCorrectnessScorer,
    GroundednessScorer,
    RetrievalHitRateScorer,
    exact_match,
    token_f1,
)
from rag_eval.task import Doc, Task, load_tasks


def test_mock_rag_eval_runs_end_to_end() -> None:
    tasks = load_tasks(Path("data/squad_sample.jsonl"))
    corpus, _ = build_corpus(tasks)
    pipeline = _pipeline(corpus, k=5)
    scorers = _scorers()

    result = run_eval(pipeline, tasks, scorers, testset_name="squad_v2")

    assert result["task_count"] == len(tasks) == 12
    assert len(result["tasks"]) == len(tasks)
    assert set(result["aggregates"]) == {"em", "f1", "groundedness", "retrieval_hit"}

    for task_result in result["tasks"]:
        assert task_result["gold_doc_id"]
        assert task_result["retrieved_doc_ids"]
        for score in task_result["scores"].values():
            assert 0.0 <= score <= 1.0


def test_compare_runs_flags_retrieval_regression(tmp_path: Path) -> None:
    tasks = [
        Task(
            id="q1",
            question="What result did the shared report announce?",
            context="The shared report announced alpha as the result.",
            reference_answer="alpha",
            gold_doc_id="doc-a",
        ),
        Task(
            id="q2",
            question="What result did the shared report announce?",
            context="The shared report announced beta as the result.",
            reference_answer="beta",
            gold_doc_id="doc-b",
        ),
        Task(
            id="q3",
            question="What result did the shared report announce?",
            context="The shared report announced gamma as the result.",
            reference_answer="gamma",
            gold_doc_id="doc-c",
        ),
    ]
    corpus = [
        Doc(id="doc-a", text=tasks[0].context),
        Doc(id="doc-b", text=tasks[1].context),
        Doc(id="doc-c", text=tasks[2].context),
    ]
    scorers = _scorers()

    good_result = run_eval(_pipeline(corpus, k=3), tasks, scorers, "squad_v2")
    bad_result = run_eval(_pipeline(corpus, k=1), tasks, scorers, "squad_v2")

    good_run = save_run(good_result, tmp_path)
    bad_run = save_run(bad_result, tmp_path)

    assert compare_runs(good_run, good_run) == []

    regressions = compare_runs(good_run, bad_run, threshold=0.01)
    assert "retrieval_hit" in regressions


def test_compare_runs_flags_synthetic_aggregate_drop(tmp_path: Path) -> None:
    good_run = tmp_path / "good.json"
    bad_run = tmp_path / "bad.json"
    good_run.write_text(
        json.dumps({"aggregates": {"groundedness": 0.9, "retrieval_hit": 0.8}}),
        encoding="utf-8",
    )
    bad_run.write_text(
        json.dumps({"aggregates": {"groundedness": 0.4, "retrieval_hit": 0.8}}),
        encoding="utf-8",
    )

    assert compare_runs(good_run, bad_run, threshold=0.01) == ["groundedness"]


def test_bm25_retrieves_lexical_match_at_top_one() -> None:
    corpus = [
        Doc(id="doc-a", text="The library stores cedar maps."),
        Doc(id="doc-b", text="The observatory records comet sketches."),
    ]

    retrieved = BM25Retriever().retrieve("Where are comet sketches recorded?", corpus, k=1)

    assert [doc.id for doc in retrieved] == ["doc-b"]


def test_retrieval_hit_rate_uses_gold_doc_id() -> None:
    task = Task(
        id="q1",
        question="Who calibrated the primary lens?",
        context="Priya Shah calibrated the primary lens.",
        reference_answer="Priya Shah",
        gold_doc_id="doc-gold",
    )
    scorer = RetrievalHitRateScorer()

    assert scorer.score(task, "", [Doc(id="doc-gold", text=task.context)]) == {
        "retrieval_hit": 1.0
    }
    assert scorer.score(task, "", [Doc(id="doc-other", text=task.context)]) == {
        "retrieval_hit": 0.0
    }


def test_answer_correctness_and_groundedness_are_explainable() -> None:
    task = Task(
        id="q1",
        question="What color was recorded?",
        context="The log recorded blue as the final color.",
        reference_answer="blue",
        gold_doc_id="doc-1",
    )

    assert exact_match("Blue.", task.reference_answer) == 1.0
    assert token_f1("The answer is blue", task.reference_answer) > 0.0
    assert GroundednessScorer().score(
        task,
        "synthetic hallucination",
        [Doc(id="doc-1", text=task.context)],
    ) == {"groundedness": 0.0}


def _pipeline(corpus, k: int) -> RAGPipeline:
    return RAGPipeline(
        BM25Retriever(),
        MockGenerator(),
        corpus=corpus,
        k=k,
        name=f"test-bm25-k{k}",
    )


def _scorers():
    return [AnswerCorrectnessScorer(), GroundednessScorer(), RetrievalHitRateScorer()]
