from __future__ import annotations

import json
from pathlib import Path

from agent_eval.compare import compare_runs
from agent_eval.runner import run_eval, save_run
from agent_eval.scorers import ExactMatchScorer, KeywordScorer
from agent_eval.systems import mock_system
from agent_eval.task import load_tasks


def test_mock_eval_runs_end_to_end(tmp_path: Path) -> None:
    testset = Path(__file__).parents[1] / "data" / "testset.jsonl"
    tasks = load_tasks(testset)
    scorers = [ExactMatchScorer(), KeywordScorer()]

    result = run_eval(mock_system, tasks, scorers)

    assert result["task_count"] == len(tasks) == 8
    assert len(result["tasks"]) == len(tasks)
    assert set(result["aggregates"]) == {"exact_match", "keyword_contains"}
    assert result["aggregates"]["exact_match"] >= 0.75

    for task_result in result["tasks"]:
        for score in task_result["scores"].values():
            assert 0.0 <= score <= 1.0

    first_run = save_run(result, tmp_path)
    second_run = save_run(result, tmp_path)

    assert first_run.exists()
    assert second_run.exists()
    assert first_run != second_run

    saved = json.loads(first_run.read_text(encoding="utf-8"))
    assert saved["task_count"] == 8
    assert compare_runs(first_run, second_run) == []
