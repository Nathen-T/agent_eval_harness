from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from rag_eval.pipeline import RAGPipeline
from rag_eval.scorers import Scorer
from rag_eval.task import Task


def run_eval(
    pipeline: RAGPipeline,
    tasks: Sequence[Task],
    scorers: Sequence[Scorer],
    testset_name: str = "unknown",
) -> dict[str, Any]:
    """Run a RAG pipeline through fixed tasks and aggregate scorer metrics."""

    task_results: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {}

    for task in tasks:
        retrieved_docs = pipeline.retrieve(task.question)
        answer = pipeline.generate(task.question, retrieved_docs)

        scores: dict[str, float] = {}
        for scorer in scorers:
            scores.update(scorer.score(task, answer, retrieved_docs))

        for metric, value in scores.items():
            metric_values.setdefault(metric, []).append(value)

        task_results.append(
            {
                "id": task.id,
                "question": task.question,
                "answer": answer,
                "reference_answer": task.reference_answer,
                "retrieved_doc_ids": [doc.id for doc in retrieved_docs],
                "scores": scores,
                "metadata": task.metadata,
            }
        )

    aggregates = {
        metric: float(np.mean(values)) if values else 0.0
        for metric, values in metric_values.items()
    }

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "system": pipeline.name,
        "testset": testset_name,
        "task_count": len(tasks),
        "tasks": task_results,
        "aggregates": aggregates,
    }


def save_run(result: dict[str, Any], runs_dir: str | Path = "runs") -> Path:
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = _available_run_path(runs_path, timestamp)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _available_run_path(runs_path: Path, timestamp: str) -> Path:
    candidate = runs_path / f"run-{timestamp}.json"
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = runs_path / f"run-{timestamp}-{suffix}.json"
        if not candidate.exists():
            return candidate
        suffix += 1
