from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from agent_eval.scorers import Scorer
from agent_eval.task import Task

SystemFn = Callable[[str], str]


def run_eval(
    system_fn: SystemFn, tasks: Sequence[Task], scorers: Sequence[Scorer]
) -> dict[str, Any]:
    """Run tasks through a system and score each output."""

    task_results: list[dict[str, Any]] = []
    totals = {scorer.name: 0.0 for scorer in scorers}

    for task in tasks:
        output = system_fn(task.input)
        scores = {scorer.name: scorer.score(task, output) for scorer in scorers}

        for scorer_name, score in scores.items():
            totals[scorer_name] += score

        task_results.append(
            {
                "id": task.id,
                "input": task.input,
                "output": output,
                "reference": task.reference,
                "metadata": task.metadata,
                "scores": scores,
            }
        )

    task_count = len(tasks)
    aggregates = {
        scorer_name: (total / task_count if task_count else 0.0)
        for scorer_name, total in totals.items()
    }

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "system": getattr(system_fn, "__name__", "system_fn"),
        "task_count": task_count,
        "tasks": task_results,
        "aggregates": aggregates,
    }


def save_run(result: dict[str, Any], runs_dir: str | Path = "runs") -> Path:
    """Persist an eval result as a timestamped JSON file."""

    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = _available_run_path(runs_path, timestamp)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


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
