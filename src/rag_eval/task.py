from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Doc:
    """A retrieved document or passage used as RAG context."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    """One RAG QA evaluation example."""

    id: str
    question: str
    context: str
    reference_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_tasks(path: str | Path) -> list[Task]:
    """Load RAG QA tasks from a JSONL file."""

    tasks: list[Task] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            try:
                tasks.append(
                    Task(
                        id=record["id"],
                        question=record["question"],
                        context=record["context"],
                        reference_answer=record["reference_answer"],
                        metadata=record.get("metadata", {}),
                    )
                )
            except KeyError as exc:
                missing = exc.args[0]
                raise ValueError(
                    f"{path}:{line_number} is missing required field {missing!r}"
                ) from exc

    return tasks


def write_tasks(path: str | Path, tasks: list[Task]) -> Path:
    """Write tasks to JSONL for reproducible local caches."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": task.id,
                "question": task.question,
                "context": task.context,
                "reference_answer": task.reference_answer,
                "metadata": task.metadata,
            },
            sort_keys=True,
        )
        for task in tasks
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
