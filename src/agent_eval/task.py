from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    """One fixed input in an evaluation set."""

    id: str
    input: str
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_tasks(path: str | Path) -> list[Task]:
    """Load evaluation tasks from a JSONL file."""

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
                        input=record["input"],
                        reference=record.get("reference"),
                        metadata=record.get("metadata", {}),
                    )
                )
            except KeyError as exc:
                missing_key = exc.args[0]
                raise ValueError(
                    f"{path}:{line_number} is missing required field {missing_key!r}"
                ) from exc

    return tasks
