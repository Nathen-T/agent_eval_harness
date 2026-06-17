from __future__ import annotations

from pathlib import Path
import sys

from rag_eval.task import Doc, Task, load_tasks, write_tasks

DEFAULT_DATA_DIR = Path("data")
DEFAULT_CORPUS_DIR = DEFAULT_DATA_DIR / "corpus"
DEFAULT_SQUAD_CACHE = DEFAULT_DATA_DIR / "squad_subset.jsonl"
DEFAULT_SQUAD_SAMPLE = DEFAULT_DATA_DIR / "squad_sample.jsonl"
DEFAULT_CUSTOM_TESTSET = DEFAULT_DATA_DIR / "custom_testset.jsonl"


def load_squad_subset(
    n: int = 40,
    cache_path: str | Path = DEFAULT_SQUAD_CACHE,
    sample_path: str | Path = DEFAULT_SQUAD_SAMPLE,
) -> list[Task]:
    """Load a cached SQuAD subset, downloading once when possible.

    If HuggingFace datasets or the network is unavailable, this falls back to
    the committed sample file so the project still runs fully offline.
    """

    cache_path = Path(cache_path)
    sample_path = Path(sample_path)

    if cache_path.exists():
        return load_tasks(cache_path)

    try:
        from datasets import load_dataset

        raw_examples = load_dataset("squad", split=f"validation[:{n}]")
        tasks: list[Task] = []
        for example in raw_examples:
            answers = example.get("answers", {}).get("text", [])
            reference_answer = answers[0] if answers else ""
            tasks.append(
                Task(
                    id=str(example["id"]),
                    question=str(example["question"]),
                    context=str(example["context"]),
                    reference_answer=str(reference_answer),
                    metadata={"source": "squad", "cached_from": "validation"},
                )
            )

        write_tasks(cache_path, tasks)
        return tasks
    except Exception as exc:  # pragma: no cover - exercised in offline environments
        print(
            f"Could not download SQuAD ({exc}). Falling back to {sample_path}.",
            file=sys.stderr,
        )
        return load_tasks(sample_path)


def load_custom_testset(
    path: str | Path = DEFAULT_CUSTOM_TESTSET,
) -> list[Task]:
    return load_tasks(path)


def build_corpus(
    tasks: list[Task],
    corpus_dir: str | Path | None = None,
) -> list[Doc]:
    """Build retrieval docs from corpus files or task contexts."""

    if corpus_dir is not None:
        return load_corpus_docs(corpus_dir)

    docs_by_text: dict[str, Doc] = {}
    for task in tasks:
        doc_id = str(task.metadata.get("doc_id") or f"task-{task.id}")
        docs_by_text.setdefault(
            task.context,
            Doc(
                id=doc_id,
                text=task.context,
                metadata={"source_task_id": task.id},
            ),
        )
    return list(docs_by_text.values())


def load_corpus_docs(corpus_dir: str | Path = DEFAULT_CORPUS_DIR) -> list[Doc]:
    corpus_path = Path(corpus_dir)
    docs: list[Doc] = []

    for path in sorted([*corpus_path.glob("*.md"), *corpus_path.glob("*.txt")]):
        docs.append(
            Doc(
                id=path.stem,
                text=path.read_text(encoding="utf-8"),
                metadata={"path": str(path)},
            )
        )

    if not docs:
        raise ValueError(f"No .md or .txt corpus files found in {corpus_path}")

    return docs
