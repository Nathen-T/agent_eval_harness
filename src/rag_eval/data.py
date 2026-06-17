from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

from rag_eval.task import Doc, Task, load_tasks, write_tasks

DEFAULT_DATA_DIR = Path("data")
DEFAULT_SQUAD_CACHE = DEFAULT_DATA_DIR / "squad_subset.jsonl"
DEFAULT_SQUAD_SAMPLE = DEFAULT_DATA_DIR / "squad_sample.jsonl"
DEFAULT_SQUAD_CORPUS = DEFAULT_DATA_DIR / "squad_corpus.jsonl"


def load_squad_v2_subset(
    n: int = 50,
    cache_path: str | Path = DEFAULT_SQUAD_CACHE,
    sample_path: str | Path = DEFAULT_SQUAD_SAMPLE,
    corpus_path: str | Path = DEFAULT_SQUAD_CORPUS,
) -> list[Task]:
    """Load a cached SQuAD v2 subset, downloading once when possible.

    If HuggingFace datasets or the network is unavailable, this falls back to
    the committed sample file so the project still runs fully offline.
    """

    cache_path = Path(cache_path)
    sample_path = Path(sample_path)
    corpus_path = Path(corpus_path)

    if cache_path.exists():
        tasks = load_tasks(cache_path)
        return _finalize_tasks(tasks, cache_path, corpus_path)

    try:
        raw_examples = _download_squad_v2_examples(n)
        tasks = [_task_from_squad_example(example) for example in raw_examples]
        return _finalize_tasks(tasks, cache_path, corpus_path)
    except Exception as exc:  # pragma: no cover - exercised in offline environments
        print(
            f"Could not download SQuAD v2 ({exc}). Falling back to {sample_path}.",
            file=sys.stderr,
        )
        tasks = load_tasks(sample_path)
        return _finalize_tasks(tasks, None, corpus_path)


def load_squad_subset(*args, **kwargs) -> list[Task]:
    """Backward-compatible alias for the squad_v2 loader."""

    return load_squad_v2_subset(*args, **kwargs)


def _download_squad_v2_examples(n: int):
    from datasets import load_dataset

    errors: list[str] = []
    for dataset_id in ("rajpurkar/squad_v2", "squad_v2"):
        try:
            return load_dataset(dataset_id, split=f"validation[:{n}]")
        except Exception as exc:
            errors.append(f"{dataset_id}: {exc}")

    raise RuntimeError("; ".join(errors))


def _task_from_squad_example(example) -> Task:
    answers = example.get("answers", {}).get("text", [])
    reference_answer = answers[0] if answers else ""
    is_impossible = bool(example.get("is_impossible", not reference_answer))
    return Task(
        id=str(example["id"]),
        question=str(example["question"]),
        context=str(example["context"]),
        reference_answer=str(reference_answer),
        metadata={
            "source": "squad_v2",
            "cached_from": "validation",
            "is_impossible": is_impossible,
        },
    )


def build_corpus(tasks: list[Task]) -> tuple[list[Doc], dict[str, str]]:
    """Pool task contexts into a de-duped retrieval corpus."""

    docs: list[Doc] = []
    doc_id_by_context: dict[str, str] = {}
    for task in tasks:
        if task.context in doc_id_by_context:
            continue

        doc_id = task.gold_doc_id or f"squad-doc-{len(docs) + 1:04d}"
        doc_id_by_context[task.context] = doc_id
        docs.append(
            Doc(
                id=doc_id,
                text=task.context,
                metadata={"source": task.metadata.get("source", "squad_v2")},
            )
        )
    return docs, doc_id_by_context


def assign_gold_doc_ids(tasks: list[Task], doc_id_by_context: dict[str, str]) -> list[Task]:
    """Attach each task's original paragraph id without passing gold text to generation."""

    return [
        task
        if task.gold_doc_id
        else replace(task, gold_doc_id=doc_id_by_context[task.context])
        for task in tasks
    ]


def write_corpus_docs(path: str | Path, docs: list[Doc]) -> Path:
    """Write pooled retrieval docs as JSONL for reviewable local artifacts."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"id": doc.id, "text": doc.text, "metadata": doc.metadata},
            sort_keys=True,
        )
        for doc in docs
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_corpus_docs_jsonl(path: str | Path = DEFAULT_SQUAD_CORPUS) -> list[Doc]:
    """Load a persisted JSONL corpus artifact."""

    docs: list[Doc] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                docs.append(
                    Doc(
                        id=record["id"],
                        text=record["text"],
                        metadata=record.get("metadata", {}),
                    )
                )
            except KeyError as exc:
                missing = exc.args[0]
                raise ValueError(
                    f"{path}:{line_number} is missing required field {missing!r}"
                ) from exc
    return docs


def _finalize_tasks(
    tasks: list[Task],
    cache_path: Path | None,
    corpus_path: Path,
) -> list[Task]:
    docs, doc_id_by_context = build_corpus(tasks)
    tasks = assign_gold_doc_ids(tasks, doc_id_by_context)

    if cache_path is not None:
        write_tasks(cache_path, tasks)
    if not corpus_path.exists() or cache_path is not None:
        write_corpus_docs(corpus_path, docs)

    return tasks
