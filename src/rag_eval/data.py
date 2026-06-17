from __future__ import annotations

import json
from pathlib import Path
import sys

from rag_eval.task import Doc, Task, load_tasks, write_tasks

DEFAULT_DATA_DIR = Path("data")
DEFAULT_SQUAD_CACHE = DEFAULT_DATA_DIR / "squad_subset.jsonl"
DEFAULT_SQUAD_SAMPLE = DEFAULT_DATA_DIR / "squad_sample.jsonl"
DEFAULT_SQUAD_CORPUS = DEFAULT_DATA_DIR / "squad_corpus.jsonl"

# The committed sample/corpus are generated with these defaults, so re-running
# online with the defaults reproduces byte-identical artifacts (no git churn).
DEFAULT_SLICE = 2000
DEFAULT_MAX_QUESTIONS = 50


def load_squad_v2(
    n: int = DEFAULT_SLICE,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    cache_path: str | Path = DEFAULT_SQUAD_CACHE,
    sample_path: str | Path = DEFAULT_SQUAD_SAMPLE,
    corpus_path: str | Path = DEFAULT_SQUAD_CORPUS,
) -> tuple[list[Task], list[Doc]]:
    """Load SQuAD v2 answerable tasks plus a pooled retrieval corpus.

    The corpus pools every unique paragraph in the downloaded slice, so most
    documents are distractors that do not answer any given question - that is
    what makes retrieval a real task. Falls back to the committed sample and
    corpus so the project runs fully offline on first clone.
    """

    cache_path = Path(cache_path)
    sample_path = Path(sample_path)
    corpus_path = Path(corpus_path)

    if cache_path.exists() and corpus_path.exists():
        return load_tasks(cache_path), load_corpus_docs_jsonl(corpus_path)

    try:
        raw_examples = _download_squad_v2_examples(n)
        tasks, corpus = _build_dataset(raw_examples, max_questions)
        write_tasks(cache_path, tasks)
        write_corpus_docs(corpus_path, corpus)
        return tasks, corpus
    except Exception as exc:  # pragma: no cover - exercised in offline environments
        print(
            f"Could not download SQuAD v2 ({exc}). Falling back to {sample_path}.",
            file=sys.stderr,
        )
        return load_tasks(sample_path), load_corpus_docs_jsonl(corpus_path)


def _download_squad_v2_examples(n: int):
    from datasets import load_dataset

    errors: list[str] = []
    for dataset_id in ("rajpurkar/squad_v2", "squad_v2"):
        try:
            return load_dataset(dataset_id, split=f"validation[:{n}]")
        except Exception as exc:
            errors.append(f"{dataset_id}: {exc}")

    raise RuntimeError("; ".join(errors))


def _build_dataset(raw_examples, max_questions: int) -> tuple[list[Task], list[Doc]]:
    corpus, doc_id_by_context = build_corpus_from_contexts(
        str(example["context"]) for example in raw_examples
    )

    tasks: list[Task] = []
    used_gold_docs: set[str] = set()
    for example in raw_examples:
        answers = example.get("answers", {}).get("text", [])
        if not answers:
            continue  # skip unanswerable rows; keep correctness metrics meaningful

        context = str(example["context"])
        gold_doc_id = doc_id_by_context[context]
        if gold_doc_id in used_gold_docs:
            continue  # spread the test set across distinct gold paragraphs

        used_gold_docs.add(gold_doc_id)
        tasks.append(
            Task(
                id=str(example["id"]),
                question=str(example["question"]),
                context=context,
                reference_answer=str(answers[0]),
                gold_doc_id=gold_doc_id,
                metadata={"source": "squad_v2", "is_impossible": False},
            )
        )
        if len(tasks) >= max_questions:
            break

    return tasks, corpus


def build_corpus_from_contexts(contexts) -> tuple[list[Doc], dict[str, str]]:
    """Pool raw paragraph strings into a de-duped retrieval corpus."""

    docs: list[Doc] = []
    doc_id_by_context: dict[str, str] = {}
    for context in contexts:
        if context in doc_id_by_context:
            continue
        doc_id = f"squad-doc-{len(docs) + 1:04d}"
        doc_id_by_context[context] = doc_id
        docs.append(Doc(id=doc_id, text=context, metadata={"source": "squad_v2"}))
    return docs, doc_id_by_context


def write_corpus_docs(path: str | Path, docs: list[Doc]) -> Path:
    """Write pooled retrieval docs as JSONL for a reviewable corpus artifact."""

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
