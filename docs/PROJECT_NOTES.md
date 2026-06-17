# rag-eval — Project Notes

A personal reference for what this project is, why it exists, and how to talk about it.

## One-line purpose

`rag-eval` is a small, offline-first evaluation harness for a RAG (retrieval-augmented generation) question-answering system that catches regressions across retriever, context-window, prompt, and model changes.

The core question it answers: **"How do I know a RAG change didn't start hallucinating or retrieving worse?"**

## Why I built it

- To understand the RAG evaluation loop end-to-end by building it from scratch, not just `pip install`-ing a framework.
- To have a portfolio project that demonstrates RAG *quality infrastructure*, not just a RAG app.
- To show I can reason about retrieval quality, groundedness/hallucination, and regression gating with a real retrieval step.

## What it does (the loop)

```
question
  -> BM25/TF-IDF retrieves top-k docs from pooled SQuAD v2 corpus
  -> mock generator writes an answer from retrieved docs only
  -> scorers grade the answer + retrieved context
  -> run saved as runs/*.json
  -> compare_runs flags any metric drop
```

The system under test is pluggable: anything with `retrieve(question, corpus, k) -> docs` and `generate(question, docs) -> answer`.

## Metrics

- `em` — exact match vs reference answer (SQuAD-style normalization).
- `f1` — token-overlap F1 vs reference answer.
- `groundedness` — fraction of answer tokens supported by retrieved context (hallucination signal).
- `retrieval_hit` — true hit@k: whether the retrieved doc ids include the task's `gold_doc_id`.

All metrics are normalized to `[0.0, 1.0]` so aggregation and run-to-run deltas stay uniform.

## Key files

- `src/rag_eval/pipeline.py` — `RAGPipeline`, `BM25Retriever`, `TfidfRetriever`, `MockGenerator` (extractive span + abstention), and stubs (`EmbeddingRetriever`, `LLMGenerator`).
- `src/rag_eval/scorers.py` — `Scorer` interface, the 3 scorers, and `LLMJudgeScorer` stub.
- `src/rag_eval/runner.py` — `run_eval()` + `save_run()`.
- `src/rag_eval/compare.py` — `compare_runs()` with regression threshold.
- `src/rag_eval/data.py` — loads SQuAD v2 data, assigns `gold_doc_id`, and builds the pooled corpus.
- `src/rag_eval/cli.py` — `run` / `compare` / `demo` commands.
- `docs/repo_overview.html` — visual overview with flowcharts.

## Data / test sets

- **SQuAD v2** (`data/squad_subset.jsonl`): `validation[:2000]` downloaded from HuggingFace `rajpurkar/squad_v2`, pooled into ~235 unique paragraphs; cached locally (gitignored).
- **Test set**: 50 answerable questions, each tagged with its gold paragraph (`gold_doc_id`). The other ~185 paragraphs are distractors, so retrieval is a real search problem.
- **Offline fallback** (`data/squad_sample.jsonl` + `data/squad_corpus.jsonl`): committed real tasks + corpus so the project runs on first clone with no network.
- The generator only sees retrieved docs, never `task.context` directly.

## How to run

```powershell
uv sync --dev
uv run python -m rag_eval run --k 5 --retriever bm25
uv run python -m rag_eval run --k 1 --retriever bm25
uv run python -m rag_eval demo      # same eval at k=5 vs k=1 (only k changes)
uv run python -m rag_eval compare   # compares the two latest runs
uv run pytest -q
```

## Headline result (measured)

Cutting `k=5 -> k=1` (BM25, 50 questions, 235-passage corpus): retrieval_hit 0.88 -> 0.72, groundedness 0.81 -> 0.79, f1 0.137 -> 0.123. Retrieval hit-rate is the strongest signal.

## How it differs from RAGAS / other tools

- RAGAS, TruLens, DeepEval, Phoenix are **metric libraries**, mostly **LLM-as-judge / embedding based**, non-deterministic, need an API key.
- `rag-eval` is a **regression harness**: offline, deterministic, run persistence + `compare_runs` gating, tiny and fully readable.
- Honest framing: not "better than RAGAS." It has fewer, simpler metrics. Its value is the end-to-end loop, regression gating, and the offline determinism. The `LLMJudgeScorer` stub is exactly where a RAGAS-style metric would plug in.

## Known limitations

- Groundedness is a lexical-overlap heuristic, not semantic entailment; it mostly catches abstention/fabrication, not wrong-passage use (that is what retrieval_hit covers).
- The mock generator is a deterministic, no-ML span extractor, so absolute EM/F1 are low; they degrade with worse retrieval but are not LLM-quality.
- The corpus is pooled from `validation[:2000]` (7 articles); a larger slice would add more topical diversity.
- `EmbeddingRetriever` and `LLMGenerator` are stubs (no real provider wired in).

## Next steps (if I pick this back up)

1. Implement `LLMGenerator` so absolute EM/F1 reflect real answer quality (keep mock as default).
2. Pool a larger / more diverse SQuAD v2 slice for a harder corpus.
3. Implement `LLMJudgeScorer` for semantic faithfulness/correctness.
4. Wire `compare` into CI to fail a PR on regression.
