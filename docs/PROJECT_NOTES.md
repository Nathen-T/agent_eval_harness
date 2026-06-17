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

- `src/rag_eval/pipeline.py` — `RAGPipeline`, `BM25Retriever`, `TfidfRetriever`, demo-only `FixedOrderRetriever`, `MockGenerator`, and stubs (`EmbeddingRetriever`, `LLMGenerator`).
- `src/rag_eval/scorers.py` — `Scorer` interface, the 3 scorers, and `LLMJudgeScorer` stub.
- `src/rag_eval/runner.py` — `run_eval()` + `save_run()`.
- `src/rag_eval/compare.py` — `compare_runs()` with regression threshold.
- `src/rag_eval/data.py` — loads SQuAD v2 data, assigns `gold_doc_id`, and builds the pooled corpus.
- `src/rag_eval/cli.py` — `run` / `compare` / `demo` commands.
- `docs/repo_overview.html` — visual overview with flowcharts.

## Data / test sets

- **SQuAD v2** (`data/squad_subset.jsonl`): 50 validation examples downloaded from HuggingFace `squad_v2` and cached locally.
- **Offline fallback** (`data/squad_sample.jsonl` + `data/squad_corpus.jsonl`): committed sample tasks plus the pooled, de-duped corpus so the project runs on first clone with no network.
- Each task has a `gold_doc_id`; the generator never receives `task.context` directly.

## How to run

```powershell
uv sync --dev
uv run python -m rag_eval run --k 5 --retriever bm25
uv run python -m rag_eval run --k 1 --retriever bm25
uv run python -m rag_eval demo      # healthy BM25 k=5 vs weakened fixed-order k=1
uv run python -m rag_eval compare   # compares the two latest runs
uv run pytest -q
```

## How it differs from RAGAS / other tools

- RAGAS, TruLens, DeepEval, Phoenix are **metric libraries**, mostly **LLM-as-judge / embedding based**, non-deterministic, need an API key.
- `rag-eval` is a **regression harness**: offline, deterministic, run persistence + `compare_runs` gating, tiny and fully readable.
- Honest framing: not "better than RAGAS." It has fewer, simpler metrics. Its value is the end-to-end loop, regression gating, and the offline determinism. The `LLMJudgeScorer` stub is exactly where a RAGAS-style metric would plug in.

## Known limitations

- Groundedness is a lexical-overlap heuristic, not semantic entailment.
- The mock generator is deterministic and extractive/fallback-based, not a real LLM.
- The committed offline sample is tiny; the 50-row SQuAD v2 download gives a more realistic retrieval space.
- `EmbeddingRetriever` and `LLMGenerator` are stubs (no real provider wired in).

## Next steps (if I pick this back up)

1. Fill the README headline X/Y values after running the 50-question demo.
2. Add distractor passages beyond the pooled gold contexts so retrieval@k is harder.
3. Implement one real adapter: a local embedding retriever and/or an OpenAI-compatible generator (keep mock as default).
4. Implement `LLMJudgeScorer` for semantic faithfulness/correctness.
5. Wire `compare` into CI to fail a PR on regression.
