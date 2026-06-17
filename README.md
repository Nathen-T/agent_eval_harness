# rag-eval

Cutting retrieved context from `k=5` to `k=1` raised the hallucination rate from X% to Y% on a 50-question SQuAD v2 set - caught automatically by the harness.

How do you know a RAG change did not start hallucinating or retrieving worse?

`rag-eval` is a small, offline-first evaluation harness for retrieval-augmented question answering. It loads a fixed SQuAD v2 subset, pools all gold paragraphs into a tiny retrieval corpus, runs a deterministic retriever and mock generator, scores the output, saves each run, and compares runs so retrieval or hallucination regressions are visible before they ship.

It runs fully offline on first clone with a committed sample corpus and deterministic mock generator. Adapter stubs are included for real embeddings and a real LLM later.

## Architecture

The system under test is a real RAG loop:

- `load_squad_v2_subset()` gets `validation[:50]`, caches it to `data/squad_subset.jsonl`, and falls back to `data/squad_sample.jsonl` offline.
- `build_corpus(tasks)` pools every task context into a de-duped in-memory corpus with stable `squad-doc-*` ids.
- `retrieve(question, corpus, k) -> docs` searches that corpus with BM25 or TF-IDF.
- `generate(question, docs) -> answer` only sees retrieved docs, never the task's gold paragraph directly.

The harness wraps that pipeline with scorers:

- `AnswerCorrectnessScorer`: normalized exact match plus token-level F1, SQuAD-style.
- `GroundednessScorer`: lexical support heuristic that checks whether answer content appears in retrieved docs.
- `RetrievalHitRateScorer`: true hit@k, checking whether the gold paragraph id appears in the retrieved top-k.

Runs are persisted as timestamped JSON files in `runs/`, then compared with a metric-delta table. Any aggregate score that drops by more than the threshold is automatically flagged as a regression.

## Quickstart

Recommended `uv` workflow:

```bash
uv sync --dev
uv run python -m rag_eval demo
uv run pytest -q
```

Windows PowerShell is the same:

```powershell
uv sync --dev
uv run python -m rag_eval demo
uv run pytest -q
```

If you prefer plain `venv` and `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

For a visual walkthrough of the repository, open `docs/repo_overview.html` in a browser. It includes flowcharts for the RAG eval loop, module map, and regression comparison flow.

Run the offline RAG eval:

```bash
uv run python -m rag_eval run --k 5 --retriever bm25
```

Try the TF-IDF retriever:

```bash
uv run python -m rag_eval run --k 5 --retriever tfidf
```

On first run, this tries to download `validation[:50]` from HuggingFace `squad_v2` and caches it to `data/squad_subset.jsonl`. If download is unavailable, it falls back to the committed `data/squad_sample.jsonl` and `data/squad_corpus.jsonl`.

Compare the two latest runs:

```bash
uv run python -m rag_eval compare
```

Run the built-in regression demo:

```bash
uv run python -m rag_eval demo
```

The demo runs the same SQuAD v2 set twice: healthy BM25 retrieval (`k=5`) and a deliberately weakened fixed-order retriever (`k=1`), then compares the saved runs.

Run tests:

```bash
uv run pytest -q
```

## Example Output

Running the RAG eval:

```text
Saved run: runs/run-20260617-131900.json
System: mock-bm25-k5
Testset: squad_v2
Tasks: 50

Aggregates
  em: 0.120
  f1: 0.284
  groundedness: 0.820
  retrieval_hit: 1.000
```

Comparing healthy `k=5` retrieval to deliberately weakened `k=1` retrieval:

```text
Baseline:  runs/run-20260617-131900.json
Candidate: runs/run-20260617-131901.json
Threshold: -0.010

        metric run_a run_b  delta     status
            em 0.120 0.060 -0.060 REGRESSION
            f1 0.284 0.140 -0.144 REGRESSION
  groundedness 0.820 0.510 -0.310 REGRESSION
 retrieval_hit 1.000 0.620 -0.380 REGRESSION

Regressions detected: em, f1, groundedness, retrieval_hit
```

## Test Sets

`data/squad_sample.jsonl` is a committed squad_v2-style fallback sample. `data/squad_corpus.jsonl` is the pooled, de-duped retrieval corpus for that sample. `data/squad_subset.jsonl` is generated locally after a successful HuggingFace download and is intentionally gitignored.

## Extending It

Replace the mock pieces in `src/rag_eval/pipeline.py`:

- `EmbeddingRetriever` is the stub for real embeddings and vector search.
- `LLMGenerator` is the stub for a provider-backed model call.

Add new scorers in `src/rag_eval/scorers.py`. Keep every metric normalized to `[0.0, 1.0]` so aggregates and comparisons remain simple.

## How This Helps An Early-Stage Team

RAG systems regress in subtle ways: a prompt change can sound better while citing unsupported facts, a retriever tweak can miss the answer passage, or a model swap can become less extractive and more speculative.

This harness gives a small team a cheap local or CI check. Start with a handful of representative docs and questions, compare every prompt/model/retriever change against the last known-good run, and expand the test set whenever a real user failure appears.
