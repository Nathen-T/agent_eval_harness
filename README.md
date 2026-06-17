# rag-eval

Shrinking retrieved context from `k=5` to `k=1` dropped retrieval hit-rate from **88% to 72%** on a 50-question SQuAD v2 set (235-passage corpus) - and groundedness and F1 fell with it - all flagged automatically by the harness.

How do you know a RAG change did not start hallucinating or retrieving worse?

`rag-eval` is a small, offline-first evaluation harness for retrieval-augmented question answering. It loads SQuAD v2, pools its paragraphs into a single retrieval corpus (most of which are distractors), runs a real BM25/TF-IDF retriever and a deterministic mock generator, scores the output, saves each run, and compares runs so retrieval or hallucination regressions are visible before they ship.

It runs fully offline on first clone with a committed real SQuAD v2 sample and corpus plus a deterministic mock generator. Adapter stubs are included for real embeddings and a real LLM later.

## Architecture

The system under test is a real RAG loop:

- `load_squad_v2()` downloads `validation[:2000]`, pools its unique paragraphs into a ~235-doc corpus, selects 50 answerable questions as the test set, caches to `data/squad_subset.jsonl`, and falls back to the committed `data/squad_sample.jsonl` + `data/squad_corpus.jsonl` offline.
- Each question's original paragraph is its `gold_doc_id`; every other paragraph is a distractor, so retrieval is a real search problem.
- `retrieve(question, corpus, k) -> docs` searches that corpus with BM25 or TF-IDF.
- `generate(question, docs) -> answer` only sees retrieved docs. It extracts a short span from the most relevant retrieved sentence, or abstains when nothing is relevant - so answers degrade when the gold paragraph is not retrieved.

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

On first run, this tries to download `validation[:2000]` from HuggingFace `rajpurkar/squad_v2`, pools the paragraphs into a corpus, and caches to `data/squad_subset.jsonl`. If download is unavailable, it falls back to the committed `data/squad_sample.jsonl` and `data/squad_corpus.jsonl`.

Compare the two latest runs:

```bash
uv run python -m rag_eval compare
```

Run the built-in regression demo:

```bash
uv run python -m rag_eval demo
```

The demo runs the same SQuAD v2 set twice with the same retriever, changing only `k`: healthy retrieval (`k=5`) versus starved retrieval (`k=1`), then compares the saved runs.

Run tests:

```bash
uv run pytest -q
```

## Example Output

Running the RAG eval (`--k 5`):

```text
Saved run: runs/run-20260617-035703.json
System: mock-bm25-k5
Testset: squad_v2
Tasks: 50

Aggregates
  em: 0.100
  f1: 0.137
  groundedness: 0.810
  retrieval_hit: 0.880
```

Comparing healthy `k=5` retrieval to starved `k=1` retrieval:

```text
Threshold: -0.010

       metric run_a run_b  delta     status
           em 0.100 0.100 +0.000
           f1 0.137 0.123 -0.013 REGRESSION
 groundedness 0.810 0.790 -0.020 REGRESSION
retrieval_hit 0.880 0.720 -0.160 REGRESSION

Regressions detected: f1, groundedness, retrieval_hit
```

Retrieval hit-rate is the headline signal: it moves the most because it directly measures whether the gold paragraph survived the smaller `k`. Absolute `em`/`f1` are low by design - the mock generator is a deterministic, no-ML span extractor, not an LLM - but they still degrade with worse retrieval, which is the point.

## Test Sets

`data/squad_sample.jsonl` (50 answerable questions) and `data/squad_corpus.jsonl` (235 pooled paragraphs) are committed real SQuAD v2 data, so the harness runs offline on first clone. `data/squad_subset.jsonl` is the local cache written after a successful HuggingFace download and is intentionally gitignored. Unanswerable SQuAD v2 questions are excluded from the test set (so `em`/`f1` stay meaningful) but their paragraphs remain in the corpus as distractors.

## Extending It

Replace the mock pieces in `src/rag_eval/pipeline.py`:

- `EmbeddingRetriever` is the stub for real embeddings and vector search.
- `LLMGenerator` is the stub for a provider-backed model call.

Add new scorers in `src/rag_eval/scorers.py`. Keep every metric normalized to `[0.0, 1.0]` so aggregates and comparisons remain simple.

## How This Helps An Early-Stage Team

RAG systems regress in subtle ways: a prompt change can sound better while citing unsupported facts, a retriever tweak can miss the answer passage, or a model swap can become less extractive and more speculative.

This harness gives a small team a cheap local or CI check. Start with a handful of representative docs and questions, compare every prompt/model/retriever change against the last known-good run, and expand the test set whenever a real user failure appears.
