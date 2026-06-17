# rag-eval

With a small local model served through LM Studio, shrinking retrieved context from `k=5` to `k=1` moved EM from **`<PLACEHOLDER_K5_EM>` to `<PLACEHOLDER_K1_EM>`** and F1 from **`<PLACEHOLDER_K5_F1>` to `<PLACEHOLDER_K1_F1>`** on a 50-question SQuAD v2 set (235-passage corpus) - all flagged automatically by the harness.

How do you know a RAG change did not start hallucinating or retrieving worse?

`rag-eval` is a small, offline-first evaluation harness for retrieval-augmented question answering. It loads SQuAD v2, pools its paragraphs into a single retrieval corpus (most of which are distractors), runs a real BM25/TF-IDF retriever and either the default deterministic mock generator or a configured OpenAI-compatible generator, scores the output, saves each run, and compares runs so retrieval or hallucination regressions are visible before they ship.

It runs fully offline on first clone with a committed real SQuAD v2 sample and corpus plus a deterministic mock generator. A real OpenAI-compatible generator is included for LM Studio or hosted APIs when you opt in.

## Architecture

The repository is split into two packages so the evaluator and the thing being
evaluated stay separate:

- `src/rag_eval/` is the **harness** - the reusable evaluator. It defines the
  `Retriever`/`Generator` protocols, the `RAGPipeline` glue, the scorers, the
  runner, and the run comparison. It never imports a concrete RAG system, so you
  can delete `example_rag` and the harness still stands on its own.
- `src/example_rag/` is the **system under test** - one concrete, swappable RAG
  system (BM25/TF-IDF retrievers + a deterministic mock generator) plus the CLI
  that wires it to the SQuAD v2 benchmark. To evaluate your own system, implement
  the two protocols in your own package and point the runner at it.

The system under test is a real RAG loop:

- `load_squad_v2()` downloads `validation[:2000]`, pools its unique paragraphs into a ~235-doc corpus, selects 50 answerable questions as the test set, caches to `data/squad_subset.jsonl`, and falls back to the committed `data/squad_sample.jsonl` + `data/squad_corpus.jsonl` offline.
- Each question's original paragraph is its `gold_doc_id`; every other paragraph is a distractor, so retrieval is a real search problem.
- `retrieve(question, corpus, k) -> docs` searches that corpus with BM25 or TF-IDF.
- `generate(question, docs) -> answer` only sees retrieved docs. The default mock extracts a short span from the most relevant retrieved sentence, while the OpenAI-compatible generator prompts the model to answer only from retrieved context or abstain when unsupported.

The harness wraps that pipeline with scorers:

- `AnswerCorrectnessScorer`: normalized exact match plus token-level F1, SQuAD-style.
- `GroundednessScorer`: lexical support heuristic that checks whether answer content appears in retrieved docs.
- `RetrievalHitRateScorer`: true hit@k, checking whether the gold paragraph id appears in the retrieved top-k.

Runs are persisted as timestamped JSON files in `runs/`, then compared with a metric-delta table. Any aggregate score that drops by more than the threshold is automatically flagged as a regression.

## Quickstart

Recommended `uv` workflow:

```bash
uv sync --dev
uv run python -m example_rag demo
uv run pytest -q
```

Windows PowerShell is the same:

```powershell
uv sync --dev
uv run python -m example_rag demo
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

For a visual walkthrough of the repository, open `docs/repo_overview.html` in a browser. It includes flowcharts for the RAG eval loop, module map, and regression comparison flow. For a full write-up of the measured numbers and what they mean, see `docs/RESULTS.md`.

Run the offline RAG eval:

```bash
uv run python -m example_rag run --k 5 --retriever bm25
```

Run the same eval with a local LM Studio model:

```bash
uv run python -m example_rag run --k 5 --retriever bm25 --generator local --model <MY_MODEL>
```

Try the TF-IDF retriever:

```bash
uv run python -m example_rag run --k 5 --retriever tfidf
```

On first run, this tries to download `validation[:2000]` from HuggingFace `rajpurkar/squad_v2`, pools the paragraphs into a corpus, and caches to `data/squad_subset.jsonl`. If download is unavailable, it falls back to the committed `data/squad_sample.jsonl` and `data/squad_corpus.jsonl`.

Compare the two latest runs:

```bash
uv run python -m example_rag compare
```

Run the built-in regression demo:

```bash
uv run python -m example_rag demo
```

The demo runs the same SQuAD v2 set twice with the same retriever, changing only `k`: healthy retrieval (`k=5`) versus starved retrieval (`k=1`), then compares the saved runs.

Run tests:

```bash
uv run pytest -q
```

## Example Output

Running the RAG eval with a small local model via LM Studio (`--k 5`):

```text
Saved run: runs/run-<PLACEHOLDER_TIMESTAMP>.json
System: lmstudio-<MY_MODEL>-k5
Testset: squad_v2
Tasks: 50

Aggregates
  em: <PLACEHOLDER_K5_EM>
  f1: <PLACEHOLDER_K5_F1>
  groundedness: <PLACEHOLDER_K5_GROUNDEDNESS>
  retrieval_hit: <PLACEHOLDER_K5_RETRIEVAL_HIT>
```

Comparing healthy `k=5` retrieval to starved `k=1` retrieval:

```text
Threshold: -0.010

       metric run_a run_b  delta     status
           em <PLACEHOLDER_K5_EM> <PLACEHOLDER_K1_EM> <PLACEHOLDER_DELTA_EM> <PLACEHOLDER_STATUS>
           f1 <PLACEHOLDER_K5_F1> <PLACEHOLDER_K1_F1> <PLACEHOLDER_DELTA_F1> <PLACEHOLDER_STATUS>
 groundedness <PLACEHOLDER_K5_GROUNDEDNESS> <PLACEHOLDER_K1_GROUNDEDNESS> <PLACEHOLDER_DELTA_GROUNDEDNESS> <PLACEHOLDER_STATUS>
retrieval_hit <PLACEHOLDER_K5_RETRIEVAL_HIT> <PLACEHOLDER_K1_RETRIEVAL_HIT> <PLACEHOLDER_DELTA_RETRIEVAL_HIT> <PLACEHOLDER_STATUS>

Regressions detected: <PLACEHOLDER_REGRESSION_METRICS>
```

These placeholders should be filled from a small local model via LM Studio. Absolute scores may be modest; the regression deltas between comparable runs are the point.

## Test Sets

`data/squad_sample.jsonl` (50 answerable questions) and `data/squad_corpus.jsonl` (235 pooled paragraphs) are committed real SQuAD v2 data, so the harness runs offline on first clone. `data/squad_subset.jsonl` is the local cache written after a successful HuggingFace download and is intentionally gitignored. Unanswerable SQuAD v2 questions are excluded from the test set (so `em`/`f1` stay meaningful) but their paragraphs remain in the corpus as distractors.

## Extending It

The harness (`src/rag_eval/`) stays untouched; you swap the system in `src/example_rag/`, or add your own package next to it:

- Implement `rag_eval.protocols.Retriever` for real embeddings or vector search - `example_rag/retrievers.py` ships an `EmbeddingRetriever` stub to start from.
- Choose a generator tier:
  - `mock` (default): deterministic, offline, no server and no API key.
  - `local`: LM Studio's OpenAI-compatible local server. Download a model, load it in LM Studio, enable the local server, then run `uv run python -m example_rag run --generator local --model <MY_MODEL>`.
  - `api`: a hosted OpenAI-compatible endpoint. Set `OPENAI_API_KEY`, optionally set `OPENAI_BASE_URL` or pass `--base-url`, then run `uv run python -m example_rag run --generator api --model <MODEL_ID>`.
- Implement `rag_eval.protocols.Generator` for a different provider if needed. `example_rag/generators.py` contains the OpenAI-compatible `LLMGenerator`.
- Hand your retriever and generator to `RAGPipeline` and run them through the same `run_eval`/`compare_runs` flow.

Add new scorers in `src/rag_eval/scorers.py`. Keep every metric normalized to `[0.0, 1.0]` so aggregates and comparisons remain simple.

## How This Helps An Early-Stage Team

RAG systems regress in subtle ways: a prompt change can sound better while citing unsupported facts, a retriever tweak can miss the answer passage, or a model swap can become less extractive and more speculative.

This harness gives a small team a cheap local or CI check. Start with a handful of representative docs and questions, compare every prompt/model/retriever change against the last known-good run, and expand the test set whenever a real user failure appears.
