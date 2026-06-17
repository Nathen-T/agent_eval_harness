# rag-eval

How do you know a RAG change did not start hallucinating or retrieving worse?

`rag-eval` is a small, opinionated evaluation harness for retrieval-augmented question answering. It runs a fixed question set through a pluggable RAG pipeline, scores answer quality and groundedness, saves each run, and compares runs so regressions are visible before they ship.

It runs fully offline on first clone with a deterministic mock retriever and mock generator. Adapter stubs are included for a real embedding retriever and real LLM generator later.

## Architecture

The system under test has two pieces:

- `retrieve(question) -> docs`
- `generate(question, docs) -> answer`

The harness wraps that pipeline with scorers:

- `AnswerCorrectnessScorer`: normalized exact match plus token-level F1, SQuAD-style.
- `GroundednessScorer`: lexical support heuristic that checks whether answer content appears in retrieved context.
- `RetrievalHitRateScorer`: checks whether retrieved docs contain the reference answer.

Runs are persisted as timestamped JSON files in `runs/`, then compared with a metric-delta table. Any aggregate score that drops by more than the threshold is flagged as a regression.

## Quickstart

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

Run the offline custom RAG eval:

```bash
python -m rag_eval run --testset custom
```

Run the SQuAD subset eval:

```bash
python -m rag_eval run --testset squad
```

On first run, this tries to download `validation[:40]` from HuggingFace `datasets` and caches it to `data/squad_subset.jsonl`. If download is unavailable, it falls back to the committed `data/squad_sample.jsonl`.

Compare the two latest runs:

```bash
python -m rag_eval compare
```

Run the built-in regression demo:

```bash
python -m rag_eval demo
```

Run tests:

```bash
pytest -q
```

## Example Output

Running the custom eval:

```text
Saved run: runs/run-20260617-111900.json
System: mock-good-k3
Testset: custom
Tasks: 8

Aggregates
  em: 1.000
  f1: 1.000
  groundedness: 1.000
  retrieval_hit: 1.000
```

Comparing a good run to a worse mock generator that ignores context:

```text
Baseline:  runs/run-20260617-111900.json
Candidate: runs/run-20260617-111901.json
Threshold: -0.010

         metric run_a run_b  delta     status
             em 1.000 0.000 -1.000 REGRESSION
             f1 1.000 0.000 -1.000 REGRESSION
   groundedness 1.000 0.250 -0.750 REGRESSION
  retrieval_hit 1.000 1.000 +0.000

Regressions detected: em, f1, groundedness
```

## Test Sets

`data/custom_testset.jsonl` contains eight hand-written question/context/reference triples over a tiny placeholder corpus in `data/corpus/`. Swap these docs and examples for a target company's public docs when you are ready.

`data/squad_sample.jsonl` is a committed SQuAD-style fallback sample. `data/squad_subset.jsonl` is generated locally after a successful HuggingFace download and is intentionally gitignored.

## Extending It

Replace the mock pieces in `src/rag_eval/pipeline.py`:

- `EmbeddingRetriever` is the stub for real embeddings and vector search.
- `LLMGenerator` is the stub for a provider-backed model call.

Add new scorers in `src/rag_eval/scorers.py`. Keep every metric normalized to `[0.0, 1.0]` so aggregates and comparisons remain simple.

## How This Helps An Early-Stage Team

RAG systems regress in subtle ways: a prompt change can sound better while citing unsupported facts, a retriever tweak can miss the answer passage, or a model swap can become less extractive and more speculative.

This harness gives a small team a cheap local or CI check. Start with a handful of representative docs and questions, compare every prompt/model/retriever change against the last known-good run, and expand the test set whenever a real user failure appears.
