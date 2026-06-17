# rag-eval — Results & Takeaways

A write-up of what this project measured, what the numbers mean, and what I learned
building an offline RAG evaluation harness. All numbers below are reproduced from
the committed offline artifacts (`data/squad_sample.jsonl` + `data/squad_corpus.jsonl`)
with the deterministic mock generator, so anyone can regenerate them on a fresh clone.

## TL;DR

Shrinking retrieved context from `k=5` to `k=1` on a 50-question SQuAD v2 set
(235-passage corpus) drops retrieval hit-rate from **88% to 72%** under BM25, and
groundedness and F1 fall with it. The harness flags every drop automatically. The
headline isn't the absolute scores (the generator is a no-ML span extractor by
design) — it's that a single knob change produces a measurable, auto-detected
regression, which is exactly what a RAG eval harness exists to catch.

## Experimental setup

- **Dataset:** SQuAD v2, `validation[:2000]` from HuggingFace `rajpurkar/squad_v2`.
- **Corpus:** every unique paragraph in that slice pooled into a single retrieval
  index — **235 documents**. Most are distractors that answer no question in the
  test set, which is what makes retrieval a real search problem.
- **Test set:** **50 answerable questions**, each tagged with the `gold_doc_id` of
  its source paragraph. Unanswerable SQuAD v2 questions are excluded from the test
  set (so EM/F1 stay meaningful) but their paragraphs stay in the corpus as noise.
- **Retrievers:** BM25 (`rank_bm25`) and a hand-rolled TF-IDF, both deterministic.
- **Generator:** a deterministic, context-only extractive span picker that abstains
  when nothing retrieved is relevant enough. No ML, no network, no API key.
- **Scorers:** exact match + token F1 (answer correctness), lexical groundedness
  (answer support in retrieved docs), and true hit@k against `gold_doc_id`.
- **Determinism:** identical inputs produce byte-identical aggregates run to run,
  so any metric delta is attributable to the change under test, not sampling noise.

## Headline result: starving retrieval (BM25, only `k` changes)

The demo runs the same eval twice with the same retriever and generator, changing
**only `k`** — healthy retrieval (`k=5`) vs starved retrieval (`k=1`):

| metric | k=5 | k=1 | delta | flagged |
|---|---|---|---|---|
| em | 0.100 | 0.100 | +0.000 | |
| f1 | 0.137 | 0.123 | -0.013 | REGRESSION |
| groundedness | 0.810 | 0.790 | -0.020 | REGRESSION |
| retrieval_hit | 0.880 | 0.720 | -0.160 | REGRESSION |

Retrieval hit-rate moves the most because it directly measures whether the gold
paragraph survived the smaller `k`. Groundedness and F1 follow because when the
gold paragraph is missed, the generator either abstains or extracts from a
distractor, so the answer stops being supported.

## Retriever comparison (TF-IDF, only `k` changes)

| metric | k=5 | k=1 | delta | flagged |
|---|---|---|---|---|
| em | 0.100 | 0.100 | +0.000 | |
| f1 | 0.137 | 0.137 | +0.000 | |
| groundedness | 0.810 | 0.810 | +0.000 | |
| retrieval_hit | 0.860 | 0.740 | -0.120 | REGRESSION |

Two things stand out against BM25:

- **BM25 ranks slightly better at `k=5`** (hit-rate 0.880 vs 0.860), and it
  degrades harder when starved (-0.160 vs -0.120). BM25's term-saturation and
  length normalization give it a small but real edge on these short questions.
- **Under TF-IDF, groundedness and F1 stay flat** while hit-rate drops. The
  questions TF-IDF loses at `k=1` are ones the generator was already abstaining or
  scoring zero on, so removing those docs doesn't change correctness — only the
  retrieval metric notices. This is the clearest argument for keeping retrieval
  hit-rate as a **separate** signal: it catches retrieval regressions that the
  answer-quality metrics are blind to.

## Takeaways

### On RAG evaluation

1. **Retrieval needs its own metric.** Answer-quality scores (EM/F1) and even
   groundedness can stay flat while retrieval quietly gets worse (see the TF-IDF
   table). A rank-aware hit@k against a known gold doc is what surfaces it.
2. **Groundedness ≠ correctness.** A lexical-overlap groundedness score mostly
   detects abstention and fabrication, not whether the *right* passage was used.
   It's a useful hallucination tripwire, but it is not a stand-in for hit-rate.
3. **Distractors are the whole point.** Earlier versions pooled only the answer
   paragraphs, which pinned hit-rate near 1.0 and made the metric meaningless.
   Pooling 235 paragraphs (most irrelevant) is what makes the number move.
4. **Determinism is a feature, not a limitation.** A no-ML generator gives low
   absolute EM/F1, but it makes every run reproducible, so a regression is
   unambiguous. For a *regression harness*, repeatability beats peak scores.

### On engineering

5. **Separate the evaluator from the system under test.** The harness
   (`rag_eval`) talks only to `Retriever`/`Generator` protocols and never imports
   the example system (`example_rag`). That boundary is what makes "bring your own
   RAG system" real rather than aspirational.
6. **Offline-first lowers the barrier to running it.** Committed sample + corpus
   plus a deterministic generator mean `pytest` and the demo work on first clone
   with no network and no API key — important for CI and for reviewers.
7. **Small, readable, explainable.** Every metric is normalized to `[0, 1]` and
   computed by hand, so the harness is auditable end to end rather than a black box.

## Honest limitations

- **Absolute EM/F1 are low (~0.10 / ~0.14).** The mock generator extracts spans
  with no model, so it is not answer-quality competitive — it exists to make the
  loop deterministic. Real correctness needs `LLMGenerator`.
- **Groundedness is lexical, not semantic.** It can be fooled by paraphrase and
  doesn't verify entailment.
- **Corpus is small and narrow.** 235 paragraphs from a 2000-row slice (a handful
  of articles); a larger, more diverse slice would be a harder retrieval test.
- **Single test set.** Results are reported on one 50-question set; no confidence
  intervals or multiple seeds (deterministic by construction, but also a single
  sample of the data).

## What I'd do next

1. Implement `LLMGenerator` so EM/F1 reflect real answer quality (keep the mock as
   the deterministic default for CI).
2. Implement `LLMJudgeScorer` for semantic faithfulness/correctness alongside the
   lexical metrics.
3. Pool a larger / more diverse SQuAD v2 slice to make retrieval harder and the
   BM25-vs-TF-IDF gap more pronounced.
4. Wire `compare` into CI to fail a PR when any aggregate regresses past threshold.

## Reproducing these numbers

```bash
uv sync --dev
uv run python -m example_rag demo                    # BM25, k=5 vs k=1 (headline table)
uv run python -m example_rag demo --retriever tfidf  # TF-IDF comparison table
uv run pytest -q                                      # 7 tests
```

The demo runs fully offline against the committed artifacts, so the tables above
reproduce exactly.
