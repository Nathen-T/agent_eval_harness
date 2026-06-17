# rag-eval Goals

Definition of done for the real retrieval version:

1. Runs offline end-to-end from a fresh clone using the committed SQuAD v2-style sample and mock generator.
2. `python -m rag_eval demo` runs healthy BM25 `k=5` vs a deliberately weakened fixed-order `k=1` config and automatically flags aggregate regressions.
3. The three core scorers are real and explainable: answer correctness, groundedness against retrieved docs, and retrieval hit@k against `gold_doc_id`.
4. The SQuAD v2 set loads from HuggingFace `validation[:50]`, caches locally, and pools contexts into a de-duped retrieval corpus.
5. `README.md` leads with the headline finding from the regression demo.
6. `pytest` passes.
