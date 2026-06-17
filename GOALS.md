# rag-eval Goals

Definition of done for the real retrieval version:

1. Runs offline end-to-end from a fresh clone using the committed real SQuAD v2 sample + corpus and the deterministic mock generator.
2. `python -m example_rag demo` runs the same eval at `k=5` vs `k=1` (same BM25 retriever, only `k` changes) and automatically flags aggregate regressions. Measured: retrieval_hit 0.88 -> 0.72, groundedness 0.81 -> 0.79, f1 0.137 -> 0.123.
3. The three core scorers are real and explainable: answer correctness (EM/F1), groundedness against retrieved docs, and retrieval hit@k against `gold_doc_id`.
4. SQuAD v2 loads from HuggingFace `rajpurkar/squad_v2` (`validation[:2000]`), caches locally, and pools paragraphs into a ~235-doc corpus (mostly distractors) with 50 answerable questions as the test set.
5. `README.md` leads with the headline finding from the regression demo.
6. `pytest` passes.
7. The harness (`src/rag_eval/`) and the system under test (`src/example_rag/`) are separate packages: the harness depends only on the `Retriever`/`Generator` protocols and never imports the concrete system, so a new system plugs in without touching the evaluator.

## Honest status / known limitations

- Retrieval hit-rate is the strong, real signal (it moves the most with `k`).
- The mock generator is a deterministic, no-ML span extractor, so absolute EM/F1 are low; they degrade with worse retrieval but are not LLM-quality. Swapping in `LLMGenerator` is the path to high absolute correctness.
- Groundedness is lexical overlap, so it mainly detects abstention/fabrication, not whether the right passage was used; that distinction is what retrieval hit-rate covers.
