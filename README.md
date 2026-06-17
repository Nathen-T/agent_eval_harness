# agent-eval

How do you know an agent change did not silently make things worse?

`agent-eval` is a small, opinionated evaluation harness for LLM and agent outputs. It runs a fixed task set through a system under test, scores each output, saves every run, and compares runs so regressions are visible before they reach users.

It is intentionally bare bones: Python 3.11+, stdlib-only runtime code, deterministic mock system, and zero API keys required on first clone.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Run the bundled offline evaluation:

```bash
python -m agent_eval run
```

This loads `data/testset.jsonl`, sends each task to the deterministic mock system, scores outputs with exact-match and keyword scorers, and writes a timestamped JSON file to `runs/`.

Compare the two latest runs:

```bash
python -m agent_eval compare
```

Or compare specific run files:

```bash
python -m agent_eval compare runs/run-20260617-092300.json runs/run-20260617-092500.json
```

Run tests:

```bash
pytest -q
```

## Example Output

Running an eval:

```text
Saved run: runs/run-20260617-092300.json
System: mock_system
Tasks: 8

Aggregates
  exact_match: 1.000
  keyword_contains: 1.000
```

Comparing two runs:

```text
Baseline:  runs/run-20260617-092300.json
Candidate: runs/run-20260617-092500.json

metric            run_a  run_b  delta   status
----------------  -----  -----  ------  ----------
exact_match       1.000  0.875  -0.125  REGRESSION
keyword_contains  1.000  1.000  +0.000

Regressions detected: exact_match
```

## What Is Included

- `Task` dataclass plus a JSONL loader.
- Eight small seed tasks in `data/testset.jsonl`.
- `ExactMatchScorer` for normalized reference matching.
- `KeywordScorer` for simple contains-style checks.
- `LLMJudgeScorer` stub for a future LLM-as-judge scorer.
- `mock_system` for deterministic offline runs.
- `real_llm_adapter` stub for a future provider-backed system.
- `run_eval()` for per-task and aggregate scoring.
- `save_run()` for timestamped JSON persistence.
- `compare_runs()` for aggregate metric deltas and regression flags.
- `python -m agent_eval run` and `python -m agent_eval compare` CLI commands.

## How This Helps An Early-Stage Team

Early agent products change quickly: prompts move, models get swapped, tools are renamed, and output formats drift. Without a fixed eval set, a demo can look better while a core workflow quietly gets worse.

This harness gives a small team a cheap regression check they can run locally or in CI. Start with a handful of high-value examples, add a scorer when a failure mode matters, and compare every meaningful prompt or model change against the last known-good run.

The mock system keeps the project runnable anywhere. When you are ready to test a real model, replace or extend `real_llm_adapter()` in `src/agent_eval/systems.py` and keep the same `str -> str` function shape.
