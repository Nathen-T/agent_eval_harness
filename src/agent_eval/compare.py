from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_runs(run_a: str | Path, run_b: str | Path) -> list[str]:
    """Print aggregate metric deltas and return names of regressed metrics."""

    baseline = load_run(run_a)
    candidate = load_run(run_b)

    baseline_scores = baseline.get("aggregates", {})
    candidate_scores = candidate.get("aggregates", {})
    metrics = sorted(set(baseline_scores) | set(candidate_scores))
    regressions: list[str] = []

    rows: list[tuple[str, str, str, str, str]] = []
    for metric in metrics:
        before = baseline_scores.get(metric)
        after = candidate_scores.get(metric)
        delta = _delta(before, after)
        status = ""
        if before is not None and after is not None and after < before:
            status = "REGRESSION"
            regressions.append(metric)
        rows.append(
            (
                metric,
                _format_score(before),
                _format_score(after),
                _format_delta(delta),
                status,
            )
        )

    print(f"Baseline:  {Path(run_a)}")
    print(f"Candidate: {Path(run_b)}")
    print()
    print(_format_table(rows))

    if regressions:
        print()
        print("Regressions detected: " + ", ".join(regressions))
    else:
        print()
        print("No aggregate regressions detected.")

    return regressions


def load_run(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_runs(runs_dir: str | Path = "runs", count: int = 2) -> list[Path]:
    runs = sorted(Path(runs_dir).glob("run-*.json"), key=lambda path: path.stat().st_mtime)
    return runs[-count:]


def _delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def _format_score(score: Any) -> str:
    if score is None:
        return "n/a"
    return f"{float(score):.3f}"


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.3f}"


def _format_table(rows: list[tuple[str, str, str, str, str]]) -> str:
    headers = ("metric", "run_a", "run_b", "delta", "status")
    all_rows = [headers, *rows]
    widths = [max(len(row[index]) for row in all_rows) for index in range(len(headers))]

    def format_row(row: tuple[str, str, str, str, str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([format_row(headers), separator, *[format_row(row) for row in rows]])
