from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def compare_runs(
    run_a: str | Path,
    run_b: str | Path,
    threshold: float = 0.01,
) -> list[str]:
    """Print metric deltas and return metrics that regressed beyond threshold."""

    baseline = load_run(run_a)
    candidate = load_run(run_b)

    baseline_scores = baseline.get("aggregates", {})
    candidate_scores = candidate.get("aggregates", {})
    metrics = sorted(set(baseline_scores) | set(candidate_scores))

    rows: list[dict[str, Any]] = []
    regressions: list[str] = []
    for metric in metrics:
        before = baseline_scores.get(metric)
        after = candidate_scores.get(metric)
        delta = None if before is None or after is None else float(after) - float(before)
        status = ""
        if delta is not None and delta < -threshold:
            status = "REGRESSION"
            regressions.append(metric)

        rows.append(
            {
                "metric": metric,
                "run_a": _format_score(before),
                "run_b": _format_score(after),
                "delta": _format_delta(delta),
                "status": status,
            }
        )

    frame = pd.DataFrame(rows, columns=["metric", "run_a", "run_b", "delta", "status"])

    print(f"Baseline:  {Path(run_a)}")
    print(f"Candidate: {Path(run_b)}")
    print(f"Threshold: -{threshold:.3f}")
    print()
    print(frame.to_string(index=False))

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


def _format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"
