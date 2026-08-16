"""CSV exporters for the dashboard's download buttons.

Reports are already persisted as JSON and Markdown, so those are served
straight from disk. CSV is the one format that has to be built: it is what
gets opened in a spreadsheet, which is why people export in the first place.

Uses the ``csv`` module rather than string joins so that commas, quotes and
newlines inside test case text — all of which occur — cannot shift columns.
"""

from __future__ import annotations

import csv
import io
from typing import Any

_RESULT_COLUMNS = [
    "id",
    "category",
    "severity",
    "passed",
    "score",
    "latency_ms",
    "evaluators",
    "input",
    "response",
]


def results_to_csv(summary: dict[str, Any]) -> str:
    """One row per test case, with the fields worth sorting or filtering on."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_RESULT_COLUMNS, lineterminator="\n")
    writer.writeheader()

    for result in summary.get("results", []):
        test_case = result.get("test_case", {})
        score = result.get("overall_score")
        writer.writerow(
            {
                "id": test_case.get("id", ""),
                "category": test_case.get("category", ""),
                "severity": test_case.get("severity", ""),
                "passed": result.get("overall_passed", False),
                "score": "" if score is None else score,
                "latency_ms": result.get("latency_ms", ""),
                "evaluators": ", ".join(e.get("evaluator", "") for e in result.get("evaluations", [])),
                "input": _flatten(test_case.get("input", "")),
                "response": result.get("response", ""),
            }
        )
    return buffer.getvalue()


def comparison_to_csv(rows: list[dict[str, Any]]) -> str:
    """Serialise an already-built comparison table, preserving its column order."""
    if not rows:
        return ""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _flatten(value: Any) -> str:
    """Render a test case input, which is either a string or a chat transcript."""
    if isinstance(value, list):
        return " | ".join(f"{turn.get('role', '?')}: {turn.get('content', '')}" for turn in value)
    return str(value)
