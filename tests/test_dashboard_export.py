"""Tests for the CSV exporters behind the dashboard's download buttons."""

from __future__ import annotations

import csv
import io

from src.dashboard.components.export import comparison_to_csv, results_to_csv


def _result(case_id: str, **overrides) -> dict:
    base = {
        "test_case": {"id": case_id, "category": "functional", "severity": "medium", "input": "What is ML?"},
        "response": "Machine learning is...",
        "overall_passed": True,
        "overall_score": 0.87,
        "latency_ms": 123.4,
        "evaluations": [{"evaluator": "rule_based", "passed": True, "score": 0.9}],
    }
    base.update(overrides)
    return base


def _parse(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


class TestResultsToCsv:
    def test_one_row_per_test_case(self) -> None:
        summary = {"run_id": "run_a", "results": [_result("a"), _result("b")]}

        rows = _parse(results_to_csv(summary))

        assert [row["id"] for row in rows] == ["a", "b"]

    def test_carries_the_fields_worth_analysing(self) -> None:
        summary = {"run_id": "run_a", "results": [_result("a")]}

        row = _parse(results_to_csv(summary))[0]

        assert row["category"] == "functional"
        assert row["severity"] == "medium"
        assert row["passed"] == "True"
        assert row["score"] == "0.87"
        assert row["latency_ms"] == "123.4"
        assert row["evaluators"] == "rule_based"

    def test_commas_and_quotes_in_text_survive_the_round_trip(self) -> None:
        """Test case text is free-form and lands in a CSV cell — unescaped, one
        comma would shift every later column of that row."""
        nasty = 'Explain "ML", briefly, with commas'
        summary = {
            "run_id": "run_a",
            "results": [_result("a", test_case={"id": "a", "category": "functional", "severity": "low", "input": nasty})],
        }

        row = _parse(results_to_csv(summary))[0]

        assert row["input"] == nasty
        assert row["severity"] == "low"

    def test_newlines_in_a_response_do_not_break_the_row_count(self) -> None:
        summary = {"run_id": "run_a", "results": [_result("a", response="line one\nline two"), _result("b")]}

        rows = _parse(results_to_csv(summary))

        assert len(rows) == 2

    def test_missing_score_becomes_an_empty_cell(self) -> None:
        summary = {"run_id": "run_a", "results": [_result("a", overall_score=None)]}

        assert _parse(results_to_csv(summary))[0]["score"] == ""

    def test_a_run_with_no_results_still_has_a_header(self) -> None:
        text = results_to_csv({"run_id": "empty", "results": []})

        assert text.splitlines()[0].startswith("id,")
        assert _parse(text) == []


class TestComparisonToCsv:
    def test_one_row_per_metric(self) -> None:
        rows_in = [
            {"Métrica": "pass_rate", "A": "0.70", "B": "0.62", "Delta": "-0.08"},
            {"Métrica": "rule_based", "A": "0.83", "B": "0.82", "Delta": "-0.01"},
        ]

        rows = _parse(comparison_to_csv(rows_in))

        assert [row["Métrica"] for row in rows] == ["pass_rate", "rule_based"]

    def test_keeps_the_column_order_it_was_given(self) -> None:
        rows_in = [{"Métrica": "pass_rate", "A": "0.70", "B": "0.62"}]

        header = comparison_to_csv(rows_in).splitlines()[0]

        assert header == "Métrica,A,B"

    def test_no_rows_produces_empty_output(self) -> None:
        assert comparison_to_csv([]) == ""
