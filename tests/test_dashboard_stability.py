"""Tests for the multi-sample stability view.

When the dashboard runs the suite more than once, the interesting part is not
the average but *which cases disagree with themselves* — a case that passes
twice and fails once is the one that will break someone's build later.
"""

from __future__ import annotations

import pytest

from src.dashboard.components.stability import sample_pattern, stability_headline, unstable_case_rows
from src.gate.baseline import build_baseline
from tests.gate_helpers import make_summary


def _baseline(*pass_maps: dict[str, bool]):
    """Build a baseline from N samples, each given as {case_id: passed}."""
    summaries = [
        make_summary(
            {case_id: {"rule_based": 1.0 if ok else 0.0} for case_id, ok in pass_map.items()},
            passed=pass_map,
            run_id=f"run_{i}",
        )
        for i, pass_map in enumerate(pass_maps)
    ]
    return build_baseline(summaries)


class TestUnstableCaseRows:
    def test_lists_only_cases_that_disagree_across_samples(self) -> None:
        baseline = _baseline(
            {"always_ok": True, "flaky": True, "always_bad": False},
            {"always_ok": True, "flaky": False, "always_bad": False},
            {"always_ok": True, "flaky": True, "always_bad": False},
        )

        rows = unstable_case_rows(baseline)

        # A case that consistently fails is a plain failure, not an unstable one.
        assert [row["Caso"] for row in rows] == ["flaky"]

    def test_orders_the_most_unstable_first(self) -> None:
        baseline = _baseline(
            {"barely": True, "coin_flip": True},
            {"barely": True, "coin_flip": False},
            {"barely": True, "coin_flip": True},
            {"barely": False, "coin_flip": False},
        )

        rows = unstable_case_rows(baseline)

        assert [row["Caso"] for row in rows] == ["coin_flip", "barely"]

    def test_is_empty_for_a_single_sample(self) -> None:
        """One sample cannot contradict itself, so nothing is ever flaky."""
        baseline = _baseline({"a": True, "b": False})

        assert unstable_case_rows(baseline) == []

    def test_row_carries_category_and_severity(self) -> None:
        baseline = _baseline({"flaky": True}, {"flaky": False})

        row = unstable_case_rows(baseline)[0]

        assert row["Categoría"] == "functional"
        assert row["Severidad"] == "medium"


class TestSamplePattern:
    def test_renders_each_sample_in_order(self) -> None:
        baseline = _baseline({"flaky": True}, {"flaky": False}, {"flaky": True})
        case = next(c for c in baseline.cases if c.id == "flaky")

        assert sample_pattern(case) == "✅❌✅"

    def test_handles_a_single_sample(self) -> None:
        baseline = _baseline({"a": False})
        case = baseline.cases[0]

        assert sample_pattern(case) == "❌"


class TestStabilityHeadline:
    def test_reports_how_many_cases_are_unstable(self) -> None:
        baseline = _baseline(
            {"a": True, "b": True, "c": True},
            {"a": True, "b": False, "c": False},
        )

        headline = stability_headline(baseline)

        assert headline["unstable"] == 2
        assert headline["total"] == 3
        assert headline["samples"] == 2

    def test_all_stable_reports_zero(self) -> None:
        baseline = _baseline({"a": True}, {"a": True})

        assert stability_headline(baseline)["unstable"] == 0

    def test_mean_flakiness_is_between_zero_and_one(self) -> None:
        baseline = _baseline({"a": True, "b": True}, {"a": False, "b": True})

        headline = stability_headline(baseline)

        assert 0.0 < headline["mean_flakiness"] <= 1.0


class TestEmptyBaseline:
    def test_headline_on_a_baseline_with_no_cases(self) -> None:
        """Guard against a division by zero if every dataset was deselected."""
        baseline = _baseline({"a": True})
        empty = baseline.model_copy(update={"cases": []})

        headline = stability_headline(empty)

        assert headline["unstable"] == 0
        assert headline["total"] == 0
        assert headline["mean_flakiness"] == pytest.approx(0.0)
