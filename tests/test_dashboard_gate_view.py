"""Tests for the Quality Gate page logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dashboard.components.gate_view import blocking_reasons, dataset_drift, list_baselines, verdict_rows
from src.gate.baseline import build_baseline, save_baseline
from src.gate.models import GatePolicy, GateVerdict, MetricComparison, MetricPolicy
from src.runner.models import TestCase
from tests.gate_helpers import make_summary


def _write_baseline(directory: Path, name: str, *, run_id: str = "run_a") -> None:
    summary = make_summary({"case_a": {"rule_based": 0.9}}, run_id=run_id)
    save_baseline(build_baseline([summary]), str(directory / f"{name}.json"))


class TestListBaselines:
    def test_returns_empty_when_directory_is_missing(self, tmp_path: Path) -> None:
        assert list_baselines(str(tmp_path / "nope")) == []

    def test_reads_name_and_metadata_from_each_file(self, tmp_path: Path) -> None:
        _write_baseline(tmp_path, "main", run_id="run_main")

        found = list_baselines(str(tmp_path))

        assert len(found) == 1
        assert found[0].name == "main"
        assert found[0].run_ids == ["run_main"]
        assert found[0].n_cases == 1
        assert found[0].samples == 1

    def test_sorts_by_name(self, tmp_path: Path) -> None:
        _write_baseline(tmp_path, "zeta")
        _write_baseline(tmp_path, "alpha")

        assert [b.name for b in list_baselines(str(tmp_path))] == ["alpha", "zeta"]

    def test_an_unreadable_file_does_not_hide_the_others(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_baseline(tmp_path, "good")
        (tmp_path / "broken.json").write_text("{not json")

        with caplog.at_level("WARNING"):
            found = list_baselines(str(tmp_path))

        assert [b.name for b in found] == ["good"]
        assert "broken.json" in caplog.text

    def test_ignores_non_json_files(self, tmp_path: Path) -> None:
        _write_baseline(tmp_path, "main")
        (tmp_path / "README.md").write_text("not a baseline")

        assert [b.name for b in list_baselines(str(tmp_path))] == ["main"]


def _case(case_id: str, text: str = "q") -> TestCase:
    return TestCase(
        id=case_id,
        category="functional",
        input=text,
        expected_behavior="answers",
        evaluation_type=["rule_based"],
        severity="medium",
    )


def _baseline_over(case_ids: list[str], text: str = "q"):
    """Baseline built from a run covering exactly ``case_ids``."""
    summary = make_summary({cid: {"rule_based": 0.9} for cid in case_ids})
    for result in summary.results:
        result.test_case.input = text
    return build_baseline([summary])


class TestDatasetDrift:
    def test_identical_cases_show_no_drift(self) -> None:
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a"), _case("b")])

        assert report.comparable is True
        assert report.drifted is False

    def test_same_id_with_changed_text_is_drift(self) -> None:
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a", "a completely different question"), _case("b")])

        assert report.comparable is True
        assert report.drifted is True

    def test_a_run_covering_extra_cases_is_not_drift(self) -> None:
        """The hash is computed over the baseline's ids only. Hashing the whole
        current dataset would flag every baseline built from a subset — a normal
        usage — and an alarm that fires almost always gets ignored."""
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a"), _case("b"), _case("c"), _case("d")])

        assert report.comparable is True
        assert report.drifted is False

    def test_a_missing_id_makes_it_not_comparable(self) -> None:
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a")])

        assert report.comparable is False
        assert report.missing_ids == ["b"]
        assert report.current_hash is None

    def test_not_comparable_is_never_reported_as_drift(self) -> None:
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a", "changed too")])

        assert report.drifted is False


def _comparison(metric: str, *, regression: float = 0.0, gated: bool = False, breaches: bool = False) -> MetricComparison:
    return MetricComparison(
        metric=metric,
        baseline_mean=0.70,
        current_mean=0.70 - regression,
        regression=regression,
        ci_low=-0.11,
        ci_high=-0.04,
        p_value=0.01,
        n_cases=43,
        significant=breaches,
        gated=gated,
        breaches=breaches,
    )


def _verdict(**kwargs) -> GateVerdict:
    defaults = dict(
        passed=True,
        comparisons=[],
        hard_rule_violations=[],
        missing_gated_metrics=[],
        new_case_ids=[],
        removed_case_ids=[],
        mean_flakiness=0.0,
        samples=3,
    )
    return GateVerdict(**{**defaults, **kwargs})


class TestVerdictRows:
    def test_one_row_per_comparison(self) -> None:
        verdict = _verdict(comparisons=[_comparison("pass_rate"), _comparison("rule_based")])

        rows = verdict_rows(verdict)

        assert [row["Métrica"] for row in rows] == ["pass_rate", "rule_based"]

    def test_marks_which_metrics_are_gated(self) -> None:
        verdict = _verdict(comparisons=[_comparison("pass_rate", gated=True), _comparison("rule_based", gated=False)])

        rows = verdict_rows(verdict)

        assert rows[0]["Gateada"] == "sí"
        assert rows[1]["Gateada"] == "no"

    def test_regression_keeps_its_sign(self) -> None:
        verdict = _verdict(comparisons=[_comparison("pass_rate", regression=0.08)])

        assert verdict_rows(verdict)[0]["Regresión"].startswith("+")

    def test_no_comparisons_gives_no_rows(self) -> None:
        assert verdict_rows(_verdict()) == []


class TestBlockingReasons:
    def test_a_passing_verdict_has_no_reasons(self) -> None:
        assert blocking_reasons(_verdict(), GatePolicy()) == []

    def test_reports_hard_rule_violations_verbatim(self) -> None:
        verdict = _verdict(passed=False, hard_rule_violations=["New critical failures: safety_004"])

        reasons = blocking_reasons(verdict, GatePolicy())

        assert any("safety_004" in reason for reason in reasons)

    def test_a_breaching_metric_quotes_its_limit(self) -> None:
        verdict = _verdict(passed=False, comparisons=[_comparison("pass_rate", regression=0.08, gated=True, breaches=True)])
        policy = GatePolicy(metrics={"pass_rate": MetricPolicy(max_regression=0.05)})

        reasons = blocking_reasons(verdict, policy)

        assert len(reasons) == 1
        assert "pass_rate" in reasons[0]
        assert "0.05" in reasons[0]

    def test_a_non_breaching_metric_is_not_a_reason(self) -> None:
        verdict = _verdict(passed=False, comparisons=[_comparison("rule_based", regression=0.01)])

        assert blocking_reasons(verdict, GatePolicy()) == []

    def test_missing_gated_metric_is_not_worded_as_a_regression(self) -> None:
        """In CI this is exit 2, a configuration error — the exact case that
        produced a false PASS in PR #11 when an evaluator silently dropped out."""
        verdict = _verdict(passed=False, missing_gated_metrics=["pass_rate"])

        reasons = blocking_reasons(verdict, GatePolicy())

        assert len(reasons) == 1
        assert "regres" not in reasons[0].lower()
        assert "compar" in reasons[0].lower()
