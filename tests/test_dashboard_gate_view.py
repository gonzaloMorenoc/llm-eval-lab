"""Tests for the Quality Gate page logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dashboard.components.gate_view import dataset_drift, list_baselines
from src.gate.baseline import build_baseline, save_baseline
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
