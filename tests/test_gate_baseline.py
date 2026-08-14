"""Tests for baseline building, hashing, saving and loading."""

from __future__ import annotations

import pytest

from src.gate.baseline import (
    BaselineError,
    build_baseline,
    compute_dataset_hash,
    load_baseline,
    save_baseline,
)
from tests.gate_helpers import make_summary


class TestComputeDatasetHash:
    def test_hash_is_stable_and_order_independent(self):
        s1 = make_summary({"a": {"rule_based": 1.0}, "b": {"rule_based": 0.5}})
        s2 = make_summary({"b": {"rule_based": 0.5}, "a": {"rule_based": 1.0}})
        cases1 = [r.test_case for r in s1.results]
        cases2 = [r.test_case for r in reversed(s2.results)]
        assert compute_dataset_hash(cases1) == compute_dataset_hash(cases2)

    def test_hash_changes_when_a_case_changes(self):
        s1 = make_summary({"a": {"rule_based": 1.0}})
        s2 = make_summary({"a": {"rule_based": 1.0}}, severities={"a": "critical"})
        h1 = compute_dataset_hash([r.test_case for r in s1.results])
        h2 = compute_dataset_hash([r.test_case for r in s2.results])
        assert h1 != h2


class TestBuildBaseline:
    def test_single_summary_flattens_metrics(self):
        summary = make_summary({"a": {"rule_based": 1.0, "ragas.answer_relevancy": 0.8}})
        baseline = build_baseline([summary])
        assert baseline.samples == 1
        case = baseline.cases[0]
        assert case.metrics == {"rule_based": 1.0, "ragas.answer_relevancy": 0.8}
        assert case.flakiness == 0.0
        assert case.pass_samples == [True]
        assert "pass_rate" in baseline.metric_set
        assert "ragas.answer_relevancy" in baseline.metric_set

    def test_multi_sample_means_variance_and_flakiness(self):
        s1 = make_summary({"a": {"rule_based": 0.8}}, passed={"a": True}, run_id="r1")
        s2 = make_summary({"a": {"rule_based": 0.4}}, passed={"a": False}, run_id="r2")
        s3 = make_summary({"a": {"rule_based": 0.6}}, passed={"a": True}, run_id="r3")
        baseline = build_baseline([s1, s2, s3])
        case = baseline.cases[0]
        assert baseline.samples == 3
        assert baseline.run_ids == ["r1", "r2", "r3"]
        assert case.metrics["rule_based"] == pytest.approx(0.6)
        assert case.metric_variance["rule_based"] == pytest.approx(0.026667, abs=1e-4)
        assert case.passed is True  # majority 2/3
        assert case.flakiness == pytest.approx(1 / 3)

    def test_tie_counts_as_passed(self):
        s1 = make_summary({"a": {"rule_based": 1.0}}, passed={"a": True}, run_id="r1")
        s2 = make_summary({"a": {"rule_based": 1.0}}, passed={"a": False}, run_id="r2")
        baseline = build_baseline([s1, s2])
        assert baseline.cases[0].passed is True
        assert baseline.cases[0].flakiness == pytest.approx(0.5)

    def test_empty_summaries_raise(self):
        with pytest.raises(BaselineError, match="At least one"):
            build_baseline([])

    def test_mismatched_case_ids_raise(self):
        s1 = make_summary({"a": {"rule_based": 1.0}}, run_id="r1")
        s2 = make_summary({"b": {"rule_based": 1.0}}, run_id="r2")
        with pytest.raises(BaselineError, match="same test case ids"):
            build_baseline([s1, s2])

    def test_mismatched_modes_raise(self):
        s1 = make_summary({"a": {"rule_based": 1.0}}, run_id="r1", chatbot_mode="plain")
        s2 = make_summary({"a": {"rule_based": 1.0}}, run_id="r2", chatbot_mode="rag")
        with pytest.raises(BaselineError, match="chatbot_mode"):
            build_baseline([s1, s2])


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        baseline = build_baseline([make_summary({"a": {"rule_based": 1.0}})])
        path = save_baseline(baseline, str(tmp_path / "baselines" / "main.json"))
        assert load_baseline(path) == baseline

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(BaselineError, match="not found"):
            load_baseline(str(tmp_path / "nope.json"))

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(BaselineError, match="Invalid baseline"):
            load_baseline(str(path))
