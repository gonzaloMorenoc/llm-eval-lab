"""Tests for policy loading and full gate evaluation."""

from __future__ import annotations

import pytest

from src.gate.baseline import build_baseline
from src.gate.comparison import CompatibilityError
from src.gate.models import BaselineCase, BaselineFile, GatePolicy, HardRules, MetricPolicy
from src.gate.policy import PolicyError, evaluate_gate, load_policy
from tests.gate_helpers import make_summary


def _baseline(case_scores, **kwargs):
    return build_baseline([make_summary(case_scores, **kwargs)])


class TestLoadPolicy:
    def test_none_returns_defaults(self):
        assert load_policy(None) == GatePolicy()

    def test_loads_yaml_with_gate_key(self, tmp_path):
        path = tmp_path / "gate.yaml"
        path.write_text("gate:\n  significance_level: 0.01\n  metrics:\n    rule_based: {max_regression: 0.2}\n  new_cases: fail\n")
        policy = load_policy(str(path))
        assert policy.significance_level == 0.01
        assert policy.metrics == {"rule_based": MetricPolicy(max_regression=0.2)}
        assert policy.new_cases == "fail"
        assert policy.min_effect_size == 0.05  # default preserved

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(PolicyError, match="Cannot read"):
            load_policy(str(tmp_path / "nope.yaml"))

    def test_invalid_content_raises(self, tmp_path):
        path = tmp_path / "gate.yaml"
        path.write_text("gate:\n  new_cases: whatever\n")
        with pytest.raises(PolicyError, match="Invalid policy"):
            load_policy(str(path))


class TestEvaluateGate:
    def test_identical_runs_pass(self):
        scores = {f"c{i}": {"rule_based": 0.9} for i in range(10)}
        verdict = evaluate_gate(_baseline(scores), _baseline(scores), GatePolicy())
        assert verdict.passed is True
        assert verdict.hard_rule_violations == []
        assert verdict.missing_gated_metrics == []
        assert verdict.samples == 1

    def test_new_critical_failure_violates_hard_rule(self):
        base = _baseline(
            {"crit": {"rule_based": 1.0}, "ok": {"rule_based": 1.0}},
            passed={"crit": True, "ok": True},
            severities={"crit": "critical"},
        )
        curr = _baseline(
            {"crit": {"rule_based": 1.0}, "ok": {"rule_based": 1.0}},
            passed={"crit": False, "ok": True},
            severities={"crit": "critical"},
        )
        verdict = evaluate_gate(base, curr, GatePolicy())
        assert verdict.passed is False
        assert any("crit" in v for v in verdict.hard_rule_violations)

    def test_already_failing_critical_case_is_not_a_new_failure(self):
        base = _baseline({"crit": {"rule_based": 0.1}}, passed={"crit": False}, severities={"crit": "critical"})
        curr = _baseline({"crit": {"rule_based": 0.1}}, passed={"crit": False}, severities={"crit": "critical"})
        verdict = evaluate_gate(base, curr, GatePolicy())
        assert verdict.hard_rule_violations == []

    def test_new_cases_report_only_vs_fail(self):
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 1.0}, "b": {"rule_based": 1.0}})
        report_only = evaluate_gate(base, curr, GatePolicy())
        assert report_only.passed is True
        assert report_only.new_case_ids == ["b"]
        failing = evaluate_gate(base, curr, GatePolicy(new_cases="fail"))
        assert failing.passed is False
        assert any("b" in v for v in failing.hard_rule_violations)

    def test_missing_gated_metric_fails_verdict(self):
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 1.0}})
        policy = GatePolicy(metrics={"ragas.answer_relevancy": MetricPolicy(max_regression=0.1)})
        verdict = evaluate_gate(base, curr, policy)
        assert verdict.missing_gated_metrics == ["ragas.answer_relevancy"]
        assert verdict.passed is False

    def test_flakiness_hard_rule(self):
        s1 = make_summary({"a": {"rule_based": 1.0}}, passed={"a": True}, run_id="r1")
        s2 = make_summary({"a": {"rule_based": 1.0}}, passed={"a": False}, run_id="r2")
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = build_baseline([s1, s2])  # flakiness 0.5 > 0.3
        verdict = evaluate_gate(base, curr, GatePolicy())
        assert verdict.mean_flakiness == pytest.approx(0.5)
        assert any("flakiness" in v.lower() for v in verdict.hard_rule_violations)
        relaxed = evaluate_gate(base, curr, GatePolicy(hard_rules=HardRules(max_flakiness=0.6)))
        assert relaxed.hard_rule_violations == []

    def test_incompatible_runs_propagate(self):
        base = _baseline({"a": {"rule_based": 1.0}}, chatbot_mode="plain")
        curr = _baseline({"a": {"rule_based": 1.0}}, chatbot_mode="rag")
        with pytest.raises(CompatibilityError):
            evaluate_gate(base, curr, GatePolicy())

    def test_non_critical_failure_does_not_violate_hard_rule(self):
        # F1 fix: verify that only CRITICAL cases trigger the hard rule
        base = _baseline(
            {
                "critical_case": {"rule_based": 1.0},
                "high_case": {"rule_based": 1.0},
            },
            passed={"critical_case": True, "high_case": True},
            severities={"critical_case": "critical", "high_case": "high"},
        )
        curr = _baseline(
            {
                "critical_case": {"rule_based": 1.0},
                "high_case": {"rule_based": 1.0},
            },
            passed={"critical_case": True, "high_case": False},
            severities={"critical_case": "critical", "high_case": "high"},
        )
        verdict = evaluate_gate(base, curr, GatePolicy())
        assert verdict.passed is True
        assert verdict.hard_rule_violations == []

    def test_missing_gated_metric_when_not_compared(self):
        # F2 fix: metric in metric_set but not compared (missing on one side)
        base_case = BaselineCase(
            id="a",
            category="test",
            severity="high",
            passed=True,
            pass_samples=[True],
            metrics={"rule_based": 0.9, "answer_relevancy": 0.8},
        )
        base = BaselineFile(
            schema_version=1,
            run_ids=["run1"],
            timestamp="2026-08-14T00:00:00Z",
            chatbot_id="test",
            chatbot_mode="plain",
            dataset_hash="hash1",
            metric_set=["rule_based", "answer_relevancy"],
            cases=[base_case],
        )
        # Current run has answer_relevancy in metric_set but NOT in the case metrics
        curr_case = BaselineCase(
            id="a",
            category="test",
            severity="high",
            passed=True,
            pass_samples=[True],
            metrics={"rule_based": 0.9},  # answer_relevancy missing
        )
        curr = BaselineFile(
            schema_version=1,
            run_ids=["run2"],
            timestamp="2026-08-14T00:00:00Z",
            chatbot_id="test",
            chatbot_mode="plain",
            dataset_hash="hash1",
            metric_set=["rule_based", "answer_relevancy"],
            cases=[curr_case],
        )
        policy = GatePolicy(metrics={"answer_relevancy": MetricPolicy(max_regression=0.1)})
        verdict = evaluate_gate(base, curr, policy)
        assert verdict.missing_gated_metrics == ["answer_relevancy"]
        assert verdict.passed is False

    def test_loads_yaml_without_gate_key(self, tmp_path):
        # F3 fix: flat YAML structure (no top-level gate: key)
        path = tmp_path / "gate.yaml"
        path.write_text("significance_level: 0.02\nmetrics:\n  rule_based: {max_regression: 0.15}\nnew_cases: fail\n")
        policy = load_policy(str(path))
        assert policy.significance_level == 0.02
        assert policy.metrics == {"rule_based": MetricPolicy(max_regression=0.15)}
        assert policy.new_cases == "fail"
        assert policy.min_effect_size == 0.05  # default preserved
