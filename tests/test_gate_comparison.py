"""Tests for case pairing and per-metric statistical comparison."""

from __future__ import annotations

import pytest

from src.gate.baseline import build_baseline
from src.gate.comparison import (
    CompatibilityError,
    compare_metrics,
    pair_cases,
    regression_deltas,
    validate_compatibility,
)
from src.gate.models import GatePolicy, MetricPolicy
from tests.gate_helpers import make_summary


def _baseline(case_scores, **kwargs):
    return build_baseline([make_summary(case_scores, **kwargs)])


class TestValidateCompatibility:
    def test_same_mode_and_shared_metrics_ok(self):
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 0.9}})
        validate_compatibility(base, curr)  # no raise

    def test_mode_mismatch_raises(self):
        base = _baseline({"a": {"rule_based": 1.0}}, chatbot_mode="plain")
        curr = _baseline({"a": {"rule_based": 1.0}}, chatbot_mode="rag")
        with pytest.raises(CompatibilityError, match="chatbot_mode"):
            validate_compatibility(base, curr)

    def test_pass_rate_always_shared(self):
        # Even with disjoint evaluator metrics, pass_rate exists on both sides.
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"llm_judge": 0.8}})
        validate_compatibility(base, curr)  # no raise


class TestPairCases:
    def test_pairs_new_and_removed(self):
        base = _baseline({"a": {"rule_based": 1.0}, "b": {"rule_based": 0.9}})
        curr = _baseline({"b": {"rule_based": 0.8}, "c": {"rule_based": 0.7}})
        pairs, new_ids, removed_ids = pair_cases(base, curr)
        assert [(p[0].id, p[1].id) for p in pairs] == [("b", "b")]
        assert new_ids == ["c"]
        assert removed_ids == ["a"]


class TestRegressionDeltas:
    def test_higher_is_better_direction(self):
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 0.8}})
        pairs, _, _ = pair_cases(base, curr)
        assert regression_deltas(pairs, "rule_based") == [pytest.approx(0.2)]

    def test_lower_is_better_direction(self):
        base = _baseline({"a": {"deepeval.toxicity": 0.1}})
        curr = _baseline({"a": {"deepeval.toxicity": 0.4}})
        pairs, _, _ = pair_cases(base, curr)
        # toxicity went UP -> regression positive
        assert regression_deltas(pairs, "deepeval.toxicity") == [pytest.approx(0.3)]

    def test_pass_rate_deltas_from_passed_flags(self):
        base = _baseline({"a": {"rule_based": 1.0}, "b": {"rule_based": 1.0}}, passed={"a": True, "b": True})
        curr = _baseline({"a": {"rule_based": 1.0}, "b": {"rule_based": 1.0}}, passed={"a": True, "b": False})
        pairs, _, _ = pair_cases(base, curr)
        assert sorted(regression_deltas(pairs, "pass_rate")) == [0.0, 1.0]

    def test_cases_missing_the_metric_are_skipped(self):
        base = _baseline({"a": {"rule_based": 1.0}, "b": {"llm_judge": 0.9}})
        curr = _baseline({"a": {"rule_based": 0.9}, "b": {"llm_judge": 0.9}})
        pairs, _, _ = pair_cases(base, curr)
        assert len(regression_deltas(pairs, "rule_based")) == 1


class TestCompareMetrics:
    def test_identical_runs_show_no_regression(self):
        scores = {f"c{i}": {"rule_based": 0.9} for i in range(10)}
        base = _baseline(scores)
        curr = _baseline(scores)
        comparisons = compare_metrics(base, curr, GatePolicy())
        by_name = {c.metric: c for c in comparisons}
        assert by_name["rule_based"].regression == 0.0
        assert by_name["rule_based"].p_value == 1.0
        assert not by_name["rule_based"].breaches
        assert by_name["pass_rate"].gated is True
        assert by_name["rule_based"].gated is False

    def test_clear_gated_regression_breaches(self):
        base = _baseline({f"c{i}": {"rule_based": 0.9} for i in range(20)})
        curr = _baseline({f"c{i}": {"rule_based": 0.5} for i in range(20)})
        policy = GatePolicy(metrics={"rule_based": MetricPolicy(max_regression=0.1)})
        comparisons = compare_metrics(base, curr, policy)
        rule = next(c for c in comparisons if c.metric == "rule_based")
        assert rule.significant is True
        assert rule.breaches is True
        assert rule.n_cases == 20

    def test_ungated_regression_reported_but_never_breaches(self):
        base = _baseline({f"c{i}": {"rule_based": 0.9} for i in range(20)})
        curr = _baseline({f"c{i}": {"rule_based": 0.5} for i in range(20)})
        comparisons = compare_metrics(base, curr, GatePolicy())  # rule_based not in policy
        rule = next(c for c in comparisons if c.metric == "rule_based")
        assert rule.significant is True
        assert rule.breaches is False

    def test_small_effect_filtered_by_min_effect_size(self):
        base = _baseline({f"c{i}": {"rule_based": 0.90} for i in range(30)})
        curr = _baseline({f"c{i}": {"rule_based": 0.88} for i in range(30)})
        policy = GatePolicy(metrics={"rule_based": MetricPolicy(max_regression=0.01)})
        comparisons = compare_metrics(base, curr, policy)
        rule = next(c for c in comparisons if c.metric == "rule_based")
        # delta constante 0.02 es "significativo" pero < min_effect_size (0.05)
        assert rule.breaches is False
