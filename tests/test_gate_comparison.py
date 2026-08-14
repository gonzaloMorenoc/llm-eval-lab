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
from src.gate.models import BaselineCase, BaselineFile, GatePolicy, MetricPolicy
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

    def test_shared_pass_rate_is_not_enough(self):
        # Used to pass: build_baseline always seeds metric_set with pass_rate, so the
        # intersection was never empty and a shrunken evaluator set slipped through.
        # pass_rate is derived from every evaluator, so dropping one changes its meaning.
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"llm_judge": 0.8}})
        assert "pass_rate" in set(base.metric_set) & set(curr.metric_set)
        with pytest.raises(CompatibilityError, match="rule_based"):
            validate_compatibility(base, curr)

    def test_metric_dropped_by_current_run_raises_and_names_it(self):
        # The real CI misconfiguration: baseline built with rule_based+safety, current
        # run only rule_based (e.g. a fork PR without OPENAI_API_KEY dropping RAGAS).
        base = _baseline({"a": {"rule_based": 1.0, "safety": 1.0}})
        curr = _baseline({"a": {"rule_based": 1.0}})
        with pytest.raises(CompatibilityError) as excinfo:
            validate_compatibility(base, curr)
        message = str(excinfo.value)
        assert "safety" in message
        assert "rule_based" not in message  # only the *missing* metric is named

    def test_extra_metric_in_current_run_is_permitted(self):
        # The opposite direction fails safe: an added evaluator can only make pass_rate
        # stricter, which surfaces as exit 1 (a regression someone looks at). An
        # evaluator that crashes on a single case must not become spurious exit-2 noise.
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 1.0, "safety": 1.0}})
        validate_compatibility(base, curr)  # no raise

    def test_zero_shared_metrics_raises(self):
        # F1: Test the defensively correct check against hand-edited or corrupted baseline JSON
        # with disjoint, pass_rate-free metric_set values.
        base_case = BaselineCase(
            id="a",
            category="functional",
            severity="medium",
            passed=True,
            pass_samples=[True],
            metrics={"rule_based": 1.0},
        )
        curr_case = BaselineCase(
            id="a",
            category="functional",
            severity="medium",
            passed=True,
            pass_samples=[True],
            metrics={"llm_judge": 0.8},
        )
        base = BaselineFile(
            schema_version=1,
            run_ids=["run1"],
            timestamp="2026-08-14T00:00:00Z",
            chatbot_id="test",
            chatbot_mode="plain",
            dataset_hash="hash1",
            metric_set=["rule_based"],  # no pass_rate
            cases=[base_case],
        )
        curr = BaselineFile(
            schema_version=1,
            run_ids=["run2"],
            timestamp="2026-08-14T00:00:00Z",
            chatbot_id="test",
            chatbot_mode="plain",
            dataset_hash="hash1",
            metric_set=["llm_judge"],  # disjoint, no pass_rate
            cases=[curr_case],
        )
        with pytest.raises(CompatibilityError, match="No shared metrics"):
            validate_compatibility(base, curr)


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

    def test_metric_missing_on_baseline_is_skipped(self):
        # F2: Metric absent in baseline, present in current → case skipped
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 0.9, "llm_judge": 0.8}})
        pairs, _, _ = pair_cases(base, curr)
        assert len(regression_deltas(pairs, "llm_judge")) == 0

    def test_metric_missing_on_current_is_skipped(self):
        # F2: Metric present in baseline, absent in current → case skipped
        base = _baseline({"a": {"rule_based": 1.0, "llm_judge": 0.9}})
        curr = _baseline({"a": {"rule_based": 0.8}})
        pairs, _, _ = pair_cases(base, curr)
        assert len(regression_deltas(pairs, "llm_judge")) == 0


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

    def test_large_effect_not_significant_does_not_breach(self):
        # F3a: isolate `significant` check — all other conjuncts satisfied but not significant
        # Deltas: [1.0, -0.7, 0.9, -0.8, 0.6, -0.4] → mean ~0.1, huge variance → not significant
        base = _baseline(
            {
                "c0": {"rule_based": 1.0},
                "c1": {"rule_based": 0.3},
                "c2": {"rule_based": 0.9},
                "c3": {"rule_based": 0.2},
                "c4": {"rule_based": 0.6},
                "c5": {"rule_based": 0.6},
            }
        )
        curr = _baseline(
            {
                "c0": {"rule_based": 0.0},
                "c1": {"rule_based": 1.0},
                "c2": {"rule_based": 0.0},
                "c3": {"rule_based": 1.0},
                "c4": {"rule_based": 0.0},
                "c5": {"rule_based": 1.0},
            }
        )
        # Set max_regression LOW (0.01) so that conjunct is satisfied: 0.1 > 0.01 = True
        policy = GatePolicy(metrics={"rule_based": MetricPolicy(max_regression=0.01)})
        comparisons = compare_metrics(base, curr, policy)
        rule = next(c for c in comparisons if c.metric == "rule_based")
        assert rule.significant is False  # Not statistically significant despite large regression
        assert rule.regression >= 0.05  # Above min_effect_size (0.05)
        assert rule.breaches is False  # Blocked only by lack of significance

    def test_significant_but_below_max_regression_does_not_breach(self):
        # F3b: isolate `> max_regression` check — all other conjuncts satisfied but not > max_regression
        # Constant delta 0.2 across 20 cases: significant, above min_effect_size, but below high max_regression
        base = _baseline({f"c{i}": {"rule_based": 0.8} for i in range(20)})
        curr = _baseline({f"c{i}": {"rule_based": 0.6} for i in range(20)})
        # Set max_regression HIGH (0.5) so regression (0.2) does not exceed it: 0.2 > 0.5 = False
        policy = GatePolicy(metrics={"rule_based": MetricPolicy(max_regression=0.5)})
        comparisons = compare_metrics(base, curr, policy)
        rule = next(c for c in comparisons if c.metric == "rule_based")
        assert rule.significant is True  # Constant deltas, 20 cases → definitely significant
        assert rule.regression == pytest.approx(0.2)  # Above min_effect_size (0.05)
        assert rule.regression < 0.5  # Below max_regression
        assert rule.breaches is False  # Blocked only by not exceeding max_regression

    def test_metric_in_both_sets_but_no_comparable_cases_omitted(self):
        # F4: Metric present in both metric_sets but no paired cases carry it → omitted from output
        # Construct: baseline has rule_based on cases a,b; current has llm_judge on same cases
        # When paired, no case has rule_based on both sides, so rule_based contributes zero deltas
        base_a = BaselineCase(
            id="a",
            category="functional",
            severity="medium",
            passed=True,
            pass_samples=[True],
            metrics={"rule_based": 0.9},
        )
        base_b = BaselineCase(
            id="b",
            category="functional",
            severity="medium",
            passed=False,  # Ensure pass_rate has a delta
            pass_samples=[False],
            metrics={"rule_based": 0.85},
        )
        curr_a = BaselineCase(
            id="a",
            category="functional",
            severity="medium",
            passed=True,
            pass_samples=[True],
            metrics={"llm_judge": 0.8},
        )
        curr_b = BaselineCase(
            id="b",
            category="functional",
            severity="medium",
            passed=True,  # Different pass status for delta
            pass_samples=[True],
            metrics={"llm_judge": 0.75},
        )
        base = BaselineFile(
            schema_version=1,
            run_ids=["run1"],
            timestamp="2026-08-14T00:00:00Z",
            chatbot_id="test",
            chatbot_mode="plain",
            dataset_hash="hash1",
            metric_set=["pass_rate", "rule_based", "llm_judge"],
            cases=[base_a, base_b],
        )
        curr = BaselineFile(
            schema_version=1,
            run_ids=["run2"],
            timestamp="2026-08-14T00:00:00Z",
            chatbot_id="test",
            chatbot_mode="plain",
            dataset_hash="hash1",
            metric_set=["pass_rate", "rule_based", "llm_judge"],
            cases=[curr_a, curr_b],
        )
        comparisons = compare_metrics(base, curr, GatePolicy())
        by_name = {c.metric: c for c in comparisons}
        # rule_based is in both metric_sets but no paired case carries it on both sides → zero deltas → omitted
        # llm_judge is in both metric_sets but no paired case carries it on both sides → zero deltas → omitted
        # pass_rate has deltas from b changing from False to True
        assert "rule_based" not in by_name
        assert "llm_judge" not in by_name
        assert "pass_rate" in by_name
