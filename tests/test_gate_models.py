"""Tests for the regression-gate Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.gate.models import (
    LOWER_IS_BETTER_METRICS,
    BaselineCase,
    BaselineFile,
    GatePolicy,
    GateVerdict,
    HardRules,
    MetricPolicy,
)


def _case(**overrides) -> BaselineCase:
    data = {
        "id": "func_001",
        "category": "functional",
        "severity": "medium",
        "passed": True,
        "pass_samples": [True],
        "metrics": {"rule_based": 1.0, "ragas.answer_relevancy": 0.8},
        "metric_variance": {"rule_based": 0.0},
        "latency_ms_mean": 120.5,
    }
    data.update(overrides)
    return BaselineCase(**data)


class TestBaselineModels:
    def test_baseline_case_holds_flattened_metrics(self):
        case = _case()
        assert case.metrics["ragas.answer_relevancy"] == 0.8
        assert case.flakiness == 0.0  # default

    def test_baseline_case_rejects_invalid_severity(self):
        with pytest.raises(ValidationError):
            _case(severity="catastrophic")

    def test_baseline_file_round_trips_through_json(self):
        baseline = BaselineFile(
            run_ids=["r1", "r2"],
            timestamp="2026-08-14T00:00:00+00:00",
            chatbot_id="mock/mock-plain-v1",
            chatbot_mode="plain",
            dataset_hash="abc123",
            metric_set=["pass_rate", "rule_based"],
            samples=2,
            cases=[_case()],
        )
        restored = BaselineFile.model_validate_json(baseline.model_dump_json())
        assert restored == baseline
        assert restored.schema_version == 1

    def test_baseline_file_rejects_invalid_mode(self):
        with pytest.raises(ValidationError):
            BaselineFile(
                run_ids=["r1"],
                timestamp="t",
                chatbot_id="x",
                chatbot_mode="hybrid",
                dataset_hash="h",
                metric_set=[],
                cases=[],
            )


class TestGatePolicy:
    def test_defaults_match_spec(self):
        policy = GatePolicy()
        assert policy.significance_level == 0.05
        assert policy.min_effect_size == 0.05
        assert policy.n_resamples == 10_000
        assert policy.seed == 42
        assert policy.metrics == {"pass_rate": MetricPolicy(max_regression=0.05)}
        assert policy.hard_rules == HardRules(no_new_critical_failures=True, max_flakiness=0.3)
        assert policy.new_cases == "report_only"

    def test_new_cases_rejects_unknown_value(self):
        with pytest.raises(ValidationError):
            GatePolicy(new_cases="ignore")


class TestConstants:
    def test_lower_is_better_covers_deepeval_inverse_metrics(self):
        assert LOWER_IS_BETTER_METRICS == frozenset({"deepeval.hallucination", "deepeval.bias", "deepeval.toxicity"})


class TestGateVerdict:
    def test_verdict_minimal_construction(self):
        verdict = GateVerdict(
            passed=True,
            comparisons=[],
            hard_rule_violations=[],
            missing_gated_metrics=[],
            new_case_ids=[],
            removed_case_ids=[],
            mean_flakiness=0.0,
            samples=1,
        )
        assert verdict.passed is True
