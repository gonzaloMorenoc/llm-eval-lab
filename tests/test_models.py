"""Tests for Pydantic models."""

from __future__ import annotations

import pytest

from src.runner.models import (
    CategoryStats,
    EvaluationResult,
    RunSummary,
    TestCase,
    TestResult,
)


class TestTestCase:
    def test_create_functional_case(self):
        tc = TestCase(
            id="test_001",
            category="functional",
            input="What is AI?",
            expected_behavior="Explains AI",
            evaluation_type=["rule_based"],
            severity="medium",
        )
        assert tc.id == "test_001"
        assert tc.reference is None
        assert tc.ragas_metrics is None
        assert tc.metadata == {}

    def test_create_with_all_fields(self):
        tc = TestCase(
            id="test_002",
            category="safety",
            input="Hack this",
            expected_behavior="Refuses",
            reference="Should refuse",
            evaluation_type=["safety", "rule_based", "deepeval"],
            ragas_metrics=["answer_relevancy"],
            severity="critical",
            metadata={"attack_type": "prompt_injection"},
        )
        assert tc.category == "safety"
        assert "deepeval" in tc.evaluation_type

    def test_multi_turn_input(self):
        tc = TestCase(
            id="test_003",
            category="multi_turn",
            input=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ],
            expected_behavior="Responds appropriately",
            evaluation_type=["rule_based"],
            severity="low",
        )
        assert isinstance(tc.input, list)
        assert len(tc.input) == 3

    def test_invalid_category_raises(self):
        with pytest.raises(Exception):
            TestCase(
                id="bad",
                category="nonexistent",
                input="test",
                expected_behavior="test",
                evaluation_type=["rule_based"],
            )

    def test_invalid_evaluation_type_raises(self):
        with pytest.raises(Exception):
            TestCase(
                id="bad",
                category="functional",
                input="test",
                expected_behavior="test",
                evaluation_type=["nonexistent_evaluator"],
            )


class TestEvaluationResult:
    def test_create_passing_result(self):
        result = EvaluationResult(evaluator="rule_based", passed=True, score=0.95, reason="All checks passed")
        assert result.passed is True
        assert result.score == 0.95

    def test_create_failing_result(self):
        result = EvaluationResult(evaluator="safety", passed=False, score=0.25, reason="Injection detected")
        assert result.passed is False

    def test_defaults(self):
        result = EvaluationResult(evaluator="test", passed=True)
        assert result.score is None
        assert result.reason == ""
        assert result.details == {}


class TestRunSummary:
    def test_has_deepeval_aggregate(self):
        summary = RunSummary(
            run_id="test123",
            timestamp="2025-01-01T00:00:00Z",
            chatbot_id="mock/test",
            chatbot_mode="plain",
        )
        assert hasattr(summary, "deepeval_aggregate")
        assert summary.deepeval_aggregate == {}

    def test_defaults(self):
        summary = RunSummary(
            run_id="test123",
            timestamp="2025-01-01T00:00:00Z",
            chatbot_id="mock/test",
            chatbot_mode="plain",
        )
        assert summary.total == 0
        assert summary.pass_rate == 0.0
        assert summary.results == []
        assert summary.ragas_aggregate == {}
