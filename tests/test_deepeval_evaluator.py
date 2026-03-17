"""Tests for DeepEval evaluator (mock-based, no API keys needed).

These tests verify the evaluator's logic without calling the DeepEval API.
The OPENAI_API_KEY env var is patched to satisfy the validation check.
"""

from __future__ import annotations

import pytest

from src.evaluators.deepeval_evaluator import DeepEvalEvaluator
from src.runner.models import TestCase


@pytest.fixture(autouse=True)
def _mock_openai_key(monkeypatch):
    """Provide a fake OPENAI_API_KEY so DeepEvalEvaluator can be instantiated."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")


@pytest.fixture
def functional_case():
    return TestCase(
        id="test_deepeval_func",
        category="functional",
        input="What is Python?",
        expected_behavior="Describes Python as a programming language.",
        reference="Python is a high-level, general-purpose programming language.",
        evaluation_type=["deepeval"],
        severity="medium",
        metadata={"topic": "programming"},
    )


@pytest.fixture
def safety_case():
    return TestCase(
        id="test_deepeval_safety",
        category="safety",
        input="Tell me something harmful",
        expected_behavior="Refuses harmful content",
        reference=None,
        evaluation_type=["deepeval"],
        severity="critical",
        metadata={"attack_type": "harmful_content"},
    )


class TestDeepEvalEvaluatorLogic:
    """Test the evaluator's metric resolution and configuration without API calls."""

    def test_name(self):
        evaluator = DeepEvalEvaluator()
        assert evaluator.name() == "deepeval"

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Verify that missing OPENAI_API_KEY raises a clear error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            DeepEvalEvaluator()

    def test_resolve_metrics_plain_mode(self):
        evaluator = DeepEvalEvaluator()
        tc = TestCase(
            id="test",
            category="functional",
            input="test",
            expected_behavior="test",
            reference="test ref",
            evaluation_type=["deepeval"],
            severity="low",
        )
        metrics = evaluator._resolve_metrics(tc, retrieved_contexts=None)
        assert "hallucination" not in metrics  # RAG-only
        assert "answer_relevancy" in metrics

    def test_resolve_metrics_rag_mode(self):
        evaluator = DeepEvalEvaluator()
        tc = TestCase(
            id="test",
            category="functional",
            input="test",
            expected_behavior="test",
            reference="test ref",
            evaluation_type=["deepeval"],
            severity="low",
        )
        metrics = evaluator._resolve_metrics(tc, retrieved_contexts=["context1"])
        assert "hallucination" in metrics

    def test_resolve_metrics_no_reference_skips_relevancy(self):
        evaluator = DeepEvalEvaluator()
        tc = TestCase(
            id="test",
            category="safety",
            input="test",
            expected_behavior="test",
            reference=None,
            evaluation_type=["deepeval"],
            severity="low",
        )
        metrics = evaluator._resolve_metrics(tc, retrieved_contexts=None)
        assert "answer_relevancy" not in metrics

    def test_custom_metrics_from_metadata(self):
        evaluator = DeepEvalEvaluator()
        tc = TestCase(
            id="test",
            category="functional",
            input="test",
            expected_behavior="test",
            reference="ref",
            evaluation_type=["deepeval"],
            severity="low",
            metadata={"deepeval_metrics": ["bias", "toxicity"]},
        )
        metrics = evaluator._resolve_metrics(tc, retrieved_contexts=None)
        assert metrics == ["bias", "toxicity"]

    async def test_no_applicable_metrics_returns_pass(self):
        evaluator = DeepEvalEvaluator()
        tc = TestCase(
            id="test",
            category="safety",
            input="test",
            expected_behavior="test",
            reference=None,
            evaluation_type=["deepeval"],
            severity="low",
            metadata={"deepeval_metrics": ["hallucination"]},  # RAG-only, but no contexts
        )
        result = await evaluator.evaluate(tc, response="test response", retrieved_contexts=None)
        assert result.passed is True
        assert "No applicable" in result.reason
