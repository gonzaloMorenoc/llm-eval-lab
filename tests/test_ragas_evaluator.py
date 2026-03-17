"""Tests for RAGAS evaluator (mock-based, no API keys needed).

These tests verify the evaluator's metric resolution, configuration logic,
and error handling without calling the RAGAS API. The OPENAI_API_KEY env var
is patched to satisfy the validation check.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.runner.models import TestCase


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_openai_key(monkeypatch):
    """Provide a fake OPENAI_API_KEY so RagasEvaluator can be instantiated."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")


@pytest.fixture
def _patch_langchain():
    """Patch LangChain wrappers so we don't need real OpenAI clients."""
    with (
        patch("src.evaluators.ragas_evaluator.ChatOpenAI") as mock_chat,
        patch("src.evaluators.ragas_evaluator.OpenAIEmbeddings") as mock_emb,
        patch("src.evaluators.ragas_evaluator.LangchainLLMWrapper") as mock_llm_wrapper,
        patch("src.evaluators.ragas_evaluator.LangchainEmbeddingsWrapper") as mock_emb_wrapper,
    ):
        mock_llm_wrapper.return_value = MagicMock()
        mock_emb_wrapper.return_value = MagicMock()
        yield {
            "chat": mock_chat,
            "embeddings": mock_emb,
            "llm_wrapper": mock_llm_wrapper,
            "emb_wrapper": mock_emb_wrapper,
        }


@pytest.fixture
def evaluator(_patch_langchain):
    from src.evaluators.ragas_evaluator import RagasEvaluator

    return RagasEvaluator()


@pytest.fixture
def functional_case():
    return TestCase(
        id="ragas_func_001",
        category="functional",
        input="What is machine learning?",
        expected_behavior="Provides a clear definition of machine learning.",
        reference="Machine learning is a subset of AI that learns from data.",
        evaluation_type=["ragas"],
        ragas_metrics=["answer_relevancy", "factual_correctness"],
        severity="medium",
    )


@pytest.fixture
def no_reference_case():
    return TestCase(
        id="ragas_no_ref",
        category="functional",
        input="Explain Python",
        expected_behavior="Describes Python programming language.",
        reference=None,
        evaluation_type=["ragas"],
        ragas_metrics=["answer_relevancy"],
        severity="low",
    )


@pytest.fixture
def multi_turn_case():
    return TestCase(
        id="ragas_mt_001",
        category="multi_turn",
        input=[
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you!"},
            {"role": "user", "content": "What is my name?"},
        ],
        expected_behavior="Recalls the name Alice.",
        reference="Alice",
        evaluation_type=["ragas"],
        ragas_metrics=["answer_relevancy"],
        severity="high",
    )


# ── Initialization Tests ─────────────────────────────────────────────────────


class TestRagasEvaluatorInit:
    def test_name(self, evaluator):
        assert evaluator.name() == "ragas"

    def test_missing_api_key_raises_error(self, monkeypatch, _patch_langchain):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from src.evaluators.ragas_evaluator import RagasEvaluator

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            RagasEvaluator()

    def test_loads_thresholds_from_config(self, evaluator):
        assert isinstance(evaluator._thresholds, dict)
        assert evaluator._pass_threshold > 0

    def test_loads_default_metrics(self, evaluator):
        assert isinstance(evaluator._default_metrics_plain, list)
        assert isinstance(evaluator._default_metrics_rag, list)
        assert len(evaluator._default_metrics_plain) > 0
        assert len(evaluator._default_metrics_rag) > 0


# ── Metric Resolution Tests ──────────────────────────────────────────────────


class TestMetricResolution:
    def test_plain_mode_excludes_rag_metrics(self, evaluator, functional_case):
        metrics = evaluator._resolve_metrics(functional_case, retrieved_contexts=None)
        from src.evaluators.ragas_evaluator import _RAG_ONLY_METRICS

        for rag_metric in _RAG_ONLY_METRICS:
            assert rag_metric not in metrics

    def test_rag_mode_includes_faithfulness(self, evaluator):
        tc = TestCase(
            id="test",
            category="functional",
            input="test",
            expected_behavior="test",
            reference="ref",
            evaluation_type=["ragas"],
            ragas_metrics=["answer_relevancy", "faithfulness"],
            severity="low",
        )
        metrics = evaluator._resolve_metrics(tc, retrieved_contexts=["ctx"])
        assert "faithfulness" in metrics

    def test_no_reference_skips_reference_required(self, evaluator, no_reference_case):
        metrics = evaluator._resolve_metrics(no_reference_case, retrieved_contexts=None)
        from src.evaluators.ragas_evaluator import _REFERENCE_REQUIRED

        for ref_metric in _REFERENCE_REQUIRED:
            assert ref_metric not in metrics

    def test_uses_test_case_ragas_metrics(self, evaluator, functional_case):
        metrics = evaluator._resolve_metrics(functional_case, retrieved_contexts=None)
        assert "answer_relevancy" in metrics
        assert "factual_correctness" in metrics

    def test_uses_defaults_when_no_ragas_metrics(self, evaluator):
        tc = TestCase(
            id="test",
            category="functional",
            input="test",
            expected_behavior="test",
            reference="ref",
            evaluation_type=["ragas"],
            ragas_metrics=None,
            severity="low",
        )
        metrics = evaluator._resolve_metrics(tc, retrieved_contexts=None)
        assert len(metrics) > 0

    def test_empty_metrics_after_filtering(self, evaluator):
        tc = TestCase(
            id="test",
            category="functional",
            input="test",
            expected_behavior="test",
            reference=None,
            evaluation_type=["ragas"],
            ragas_metrics=["factual_correctness"],  # Needs reference
            severity="low",
        )
        metrics = evaluator._resolve_metrics(tc, retrieved_contexts=None)
        assert metrics == []


# ── Build Metric Tests ────────────────────────────────────────────────────────


class TestBuildMetric:
    def test_builds_answer_relevancy(self, evaluator):
        metric = evaluator._build_metric("answer_relevancy")
        assert metric is not None

    def test_builds_factual_correctness(self, evaluator):
        metric = evaluator._build_metric("factual_correctness")
        assert metric is not None

    def test_builds_faithfulness(self, evaluator):
        metric = evaluator._build_metric("faithfulness")
        assert metric is not None

    def test_builds_bleu_score(self, evaluator):
        try:
            metric = evaluator._build_metric("bleu_score")
            assert metric is not None
        except ImportError:
            pytest.skip("sacrebleu not installed")

    def test_builds_rouge_score(self, evaluator):
        try:
            metric = evaluator._build_metric("rouge_score")
            assert metric is not None
        except ImportError:
            pytest.skip("rouge_score not installed")

    def test_returns_none_for_unknown(self, evaluator):
        metric = evaluator._build_metric("nonexistent_metric")
        assert metric is None


# ── Evaluate Tests ────────────────────────────────────────────────────────────


class TestEvaluate:
    async def test_no_applicable_metrics_returns_pass(self, evaluator):
        tc = TestCase(
            id="test",
            category="functional",
            input="test",
            expected_behavior="test",
            reference=None,
            evaluation_type=["ragas"],
            ragas_metrics=["factual_correctness"],  # Needs reference, will be filtered out
            severity="low",
        )
        result = await evaluator.evaluate(tc, response="some response", retrieved_contexts=None)
        assert result.passed is True
        assert "No applicable" in result.reason

    async def test_metric_error_is_captured(self, evaluator, functional_case):
        """When a metric raises an exception, it should be captured in errors."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(side_effect=RuntimeError("API error"))

        with patch.object(evaluator, "_build_metric", return_value=mock_metric):
            result = await evaluator.evaluate(
                functional_case, response="Machine learning is AI.", retrieved_contexts=None
            )
        assert "errors" in result.details
        assert len(result.details["errors"]) > 0

    async def test_successful_metric_scoring(self, evaluator, functional_case):
        """When metrics succeed, scores and thresholds should be computed."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.85)

        with patch.object(evaluator, "_build_metric", return_value=mock_metric):
            result = await evaluator.evaluate(
                functional_case, response="Machine learning is AI.", retrieved_contexts=None
            )
        assert result.score is not None
        assert result.score > 0
        assert "metric_scores" in result.details
        assert "threshold_results" in result.details

    async def test_below_threshold_fails(self, evaluator, functional_case):
        """Score below threshold should result in failed evaluation."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.1)

        with patch.object(evaluator, "_build_metric", return_value=mock_metric):
            result = await evaluator.evaluate(
                functional_case, response="Irrelevant answer", retrieved_contexts=None
            )
        assert result.passed is False
        assert "FAIL" in result.reason

    async def test_above_threshold_passes(self, evaluator, functional_case):
        """Score above threshold should result in passed evaluation."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.95)

        with patch.object(evaluator, "_build_metric", return_value=mock_metric):
            result = await evaluator.evaluate(
                functional_case, response="Great answer", retrieved_contexts=None
            )
        assert result.passed is True
        assert "PASS" in result.reason

    async def test_multi_turn_uses_last_user_message(self, evaluator, multi_turn_case):
        """Multi-turn test cases should extract the last user message."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.9)

        with patch.object(evaluator, "_build_metric", return_value=mock_metric):
            result = await evaluator.evaluate(
                multi_turn_case, response="Your name is Alice", retrieved_contexts=None
            )
        assert result.evaluator == "ragas"
        assert result.score is not None

    async def test_all_metrics_error_still_returns_result(self, evaluator, functional_case):
        """Even if all metrics fail, the evaluator should return a result (not crash)."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(side_effect=Exception("Total failure"))

        with patch.object(evaluator, "_build_metric", return_value=mock_metric):
            result = await evaluator.evaluate(
                functional_case, response="test", retrieved_contexts=None
            )
        assert result.score is None or result.score == 0.0
        assert len(result.details["errors"]) > 0

    async def test_rag_mode_with_contexts(self, evaluator):
        """RAG mode should pass retrieved_contexts to the sample."""
        tc = TestCase(
            id="rag_test",
            category="functional",
            input="What is ML?",
            expected_behavior="Explains ML",
            reference="ML is a subset of AI",
            evaluation_type=["ragas"],
            ragas_metrics=["answer_relevancy"],
            severity="low",
        )
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.88)

        with patch.object(evaluator, "_build_metric", return_value=mock_metric):
            result = await evaluator.evaluate(
                tc,
                response="ML is AI that learns from data",
                retrieved_contexts=["ML is a subset of AI focusing on data."],
            )
        assert result.score is not None
        assert result.evaluator == "ragas"
