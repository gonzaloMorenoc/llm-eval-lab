"""Tests for runner error scenarios — retry logic, malformed datasets, evaluator failures.

These tests cover the error paths that test_runner.py doesn't exercise:
retry with backoff, dataset validation, evaluator crashes, and edge cases.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.chatbots.base import BaseChatbot, ChatbotResponse
from src.chatbots.mock_adapter import MockChatbot
from src.evaluators.base import BaseEvaluator
from src.evaluators.rule_based import RuleBasedEvaluator
from src.runner.models import EvaluationResult, TestCase
from src.runner.runner import EvalRunner, load_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def functional_case():
    return TestCase(
        id="err_func_001",
        category="functional",
        input="What is Python?",
        expected_behavior="Describes Python.",
        reference="Python is a programming language.",
        evaluation_type=["rule_based"],
        severity="medium",
    )


@pytest.fixture
def mock_chatbot():
    return MockChatbot(latency_range=(10.0, 20.0))


class CrashingEvaluator(BaseEvaluator):
    """An evaluator that always raises an exception."""

    def name(self) -> str:
        return "crashing"

    async def evaluate(self, test_case, response, retrieved_contexts=None, latency_ms=0.0):
        raise RuntimeError("Evaluator exploded!")


class FailingChatbot(BaseChatbot):
    """A chatbot that always fails."""

    def __init__(self, error_type: str = "generic"):
        self._error_type = error_type
        self._call_count = 0

    async def complete(self, messages: list[dict], **kwargs) -> ChatbotResponse:
        self._call_count += 1
        if self._error_type == "rate_limit":
            raise Exception("429 Too Many Requests")
        if self._error_type == "timeout":
            raise Exception("Connection timeout")
        raise Exception("Generic API error")

    def get_id(self) -> str:
        return "failing/chatbot"


class RecoveringChatbot(BaseChatbot):
    """A chatbot that fails N times then succeeds."""

    def __init__(self, fail_count: int = 2):
        self._fail_count = fail_count
        self._call_count = 0

    async def complete(self, messages: list[dict], **kwargs) -> ChatbotResponse:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise Exception("429 Rate limit exceeded")
        return ChatbotResponse(
            content="Recovered response",
            retrieved_contexts=None,
            provider="mock",
            model="recovering-v1",
            latency_ms=100.0,
        )

    def get_id(self) -> str:
        return "mock/recovering-v1"


# ── Dataset Loading Error Tests ───────────────────────────────────────────────


class TestDatasetLoadingErrors:
    def test_malformed_json_reports_line_number(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write('{"id": "ok", "category": "functional", "input": "test", '
                  '"expected_behavior": "test", "evaluation_type": ["rule_based"]}\n')
        tmp.write("this is not json\n")
        tmp.close()
        try:
            with pytest.raises(ValueError, match="line 2"):
                load_dataset(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_invalid_test_case_reports_line_number(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write('{"id": "bad", "category": "invalid_category", "input": "test", '
                  '"expected_behavior": "test", "evaluation_type": ["rule_based"]}\n')
        tmp.close()
        try:
            with pytest.raises(ValueError, match="line 1"):
                load_dataset(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_empty_file_returns_empty_list(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write("\n\n  \n")
        tmp.close()
        try:
            cases = load_dataset(tmp.name)
            assert cases == []
        finally:
            os.unlink(tmp.name)

    def test_valid_file_loads_correctly(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        entry = {
            "id": "valid_001",
            "category": "functional",
            "input": "What is AI?",
            "expected_behavior": "Explains AI.",
            "evaluation_type": ["rule_based"],
            "severity": "low",
        }
        tmp.write(json.dumps(entry) + "\n")
        tmp.close()
        try:
            cases = load_dataset(tmp.name)
            assert len(cases) == 1
            assert cases[0].id == "valid_001"
        finally:
            os.unlink(tmp.name)


# ── Retry Logic Tests ────────────────────────────────────────────────────────


class TestRetryLogic:
    async def test_generic_error_no_retry(self, functional_case):
        """Non-retryable errors should fail immediately without retrying."""
        chatbot = FailingChatbot(error_type="generic")
        runner = EvalRunner(
            chatbot=chatbot,
            evaluators={"rule_based": RuleBasedEvaluator()},
        )
        content, contexts, latency, error = await runner._call_chatbot(
            [{"role": "user", "content": "test"}]
        )
        assert error is not None
        assert "Generic API error" in error
        assert chatbot._call_count == 1  # No retry

    async def test_rate_limit_retries(self, functional_case):
        """Rate limit errors should trigger retries."""
        chatbot = FailingChatbot(error_type="rate_limit")
        runner = EvalRunner(
            chatbot=chatbot,
            evaluators={"rule_based": RuleBasedEvaluator()},
            config={"runner": {"max_concurrent": 1, "retry_attempts": 3, "retry_backoff_base": 0.01}},
        )
        content, contexts, latency, error = await runner._call_chatbot(
            [{"role": "user", "content": "test"}]
        )
        assert error is not None
        assert chatbot._call_count == 3  # All retries exhausted

    async def test_timeout_retries(self, functional_case):
        """Timeout errors should trigger retries."""
        chatbot = FailingChatbot(error_type="timeout")
        runner = EvalRunner(
            chatbot=chatbot,
            evaluators={},
            config={"runner": {"max_concurrent": 1, "retry_attempts": 2, "retry_backoff_base": 0.01}},
        )
        content, contexts, latency, error = await runner._call_chatbot(
            [{"role": "user", "content": "test"}]
        )
        assert error is not None
        assert chatbot._call_count == 2

    async def test_recovery_after_retries(self, functional_case):
        """Chatbot that recovers after N failures should succeed."""
        chatbot = RecoveringChatbot(fail_count=2)
        runner = EvalRunner(
            chatbot=chatbot,
            evaluators={"rule_based": RuleBasedEvaluator()},
            config={"runner": {"max_concurrent": 1, "retry_attempts": 3, "retry_backoff_base": 0.01}},
        )
        content, contexts, latency, error = await runner._call_chatbot(
            [{"role": "user", "content": "test"}]
        )
        assert error is None
        assert content == "Recovered response"
        assert chatbot._call_count == 3  # 2 failures + 1 success


# ── Evaluator Error Tests ────────────────────────────────────────────────────


class TestEvaluatorErrors:
    async def test_crashing_evaluator_doesnt_crash_run(self, functional_case):
        """If an evaluator crashes, the run should continue with an error result."""
        tc = functional_case.model_copy()
        tc.evaluation_type = ["crashing"]

        runner = EvalRunner(
            chatbot=MockChatbot(latency_range=(10.0, 20.0)),
            evaluators={"crashing": CrashingEvaluator()},
        )
        summary = await runner.run([tc])

        assert summary.total == 1
        result = summary.results[0]
        assert result.overall_passed is False

        # The error should be captured in the evaluation
        crash_eval = [e for e in result.evaluations if e.evaluator == "crashing"]
        assert len(crash_eval) == 1
        assert "Evaluator error" in crash_eval[0].reason
        assert "exploded" in crash_eval[0].reason.lower()

    async def test_missing_evaluator_is_skipped(self, functional_case):
        """If evaluation_type references a non-registered evaluator, skip it."""
        tc = functional_case.model_copy()
        tc.evaluation_type = ["nonexistent", "rule_based"]

        runner = EvalRunner(
            chatbot=MockChatbot(latency_range=(10.0, 20.0)),
            evaluators={"rule_based": RuleBasedEvaluator()},
        )
        summary = await runner.run([tc])

        result = summary.results[0]
        evaluator_names = [e.evaluator for e in result.evaluations]
        assert "nonexistent" not in evaluator_names
        assert "rule_based" in evaluator_names


# ── Edge Case Tests ───────────────────────────────────────────────────────────


class TestRunnerEdgeCases:
    async def test_empty_test_cases_list(self, mock_chatbot):
        """Running with an empty list should return a valid summary."""
        runner = EvalRunner(
            chatbot=mock_chatbot,
            evaluators={"rule_based": RuleBasedEvaluator()},
        )
        summary = await runner.run([])
        assert summary.total == 0
        assert summary.passed == 0
        assert summary.failed == 0

    async def test_chatbot_error_produces_error_result(self, functional_case):
        """If chatbot fails completely, the result should have an error field."""
        chatbot = FailingChatbot(error_type="generic")
        runner = EvalRunner(
            chatbot=chatbot,
            evaluators={"rule_based": RuleBasedEvaluator()},
        )
        summary = await runner.run([functional_case])

        assert summary.errors == 1
        result = summary.results[0]
        assert result.error is not None
        assert len(result.evaluations) == 0  # No evaluations run on error

    async def test_error_preserves_traceback(self, functional_case):
        """Error messages should contain the exception type for debugging."""
        chatbot = FailingChatbot(error_type="generic")
        runner = EvalRunner(
            chatbot=chatbot,
            evaluators={},
        )
        content, contexts, latency, error = await runner._call_chatbot(
            [{"role": "user", "content": "test"}]
        )
        assert "Exception:" in error
        assert "Generic API error" in error

    async def test_single_concurrent(self, functional_case):
        """Runner with max_concurrent=1 should still work."""
        runner = EvalRunner(
            chatbot=MockChatbot(latency_range=(1.0, 2.0)),
            evaluators={"rule_based": RuleBasedEvaluator()},
            config={"runner": {"max_concurrent": 1, "retry_attempts": 1, "retry_backoff_base": 1}},
        )
        cases = [functional_case.model_copy() for _ in range(3)]
        for i, tc in enumerate(cases):
            tc.id = f"concurrent_{i}"

        summary = await runner.run(cases)
        assert summary.total == 3
