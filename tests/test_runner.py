"""Tests for the evaluation runner."""

from __future__ import annotations

import pytest

from src.chatbots.mock_adapter import MockChatbot, MockRAGChatbot
from src.evaluators.rule_based import RuleBasedEvaluator
from src.evaluators.safety import SafetyEvaluator
from src.runner.models import RunSummary, TestCase, TestResult
from src.runner.runner import EvalRunner, load_all_datasets, load_dataset


class TestDatasetLoading:
    def test_load_all_datasets(self):
        cases = load_all_datasets()
        assert len(cases) > 0
        assert all(isinstance(c, TestCase) for c in cases)

    def test_all_categories_present(self):
        cases = load_all_datasets()
        categories = {c.category for c in cases}
        assert "functional" in categories
        assert "safety" in categories
        assert "multi_turn" in categories
        assert "regression" in categories

    def test_load_functional_dataset(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "datasets", "functional.jsonl")
        cases = load_dataset(os.path.abspath(path))
        assert len(cases) >= 5
        assert all(c.category == "functional" for c in cases)


class TestEvalRunner:
    @pytest.fixture
    def plain_evaluators(self):
        return {
            "rule_based": RuleBasedEvaluator(),
            "safety": SafetyEvaluator(),
        }

    async def test_run_plain_mode(self, mock_chatbot: MockChatbot, plain_evaluators: dict):
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=plain_evaluators)
        cases = load_all_datasets()
        summary = await runner.run(cases)

        assert isinstance(summary, RunSummary)
        assert summary.chatbot_mode == "plain"
        assert summary.total == len(cases)
        assert summary.total == summary.passed + summary.failed + summary.errors
        assert 0 <= summary.pass_rate <= 1

    async def test_run_rag_mode(self, mock_rag_chatbot: MockRAGChatbot, plain_evaluators: dict):
        runner = EvalRunner(chatbot=mock_rag_chatbot, evaluators=plain_evaluators)
        cases = load_all_datasets()
        summary = await runner.run(cases)

        assert isinstance(summary, RunSummary)
        assert summary.chatbot_mode == "rag"
        assert summary.total == len(cases)

        # RAG mode results should have retrieved_contexts
        for result in summary.results:
            assert result.chatbot_mode == "rag"
            assert result.retrieved_contexts is not None

    async def test_run_produces_category_stats(self, mock_chatbot: MockChatbot, plain_evaluators: dict):
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=plain_evaluators)
        cases = load_all_datasets()
        summary = await runner.run(cases)

        assert len(summary.by_category) > 0
        for cat, stats in summary.by_category.items():
            assert stats.total > 0
            assert stats.total == stats.passed + stats.failed

    async def test_run_single_case(self, mock_chatbot: MockChatbot, plain_evaluators: dict, functional_test_case: TestCase):
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=plain_evaluators)
        summary = await runner.run([functional_test_case])

        assert summary.total == 1
        assert len(summary.results) == 1
        result = summary.results[0]
        assert isinstance(result, TestResult)
        assert result.response
        assert len(result.evaluations) > 0

    async def test_plain_vs_rag_same_dataset(
        self,
        mock_chatbot: MockChatbot,
        mock_rag_chatbot: MockRAGChatbot,
        plain_evaluators: dict,
    ):
        """Same dataset works for both plain and RAG mode."""
        cases = load_all_datasets()

        runner_plain = EvalRunner(chatbot=mock_chatbot, evaluators=plain_evaluators)
        summary_plain = await runner_plain.run(cases)

        runner_rag = EvalRunner(chatbot=mock_rag_chatbot, evaluators=plain_evaluators)
        summary_rag = await runner_rag.run(cases)

        assert summary_plain.chatbot_mode == "plain"
        assert summary_rag.chatbot_mode == "rag"
        assert summary_plain.total == summary_rag.total
