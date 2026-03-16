"""Tests for the evaluation runner."""

from __future__ import annotations

import pytest

from src.chatbots.mock_adapter import MockChatbot, MockRAGChatbot
from src.evaluators.rule_based import RuleBasedEvaluator
from src.evaluators.safety import SafetyEvaluator
from src.evaluators.consistency import ConsistencyEvaluator
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
        assert len(cases) >= 10  # Expanded dataset
        assert all(c.category == "functional" for c in cases)

    def test_load_safety_dataset(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "datasets", "safety.jsonl")
        cases = load_dataset(os.path.abspath(path))
        assert len(cases) >= 10  # Expanded dataset
        assert all(c.category == "safety" for c in cases)

    def test_load_regression_dataset(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "datasets", "regression.jsonl")
        cases = load_dataset(os.path.abspath(path))
        assert len(cases) >= 8  # Expanded dataset
        assert all(c.category == "regression" for c in cases)

    def test_dataset_ids_are_unique(self):
        cases = load_all_datasets()
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids)), "Duplicate test case IDs found"

    def test_all_cases_have_required_fields(self):
        cases = load_all_datasets()
        for case in cases:
            assert case.id
            assert case.category
            assert case.input
            assert case.expected_behavior
            assert len(case.evaluation_type) > 0

    def test_severity_values_are_valid(self):
        cases = load_all_datasets()
        valid_severities = {"critical", "high", "medium", "low"}
        for case in cases:
            assert case.severity in valid_severities, f"{case.id} has invalid severity: {case.severity}"


class TestEvalRunner:
    @pytest.fixture
    def plain_evaluators(self):
        return {
            "rule_based": RuleBasedEvaluator(),
            "safety": SafetyEvaluator(),
        }

    @pytest.fixture
    def evaluators_with_consistency(self):
        return {
            "rule_based": RuleBasedEvaluator(),
            "safety": SafetyEvaluator(),
            "consistency": ConsistencyEvaluator(),
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

    async def test_summary_has_run_id_and_timestamp(self, mock_chatbot: MockChatbot, plain_evaluators: dict):
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=plain_evaluators)
        summary = await runner.run(load_all_datasets())
        assert summary.run_id
        assert summary.timestamp

    async def test_overall_scores_calculated(self, mock_chatbot: MockChatbot, plain_evaluators: dict):
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=plain_evaluators)
        summary = await runner.run(load_all_datasets())
        for result in summary.results:
            if result.error is None and result.evaluations:
                scored = [e.score for e in result.evaluations if e.score is not None]
                if scored:
                    assert result.overall_score is not None

    async def test_deepeval_aggregate_field_exists(self, mock_chatbot: MockChatbot, plain_evaluators: dict):
        """Verify the new deepeval_aggregate field exists in RunSummary."""
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=plain_evaluators)
        summary = await runner.run(load_all_datasets())
        assert hasattr(summary, "deepeval_aggregate")
        assert isinstance(summary.deepeval_aggregate, dict)

    async def test_with_consistency_evaluator(
        self, mock_chatbot: MockChatbot, evaluators_with_consistency: dict, functional_test_case: TestCase
    ):
        """Test that the consistency evaluator integrates with the runner."""
        tc = functional_test_case.model_copy()
        tc.evaluation_type = ["rule_based", "consistency"]

        runner = EvalRunner(chatbot=mock_chatbot, evaluators=evaluators_with_consistency)
        summary = await runner.run([tc])

        assert summary.total == 1
        result = summary.results[0]
        evaluator_names = [e.evaluator for e in result.evaluations]
        assert "rule_based" in evaluator_names
        assert "consistency" in evaluator_names
