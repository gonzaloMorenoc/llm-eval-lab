"""Tests for report generation."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.chatbots.mock_adapter import MockChatbot, MockRAGChatbot
from src.evaluators.rule_based import RuleBasedEvaluator
from src.evaluators.safety import SafetyEvaluator
from src.reporting.json_reporter import generate_json_report
from src.reporting.markdown_reporter import generate_markdown_report
from src.runner.runner import EvalRunner, load_all_datasets


class TestJsonReporter:
    async def test_generates_json_file(self, mock_chatbot: MockChatbot):
        evaluators = {"rule_based": RuleBasedEvaluator(), "safety": SafetyEvaluator()}
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=evaluators)
        cases = load_all_datasets()
        summary = await runner.run(cases)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_json_report(summary, tmpdir)
            assert os.path.exists(path)
            assert path.endswith("report.json")

            with open(path) as f:
                data = json.load(f)
            assert data["chatbot_mode"] == "plain"
            assert data["total"] == len(cases)
            assert "results" in data
            assert "by_category" in data

    async def test_json_contains_deepeval_aggregate(self, mock_chatbot: MockChatbot):
        evaluators = {"rule_based": RuleBasedEvaluator(), "safety": SafetyEvaluator()}
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=evaluators)
        cases = load_all_datasets()
        summary = await runner.run(cases)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_json_report(summary, tmpdir)
            with open(path) as f:
                data = json.load(f)
            assert "deepeval_aggregate" in data

    async def test_json_report_is_valid_json(self, mock_chatbot: MockChatbot):
        evaluators = {"rule_based": RuleBasedEvaluator(), "safety": SafetyEvaluator()}
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=evaluators)
        cases = load_all_datasets()[:5]
        summary = await runner.run(cases)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_json_report(summary, tmpdir)
            with open(path) as f:
                data = json.load(f)  # Should not raise
            assert isinstance(data, dict)


class TestMarkdownReporter:
    async def test_generates_markdown_file_plain(self, mock_chatbot: MockChatbot):
        evaluators = {"rule_based": RuleBasedEvaluator(), "safety": SafetyEvaluator()}
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=evaluators)
        cases = load_all_datasets()
        summary = await runner.run(cases)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_markdown_report(summary, tmpdir)
            assert os.path.exists(path)
            assert path.endswith("report.md")

            with open(path) as f:
                content = f.read()
            assert "# Run Report" in content
            assert "Executive Overview" in content
            assert "Results by Category" in content
            assert "plain" in content

    async def test_generates_markdown_file_rag(self, mock_rag_chatbot: MockRAGChatbot):
        evaluators = {"rule_based": RuleBasedEvaluator(), "safety": SafetyEvaluator()}
        runner = EvalRunner(chatbot=mock_rag_chatbot, evaluators=evaluators)
        cases = load_all_datasets()
        summary = await runner.run(cases)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_markdown_report(summary, tmpdir)
            assert os.path.exists(path)

            with open(path) as f:
                content = f.read()
            assert "rag" in content

    async def test_recommendations_generated(self, mock_chatbot: MockChatbot):
        evaluators = {"rule_based": RuleBasedEvaluator(), "safety": SafetyEvaluator()}
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=evaluators)
        cases = load_all_datasets()
        summary = await runner.run(cases)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_markdown_report(summary, tmpdir)
            with open(path) as f:
                content = f.read()
            assert "Recommendations" in content

    async def test_markdown_contains_all_categories(self, mock_chatbot: MockChatbot):
        evaluators = {"rule_based": RuleBasedEvaluator(), "safety": SafetyEvaluator()}
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=evaluators)
        cases = load_all_datasets()
        summary = await runner.run(cases)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_markdown_report(summary, tmpdir)
            with open(path) as f:
                content = f.read()
            assert "functional" in content
            assert "safety" in content
            assert "regression" in content
            assert "multi_turn" in content

    async def test_markdown_report_directory_created(self, mock_chatbot: MockChatbot):
        """Report should create output directory if it doesn't exist."""
        evaluators = {"rule_based": RuleBasedEvaluator(), "safety": SafetyEvaluator()}
        runner = EvalRunner(chatbot=mock_chatbot, evaluators=evaluators)
        cases = load_all_datasets()[:3]
        summary = await runner.run(cases)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "output")
            path = generate_markdown_report(summary, nested_dir)
            assert os.path.exists(path)
