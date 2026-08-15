"""Tests for the shared multi-sample execution helper.

``run_samples`` is the single implementation of "run the suite N times and
persist each report", used by both the CLI (``--samples``) and the dashboard,
so the two cannot drift apart.
"""

from __future__ import annotations

import json
import os

import pytest

from src.chatbots.mock_adapter import MockChatbot
from src.evaluators.rule_based import RuleBasedEvaluator
from src.runner.execution import run_samples
from src.runner.models import TestCase


@pytest.fixture
def evaluators():
    return {"rule_based": RuleBasedEvaluator()}


@pytest.fixture
def cases() -> list[TestCase]:
    return [
        TestCase(
            id=f"case_{i}",
            category="functional",
            input=f"Question {i}?",
            expected_behavior="Answers the question.",
            evaluation_type=["rule_based"],
            severity="medium",
        )
        for i in range(3)
    ]


class TestRunSamples:
    async def test_runs_the_suite_once_per_sample(self, evaluators, cases, tmp_path):
        summaries = await run_samples(
            chatbot=MockChatbot(latency_range=(1.0, 2.0)),
            evaluators=evaluators,
            test_cases=cases,
            samples=3,
            results_dir=str(tmp_path),
        )

        assert len(summaries) == 3
        assert len({s.run_id for s in summaries}) == 3, "each sample must be its own run"
        assert all(s.total == len(cases) for s in summaries)

    async def test_persists_json_and_markdown_for_every_sample(self, evaluators, cases, tmp_path):
        summaries = await run_samples(
            chatbot=MockChatbot(latency_range=(1.0, 2.0)),
            evaluators=evaluators,
            test_cases=cases,
            samples=2,
            results_dir=str(tmp_path),
        )

        for summary in summaries:
            run_dir = tmp_path / summary.run_id
            assert (run_dir / "report.json").exists()
            assert (run_dir / "report.md").exists()
            persisted = json.loads((run_dir / "report.json").read_text())
            assert persisted["run_id"] == summary.run_id

    async def test_progress_spans_every_sample_without_restarting(self, evaluators, cases, tmp_path):
        """The count must run 1..(samples * cases) straight through. A per-run
        counter would send the bar back to zero at each sample boundary."""
        calls: list[tuple[int, int]] = []

        await run_samples(
            chatbot=MockChatbot(latency_range=(1.0, 2.0)),
            evaluators=evaluators,
            test_cases=cases,
            samples=3,
            results_dir=str(tmp_path),
            on_progress=lambda done, total: calls.append((done, total)),
        )

        expected_total = 3 * len(cases)
        assert [done for done, _ in calls] == list(range(1, expected_total + 1))
        assert {total for _, total in calls} == {expected_total}

    async def test_announces_each_sample_before_it_starts(self, evaluators, cases, tmp_path):
        """The CLI labels each sample in its console output; the announcement has
        to fire before the run, not after, to head its progress bar."""
        announced: list[tuple[int, int]] = []

        await run_samples(
            chatbot=MockChatbot(latency_range=(1.0, 2.0)),
            evaluators=evaluators,
            test_cases=cases,
            samples=3,
            results_dir=str(tmp_path),
            on_sample_start=lambda index, total: announced.append((index, total)),
        )

        assert announced == [(1, 3), (2, 3), (3, 3)]

    async def test_rejects_a_non_positive_sample_count(self, evaluators, cases, tmp_path):
        with pytest.raises(ValueError, match="samples"):
            await run_samples(
                chatbot=MockChatbot(latency_range=(1.0, 2.0)),
                evaluators=evaluators,
                test_cases=cases,
                samples=0,
                results_dir=str(tmp_path),
            )

        assert os.listdir(tmp_path) == [], "nothing should be written when the input is rejected"
