"""CLI tests using typer's CliRunner and the mock provider (no API keys)."""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_llm_evaluators(monkeypatch):
    # Never let a real .env / OPENAI_API_KEY on a developer machine push these
    # tests into constructing a real RagasEvaluator and making paid API calls.
    monkeypatch.setattr("src.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("USE_DEEPEVAL", "false")
    monkeypatch.setenv("USE_CONSISTENCY", "false")
    monkeypatch.setenv("USE_LLM_JUDGE", "false")


class TestRunCommand:
    def test_run_with_mock_provider_writes_reports(self, tmp_path):
        result = runner.invoke(
            app,
            ["run", "--provider", "mock", "--datasets", "functional", "--results-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        run_dirs = os.listdir(tmp_path)
        assert len(run_dirs) == 1
        assert os.path.exists(tmp_path / run_dirs[0] / "report.json")
        assert os.path.exists(tmp_path / run_dirs[0] / "report.md")
        assert "Estimated API calls" in result.output

    def test_run_with_samples_creates_one_run_per_sample(self, tmp_path):
        result = runner.invoke(
            app,
            ["run", "--provider", "mock", "--datasets", "functional", "--samples", "2", "--results-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert len(os.listdir(tmp_path)) == 2

    def test_unknown_dataset_exits_2(self, tmp_path):
        result = runner.invoke(
            app,
            ["run", "--provider", "mock", "--datasets", "nope", "--results-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        assert "Unknown dataset" in result.output

    def test_unknown_evaluator_exits_2(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "run",
                "--provider",
                "mock",
                "--datasets",
                "functional",
                "--evaluators",
                "ghost_eval",
                "--results-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        assert "Unknown or unavailable evaluators" in result.output

    def test_evaluator_filter_limits_evaluations(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "run",
                "--provider",
                "mock",
                "--datasets",
                "safety",
                "--evaluators",
                "rule_based",
                "--results-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
