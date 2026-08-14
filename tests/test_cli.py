"""CLI tests using typer's CliRunner and the mock provider (no API keys)."""

from __future__ import annotations

import json
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
        # The safety dataset's cases all request evaluation_type ["safety", "rule_based"].
        # An unfiltered run would produce both "safety" and "rule_based" evaluations; with
        # --evaluators rule_based, only "rule_based" evaluations must be persisted.
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
        run_dirs = os.listdir(tmp_path)
        assert len(run_dirs) == 1
        report = json.loads((tmp_path / run_dirs[0] / "report.json").read_text())
        evaluator_names = {evaluation["evaluator"] for test_result in report["results"] for evaluation in test_result["evaluations"]}
        assert evaluator_names == {"rule_based"}

    def test_load_summary_with_malformed_json_exits_2(self, tmp_path):
        """Test that malformed report.json raises exit code 2 with clear message."""
        results_dir = tmp_path / "results"
        run_id = "test_run_123"
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True)
        # Write malformed JSON to report.json
        (run_dir / "report.json").write_text('{"invalid": json}')
        result = runner.invoke(
            app,
            ["run", "--provider", "mock", "--datasets", "functional", "--results-dir", str(results_dir)],
        )
        # Now try to baseline save with the malformed report
        result = runner.invoke(
            app,
            ["baseline", "save", run_id, "--results-dir", str(results_dir), "--baselines-dir", str(tmp_path / "baselines")],
        )
        assert result.exit_code == 2
        assert "report.json" in result.output
        assert "Failed to parse" in result.output


def _do_run(tmp_path, datasets="functional", samples=1):
    """Run the mock provider and return the created run ids (oldest first)."""
    results_dir = tmp_path / "results"
    args = ["run", "--provider", "mock", "--datasets", datasets, "--results-dir", str(results_dir)]
    if samples > 1:
        args += ["--samples", str(samples)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return results_dir, sorted(os.listdir(results_dir))


class TestBaselineSave:
    def test_save_single_run(self, tmp_path):
        results_dir, run_ids = _do_run(tmp_path)
        baselines_dir = tmp_path / "baselines"
        result = runner.invoke(
            app,
            [
                "baseline",
                "save",
                run_ids[0],
                "--name",
                "main",
                "--results-dir",
                str(results_dir),
                "--baselines-dir",
                str(baselines_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (baselines_dir / "main.json").exists()

    def test_save_multiple_runs_records_samples(self, tmp_path):
        from src.gate.models import BaselineFile

        results_dir, run_ids = _do_run(tmp_path, samples=2)
        baselines_dir = tmp_path / "baselines"
        result = runner.invoke(
            app,
            ["baseline", "save", *run_ids, "--results-dir", str(results_dir), "--baselines-dir", str(baselines_dir)],
        )
        assert result.exit_code == 0, result.output
        baseline = BaselineFile.model_validate_json((baselines_dir / "main.json").read_text())
        assert baseline.samples == 2

    def test_save_unknown_run_exits_2(self, tmp_path):
        result = runner.invoke(
            app,
            ["baseline", "save", "no_such_run", "--results-dir", str(tmp_path), "--baselines-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        assert "Run not found" in result.output


class TestCompareCommand:
    def test_compare_two_runs_renders_table(self, tmp_path):
        results_dir, run_ids = _do_run(tmp_path, samples=2)
        result = runner.invoke(app, ["compare", run_ids[0], run_ids[1], "--results-dir", str(results_dir)])
        assert result.exit_code == 0, result.output
        assert "Current" in result.output
        assert "Regression gate" in result.output

    def test_compare_missing_run_exits_2(self, tmp_path):
        results_dir, run_ids = _do_run(tmp_path)
        result = runner.invoke(app, ["compare", run_ids[0], "ghost", "--results-dir", str(results_dir)])
        assert result.exit_code == 2
