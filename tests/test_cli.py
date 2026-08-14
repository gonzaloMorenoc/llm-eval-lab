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
        # Try to baseline save with the malformed report
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

    def test_save_mismatched_test_cases_exits_2(self, tmp_path):
        """Test that combining runs with different test cases triggers BaselineError with exit 2."""
        results_dir, run_ids = _do_run(tmp_path, datasets="functional", samples=2)
        # Manually edit one run's report to remove a test case
        import json

        report_path = results_dir / run_ids[1] / "report.json"
        report_data = json.loads(report_path.read_text())
        # Remove the first result to create a mismatch
        report_data["results"] = report_data["results"][1:]
        report_path.write_text(json.dumps(report_data))

        # Try to baseline save with runs having different test cases
        result = runner.invoke(
            app,
            [
                "baseline",
                "save",
                run_ids[0],
                run_ids[1],
                "--results-dir",
                str(results_dir),
                "--baselines-dir",
                str(tmp_path / "baselines"),
            ],
        )
        assert result.exit_code == 2
        assert "same test case ids" in result.output


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
        assert "Run not found" in result.output

    def test_compare_incompatible_chatbot_modes_exits_2(self, tmp_path):
        """Test that comparing plain vs rag mode runs triggers CompatibilityError with exit 2."""
        results_dir = tmp_path / "results"
        # Run in plain mode (default)
        args_plain = ["run", "--provider", "mock", "--datasets", "functional", "--results-dir", str(results_dir)]
        result = runner.invoke(app, args_plain)
        assert result.exit_code == 0, result.output
        plain_run_ids = sorted(os.listdir(results_dir))
        # Run in rag mode
        args_rag = [
            "run",
            "--provider",
            "mock",
            "--mode",
            "rag",
            "--datasets",
            "functional",
            "--results-dir",
            str(results_dir),
        ]
        result = runner.invoke(app, args_rag)
        assert result.exit_code == 0, result.output
        rag_run_ids = sorted(os.listdir(results_dir))
        # Identify the new rag run
        new_run = next(rid for rid in rag_run_ids if rid not in plain_run_ids)
        # Try to compare plain vs rag
        result = runner.invoke(
            app,
            ["compare", plain_run_ids[0], new_run, "--results-dir", str(results_dir)],
        )
        assert result.exit_code == 2
        assert "chatbot_mode" in result.output

    def test_compare_failing_verdict_still_exits_0(self, tmp_path):
        """Test that compare exits 0 even when verdict fails (no special gating)."""
        # Use safety dataset which has critical-severity cases (functional does not)
        results_dir, run_ids = _do_run(tmp_path, datasets="safety", samples=2)
        import json

        # Edit both runs to control the verdict outcome:
        # - baseline (run_a): critical case must PASS
        # - current (run_b): same critical case must FAIL
        # This triggers no_new_critical_failures hard rule, making verdict fail
        baseline_path = results_dir / run_ids[0] / "report.json"
        current_path = results_dir / run_ids[1] / "report.json"

        baseline_data = json.loads(baseline_path.read_text())
        current_data = json.loads(current_path.read_text())

        critical_case_id = None
        baseline_edited = False
        current_edited = False

        # Find a critical case and ensure it passes in baseline
        for result in baseline_data["results"]:
            if result["test_case"]["severity"] == "critical":
                critical_case_id = result["test_case"]["id"]
                result["overall_passed"] = True
                baseline_edited = True
                break

        assert critical_case_id is not None, "safety dataset must have critical cases"
        assert baseline_edited, "Failed to edit baseline critical case"

        # Find same case in current and make it fail
        for result in current_data["results"]:
            if result["test_case"]["id"] == critical_case_id:
                result["overall_passed"] = False
                current_edited = True
                break

        assert current_edited, f"Failed to find critical case {critical_case_id} in current run"

        baseline_path.write_text(json.dumps(baseline_data))
        current_path.write_text(json.dumps(current_data))

        result = runner.invoke(app, ["compare", run_ids[0], run_ids[1], "--results-dir", str(results_dir)])

        # Verdict must FAIL due to hard rule violation (new critical failure)
        assert "❌ FAIL" in result.output, "Verdict should fail on new critical failure"
        assert "New critical failures" in result.output, "Should report the hard rule violation"
        assert critical_case_id in result.output, "Should name the failing case"

        # But compare should still exit 0 (no special gating)
        assert result.exit_code == 0, result.output
        assert "Regression gate" in result.output


class TestCheckCommand:
    def _run_and_save_baseline(self, tmp_path, datasets="safety"):
        results_dir, run_ids = _do_run(tmp_path, datasets=datasets)
        baselines_dir = tmp_path / "baselines"
        result = runner.invoke(
            app,
            ["baseline", "save", run_ids[0], "--results-dir", str(results_dir), "--baselines-dir", str(baselines_dir)],
        )
        assert result.exit_code == 0, result.output
        return results_dir, baselines_dir / "main.json"

    def test_check_against_own_baseline_passes(self, tmp_path):
        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        # run_id timestamps only have second resolution, so a baseline run and the
        # check's own run can land in the same second; pick the check's run by set
        # difference rather than trusting alphabetical order to reflect recency.
        pre_existing_run_ids = set(os.listdir(results_dir))
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline",
                str(baseline_path),
                "--provider",
                "mock",
                "--datasets",
                "safety",
                "--results-dir",
                str(results_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output
        new_run_id = next(rid for rid in os.listdir(results_dir) if rid not in pre_existing_run_ids)
        assert os.path.exists(results_dir / new_run_id / "gate_report.md")

    def test_check_detects_new_critical_failure(self, tmp_path):
        from src.gate.models import BaselineFile

        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        baseline = BaselineFile.model_validate_json(baseline_path.read_text())
        critical_failed = [c for c in baseline.cases if c.severity == "critical" and not c.passed]
        # El mock responde de forma insegura a los triggers de safety.jsonl a propósito;
        # si esta precondición falla, ampliar _UNSAFE_TRIGGERS en mock_adapter.py
        # para cubrir al menos un caso critical del dataset safety.
        assert critical_failed, "expected the mock to fail at least one critical safety case"
        flipped = [
            c.model_copy(update={"passed": True, "pass_samples": [True]}) if (c.severity == "critical" and not c.passed) else c
            for c in baseline.cases
        ]
        baseline_path.write_text(baseline.model_copy(update={"cases": flipped}).model_dump_json(indent=2))
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline",
                str(baseline_path),
                "--provider",
                "mock",
                "--datasets",
                "safety",
                "--results-dir",
                str(results_dir),
            ],
        )
        assert result.exit_code == 1, result.output
        assert "New critical failures" in result.output

    def test_check_missing_baseline_exits_2(self, tmp_path):
        result = runner.invoke(
            app,
            ["check", "--baseline", str(tmp_path / "ghost.json"), "--provider", "mock", "--results-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        assert "not found" in result.output

    def test_check_mode_mismatch_exits_2(self, tmp_path):
        results_dir = tmp_path / "results"
        r = runner.invoke(
            app,
            ["run", "--provider", "mock", "--mode", "rag", "--datasets", "functional", "--results-dir", str(results_dir)],
        )
        assert r.exit_code == 0, r.output
        run_id = os.listdir(results_dir)[0]
        baselines_dir = tmp_path / "baselines"
        r = runner.invoke(
            app,
            ["baseline", "save", run_id, "--results-dir", str(results_dir), "--baselines-dir", str(baselines_dir)],
        )
        assert r.exit_code == 0, r.output
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline",
                str(baselines_dir / "main.json"),
                "--provider",
                "mock",
                "--mode",
                "plain",
                "--datasets",
                "functional",
                "--results-dir",
                str(results_dir),
            ],
        )
        assert result.exit_code == 2
        assert "chatbot_mode" in result.output

    def test_check_resolves_baseline_by_name(self, tmp_path):
        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline",
                "main",
                "--baselines-dir",
                str(baseline_path.parent),
                "--provider",
                "mock",
                "--datasets",
                "safety",
                "--results-dir",
                str(results_dir),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_check_bad_policy_exits_2(self, tmp_path):
        # Not in the brief's Step 1, added to prove the bad-policy exit-2 path (which
        # otherwise has no coverage): a nonexistent --policy file must surface PolicyError
        # via check, distinctly from the bad-baseline and incompatible-modes paths.
        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline",
                str(baseline_path),
                "--provider",
                "mock",
                "--datasets",
                "safety",
                "--results-dir",
                str(results_dir),
                "--policy",
                str(tmp_path / "ghost_policy.yaml"),
            ],
        )
        assert result.exit_code == 2
        assert "Cannot read policy file" in result.output

    def test_check_evaluation_failure_exits_2(self, tmp_path):
        # Not in the brief's Step 1, added to prove the generic-evaluation-failure exit-2
        # path: an unknown provider makes _build_chatbot raise a bare KeyError (no network
        # call — config.yaml simply has no "ghost_provider" entry), which check's broad
        # except must convert to exit 2 rather than letting it propagate as a crash.
        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline",
                str(baseline_path),
                "--provider",
                "ghost_provider",
                "--datasets",
                "safety",
                "--results-dir",
                str(results_dir),
            ],
        )
        assert result.exit_code == 2
        assert "Evaluation failed" in result.output

    def test_check_bad_datasets_reraises_typer_exit_untouched(self, tmp_path):
        # Not in the brief's Step 1, added to prove the `except typer.Exit: raise` line:
        # a bad --datasets value already exits 2 via _select_datasets's own typer.Exit(2),
        # and check must re-raise it untouched rather than let the broad `except Exception`
        # below also catch it and print a spurious extra "Evaluation failed:" line.
        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline",
                str(baseline_path),
                "--provider",
                "mock",
                "--datasets",
                "nope",
                "--results-dir",
                str(results_dir),
            ],
        )
        assert result.exit_code == 2
        assert "Unknown dataset" in result.output
        assert "Evaluation failed" not in result.output

    def test_check_missing_gated_metric_exits_2(self, tmp_path):
        # Not in the brief's Step 1, added to prove the missing_gated_metrics ordering:
        # this policy gates a metric the mock run never produces, so missing_gated_metrics
        # is non-empty AND verdict.passed is False. If the CLI checked verdict.passed
        # first, this would exit 1 instead of 2 — assert the specific code to catch that.
        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        policy_path = tmp_path / "gate.yaml"
        policy_path.write_text("gate:\n  metrics:\n    ragas.answer_relevancy: {max_regression: 0.1}\n")
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline",
                str(baseline_path),
                "--provider",
                "mock",
                "--datasets",
                "safety",
                "--results-dir",
                str(results_dir),
                "--policy",
                str(policy_path),
            ],
        )
        assert result.exit_code == 2
        assert "Gated metric not comparable" in result.output
        assert "ragas.answer_relevancy" in result.output
