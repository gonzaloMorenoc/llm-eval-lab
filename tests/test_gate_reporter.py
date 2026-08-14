"""Tests for the gate console and Markdown reporters."""

from __future__ import annotations

import os

from rich.console import Console

from src.gate.models import GateVerdict, MetricComparison
from src.reporting.gate_reporter import generate_gate_markdown, render_gate_console


def _verdict(
    *,
    passed: bool = True,
    samples: int = 1,
    missing_gated_metrics: list[str] | None = None,
    removed_case_ids: list[str] | None = None,
) -> GateVerdict:
    return GateVerdict(
        passed=passed,
        comparisons=[
            MetricComparison(
                metric="pass_rate",
                baseline_mean=0.9,
                current_mean=0.8,
                regression=0.1,
                ci_low=0.02,
                ci_high=0.18,
                p_value=0.01,
                n_cases=40,
                significant=True,
                gated=True,
                breaches=not passed,
            )
        ],
        hard_rule_violations=[] if passed else ["New critical failures: safety_001"],
        missing_gated_metrics=missing_gated_metrics or [],
        new_case_ids=["new_1"],
        removed_case_ids=removed_case_ids or [],
        mean_flakiness=0.05,
        samples=samples,
    )


class TestConsoleReporter:
    def test_renders_table_and_verdict(self):
        console = Console(record=True, width=200)
        render_gate_console(_verdict(passed=False), console)
        output = console.export_text()
        assert "pass_rate" in output
        assert "FAIL" in output
        assert "New critical failures" in output
        assert "new_1" in output

    def test_renders_metric_row_values(self):
        """Verify exact numeric values and icons are rendered (F2)."""
        console = Console(record=True, width=200)
        render_gate_console(_verdict(passed=False), console)
        output = console.export_text()
        # Fixture has baseline=0.9, current=0.8, regression=0.1, ci_low=0.02, ci_high=0.18, p_value=0.01, gated=yes, breaches=regression
        # Pin baseline before current via index ordering to catch transposition (F2)
        assert output.index("0.9000") < output.index("0.8000")
        assert "+0.1000" in output  # regression (signed)
        # Pin CI bounds as complete bracket to catch ci_low/ci_high swap (F2)
        assert "[+0.0200, +0.1800]" in output
        assert "0.0100" in output  # p_value
        assert "yes" in output  # gated
        assert "❌ regression" in output  # breaches icon

    def test_renders_non_breaching_verdict_icon(self):
        """Verify ✅ ok icon for non-breaching metrics (F2)."""
        console = Console(record=True, width=200)
        render_gate_console(_verdict(passed=True), console)
        output = console.export_text()
        assert "✅ ok" in output

    def test_single_sample_warning(self):
        console = Console(record=True, width=200)
        render_gate_console(_verdict(samples=1), console)
        assert "low statistical power" in console.export_text()

    def test_no_warning_with_multiple_samples(self):
        console = Console(record=True, width=200)
        render_gate_console(_verdict(samples=3), console)
        assert "low statistical power" not in console.export_text()

    def test_renders_missing_gated_metrics(self):
        """Verify missing gated metrics appear in output (F3)."""
        console = Console(record=True, width=200)
        render_gate_console(
            _verdict(missing_gated_metrics=["answer_relevancy", "faithfulness"]),
            console,
        )
        output = console.export_text()
        assert "Gated metric not comparable" in output
        assert "answer_relevancy" in output
        assert "faithfulness" in output

    def test_renders_removed_case_ids(self):
        """Verify removed case IDs appear in output (F3)."""
        console = Console(record=True, width=200)
        render_gate_console(_verdict(removed_case_ids=["old_case_1", "old_case_2"]), console)
        output = console.export_text()
        assert "Cases removed since baseline" in output
        assert "old_case_1" in output
        assert "old_case_2" in output


class TestMarkdownReporter:
    def test_writes_markdown_table(self, tmp_path):
        path = generate_gate_markdown(_verdict(passed=False), str(tmp_path))
        assert os.path.basename(path) == "gate_report.md"
        content = open(path).read()
        assert "# Regression gate: ❌ FAIL" in content
        assert "| pass_rate |" in content
        assert "New critical failures: safety_001" in content
        assert "new_1" in content

    def test_markdown_metric_row_values(self, tmp_path):
        """Verify exact numeric values and icons in markdown table (F2)."""
        path = generate_gate_markdown(_verdict(passed=False), str(tmp_path))
        content = open(path).read()
        # Fixture: baseline=0.9, current=0.8, regression=0.1, ci_low=0.02, ci_high=0.18, p_value=0.01, gated=yes, breaches=regression
        # Assert exact row to catch baseline/current transposition, ci_low/ci_high swap, and icon inversion (F2)
        expected_row = "| pass_rate | 0.9000 | 0.8000 | +0.1000 | [+0.0200, +0.1800] | 0.0100 | yes | ❌ regression |"
        assert expected_row in content

    def test_markdown_non_breaching_verdict_icon(self, tmp_path):
        """Verify ✅ ok icon for non-breaching metrics (F2)."""
        path = generate_gate_markdown(_verdict(passed=True), str(tmp_path))
        content = open(path).read()
        assert "✅ ok" in content

    def test_markdown_low_power_note_with_single_sample(self, tmp_path):
        """Verify low-power note appears when samples=1 (F1)."""
        path = generate_gate_markdown(_verdict(samples=1), str(tmp_path))
        content = open(path).read()
        assert "low statistical power" in content

    def test_markdown_no_low_power_note_with_multiple_samples(self, tmp_path):
        """Verify low-power note does NOT appear when samples>1 (F1)."""
        path = generate_gate_markdown(_verdict(samples=3), str(tmp_path))
        content = open(path).read()
        assert "low statistical power" not in content

    def test_appends_to_github_step_summary(self, tmp_path, monkeypatch):
        summary_file = tmp_path / "step_summary.md"
        summary_file.write_text("previous content\n")
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
        generate_gate_markdown(_verdict(), str(tmp_path / "out"))
        content = summary_file.read_text()
        assert content.startswith("previous content\n")
        assert "# Regression gate: ✅ PASS" in content

    def test_no_step_summary_env_is_fine(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        path = generate_gate_markdown(_verdict(), str(tmp_path))
        assert os.path.exists(path)

    def test_markdown_missing_gated_metrics(self, tmp_path):
        """Verify missing gated metrics appear in markdown (F3)."""
        path = generate_gate_markdown(
            _verdict(missing_gated_metrics=["answer_relevancy", "faithfulness"]),
            str(tmp_path),
        )
        content = open(path).read()
        assert "## Gated metrics not comparable" in content
        assert "- answer_relevancy" in content
        assert "- faithfulness" in content

    def test_markdown_removed_case_ids(self, tmp_path):
        """Verify removed case IDs appear in markdown (F3)."""
        path = generate_gate_markdown(
            _verdict(removed_case_ids=["old_case_1", "old_case_2"]),
            str(tmp_path),
        )
        content = open(path).read()
        assert "Cases removed since baseline" in content
        assert "old_case_1" in content
        assert "old_case_2" in content
