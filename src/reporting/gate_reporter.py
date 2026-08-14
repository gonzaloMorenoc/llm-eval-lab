"""Gate verdict reporters — rich console table and Markdown for PR comments."""

from __future__ import annotations

import os

from rich.console import Console
from rich.table import Table

from src.gate.models import GateVerdict

_STATUS = {True: "✅ PASS", False: "❌ FAIL"}
_LOW_POWER_NOTE = "Note: --samples 1 has low statistical power; use --samples 3 or more for reliable significance."


def render_gate_console(verdict: GateVerdict, console: Console) -> None:
    """Print the verdict as a rich table plus hard-rule and case-diff notes."""
    table = Table(title=f"Regression gate — {_STATUS[verdict.passed]} (samples: {verdict.samples})")
    for column in ("Metric", "Baseline", "Current", "Regression", "95% CI", "p-value", "Gated", "Verdict"):
        table.add_column(column)
    for c in verdict.comparisons:
        table.add_row(
            c.metric,
            f"{c.baseline_mean:.4f}",
            f"{c.current_mean:.4f}",
            f"{c.regression:+.4f}",
            f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]",
            f"{c.p_value:.4f}",
            "yes" if c.gated else "no",
            "❌ regression" if c.breaches else "✅ ok",
        )
    console.print(table)
    for violation in verdict.hard_rule_violations:
        console.print(f"[red]Hard rule violated:[/red] {violation}")
    for metric in verdict.missing_gated_metrics:
        console.print(f"[red]Gated metric not comparable:[/red] {metric}")
    if verdict.new_case_ids:
        console.print(f"[yellow]Cases without baseline:[/yellow] {', '.join(verdict.new_case_ids)}")
    if verdict.removed_case_ids:
        console.print(f"[yellow]Cases removed since baseline:[/yellow] {', '.join(verdict.removed_case_ids)}")
    if verdict.samples == 1:
        console.print(f"[yellow]{_LOW_POWER_NOTE}[/yellow]")


def generate_gate_markdown(verdict: GateVerdict, output_dir: str) -> str:
    """Write gate_report.md (and append to $GITHUB_STEP_SUMMARY when set). Returns the path."""
    lines = [
        f"# Regression gate: {_STATUS[verdict.passed]}",
        "",
        f"Samples per case: {verdict.samples} · Mean flakiness: {verdict.mean_flakiness:.2f}",
        "",
        "| Metric | Baseline | Current | Regression | 95% CI | p-value | Gated | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in verdict.comparisons:
        lines.append(
            f"| {c.metric} | {c.baseline_mean:.4f} | {c.current_mean:.4f} | {c.regression:+.4f} "
            f"| [{c.ci_low:+.4f}, {c.ci_high:+.4f}] | {c.p_value:.4f} "
            f"| {'yes' if c.gated else 'no'} | {'❌ regression' if c.breaches else '✅ ok'} |"
        )
    if verdict.hard_rule_violations:
        lines += ["", "## Hard rule violations", ""] + [f"- {v}" for v in verdict.hard_rule_violations]
    if verdict.missing_gated_metrics:
        lines += ["", "## Gated metrics not comparable", ""] + [f"- {m}" for m in verdict.missing_gated_metrics]
    if verdict.new_case_ids:
        lines += ["", f"Cases without baseline: {', '.join(verdict.new_case_ids)}"]
    if verdict.removed_case_ids:
        lines += ["", f"Cases removed since baseline: {', '.join(verdict.removed_case_ids)}"]
    if verdict.samples == 1:
        lines += ["", f"_{_LOW_POWER_NOTE}_"]
    content = "\n".join(lines) + "\n"

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "gate_report.md")
    with open(path, "w") as f:
        f.write(content)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(content)
    return path
