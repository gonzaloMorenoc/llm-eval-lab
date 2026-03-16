"""Markdown reporter — generates a human-readable report from RunSummary."""

from __future__ import annotations

import os

import yaml

from src.runner.models import RunSummary


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


def generate_markdown_report(summary: RunSummary, output_dir: str) -> str:
    """Write a Markdown report. Returns the file path."""
    config = _load_config()
    thresholds = config.get("ragas", {}).get("thresholds", {})

    lines: list[str] = []
    w = lines.append

    w(f"# Run Report — {summary.chatbot_id} — {summary.chatbot_mode} — {summary.timestamp}\n")

    # Executive overview
    w("## Executive Overview\n")
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Pass Rate | {summary.pass_rate:.1%} |")
    w(f"| Avg Score | {summary.avg_score:.3f} |")
    w(f"| Critical Failures | {summary.critical_failures} |")
    w(f"| Avg Latency | {summary.avg_latency_ms:.0f}ms |")
    w(f"| Chatbot Mode | {summary.chatbot_mode} |")
    w(f"| Total Tests | {summary.total} |")
    w(f"| Passed | {summary.passed} |")
    w(f"| Failed | {summary.failed} |")
    w(f"| Errors | {summary.errors} |")
    w("")

    # RAGAS metrics summary
    if summary.ragas_aggregate:
        w("## RAGAS Metrics Summary\n")
        w("| Metric | Avg Score | Threshold | Status | Note |")
        w("|--------|-----------|-----------|--------|------|")
        rag_only = {"faithfulness", "context_precision", "context_recall"}
        for metric, avg in sorted(summary.ragas_aggregate.items()):
            t = thresholds.get(metric, 0.65)
            status = "PASS" if avg >= t else "FAIL"
            note = "RAG only" if metric in rag_only else "plain + RAG"
            w(f"| {metric} | {avg:.3f} | {t} | {status} | {note} |")
        w("")

    # DeepEval metrics summary
    if summary.deepeval_aggregate:
        w("## DeepEval Metrics Summary\n")
        w("| Metric | Avg Score | Threshold | Status |")
        w("|--------|-----------|-----------|--------|")
        deepeval_thresholds = config.get("deepeval", {}).get("thresholds", {})
        for metric, avg in sorted(summary.deepeval_aggregate.items()):
            t = deepeval_thresholds.get(metric, 0.5)
            status = "PASS" if avg >= t else "FAIL"
            w(f"| {metric} | {avg:.3f} | {t} | {status} |")
        w("")

    # Results by category
    w("## Results by Category\n")
    w("| Category | Total | Passed | Failed | Pass Rate |")
    w("|----------|-------|--------|--------|-----------|")
    for cat, stats in sorted(summary.by_category.items()):
        w(f"| {cat} | {stats.total} | {stats.passed} | {stats.failed} | {stats.pass_rate:.1%} |")
    w("")

    # Critical & high failures
    critical_high = [
        r for r in summary.results
        if not r.overall_passed and r.test_case.severity in ("critical", "high")
    ]
    if critical_high:
        w("## Critical & High Failures\n")
        for r in critical_high:
            tc = r.test_case
            input_display = tc.input if isinstance(tc.input, str) else str(tc.input)
            w(f"### {tc.id} ({tc.severity})\n")
            w(f"- **Input**: {input_display}")
            w(f"- **Expected**: {tc.expected_behavior}")
            w(f"- **Response**: {r.response[:300]}{'...' if len(r.response) > 300 else ''}")
            if r.retrieved_contexts:
                w(f"- **Retrieved Contexts**: {len(r.retrieved_contexts)} documents")
                for i, ctx in enumerate(r.retrieved_contexts[:3]):
                    w(f"  - Context {i+1}: {ctx[:150]}...")
            for ev in r.evaluations:
                w(f"- **{ev.evaluator}**: {'PASS' if ev.passed else 'FAIL'} "
                  f"(score: {ev.score}) — {ev.reason[:200]}")
            w("")

    # Problematic response examples (3 worst by overall_score)
    scored_results = [r for r in summary.results if r.overall_score is not None and not r.overall_passed]
    scored_results.sort(key=lambda r: r.overall_score or 0.0)
    worst = scored_results[:3]
    if worst:
        w("## Problematic Response Examples\n")
        for r in worst:
            tc = r.test_case
            input_display = tc.input if isinstance(tc.input, str) else str(tc.input)
            w(f"### {tc.id} (score: {r.overall_score:.3f})\n")
            w(f"- **Input**: {input_display}")
            w(f"- **Response**: {r.response[:300]}{'...' if len(r.response) > 300 else ''}")
            if r.retrieved_contexts:
                w(f"- **Retrieved Contexts**: {[c[:80] + '...' for c in r.retrieved_contexts[:3]]}")
            w("")

    # Recommendations
    w("## Recommendations\n")
    recommendations = _generate_recommendations(summary, thresholds)
    if recommendations:
        for rec in recommendations:
            w(f"- {rec}")
    else:
        w("- All metrics are within acceptable thresholds. No immediate action needed.")
    w("")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.md")
    content = "\n".join(lines)
    with open(path, "w") as f:
        f.write(content)
    return path


def _generate_recommendations(summary: RunSummary, thresholds: dict) -> list[str]:
    """Auto-generate recommendations based on failure patterns and low RAGAS scores."""
    recs: list[str] = []

    if summary.pass_rate < 0.5:
        recs.append("Overall pass rate is below 50%. Consider reviewing the chatbot's base capabilities.")

    if summary.critical_failures > 0:
        recs.append(
            f"{summary.critical_failures} critical test(s) failed. "
            "Prioritize fixing these before other improvements."
        )

    for metric, avg in summary.ragas_aggregate.items():
        t = thresholds.get(metric, 0.65)
        if avg < t:
            if metric == "faithfulness":
                recs.append(
                    f"Faithfulness score ({avg:.3f}) is below threshold ({t}). "
                    "The chatbot generates claims not supported by the retrieved context."
                )
            elif metric == "context_precision":
                recs.append(
                    f"Context Precision ({avg:.3f}) is below threshold ({t}). "
                    "The retriever is returning irrelevant documents."
                )
            elif metric == "context_recall":
                recs.append(
                    f"Context Recall ({avg:.3f}) is below threshold ({t}). "
                    "The retriever is missing relevant documents."
                )
            elif metric == "answer_relevancy":
                recs.append(
                    f"Answer Relevancy ({avg:.3f}) is below threshold ({t}). "
                    "Responses are not addressing the user's questions directly."
                )
            elif metric == "factual_correctness":
                recs.append(
                    f"Factual Correctness ({avg:.3f}) is below threshold ({t}). "
                    "Responses contain factual inaccuracies compared to ground truth."
                )
            else:
                recs.append(f"{metric} ({avg:.3f}) is below threshold ({t}).")

    # Check for safety failures
    safety_cat = summary.by_category.get("safety")
    if safety_cat and safety_cat.failed > 0:
        recs.append(
            f"{safety_cat.failed} safety test(s) failed. "
            "Review the chatbot's safety guardrails and system prompt."
        )

    return recs
