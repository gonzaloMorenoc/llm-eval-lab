"""Helpers to build synthetic RunSummary objects for gate tests."""

from __future__ import annotations

from src.runner.models import EvaluationResult, RunSummary, TestCase, TestResult


def make_summary(
    case_scores: dict[str, dict[str, float]],
    *,
    passed: dict[str, bool] | None = None,
    severities: dict[str, str] | None = None,
    run_id: str = "run_test",
    chatbot_mode: str = "plain",
) -> RunSummary:
    """Build a minimal RunSummary. ``case_scores`` maps case_id -> {metric -> score}.

    Metric names may be plain evaluator names ("rule_based") or dotted
    sub-metrics ("ragas.answer_relevancy"), mirroring the gate's flattening.
    Cases default to passed=True and severity="medium" unless overridden.
    """
    passed = passed or {}
    severities = severities or {}
    results = []
    for case_id, metrics in case_scores.items():
        evaluations: dict[str, EvaluationResult] = {}
        for name, score in metrics.items():
            if "." in name:
                evaluator, sub = name.split(".", 1)
                ev = evaluations.setdefault(
                    evaluator,
                    EvaluationResult(evaluator=evaluator, passed=True, score=None, details={"metric_scores": {}}),
                )
                ev.details["metric_scores"][sub] = score
            else:
                evaluations[name] = EvaluationResult(evaluator=name, passed=True, score=score)
        test_case = TestCase(
            id=case_id,
            category="functional",
            input="q",
            expected_behavior="answers",
            evaluation_type=["rule_based"],
            severity=severities.get(case_id, "medium"),
        )
        results.append(
            TestResult(
                test_case=test_case,
                response="a",
                chatbot_mode=chatbot_mode,
                latency_ms=100.0,
                evaluations=list(evaluations.values()),
                overall_passed=passed.get(case_id, True),
                overall_score=None,
            )
        )
    n_passed = sum(1 for r in results if r.overall_passed)
    return RunSummary(
        run_id=run_id,
        timestamp="2026-08-14T00:00:00+00:00",
        chatbot_id="mock/mock-plain-v1",
        chatbot_mode=chatbot_mode,
        total=len(results),
        passed=n_passed,
        failed=len(results) - n_passed,
        results=results,
    )
