"""Build, persist and load regression-gate baselines from run summaries."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence

from src.gate.models import PASS_RATE_METRIC, BaselineCase, BaselineFile
from src.gate.statistics import case_flakiness
from src.runner.models import RunSummary, TestCase, TestResult


class BaselineError(Exception):
    """Raised when a baseline cannot be built, loaded or validated."""


def compute_dataset_hash(test_cases: Sequence[TestCase]) -> str:
    """SHA-256 over the canonical JSON of the test cases, sorted by id."""
    canonical = json.dumps(
        [tc.model_dump() for tc in sorted(test_cases, key=lambda t: t.id)],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _case_metrics(result: TestResult) -> dict[str, float]:
    """Flatten evaluator scores into {metric_name: score} using the gate naming convention."""
    metrics: dict[str, float] = {}
    for ev in result.evaluations:
        if ev.score is not None:
            metrics[ev.evaluator] = ev.score
        metric_scores = ev.details.get("metric_scores") or {}
        for name, score in metric_scores.items():
            metrics[f"{ev.evaluator}.{name}"] = float(score)
    return metrics


def build_baseline(summaries: Sequence[RunSummary]) -> BaselineFile:
    """Aggregate N sampled runs over the same dataset into one BaselineFile."""
    if not summaries:
        raise BaselineError("At least one run summary is required")
    first = summaries[0]
    ids = {r.test_case.id for r in first.results}
    for s in summaries[1:]:
        if s.chatbot_mode != first.chatbot_mode:
            raise BaselineError("All samples must share the same chatbot_mode")
        if {r.test_case.id for r in s.results} != ids:
            raise BaselineError("All samples must cover the same test case ids")

    per_case: dict[str, list[TestResult]] = {}
    for s in summaries:
        for r in s.results:
            per_case.setdefault(r.test_case.id, []).append(r)

    cases: list[BaselineCase] = []
    metric_set: set[str] = {PASS_RATE_METRIC}
    for case_id in sorted(per_case):
        results = per_case[case_id]
        test_case = results[0].test_case
        sample_metrics = [_case_metrics(r) for r in results]
        metric_names = sorted({name for m in sample_metrics for name in m})
        means: dict[str, float] = {}
        variances: dict[str, float] = {}
        for name in metric_names:
            values = [m[name] for m in sample_metrics if name in m]
            mean = sum(values) / len(values)
            means[name] = round(mean, 6)
            variances[name] = round(sum((v - mean) ** 2 for v in values) / len(values), 6)
        pass_samples = [r.overall_passed for r in results]
        latencies = [r.latency_ms for r in results]
        metric_set.update(metric_names)
        cases.append(
            BaselineCase(
                id=case_id,
                category=test_case.category,
                severity=test_case.severity,
                passed=sum(pass_samples) * 2 >= len(pass_samples),
                pass_samples=pass_samples,
                flakiness=case_flakiness(pass_samples),
                metrics=means,
                metric_variance=variances,
                latency_ms_mean=round(sum(latencies) / len(latencies), 2),
            )
        )

    return BaselineFile(
        run_ids=[s.run_id for s in summaries],
        timestamp=first.timestamp,
        chatbot_id=first.chatbot_id,
        chatbot_mode=first.chatbot_mode,
        dataset_hash=compute_dataset_hash([r.test_case for r in first.results]),
        metric_set=sorted(metric_set),
        samples=len(summaries),
        cases=cases,
    )


def save_baseline(baseline: BaselineFile, path: str) -> str:
    """Write the baseline as indented JSON (small, diffable). Returns the path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(baseline.model_dump_json(indent=2))
    return path


def load_baseline(path: str) -> BaselineFile:
    """Load and validate a baseline file. Raises BaselineError on any problem."""
    if not os.path.exists(path):
        raise BaselineError(f"Baseline file not found: {path}")
    try:
        with open(path) as f:
            return BaselineFile.model_validate_json(f.read())
    except Exception as e:
        raise BaselineError(f"Invalid baseline file {path}: {e}") from e
