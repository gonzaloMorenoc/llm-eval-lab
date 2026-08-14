"""Pair baseline vs current cases and compute per-metric statistical comparisons."""

from __future__ import annotations

from src.gate.models import (
    LOWER_IS_BETTER_METRICS,
    PASS_RATE_METRIC,
    BaselineCase,
    BaselineFile,
    GatePolicy,
    MetricComparison,
)
from src.gate.statistics import paired_bootstrap


class CompatibilityError(Exception):
    """Raised when two runs cannot be meaningfully compared."""


CasePair = tuple[BaselineCase, BaselineCase]


def validate_compatibility(baseline: BaselineFile, current: BaselineFile) -> None:
    """Fail fast when runs are not comparable (spec §4: exit 2 at the CLI layer)."""
    if baseline.chatbot_mode != current.chatbot_mode:
        raise CompatibilityError(f"chatbot_mode mismatch: baseline={baseline.chatbot_mode}, current={current.chatbot_mode}")
    if not set(baseline.metric_set) & set(current.metric_set):
        raise CompatibilityError("No shared metrics between baseline and current run")
    # A shared metric_set is not enough: pass_rate (the default gated metric) is derived
    # from *every* evaluator, so a current run that lost one of the baseline's metrics
    # measures something weaker under the same name and would gate a green build on it.
    # The opposite direction (current has extra metrics) fails safe — a stricter
    # pass_rate surfaces as exit 1 — so extra metrics stay permitted.
    missing = sorted(set(baseline.metric_set) - set(current.metric_set))
    if missing:
        raise CompatibilityError(
            f"Metrics missing from the current run: {', '.join(missing)}. "
            f"The evaluator set shrank since the baseline, so its metrics are not comparable."
        )


def pair_cases(baseline: BaselineFile, current: BaselineFile) -> tuple[list[CasePair], list[str], list[str]]:
    """Pair cases by id. Returns (pairs, new_case_ids, removed_case_ids)."""
    base_by_id = {c.id: c for c in baseline.cases}
    curr_by_id = {c.id: c for c in current.cases}
    pairs = [(base_by_id[i], curr_by_id[i]) for i in sorted(base_by_id.keys() & curr_by_id.keys())]
    new_ids = sorted(curr_by_id.keys() - base_by_id.keys())
    removed_ids = sorted(base_by_id.keys() - curr_by_id.keys())
    return pairs, new_ids, removed_ids


def regression_deltas(pairs: list[CasePair], metric: str) -> list[float]:
    """Per-case regression (positive = worse), skipping pairs missing the metric."""
    deltas: list[float] = []
    for base, curr in pairs:
        if metric == PASS_RATE_METRIC:
            deltas.append(float(base.passed) - float(curr.passed))
            continue
        if metric not in base.metrics or metric not in curr.metrics:
            continue
        if metric in LOWER_IS_BETTER_METRICS:
            deltas.append(curr.metrics[metric] - base.metrics[metric])
        else:
            deltas.append(base.metrics[metric] - curr.metrics[metric])
    return deltas


def _metric_mean(cases: list[BaselineCase], metric: str) -> float:
    if metric == PASS_RATE_METRIC:
        return sum(1.0 for c in cases if c.passed) / len(cases) if cases else 0.0
    values = [c.metrics[metric] for c in cases if metric in c.metrics]
    return sum(values) / len(values) if values else 0.0


def compare_metrics(baseline: BaselineFile, current: BaselineFile, policy: GatePolicy) -> list[MetricComparison]:
    """Compare every metric shared by both runs; only policy-listed metrics can breach."""
    pairs, _, _ = pair_cases(baseline, current)
    comparisons: list[MetricComparison] = []
    for metric in sorted(set(baseline.metric_set) & set(current.metric_set)):
        deltas = regression_deltas(pairs, metric)
        # Metric with no comparable pairs is omitted here; if gated, it is surfaced by the policy layer's missing_gated_metrics check.
        if not deltas:
            continue
        boot = paired_bootstrap(deltas, n_resamples=policy.n_resamples, seed=policy.seed)
        significant = boot.p_value < policy.significance_level
        metric_policy = policy.metrics.get(metric)
        breaches = False
        if metric_policy is not None and significant:
            breaches = boot.mean_delta > policy.min_effect_size and boot.mean_delta > metric_policy.max_regression
        comparisons.append(
            MetricComparison(
                metric=metric,
                baseline_mean=round(_metric_mean(baseline.cases, metric), 4),
                current_mean=round(_metric_mean(current.cases, metric), 4),
                regression=round(boot.mean_delta, 4),
                ci_low=round(boot.ci_low, 4),
                ci_high=round(boot.ci_high, 4),
                p_value=round(boot.p_value, 4),
                n_cases=len(deltas),
                significant=significant,
                gated=metric_policy is not None,
                breaches=breaches,
            )
        )
    return comparisons
