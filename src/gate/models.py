"""Pydantic models for the regression gate: baselines, comparisons, policy, verdict."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Metrics whose threshold semantics in config.yaml are "lower is better"
# (DeepEval thresholds use `< 0.5`); for these, regression means the score went UP.
LOWER_IS_BETTER_METRICS = frozenset({"deepeval.hallucination", "deepeval.bias", "deepeval.toxicity"})

# Synthetic metric derived from per-case pass/fail, always present in metric_set.
PASS_RATE_METRIC = "pass_rate"  # noqa: S105


class BaselineCase(BaseModel):
    """Per-test-case aggregate stored in a baseline (means across samples)."""

    id: str
    category: str
    severity: Literal["critical", "high", "medium", "low"]
    passed: bool  # majority vote across samples (ties count as passed)
    pass_samples: list[bool]
    flakiness: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_variance: dict[str, float] = Field(default_factory=dict)
    latency_ms_mean: float = 0.0


class BaselineFile(BaseModel):
    """Versioned, diffable snapshot of a run (or N sampled runs) for gating."""

    schema_version: int = 1
    run_ids: list[str]
    timestamp: str
    chatbot_id: str
    chatbot_mode: Literal["plain", "rag"]
    dataset_hash: str
    metric_set: list[str]
    samples: int = 1
    cases: list[BaselineCase]


class BootstrapResult(BaseModel):
    """Outcome of a paired bootstrap over per-case regression deltas."""

    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float


class MetricComparison(BaseModel):
    """Statistical comparison of one metric between baseline and current run."""

    metric: str
    baseline_mean: float
    current_mean: float
    regression: float  # positive = worse, direction-normalized
    ci_low: float
    ci_high: float
    p_value: float
    n_cases: int
    significant: bool
    gated: bool  # listed in policy.metrics
    breaches: bool  # significant AND > min_effect_size AND > max_regression


class MetricPolicy(BaseModel):
    max_regression: float


class HardRules(BaseModel):
    no_new_critical_failures: bool = True
    max_flakiness: float = 0.3


class GatePolicy(BaseModel):
    """Gate policy — loaded from gate.yaml or built-in defaults."""

    significance_level: float = 0.05
    min_effect_size: float = 0.05
    n_resamples: int = 10_000
    seed: int = 42
    metrics: dict[str, MetricPolicy] = Field(default_factory=lambda: {PASS_RATE_METRIC: MetricPolicy(max_regression=0.05)})
    hard_rules: HardRules = Field(default_factory=HardRules)
    new_cases: Literal["report_only", "fail"] = "report_only"


class GateVerdict(BaseModel):
    """Final gate outcome consumed by reporters and the CLI exit-code logic."""

    passed: bool
    comparisons: list[MetricComparison]
    hard_rule_violations: list[str]
    missing_gated_metrics: list[str]
    new_case_ids: list[str]
    removed_case_ids: list[str]
    mean_flakiness: float
    samples: int
