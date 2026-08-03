"""Pydantic v2 models for test cases, results, and run summaries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single test case loaded from a JSONL dataset."""

    __test__ = False  # Prevent pytest from trying to collect this Pydantic model

    id: str
    category: Literal["functional", "multi_turn", "safety", "regression"]
    input: str | list[dict]  # str for single-turn, list[dict] for multi-turn
    expected_behavior: str
    reference: str | None = None
    evaluation_type: list[Literal["rule_based", "llm_judge", "ragas", "safety", "deepeval", "consistency"]]
    ragas_metrics: list[str] | None = None  # None = use defaults from config.yaml
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    metadata: dict = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result from a single evaluator."""

    evaluator: str
    passed: bool
    score: float | None = None  # 0.0–1.0
    reason: str = ""
    details: dict = Field(default_factory=dict)


class TestResult(BaseModel):
    """Full result for one test case including all evaluations."""

    __test__ = False  # Prevent pytest from trying to collect this Pydantic model

    test_case: TestCase
    response: str
    retrieved_contexts: list[str] | None = None
    chatbot_mode: Literal["plain", "rag"]
    latency_ms: float = 0.0
    error: str | None = None
    evaluations: list[EvaluationResult] = Field(default_factory=list)
    # Evaluators the test case asked for that weren't registered for this run
    # (e.g. "ragas" requested without OPENAI_API_KEY set). Recorded so a case
    # that ends up with no evaluations at all isn't silently reported as failed.
    skipped_evaluators: list[str] = Field(default_factory=list)
    overall_passed: bool = False
    overall_score: float | None = None


class CategoryStats(BaseModel):
    """Aggregated stats for a test category."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    avg_score: float | None = None


class RunSummary(BaseModel):
    """Complete summary of a test run."""

    run_id: str
    timestamp: str
    chatbot_id: str
    chatbot_mode: Literal["plain", "rag"]
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    pass_rate: float = 0.0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    critical_failures: int = 0
    by_category: dict[str, CategoryStats] = Field(default_factory=dict)
    ragas_aggregate: dict[str, float] = Field(default_factory=dict)
    deepeval_aggregate: dict[str, float] = Field(default_factory=dict)
    # Evaluators requested by at least one test case but not registered for the
    # run, mapped to how many cases asked for them.
    skipped_evaluators: dict[str, int] = Field(default_factory=dict)
    # Cases left with zero evaluations because every evaluator they asked for
    # was missing. These count as failures but nothing actually evaluated them.
    unevaluated: int = 0
    results: list[TestResult] = Field(default_factory=list)
