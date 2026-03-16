"""Pydantic v2 models for test cases, results, and run summaries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single test case loaded from a JSONL dataset."""

    id: str
    category: Literal["functional", "multi_turn", "safety", "regression"]
    input: str | list[dict]  # str for single-turn, list[dict] for multi-turn
    expected_behavior: str
    reference: str | None = None
    evaluation_type: list[Literal["rule_based", "llm_judge", "ragas", "safety"]]
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

    test_case: TestCase
    response: str
    retrieved_contexts: list[str] | None = None
    chatbot_mode: Literal["plain", "rag"]
    latency_ms: float = 0.0
    error: str | None = None
    evaluations: list[EvaluationResult] = Field(default_factory=list)
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
    results: list[TestResult] = Field(default_factory=list)
