"""RAGAS evaluator wrapper — encapsulates all RAGAS metrics behind BaseEvaluator."""

from __future__ import annotations

import os
from typing import Any

import yaml
from openai import AsyncOpenAI

from src.evaluators.base import BaseEvaluator
from src.runner.models import EvaluationResult, TestCase

# RAGAS imports
from ragas import SingleTurnSample
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    FactualCorrectness,
    BleuScore,
    RougeScore,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain wrappers for RAGAS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


# Metrics that require retrieved_contexts (RAG-only)
_RAG_ONLY_METRICS = {"faithfulness", "context_precision", "context_recall"}

# Metrics that require a reference answer
_REFERENCE_REQUIRED = {"factual_correctness", "context_precision", "context_recall", "bleu_score", "rouge_score"}

# Non-LLM metrics (no evaluator LLM needed)
_NON_LLM_METRICS = {"bleu_score", "rouge_score"}


class RagasEvaluator(BaseEvaluator):
    """Wraps RAGAS metrics. Activates RAG-only metrics when retrieved_contexts is provided."""

    def __init__(self) -> None:
        config = _load_config()
        ragas_cfg = config.get("ragas", {})

        self._thresholds = ragas_cfg.get("thresholds", {})
        self._pass_threshold = ragas_cfg.get("pass_threshold", 0.65)
        self._default_metrics_plain = ragas_cfg.get("default_metrics_plain", ["answer_relevancy"])
        self._default_metrics_rag = ragas_cfg.get("default_metrics_rag", ["answer_relevancy", "faithfulness"])

        # Initialize RAGAS evaluator LLM and embeddings
        evaluator_model = ragas_cfg.get("evaluator_llm", "gpt-4o-mini")
        embeddings_model = ragas_cfg.get("embeddings_model", "text-embedding-3-small")
        openai_key = os.getenv("OPENAI_API_KEY", "")

        self._llm = LangchainLLMWrapper(ChatOpenAI(model=evaluator_model, api_key=openai_key))
        self._embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=embeddings_model, api_key=openai_key)
        )

    def name(self) -> str:
        return "ragas"

    def _build_metric(self, metric_name: str) -> Any:
        """Instantiate a RAGAS metric by name, injecting LLM/embeddings as needed."""
        constructors: dict[str, type] = {
            "answer_relevancy": AnswerRelevancy,
            "factual_correctness": FactualCorrectness,
            "faithfulness": Faithfulness,
            "bleu_score": BleuScore,
            "rouge_score": RougeScore,
        }

        # Try importing context_precision and context_recall (may vary by ragas version)
        try:
            from ragas.metrics import LLMContextPrecisionWithReference
            constructors["context_precision"] = LLMContextPrecisionWithReference
        except ImportError:
            try:
                from ragas.metrics import ContextPrecision
                constructors["context_precision"] = ContextPrecision
            except ImportError:
                pass

        try:
            from ragas.metrics import LLMContextRecall
            constructors["context_recall"] = LLMContextRecall
        except ImportError:
            try:
                from ragas.metrics import ContextRecall
                constructors["context_recall"] = ContextRecall
            except ImportError:
                pass

        if metric_name not in constructors:
            return None

        cls = constructors[metric_name]

        if metric_name in _NON_LLM_METRICS:
            return cls()

        # LLM-based metrics need llm; AnswerRelevancy also needs embeddings
        if metric_name == "answer_relevancy":
            return cls(llm=self._llm, embeddings=self._embeddings)
        return cls(llm=self._llm)

    def _resolve_metrics(
        self,
        test_case: TestCase,
        retrieved_contexts: list[str] | None,
    ) -> list[str]:
        """Determine which metrics to run based on the test case and mode."""
        is_rag = retrieved_contexts is not None

        # Start with test-case-specific metrics or defaults
        if test_case.ragas_metrics:
            requested = test_case.ragas_metrics
        else:
            requested = list(self._default_metrics_rag if is_rag else self._default_metrics_plain)

        resolved = []
        for m in requested:
            # Skip RAG-only metrics in plain mode
            if m in _RAG_ONLY_METRICS and not is_rag:
                continue
            # Skip metrics needing reference when reference is absent
            if m in _REFERENCE_REQUIRED and test_case.reference is None:
                continue
            resolved.append(m)
        return resolved

    async def evaluate(
        self,
        test_case: TestCase,
        response: str,
        retrieved_contexts: list[str] | None = None,
        latency_ms: float = 0.0,
    ) -> EvaluationResult:
        metrics_to_run = self._resolve_metrics(test_case, retrieved_contexts)
        if not metrics_to_run:
            return EvaluationResult(
                evaluator=self.name(),
                passed=True,
                score=None,
                reason="No applicable RAGAS metrics for this test case.",
                details={},
            )

        # Build the SingleTurnSample
        user_input = test_case.input
        if isinstance(user_input, list):
            # For multi-turn, use the last user message
            for msg in reversed(user_input):
                if msg.get("role") == "user":
                    user_input = msg["content"]
                    break
            else:
                user_input = str(test_case.input)

        sample_kwargs: dict[str, Any] = {
            "user_input": user_input,
            "response": response,
        }
        if test_case.reference is not None:
            sample_kwargs["reference"] = test_case.reference
        if retrieved_contexts is not None:
            sample_kwargs["retrieved_contexts"] = retrieved_contexts

        sample = SingleTurnSample(**sample_kwargs)

        # Score each metric
        metric_scores: dict[str, float] = {}
        errors: dict[str, str] = {}

        for metric_name in metrics_to_run:
            metric = self._build_metric(metric_name)
            if metric is None:
                errors[metric_name] = f"Metric '{metric_name}' not available in this RAGAS version"
                continue
            try:
                score = await metric.single_turn_ascore(sample)
                metric_scores[metric_name] = float(score)
            except Exception as e:
                errors[metric_name] = str(e)

        # Calculate aggregate score and pass/fail
        if metric_scores:
            avg_score = sum(metric_scores.values()) / len(metric_scores)
        else:
            avg_score = 0.0

        # Check per-metric thresholds
        threshold_results: dict[str, bool] = {}
        for m, s in metric_scores.items():
            threshold = self._thresholds.get(m, self._pass_threshold)
            threshold_results[m] = s >= threshold

        all_passed = all(threshold_results.values()) if threshold_results else len(errors) == 0

        reasons = []
        for m, s in metric_scores.items():
            t = self._thresholds.get(m, self._pass_threshold)
            status = "PASS" if s >= t else "FAIL"
            reasons.append(f"{m}: {s:.3f} (threshold {t}, {status})")
        for m, e in errors.items():
            reasons.append(f"{m}: ERROR — {e}")

        return EvaluationResult(
            evaluator=self.name(),
            passed=all_passed,
            score=round(avg_score, 4) if metric_scores else None,
            reason="; ".join(reasons),
            details={
                "metric_scores": metric_scores,
                "threshold_results": threshold_results,
                "errors": errors,
            },
        )
