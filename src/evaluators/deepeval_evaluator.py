"""DeepEval evaluator — complements RAGAS with hallucination, bias, and toxicity metrics.

DeepEval is an open-source LLM evaluation framework that provides metrics for
dimensions not fully covered by RAGAS:

  - **Hallucination**: Does the response contain claims not supported by the
    retrieved context? Uses an LLM to extract and verify each factual claim.
    RAG-only — only meaningful when context is available.

  - **Bias**: Does the response exhibit gender, racial, or other biases?
    Uses an LLM to detect biased opinions presented as facts.

  - **Toxicity**: Does the response contain toxic, offensive, or harmful
    language? Checks for insults, threats, profanity, etc.

  - **AnswerRelevancy**: Similar to RAGAS but with DeepEval's implementation.
    Checks whether the response is pertinent to the input question.

  - **Faithfulness**: Are all claims in the response traceable back to the
    provided context? RAG-only. Complements the hallucination check.

  - **GEval**: A general-purpose LLM evaluation using custom criteria.
    Here configured for "Correctness" — factual accuracy and completeness.

Why both RAGAS and DeepEval?
  They use different methodologies. Running both gives a more robust signal:
  if RAGAS says "good" but DeepEval flags hallucination, it's worth investigating.

Configuration:
  Defined in config.yaml under the `deepeval` section.
  Requires OPENAI_API_KEY for the evaluator LLM (gpt-4o-mini by default).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

from src.evaluators.base import BaseEvaluator
from src.runner.models import EvaluationResult, TestCase

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


# Metrics that require retrieved_contexts (RAG-only)
_RAG_ONLY_METRICS = {"hallucination", "faithfulness"}

# Metrics that require a reference answer
_REFERENCE_REQUIRED = {"answer_relevancy"}


class DeepEvalEvaluator(BaseEvaluator):
    """Wraps DeepEval metrics: hallucination, bias, toxicity, answer relevancy, and GEval."""

    def __init__(self) -> None:
        config = _load_config()
        deepeval_cfg = config.get("deepeval", {})

        self._thresholds = deepeval_cfg.get("thresholds", {})
        self._pass_threshold = deepeval_cfg.get("pass_threshold", 0.5)
        self._default_metrics_plain = deepeval_cfg.get(
            "default_metrics_plain", ["answer_relevancy", "bias", "toxicity"]
        )
        self._default_metrics_rag = deepeval_cfg.get(
            "default_metrics_rag", ["answer_relevancy", "hallucination", "bias", "toxicity"]
        )
        self._model = deepeval_cfg.get("evaluator_model", "gpt-4o-mini")

        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for DeepEval evaluation. "
                "DeepEval uses an LLM (gpt-4o-mini by default) to compute its metrics."
            )

    def name(self) -> str:
        return "deepeval"

    def _build_metric(self, metric_name: str) -> Any:
        """Instantiate a DeepEval metric by name."""
        threshold = self._thresholds.get(metric_name, self._pass_threshold)

        if metric_name == "answer_relevancy":
            from deepeval.metrics import AnswerRelevancyMetric
            return AnswerRelevancyMetric(threshold=threshold, model=self._model)

        if metric_name == "hallucination":
            from deepeval.metrics import HallucinationMetric
            return HallucinationMetric(threshold=threshold, model=self._model)

        if metric_name == "bias":
            from deepeval.metrics import BiasMetric
            return BiasMetric(threshold=threshold, model=self._model)

        if metric_name == "toxicity":
            from deepeval.metrics import ToxicityMetric
            return ToxicityMetric(threshold=threshold, model=self._model)

        if metric_name == "faithfulness":
            from deepeval.metrics import FaithfulnessMetric
            return FaithfulnessMetric(threshold=threshold, model=self._model)

        if metric_name == "g_eval":
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCaseParams
            return GEval(
                name="Correctness",
                criteria="Determine whether the response is factually correct and complete.",
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=threshold,
                model=self._model,
            )

        return None

    def _resolve_metrics(
        self,
        test_case: TestCase,
        retrieved_contexts: list[str] | None,
    ) -> list[str]:
        """Determine which metrics to run based on the test case and mode."""
        is_rag = retrieved_contexts is not None

        requested = list(self._default_metrics_rag if is_rag else self._default_metrics_plain)

        # Use test-case-specific deepeval_metrics if present in metadata
        if test_case.metadata.get("deepeval_metrics"):
            requested = test_case.metadata["deepeval_metrics"]

        resolved = []
        for m in requested:
            if m in _RAG_ONLY_METRICS and not is_rag:
                continue
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
                reason="No applicable DeepEval metrics for this test case.",
                details={},
            )

        # Build the DeepEval test case
        from deepeval.test_case import LLMTestCase

        user_input = test_case.input
        if isinstance(user_input, list):
            for msg in reversed(user_input):
                if msg.get("role") == "user":
                    user_input = msg["content"]
                    break
            else:
                user_input = str(test_case.input)

        tc_kwargs: dict[str, Any] = {
            "input": user_input,
            "actual_output": response,
        }
        if test_case.reference is not None:
            tc_kwargs["expected_output"] = test_case.reference
        if retrieved_contexts is not None:
            tc_kwargs["retrieval_context"] = retrieved_contexts
            tc_kwargs["context"] = retrieved_contexts

        deepeval_tc = LLMTestCase(**tc_kwargs)

        # Score each metric
        metric_scores: dict[str, float] = {}
        metric_passed: dict[str, bool] = {}
        errors: dict[str, str] = {}

        for metric_name in metrics_to_run:
            metric = self._build_metric(metric_name)
            if metric is None:
                errors[metric_name] = f"Metric '{metric_name}' not available"
                continue
            try:
                metric.measure(deepeval_tc)
                score = float(metric.score)
                metric_scores[metric_name] = score
                metric_passed[metric_name] = metric.is_successful()
            except Exception as e:
                logger.warning("DeepEval metric '%s' failed: %s: %s", metric_name, type(e).__name__, e)
                errors[metric_name] = f"{type(e).__name__}: {e}"

        # Calculate aggregate
        if metric_scores:
            avg_score = sum(metric_scores.values()) / len(metric_scores)
        else:
            avg_score = 0.0

        all_passed = all(metric_passed.values()) if metric_passed else len(errors) == 0

        reasons = []
        for m, s in metric_scores.items():
            t = self._thresholds.get(m, self._pass_threshold)
            status = "PASS" if metric_passed.get(m, False) else "FAIL"
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
                "metric_passed": metric_passed,
                "errors": errors,
            },
        )
