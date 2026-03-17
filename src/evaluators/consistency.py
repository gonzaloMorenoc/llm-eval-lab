"""Consistency evaluator — measures response stability across multiple runs.

A good chatbot should give similar answers to the same question. This evaluator
measures that stability using string similarity.

Algorithm:
  Uses Python's SequenceMatcher (Ratcliff/Obershelp algorithm) to compute
  similarity between pairs of responses. The algorithm finds the longest
  common subsequence and returns a ratio from 0.0 (completely different)
  to 1.0 (identical).

  Complexity: O(n²) where n is the length of the strings. For typical chatbot
  responses (<2000 chars), this is fast enough. For very long responses,
  consider switching to cosine similarity on TF-IDF vectors or embeddings.

Two modes:
  1. **Multi-response**: If the test case provides multiple prior responses
     (via metadata.consistency_responses), computes pairwise similarity
     across ALL responses including the current one.
  2. **Single vs reference**: If only one response and a reference answer
     are available, computes similarity between the response and the
     reference. Useful for regression testing.

The similarity threshold (default 0.6) is configurable in config.yaml.
A threshold of 0.6 is lenient — it allows paraphrasing while catching
completely different answers.
"""

from __future__ import annotations

import os
from difflib import SequenceMatcher

import yaml

from src.evaluators.base import BaseEvaluator
from src.runner.models import EvaluationResult, TestCase


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


def _similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings (0.0 to 1.0)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def compute_consistency_score(responses: list[str]) -> tuple[float, dict]:
    """Compute pairwise similarity across multiple responses.

    Returns (avg_similarity, details_dict).
    """
    if len(responses) < 2:
        return 1.0, {"pairs": 0, "note": "Single response, consistency is trivially 1.0"}

    similarities: list[float] = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            similarities.append(_similarity(responses[i], responses[j]))

    avg = sum(similarities) / len(similarities)
    return round(avg, 4), {
        "pairs": len(similarities),
        "min_similarity": round(min(similarities), 4),
        "max_similarity": round(max(similarities), 4),
        "all_similarities": [round(s, 4) for s in similarities],
    }


class ConsistencyEvaluator(BaseEvaluator):
    """Evaluates whether a chatbot gives consistent responses to the same input.

    This evaluator is designed to be used with pre-collected multiple responses
    passed via the `details` field in metadata, or with a single response where
    it checks consistency against the reference answer.
    """

    def __init__(self) -> None:
        config = _load_config()
        consistency_cfg = config.get("consistency", {})
        self._threshold = consistency_cfg.get("similarity_threshold", 0.6)

    def name(self) -> str:
        return "consistency"

    async def evaluate(
        self,
        test_case: TestCase,
        response: str,
        retrieved_contexts: list[str] | None = None,
        latency_ms: float = 0.0,
    ) -> EvaluationResult:
        # If multiple responses were pre-collected, use them
        prior_responses = test_case.metadata.get("consistency_responses", [])

        if prior_responses:
            all_responses = prior_responses + [response]
            score, details = compute_consistency_score(all_responses)
            passed = score >= self._threshold

            return EvaluationResult(
                evaluator=self.name(),
                passed=passed,
                score=score,
                reason=(
                    f"Consistency score {score:.3f} across {len(all_responses)} responses "
                    f"({'PASS' if passed else 'FAIL'}, threshold {self._threshold})"
                ),
                details=details,
            )

        # Single response: compare against reference if available
        if test_case.reference:
            score = _similarity(response, test_case.reference)
            passed = score >= self._threshold

            return EvaluationResult(
                evaluator=self.name(),
                passed=passed,
                score=round(score, 4),
                reason=(
                    f"Response-reference similarity {score:.3f} "
                    f"({'PASS' if passed else 'FAIL'}, threshold {self._threshold})"
                ),
                details={"mode": "single_vs_reference", "similarity": round(score, 4)},
            )

        return EvaluationResult(
            evaluator=self.name(),
            passed=True,
            score=None,
            reason="No reference or prior responses available for consistency check.",
            details={},
        )
