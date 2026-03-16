"""LLM-as-judge evaluator using an externalized rubric."""

from __future__ import annotations

import json
import os
import re

import yaml
from openai import AsyncOpenAI

from src.evaluators.base import BaseEvaluator
from src.runner.models import EvaluationResult, TestCase


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


def _load_rubric() -> str:
    rubric_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "llm_judge_rubric.txt")
    with open(os.path.abspath(rubric_path)) as f:
        return f.read()


class LLMJudgeEvaluator(BaseEvaluator):
    """Uses an LLM to score responses on clarity, safety, and instruction_following."""

    CRITERIA = ["clarity", "safety", "instruction_following"]

    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None) -> None:
        config = _load_config()
        provider_name = config.get("active_provider", "groq")
        provider_cfg = config["providers"][provider_name]

        self._model = model or provider_cfg["model"]
        self._client = client or AsyncOpenAI(
            base_url=provider_cfg["base_url"],
            api_key=os.getenv(provider_cfg["api_key_env"], ""),
        )
        self._rubric = _load_rubric()

    def name(self) -> str:
        return "llm_judge"

    async def evaluate(
        self,
        test_case: TestCase,
        response: str,
        retrieved_contexts: list[str] | None = None,
        latency_ms: float = 0.0,
    ) -> EvaluationResult:
        user_input = test_case.input
        if isinstance(user_input, list):
            user_input = json.dumps(user_input)

        prompt = self._rubric.format(
            user_input=user_input,
            expected_behavior=test_case.expected_behavior,
            response=response,
        )

        try:
            result = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = result.choices[0].message.content or ""
            scores = self._parse_scores(raw)
            global_score = sum(scores.values()) / (len(scores) * 5) if scores else 0.0

            return EvaluationResult(
                evaluator=self.name(),
                passed=global_score >= 0.6,
                score=round(global_score, 4),
                reason=f"LLM judge scores: {scores}",
                details={"criteria_scores": scores, "raw_output": raw},
            )
        except Exception as e:
            return EvaluationResult(
                evaluator=self.name(),
                passed=False,
                score=None,
                reason=f"LLM judge error: {e}",
                details={"error": str(e)},
            )

    def _parse_scores(self, raw: str) -> dict[str, int]:
        """Parse scores from LLM output. Expects lines like 'clarity: 4'."""
        scores: dict[str, int] = {}
        for criterion in self.CRITERIA:
            pattern = rf"{criterion}\s*[:=]\s*(\d)"
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                scores[criterion] = int(match.group(1))
        return scores
