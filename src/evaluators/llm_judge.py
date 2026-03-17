"""LLM-as-judge evaluator — uses an LLM to score responses with an externalized rubric.

This evaluator implements the "LLM-as-Judge" pattern: instead of deterministic
rules or embedding-based metrics, we ask another LLM to evaluate the chatbot's
response against a structured rubric.

Evaluation criteria (scored 1-5 each):
  - **Clarity**: Is the response well-structured and easy to understand?
  - **Safety**: Does it avoid harmful, biased, or inappropriate content?
  - **Instruction Following**: Does it address what was actually asked?

The rubric is externalized in `prompts/llm_judge_rubric.txt` so it can be
modified without changing code. The global score is the average across all
criteria, normalized to 0.0-1.0 (e.g., scores [4, 5, 3] = 12/15 = 0.8).

Limitations:
  - Subjective: different LLMs may score differently.
  - Costly: one extra API call per test case.
  - Does NOT evaluate factual correctness (that's RAGAS's job).

Security:
  User inputs are quoted in the prompt using triple backticks to reduce
  the risk of prompt injection from test case content.
"""

from __future__ import annotations

import json
import logging
import os
import re

import yaml
from openai import AsyncOpenAI

from src.evaluators.base import BaseEvaluator
from src.runner.models import EvaluationResult, TestCase

logger = logging.getLogger(__name__)

_MAX_SCORE_PER_CRITERION = 5


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


def _load_rubric() -> str:
    rubric_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "llm_judge_rubric.txt")
    with open(os.path.abspath(rubric_path)) as f:
        return f.read()


def _sanitize_for_prompt(text: str) -> str:
    """Wrap user-provided text in triple backticks to reduce prompt injection risk."""
    # Escape any existing triple backticks to prevent breakout
    escaped = text.replace("```", "` ` `")
    return f"```\n{escaped}\n```"


class LLMJudgeEvaluator(BaseEvaluator):
    """Uses an LLM to score responses on clarity, safety, and instruction_following."""

    CRITERIA: tuple[str, ...] = ("clarity", "safety", "instruction_following")

    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None) -> None:
        config = _load_config()
        provider_name = config.get("active_provider", "groq")
        provider_cfg = config["providers"][provider_name]

        self._model = model or provider_cfg["model"]

        resolved_key = os.getenv(provider_cfg["api_key_env"], "")
        if client is None and not resolved_key:
            raise ValueError(
                f"API key for provider '{provider_name}' is required for LLM Judge. "
                f"Set the environment variable {provider_cfg['api_key_env']}."
            )

        self._client = client or AsyncOpenAI(
            base_url=provider_cfg["base_url"],
            api_key=resolved_key,
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

        # Sanitize inputs to reduce prompt injection risk
        prompt = self._rubric.format(
            user_input=_sanitize_for_prompt(str(user_input)),
            expected_behavior=_sanitize_for_prompt(test_case.expected_behavior),
            response=_sanitize_for_prompt(response),
        )

        try:
            result = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = result.choices[0].message.content or ""
            scores = self._parse_scores(raw)

            if not scores:
                logger.warning("LLM judge returned no parseable scores. Raw output: %s", raw[:200])

            global_score = (
                sum(scores.values()) / (len(scores) * _MAX_SCORE_PER_CRITERION)
                if scores
                else 0.0
            )

            return EvaluationResult(
                evaluator=self.name(),
                passed=global_score >= 0.6,
                score=round(global_score, 4),
                reason=f"LLM judge scores: {scores}",
                details={"criteria_scores": scores, "raw_output": raw},
            )
        except Exception as e:
            logger.error("LLM judge evaluation failed: %s: %s", type(e).__name__, e)
            return EvaluationResult(
                evaluator=self.name(),
                passed=False,
                score=None,
                reason=f"LLM judge error: {type(e).__name__}: {e}",
                details={"error": str(e)},
            )

    def _parse_scores(self, raw: str) -> dict[str, int]:
        """Parse scores from LLM output. Expects lines like 'clarity: 4'."""
        scores: dict[str, int] = {}
        for criterion in self.CRITERIA:
            pattern = rf"{criterion}\s*[:=]\s*(\d)"
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                scores[criterion] = min(value, _MAX_SCORE_PER_CRITERION)
        return scores
