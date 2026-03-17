"""Abstract base class for all evaluators.

Every evaluator in this framework implements the same interface: given a test
case and a chatbot response, return an EvaluationResult with a pass/fail
verdict, a 0.0-1.0 score, and a human-readable reason.

The framework ships with six evaluators, each targeting a different quality
dimension:

  | Evaluator          | What it measures              | Needs API key? |
  |--------------------|-------------------------------|----------------|
  | RuleBasedEvaluator | Non-empty, length, keywords   | No             |
  | SafetyEvaluator    | Injection, leaks, refusal     | No             |
  | ConsistencyEval.   | Response stability            | No             |
  | RagasEvaluator     | RAG quality (RAGAS metrics)   | OPENAI_API_KEY |
  | DeepEvalEvaluator  | Hallucination, bias, toxicity | OPENAI_API_KEY |
  | LLMJudgeEvaluator  | Clarity, safety, following    | Provider key   |

To create a custom evaluator:
  1. Subclass BaseEvaluator
  2. Implement evaluate() and name()
  3. Register it in the runner's evaluator dict
"""

from abc import ABC, abstractmethod

from src.runner.models import EvaluationResult, TestCase


class BaseEvaluator(ABC):
    """Contract that every evaluator must implement."""

    @abstractmethod
    async def evaluate(
        self,
        test_case: TestCase,
        response: str,
        retrieved_contexts: list[str] | None = None,
        latency_ms: float = 0.0,
    ) -> EvaluationResult:
        """Evaluate a chatbot response against a test case.

        Args:
            test_case: The test case being evaluated (includes expected_behavior, reference, etc.).
            response: The chatbot's generated response text.
            retrieved_contexts: Documents retrieved by RAG (None in plain mode).
            latency_ms: Time taken by the chatbot to respond, in milliseconds.

        Returns:
            EvaluationResult with passed (bool), score (0.0-1.0 or None), and reason.
        """

    @abstractmethod
    def name(self) -> str:
        """Return the evaluator's identifier (e.g. 'rule_based', 'ragas')."""
