"""Abstract base class for all evaluators."""

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
        """Evaluate a chatbot response against a test case."""

    @abstractmethod
    def name(self) -> str:
        """Return the evaluator's identifier for reports."""
