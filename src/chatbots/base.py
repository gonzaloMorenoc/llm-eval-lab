"""Abstract base classes for chatbot adapters.

The chatbot layer defines a common interface for interacting with any LLM,
regardless of provider or mode (plain vs RAG). This decoupling is key:

  - **BaseChatbot**: Plain LLM interaction. Takes messages, returns a response.
    The `is_rag` property returns False, and `retrieved_contexts` is always None.

  - **BaseRAGChatbot**: Extends BaseChatbot with a `retrieve()` step before
    generation. The `complete()` method must (1) retrieve relevant documents,
    (2) inject them into the prompt, and (3) return them in `retrieved_contexts`
    so RAG-specific evaluators (Faithfulness, ContextPrecision) can assess
    retrieval quality.

  - **ChatbotResponse**: Standardized output containing the response text,
    provider/model metadata, latency in milliseconds, and optionally the
    retrieved contexts.

Why this abstraction?
  The EvalRunner doesn't care which provider or mode is used. It calls
  `chatbot.complete(messages)` and gets a ChatbotResponse. This lets us
  swap providers via config without touching evaluation logic.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class ChatbotResponse(BaseModel):
    """Standardized response from any chatbot adapter."""

    content: str
    retrieved_contexts: list[str] | None = None
    provider: str
    model: str
    latency_ms: float


class BaseChatbot(ABC):
    """Base interface for plain LLM chatbots (no retrieval)."""

    @abstractmethod
    async def complete(self, messages: list[dict], **kwargs) -> ChatbotResponse:
        """Generate a response. retrieved_contexts will always be None."""

    @abstractmethod
    def get_id(self) -> str:
        """Return '{provider}/{model}'."""

    @property
    def is_rag(self) -> bool:
        return False


class BaseRAGChatbot(BaseChatbot, ABC):
    """Extends BaseChatbot with document retrieval.

    The complete() implementation must:
    1. Call retrieve() to get relevant context
    2. Inject that context into the LLM prompt
    3. Return ChatbotResponse with retrieved_contexts populated
    """

    @abstractmethod
    async def retrieve(self, query: str) -> list[str]:
        """Retrieve relevant documents for the query."""

    @property
    def is_rag(self) -> bool:
        return True
