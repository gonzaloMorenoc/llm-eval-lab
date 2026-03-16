"""Abstract base classes for chatbot adapters."""

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
