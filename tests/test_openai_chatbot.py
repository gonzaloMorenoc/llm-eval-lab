"""Tests for OpenAICompatibleChatbot (mock-based, no API keys needed).

These tests verify API key validation, error handling, and response
processing without making actual API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.chatbots.base import BaseChatbot, ChatbotResponse


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_groq_key(monkeypatch):
    """Provide a fake GROQ_API_KEY for the default provider."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-groq")


def _make_chat_response(content: str) -> MagicMock:
    """Create a mock chat completion response."""
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response.choices = [choice]
    return response


def _make_empty_choices_response() -> MagicMock:
    """Create a mock response with empty choices."""
    response = MagicMock()
    response.choices = []
    return response


# ── Initialization Tests ─────────────────────────────────────────────────────


class TestOpenAIChatbotInit:
    def test_creates_with_valid_key(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot()
        assert isinstance(chatbot, BaseChatbot)

    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        with pytest.raises(ValueError, match="API key.*missing"):
            OpenAICompatibleChatbot()

    def test_accepts_explicit_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot(api_key="explicit-key")
        assert chatbot.get_id() == "groq/llama-3.3-70b-versatile"

    def test_get_id_format(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot()
        assert "/" in chatbot.get_id()
        assert chatbot.get_id().startswith("groq/")

    def test_is_not_rag(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot()
        assert chatbot.is_rag is False

    def test_custom_provider(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-gemini")
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot(provider_name="gemini")
        assert "gemini" in chatbot.get_id()

    def test_custom_model_override(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot(model="custom-model-v2")
        assert "custom-model-v2" in chatbot.get_id()


# ── Complete Method Tests ─────────────────────────────────────────────────────


class TestOpenAIChatbotComplete:
    async def test_successful_response(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot()
        mock_response = _make_chat_response("Python is a programming language.")
        chatbot._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "What is Python?"}]
        result = await chatbot.complete(messages)

        assert isinstance(result, ChatbotResponse)
        assert result.content == "Python is a programming language."
        assert result.retrieved_contexts is None
        assert result.provider == "groq"
        assert result.latency_ms >= 0

    async def test_api_error_raises_runtime_error(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot()
        chatbot._client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(RuntimeError, match="API call.*failed"):
            await chatbot.complete(messages)

    async def test_empty_choices_raises_error(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot()
        mock_response = _make_empty_choices_response()
        chatbot._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(RuntimeError, match="empty choices"):
            await chatbot.complete(messages)

    async def test_none_content_returns_empty_string(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot()
        mock_response = _make_chat_response(None)
        mock_response.choices[0].message.content = None
        chatbot._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        result = await chatbot.complete(messages)
        assert result.content == ""

    async def test_latency_is_measured(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot()
        mock_response = _make_chat_response("Hi")
        chatbot._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        result = await chatbot.complete(messages)
        assert result.latency_ms >= 0
        assert isinstance(result.latency_ms, float)

    async def test_multi_turn_messages(self):
        from src.chatbots.openai_compatible import OpenAICompatibleChatbot

        chatbot = OpenAICompatibleChatbot()
        mock_response = _make_chat_response("Your name is Alice.")
        chatbot._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you!"},
            {"role": "user", "content": "What is my name?"},
        ]
        result = await chatbot.complete(messages)
        assert result.content == "Your name is Alice."
