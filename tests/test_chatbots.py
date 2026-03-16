"""Tests for chatbot adapters."""

from __future__ import annotations

import pytest

from src.chatbots.base import BaseChatbot, BaseRAGChatbot, ChatbotResponse
from src.chatbots.mock_adapter import MockChatbot, MockRAGChatbot


class TestMockChatbot:
    async def test_is_not_rag(self, mock_chatbot: MockChatbot):
        assert mock_chatbot.is_rag is False

    async def test_get_id(self, mock_chatbot: MockChatbot):
        assert mock_chatbot.get_id() == "mock/mock-plain-v1"

    async def test_complete_returns_chatbot_response(self, mock_chatbot: MockChatbot):
        messages = [{"role": "user", "content": "What is machine learning?"}]
        response = await mock_chatbot.complete(messages)
        assert isinstance(response, ChatbotResponse)
        assert response.content
        assert response.retrieved_contexts is None
        assert response.provider == "mock"
        assert response.latency_ms > 0

    async def test_complete_keyword_matching(self, mock_chatbot: MockChatbot):
        messages = [{"role": "user", "content": "Tell me about Python"}]
        response = await mock_chatbot.complete(messages)
        assert "Python" in response.content

    async def test_complete_default_response(self, mock_chatbot: MockChatbot):
        messages = [{"role": "user", "content": "xyzzy nonsense query"}]
        response = await mock_chatbot.complete(messages)
        assert response.content  # Should return default response

    async def test_is_instance_of_base(self, mock_chatbot: MockChatbot):
        assert isinstance(mock_chatbot, BaseChatbot)
        assert not isinstance(mock_chatbot, BaseRAGChatbot)

    @pytest.mark.parametrize(
        "query,expected_keyword",
        [
            ("What is machine learning?", "learning"),
            ("Tell me about Python", "Python"),
        ],
    )
    async def test_keyword_responses(self, mock_chatbot: MockChatbot, query: str, expected_keyword: str):
        messages = [{"role": "user", "content": query}]
        response = await mock_chatbot.complete(messages)
        assert expected_keyword.lower() in response.content.lower()

    async def test_response_latency_is_positive(self, mock_chatbot: MockChatbot):
        messages = [{"role": "user", "content": "Hello"}]
        response = await mock_chatbot.complete(messages)
        assert response.latency_ms > 0

    async def test_multi_turn_messages(self, mock_chatbot: MockChatbot):
        """Mock chatbot should handle multi-turn message lists."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you!"},
            {"role": "user", "content": "What is Python?"},
        ]
        response = await mock_chatbot.complete(messages)
        assert isinstance(response, ChatbotResponse)
        assert response.content


class TestMockRAGChatbot:
    async def test_is_rag(self, mock_rag_chatbot: MockRAGChatbot):
        assert mock_rag_chatbot.is_rag is True

    async def test_get_id(self, mock_rag_chatbot: MockRAGChatbot):
        assert mock_rag_chatbot.get_id() == "mock/mock-rag-v1"

    async def test_complete_returns_contexts(self, mock_rag_chatbot: MockRAGChatbot):
        messages = [{"role": "user", "content": "What is machine learning?"}]
        response = await mock_rag_chatbot.complete(messages)
        assert isinstance(response, ChatbotResponse)
        assert response.retrieved_contexts is not None
        assert len(response.retrieved_contexts) > 0
        assert response.provider == "mock"

    async def test_retrieve(self, mock_rag_chatbot: MockRAGChatbot):
        contexts = await mock_rag_chatbot.retrieve("Tell me about Python")
        assert isinstance(contexts, list)
        assert len(contexts) > 0

    async def test_is_instance_of_base_rag(self, mock_rag_chatbot: MockRAGChatbot):
        assert isinstance(mock_rag_chatbot, BaseChatbot)
        assert isinstance(mock_rag_chatbot, BaseRAGChatbot)

    async def test_plain_vs_rag_difference(self, mock_chatbot: MockChatbot, mock_rag_chatbot: MockRAGChatbot):
        """Same query, plain returns no contexts, RAG returns contexts."""
        messages = [{"role": "user", "content": "What is machine learning?"}]
        plain_resp = await mock_chatbot.complete(messages)
        rag_resp = await mock_rag_chatbot.complete(messages)
        assert plain_resp.retrieved_contexts is None
        assert rag_resp.retrieved_contexts is not None

    async def test_contexts_are_strings(self, mock_rag_chatbot: MockRAGChatbot):
        messages = [{"role": "user", "content": "What is Python?"}]
        response = await mock_rag_chatbot.complete(messages)
        assert all(isinstance(ctx, str) for ctx in response.retrieved_contexts)
