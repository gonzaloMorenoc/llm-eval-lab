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
