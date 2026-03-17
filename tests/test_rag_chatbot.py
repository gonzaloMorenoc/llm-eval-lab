"""Tests for DemoRAGChatbot (mock-based, no API keys needed).

These tests verify knowledge base loading, retrieval, error handling,
and the RAG pipeline without making actual API calls.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.chatbots.base import BaseRAGChatbot, ChatbotResponse


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


def _create_knowledge_base(entries: list[dict]) -> str:
    """Create a temporary JSONL knowledge base file and return its path.

    Automatically adds a non-empty metadata dict if not present,
    as ChromaDB requires metadata to be a non-empty dict.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for entry in entries:
        if "metadata" not in entry or not entry["metadata"]:
            entry = {**entry, "metadata": {"source": "test"}}
        tmp.write(json.dumps(entry) + "\n")
    tmp.close()
    return tmp.name


# ── Initialization Tests ─────────────────────────────────────────────────────


class TestRAGChatbotInit:
    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        with pytest.raises(ValueError, match="API key.*missing"):
            DemoRAGChatbot()

    def test_accepts_explicit_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        chatbot = DemoRAGChatbot(api_key="explicit-key")
        assert isinstance(chatbot, BaseRAGChatbot)

    def test_is_rag(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        chatbot = DemoRAGChatbot()
        assert chatbot.is_rag is True

    def test_get_id_format(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        chatbot = DemoRAGChatbot()
        assert "/" in chatbot.get_id()


# ── Knowledge Base Loading Tests ──────────────────────────────────────────────


class TestKnowledgeBaseLoading:
    def test_loads_valid_knowledge_base(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        kb_path = _create_knowledge_base([
            {"id": "kb_001", "title": "Python", "content": "Python is a language.", "metadata": {}},
            {"id": "kb_002", "title": "Java", "content": "Java is a language.", "metadata": {}},
        ])
        try:
            chatbot = DemoRAGChatbot(knowledge_base_path=kb_path)
            assert chatbot._collection.count() >= 2
        finally:
            os.unlink(kb_path)

    def test_handles_nonexistent_kb_file(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        chatbot = DemoRAGChatbot(knowledge_base_path="/nonexistent/path.jsonl")
        # Should not crash, just log a warning.
        # Note: ChromaDB collection may already have docs from default KB
        # loaded by other tests, so we just verify no exception was raised.
        assert chatbot is not None

    def test_malformed_json_raises_error(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write("not valid json\n")
        tmp.close()
        try:
            with pytest.raises(ValueError, match="Malformed JSON.*line 1"):
                DemoRAGChatbot(knowledge_base_path=tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_missing_id_field_raises_error(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        kb_path = _create_knowledge_base([
            {"title": "No ID", "content": "Missing id field."},
        ])
        try:
            with pytest.raises(ValueError, match="missing required 'id' field"):
                DemoRAGChatbot(knowledge_base_path=kb_path)
        finally:
            os.unlink(kb_path)

    def test_skips_empty_lines(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write(json.dumps({"id": "kb_001", "content": "First"}) + "\n")
        tmp.write("\n")  # Empty line
        tmp.write("   \n")  # Whitespace line
        tmp.write(json.dumps({"id": "kb_002", "content": "Second"}) + "\n")
        tmp.close()
        try:
            chatbot = DemoRAGChatbot(knowledge_base_path=tmp.name)
            assert chatbot._collection.count() >= 2
        finally:
            os.unlink(tmp.name)

    def test_skips_duplicate_ids_in_same_file(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        kb_path = _create_knowledge_base([
            {"id": "kb_dedup_1", "content": "First version"},
            {"id": "kb_dedup_1", "content": "Duplicate — should be skipped"},
            {"id": "kb_dedup_2", "content": "Second unique entry"},
        ])
        try:
            chatbot = DemoRAGChatbot(knowledge_base_path=kb_path)
            # Verify only unique IDs were added (dedup_1 + dedup_2 = 2)
            result = chatbot._collection.get(ids=["kb_dedup_1", "kb_dedup_2"])
            assert len(result["ids"]) == 2
        finally:
            os.unlink(kb_path)


# ── Retrieval Tests ───────────────────────────────────────────────────────────


class TestRetrieval:
    async def test_retrieve_returns_list(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        kb_path = _create_knowledge_base([
            {"id": "kb_ret_1", "content": "Machine learning is AI."},
            {"id": "kb_ret_2", "content": "Python is a programming language."},
        ])
        try:
            chatbot = DemoRAGChatbot(knowledge_base_path=kb_path)
            results = await chatbot.retrieve("machine learning")
            assert isinstance(results, list)
            assert len(results) > 0
            assert all(isinstance(r, str) for r in results)
        finally:
            os.unlink(kb_path)

    async def test_retrieve_returns_strings(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        kb_path = _create_knowledge_base([
            {"id": "kb_str_1", "content": "Strings are text."},
        ])
        try:
            chatbot = DemoRAGChatbot(knowledge_base_path=kb_path)
            results = await chatbot.retrieve("strings")
            assert isinstance(results, list)
            assert all(isinstance(r, str) for r in results)
        finally:
            os.unlink(kb_path)


# ── Complete Method Tests ─────────────────────────────────────────────────────


class TestRAGComplete:
    async def test_successful_rag_response(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        kb_path = _create_knowledge_base([
            {"id": "kb_comp_1", "content": "ML is artificial intelligence."},
        ])
        try:
            chatbot = DemoRAGChatbot(knowledge_base_path=kb_path)
            mock_response = _make_chat_response("ML is a branch of AI.")
            chatbot._client.chat.completions.create = AsyncMock(return_value=mock_response)

            messages = [{"role": "user", "content": "What is ML?"}]
            result = await chatbot.complete(messages)

            assert isinstance(result, ChatbotResponse)
            assert result.content == "ML is a branch of AI."
            assert result.retrieved_contexts is not None
            assert len(result.retrieved_contexts) > 0
            assert result.latency_ms >= 0
        finally:
            os.unlink(kb_path)

    async def test_api_error_raises_runtime_error(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        chatbot = DemoRAGChatbot(knowledge_base_path="/nonexistent.jsonl")
        chatbot._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API down")
        )

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(RuntimeError, match="API call.*failed"):
            await chatbot.complete(messages)

    async def test_empty_choices_raises_error(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        chatbot = DemoRAGChatbot(knowledge_base_path="/nonexistent.jsonl")
        mock_response = MagicMock()
        mock_response.choices = []
        chatbot._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(RuntimeError, match="empty choices"):
            await chatbot.complete(messages)

    async def test_extracts_last_user_message_for_retrieval(self):
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        kb_path = _create_knowledge_base([
            {"id": "kb_ext_1", "content": "Python is great."},
        ])
        try:
            chatbot = DemoRAGChatbot(knowledge_base_path=kb_path)
            mock_response = _make_chat_response("Python!")
            chatbot._client.chat.completions.create = AsyncMock(return_value=mock_response)

            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "Tell me about Python"},
            ]
            result = await chatbot.complete(messages)
            assert result.content == "Python!"
        finally:
            os.unlink(kb_path)
