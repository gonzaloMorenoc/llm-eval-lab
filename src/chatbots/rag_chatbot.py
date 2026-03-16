"""Demo RAG chatbot using ChromaDB as a local vector store."""

from __future__ import annotations

import json
import os
import time

import chromadb
import yaml
from openai import AsyncOpenAI

from src.chatbots.base import BaseRAGChatbot, ChatbotResponse


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


class DemoRAGChatbot(BaseRAGChatbot):
    """RAG chatbot that retrieves from ChromaDB and generates via any OpenAI-compatible provider."""

    def __init__(
        self,
        provider_name: str | None = None,
        knowledge_base_path: str | None = None,
        api_key: str | None = None,
    ) -> None:
        config = _load_config()
        provider_name = provider_name or config.get("active_provider", "groq")
        provider_cfg = config["providers"][provider_name]
        rag_cfg = config.get("rag", {})

        self._provider = provider_name
        self._model = provider_cfg["model"]
        self._top_k = rag_cfg.get("top_k", 3)
        self._context_template = rag_cfg.get(
            "context_template",
            "Use the following context to answer the user's question.\n\nContext:\n{context}",
        )

        resolved_key = api_key or os.getenv(provider_cfg["api_key_env"], "")
        self._client = AsyncOpenAI(
            base_url=provider_cfg["base_url"],
            api_key=resolved_key,
        )

        # Initialize ChromaDB in-memory
        self._chroma_client = chromadb.Client()
        collection_name = rag_cfg.get("collection_name", "demo_knowledge_base")
        self._collection = self._chroma_client.get_or_create_collection(name=collection_name)

        # Load knowledge base
        kb_path = knowledge_base_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "datasets", "rag_knowledge_base.jsonl"
        )
        self._load_knowledge_base(os.path.abspath(kb_path))

    def _load_knowledge_base(self, path: str) -> None:
        if not os.path.exists(path):
            return
        docs, ids, metadatas = [], [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                doc_id = entry["id"]
                # Skip if already exists
                existing = self._collection.get(ids=[doc_id])
                if existing["ids"]:
                    continue
                content = entry.get("content", "")
                title = entry.get("title", "")
                docs.append(f"{title}\n{content}" if title else content)
                ids.append(doc_id)
                metadatas.append(entry.get("metadata", {}))
        if docs:
            self._collection.add(documents=docs, ids=ids, metadatas=metadatas)

    async def retrieve(self, query: str) -> list[str]:
        results = self._collection.query(query_texts=[query], n_results=self._top_k)
        documents = results.get("documents", [[]])
        return documents[0] if documents else []

    async def complete(self, messages: list[dict], **kwargs) -> ChatbotResponse:
        # Extract the latest user query for retrieval
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        # Retrieve context
        contexts = await self.retrieve(query)

        # Build augmented messages with context in system prompt
        context_block = "\n\n---\n\n".join(contexts)
        system_content = self._context_template.format(context=context_block)
        augmented_messages = [{"role": "system", "content": system_content}] + messages

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=augmented_messages,
            **kwargs,
        )
        latency = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""

        return ChatbotResponse(
            content=content,
            retrieved_contexts=contexts,
            provider=self._provider,
            model=self._model,
            latency_ms=round(latency, 2),
        )

    def get_id(self) -> str:
        return f"{self._provider}/{self._model}"
