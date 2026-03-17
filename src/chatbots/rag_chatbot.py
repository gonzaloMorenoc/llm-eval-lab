"""Demo RAG chatbot using ChromaDB as a local vector store.

RAG (Retrieval-Augmented Generation) combines document retrieval with LLM
generation. The pipeline:

  1. **Retrieve**: given a user query, find the top-k most relevant documents
     from a vector store (here ChromaDB with its built-in embedding function).
  2. **Augment**: inject the retrieved documents into the LLM prompt as a
     system message so the model can ground its answer in factual context.
  3. **Generate**: call the LLM with the augmented prompt and return both the
     response and the retrieved contexts (needed by RAG-specific evaluators
     like Faithfulness or ContextPrecision).

Why a system message for context?
  Placing context in the system message keeps it separate from the user's
  conversational history, reducing the chance the model confuses user
  instructions with reference material.
"""

from __future__ import annotations

import json
import logging
import os
import time

import chromadb
import yaml
from openai import AsyncOpenAI

from src.chatbots.base import BaseRAGChatbot, ChatbotResponse

logger = logging.getLogger(__name__)


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
        if not resolved_key:
            raise ValueError(
                f"API key for provider '{provider_name}' is missing. "
                f"Set the environment variable {provider_cfg['api_key_env']} or pass api_key directly."
            )

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
            logger.warning("Knowledge base file not found: %s", path)
            return
        docs, ids, metadatas = [], [], []
        with open(path) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Malformed JSON in knowledge base at line {line_num}: {e}"
                    ) from e

                if "id" not in entry:
                    raise ValueError(
                        f"Knowledge base entry at line {line_num} missing required 'id' field"
                    )

                doc_id = entry["id"]
                # Skip if already exists in collection or in current batch
                existing = self._collection.get(ids=[doc_id])
                if existing["ids"] or doc_id in ids:
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
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=augmented_messages,
                **kwargs,
            )
        except Exception as e:
            logger.error("RAG generation failed for %s/%s: %s", self._provider, self._model, e)
            raise RuntimeError(
                f"API call to {self._provider}/{self._model} failed: {type(e).__name__}: {e}"
            ) from e

        latency = (time.perf_counter() - start) * 1000

        if not response.choices:
            raise RuntimeError(f"API returned empty choices for {self._provider}/{self._model}")

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
