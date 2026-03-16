"""Generic adapter for all OpenAI-compatible providers."""

from __future__ import annotations

import os
import time

import yaml
from openai import AsyncOpenAI

from src.chatbots.base import BaseChatbot, ChatbotResponse


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


class OpenAICompatibleChatbot(BaseChatbot):
    """Single adapter that works with Groq, Gemini, Mistral, OpenRouter, etc."""

    def __init__(
        self,
        provider_name: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        config = load_config()
        provider_name = provider_name or config.get("active_provider", "groq")
        provider_cfg = config["providers"][provider_name]

        self._provider = provider_name
        self._model = model or provider_cfg["model"]
        self._base_url = base_url or provider_cfg["base_url"]

        resolved_key = api_key or os.getenv(provider_cfg["api_key_env"], "")
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=resolved_key,
        )

    async def complete(self, messages: list[dict], **kwargs) -> ChatbotResponse:
        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        latency = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""
        return ChatbotResponse(
            content=content,
            retrieved_contexts=None,
            provider=self._provider,
            model=self._model,
            latency_ms=round(latency, 2),
        )

    def get_id(self) -> str:
        return f"{self._provider}/{self._model}"
