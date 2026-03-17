"""Generic adapter for all OpenAI-compatible providers.

This adapter implements a single interface that works with any LLM provider
exposing an OpenAI-compatible API (e.g., Groq, Gemini, Mistral, OpenRouter).
Instead of writing a separate adapter per provider, all providers share this
class and differ only in configuration (base_url, model, api_key_env).

Why OpenAI-compatible?
  Most LLM providers adopted the OpenAI chat-completions format as a de-facto
  standard. By targeting this interface, we can swap providers via config.yaml
  without changing any code.
"""

from __future__ import annotations

import logging
import os
import time

import yaml
from openai import AsyncOpenAI

from src.chatbots.base import BaseChatbot, ChatbotResponse

logger = logging.getLogger(__name__)


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
        if not resolved_key:
            raise ValueError(
                f"API key for provider '{provider_name}' is missing. "
                f"Set the environment variable {provider_cfg['api_key_env']} or pass api_key directly."
            )

        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=resolved_key,
        )

    async def complete(self, messages: list[dict], **kwargs) -> ChatbotResponse:
        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            logger.error("API call failed for %s/%s: %s", self._provider, self._model, e)
            raise RuntimeError(
                f"API call to {self._provider}/{self._model} failed: {type(e).__name__}: {e}"
            ) from e

        latency = (time.perf_counter() - start) * 1000

        if not response.choices:
            raise RuntimeError(f"API returned empty choices for {self._provider}/{self._model}")

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
