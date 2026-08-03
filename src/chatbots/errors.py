"""Chatbot adapter exceptions.

Adapters wrap provider SDK errors so the message carries the provider/model
that failed. Plain ``RuntimeError`` would lose the information the runner uses
to decide whether an error is worth retrying — most importantly the HTTP
``status_code`` that distinguishes a 429 (retry) from a 400 (don't).

``ChatbotAPIError`` keeps that context: it copies ``status_code`` from the
original exception and is always raised with ``from e`` so the runner can walk
``__cause__`` to inspect the underlying SDK error type.
"""

from __future__ import annotations


class ChatbotAPIError(RuntimeError):
    """Raised when a provider API call fails. Preserves the original status code."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def wrap_api_error(exc: Exception, provider: str, model: str) -> ChatbotAPIError:
    """Build a ChatbotAPIError describing a failed call to ``provider/model``."""
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None
    return ChatbotAPIError(
        f"API call to {provider}/{model} failed: {type(exc).__name__}: {exc}",
        status_code=status_code,
    )
