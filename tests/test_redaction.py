"""Tests for secret redaction and error summarization.

These guarantee that API keys never leak into persisted ``report.json`` files,
whether the error surfaced through the runner or was swallowed by an evaluator
that reports its own failures (the LLM judge).
"""

from __future__ import annotations

import pytest

from src.redaction import redact_secrets, summarize_error


class TestRedactSecrets:
    @pytest.mark.parametrize(
        "raw,leak",
        [
            ("HTTP 401: invalid sk-abc123XYZdefghi from openai", "sk-abc123XYZdefghi"),
            ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9.payload.sig"),
            ("Using gsk_aBcDeF12345xyz9876 for Groq", "gsk_aBcDeF12345xyz9876"),
            ("Google key: AIzaSyA-fakekeyfortest123", "AIzaSyA-fakekeyfortest123"),
            ('config: {"api_key": "supersecret-token-xx"}', "supersecret-token-xx"),
        ],
    )
    def test_known_secret_patterns_are_redacted(self, raw: str, leak: str) -> None:
        out = redact_secrets(raw)
        assert leak not in out
        assert "[REDACTED]" in out

    def test_innocuous_text_is_untouched(self) -> None:
        assert redact_secrets("connection refused on port 8080") == "connection refused on port 8080"


class TestSummarizeError:
    def test_includes_type_and_message(self) -> None:
        exc = ValueError("payload too large")
        out = summarize_error(exc)
        assert "ValueError" in out
        assert "payload too large" in out

    def test_redacts_secrets_in_message(self) -> None:
        exc = RuntimeError("auth failed for sk-leaked-key-1234567890")
        out = summarize_error(exc)
        assert "sk-leaked-key-1234567890" not in out
        assert "[REDACTED]" in out

    def test_truncates_long_messages(self) -> None:
        exc = RuntimeError("x" * 1000)
        out = summarize_error(exc, max_len=100)
        assert len(out) <= 100 + len("RuntimeError: ") + 1  # +1 for the ellipsis
        assert out.endswith("…")
