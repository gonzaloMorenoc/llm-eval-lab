"""Tests for the runner-level safety helpers: secret redaction, error
summarization, rate-limit and network-error heuristics.

These guarantees prevent two regressions:
  - API keys leaking into persisted ``report.json`` files (A3).
  - Substring matches like ``"separated"`` being misclassified as rate-limit
    errors and triggering pointless retries (A6).
"""

from __future__ import annotations

import pytest

from src.runner.runner import (
    _is_network_error,
    _is_rate_limit,
    _redact_secrets,
    _summarize_error,
)


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
        out = _redact_secrets(raw)
        assert leak not in out
        assert "[REDACTED]" in out

    def test_innocuous_text_is_untouched(self) -> None:
        assert _redact_secrets("connection refused on port 8080") == "connection refused on port 8080"


class TestSummarizeError:
    def test_includes_type_and_message(self) -> None:
        exc = ValueError("payload too large")
        out = _summarize_error(exc)
        assert "ValueError" in out
        assert "payload too large" in out

    def test_redacts_secrets_in_message(self) -> None:
        exc = RuntimeError("auth failed for sk-leaked-key-1234567890")
        out = _summarize_error(exc)
        assert "sk-leaked-key-1234567890" not in out
        assert "[REDACTED]" in out

    def test_truncates_long_messages(self) -> None:
        exc = RuntimeError("x" * 1000)
        out = _summarize_error(exc, max_len=100)
        assert len(out) <= 100 + len("RuntimeError: ") + 1  # +1 for the ellipsis
        assert out.endswith("…")


class TestIsRateLimit:
    def test_detects_429_status_code(self) -> None:
        exc = RuntimeError("upstream error")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_rate_limit(exc) is True

    def test_detects_class_name(self) -> None:
        class RateLimitError(Exception):
            pass

        assert _is_rate_limit(RateLimitError("slow down")) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "429 too many requests",
            "rate limit exceeded for org-foo",
            "Rate-Limit hit after 100 requests",
        ],
    )
    def test_detects_message_substrings(self, msg: str) -> None:
        assert _is_rate_limit(RuntimeError(msg)) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "narrate the response",  # contains 'rate' as a substring of 'narrate'
            "separated by commas",
            "operator timed out",
        ],
    )
    def test_no_false_positives_on_substrings(self, msg: str) -> None:
        assert _is_rate_limit(RuntimeError(msg)) is False


class TestIsNetworkError:
    def test_detects_timeout(self) -> None:
        assert _is_network_error(RuntimeError("request timed out after 30s")) is True

    def test_detects_connection_error(self) -> None:
        assert _is_network_error(ConnectionError("connection refused")) is True

    def test_does_not_match_unrelated_text(self) -> None:
        assert _is_network_error(ValueError("malformed JSON")) is False
