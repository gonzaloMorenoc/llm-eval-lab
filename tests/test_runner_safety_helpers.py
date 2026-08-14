"""Tests for the runner's retry heuristics.

These prevent substring matches like ``"separated"`` from being misclassified
as rate-limit errors and triggering pointless retries (A6). Secret redaction
moved to ``src.redaction``; see ``tests/test_redaction.py``.
"""

from __future__ import annotations

import pytest

from src.runner.runner import _is_network_error, _is_rate_limit


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
