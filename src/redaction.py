"""Secret redaction and error summarization for anything we persist.

Every string that ends up in ``results/<run_id>/report.json`` passes through
here first. Exception messages from LLM SDKs routinely echo back request
headers, URLs with query-string tokens, or the raw key that failed to
authenticate — none of which should survive into a file a user might commit
or share.

Two entry points:
  - ``redact_secrets()``: replaces anything shaped like an API key or bearer
    token with ``[REDACTED]``.
  - ``summarize_error()``: builds a short, redacted, type-prefixed string from
    an exception. Full tracebacks stay in the application log (server-side).
"""

from __future__ import annotations

import re

# Patterns that look like secrets (OpenAI/Anthropic/Groq/etc. API keys, bearer tokens).
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"gsk_[A-Za-z0-9_\-]{12,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"(?i)api[_-]?key[\"'=:\s]+[A-Za-z0-9_\-]{8,}"),
)

DEFAULT_MAX_LEN = 240


def redact_secrets(text: str) -> str:
    """Strip anything that looks like an API key or bearer token from a string."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def summarize_error(exc: BaseException, max_len: int = DEFAULT_MAX_LEN) -> str:
    """Build a short, redacted error string safe to persist in reports."""
    msg = redact_secrets(str(exc))
    if len(msg) > max_len:
        msg = msg[:max_len] + "…"
    return f"{type(exc).__name__}: {msg}"
