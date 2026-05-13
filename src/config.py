"""Project-wide configuration loader.

A single LRU-cached entry point for ``config/config.yaml`` so the file is
parsed only once per process instead of once per evaluator / chatbot / report
constructor call (previously this happened dozens of times per run).

Call ``load_config()`` for the cached read, or ``load_config(refresh=True)``
to force a re-read (useful in tests and when the file changes at runtime).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))


@lru_cache(maxsize=1)
def _read_config_file() -> dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def load_config(refresh: bool = False) -> dict[str, Any]:
    """Return the parsed ``config.yaml`` content.

    The result is cached. Pass ``refresh=True`` to invalidate the cache and
    re-read the file from disk.
    """
    if refresh:
        _read_config_file.cache_clear()
    return _read_config_file()
