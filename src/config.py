"""Project-wide configuration loader.

A single LRU-cached entry point for ``config/config.yaml`` so the file is
parsed only once per process instead of once per evaluator / chatbot / report
constructor call (previously this happened dozens of times per run).

Call ``load_config()`` for the cached read, or ``load_config(refresh=True)``
to force a re-read (useful in tests and when the file changes at runtime).
"""

from __future__ import annotations

import os
from copy import deepcopy
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

    Parsing is cached, but every caller gets its own deep copy. Handing out the
    cached dict itself would let any consumer that tweaks a value — the
    dashboard overriding ``runner.max_concurrent`` for one run, say — silently
    rewrite the configuration seen by every other component in the process.

    Pass ``refresh=True`` to invalidate the cache and re-read the file.
    """
    if refresh:
        _read_config_file.cache_clear()
    return deepcopy(_read_config_file())
