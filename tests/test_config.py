"""Tests for the centralized config loader.

Ensures the LRU cache is in effect (no repeated YAML parsing per call) and
that ``refresh=True`` invalidates it correctly.
"""

from __future__ import annotations

import pytest

import src.config as config_module


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Each test starts with a cold cache so it can introspect cache_info."""
    config_module._read_config_file.cache_clear()


def test_load_config_returns_expected_shape() -> None:
    cfg = config_module.load_config()
    assert "providers" in cfg
    assert "ragas" in cfg
    assert "runner" in cfg


def test_load_config_is_cached() -> None:
    config_module.load_config()
    config_module.load_config()
    config_module.load_config()
    info = config_module._read_config_file.cache_info()
    assert info.hits == 2
    assert info.misses == 1


def test_refresh_invalidates_cache() -> None:
    first = config_module.load_config()
    # refresh=True clears the cache then performs a fresh read; the prior load
    # is therefore no longer counted in cache_info().
    refreshed = config_module.load_config(refresh=True)
    info = config_module._read_config_file.cache_info()
    # The refreshed call rebuilt the cache from scratch (1 miss, 0 hits).
    assert info.misses == 1
    assert info.hits == 0
    # And it returns the freshly-loaded dict, not a stale reference.
    assert first is not refreshed


def test_returns_independent_copies() -> None:
    """Callers must not be able to reach the cached dict.

    The dashboard overrides ``runner.max_concurrent`` on the config it gets
    back; if that were the cached object, the override would leak into every
    other consumer in the process.
    """
    first = config_module.load_config()
    second = config_module.load_config()
    assert first is not second
    assert first == second


def test_mutating_result_does_not_affect_later_loads() -> None:
    first = config_module.load_config()
    original = first["runner"]["max_concurrent"]
    first["runner"]["max_concurrent"] = 999
    assert config_module.load_config()["runner"]["max_concurrent"] == original
