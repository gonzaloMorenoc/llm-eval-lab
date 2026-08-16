"""Shared utilities for the dashboard — eliminates code duplication across pages.

This module centralizes:
  - Run listing from the results directory
  - Category/severity icon and color mappings
  - Config loading with caching
"""

from __future__ import annotations

import html
import json
import logging
import os
from typing import Any

import streamlit as st

from src.dashboard.components.theme import PALETTE

logger = logging.getLogger(__name__)


def safe(value: Any) -> str:
    """Escape any value for safe interpolation into HTML rendered via
    ``unsafe_allow_html=True``. Use whenever the value originates from a
    persisted dataset, user input, model response, or any other untrusted
    source. Non-string values are stringified first.
    """
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def append_jsonl(path: str, entry: dict[str, Any]) -> None:
    """Append a single JSON object as a canonical JSONL line.

    Behaves correctly whether or not the existing file ends with a newline
    (a previous bug used ``"\\n" + json.dumps(...)`` which both produced an
    extra blank line in the middle of well-formed files and missed the
    trailing newline at EOF, slowly desynchronising the format).
    """
    needs_newline = False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS_DIR = os.path.join(_ROOT_DIR, "results")
DATASETS_DIR = os.path.join(_ROOT_DIR, "datasets")
CONFIG_PATH = os.path.join(_ROOT_DIR, "config", "config.yaml")
BASELINES_DIR = os.path.join(_ROOT_DIR, "baselines")


# ── Config Loading (cached) ──────────────────────────────────────────────────


@st.cache_data(ttl=60)
def load_config() -> dict:
    """Load config.yaml with Streamlit caching (refreshes every 60s).

    Delegates to the project-wide cached loader so the YAML file is parsed
    at most once per process, with Streamlit's TTL cache on top so the
    dashboard picks up edits without a full restart.
    """
    from src.config import load_config as _project_load_config

    return _project_load_config(refresh=True)


# ── Run Listing ───────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _read_report(path: str, mtime: float) -> dict | None:
    """Parse one run's ``report.json``, or ``None`` if it cannot be read.

    ``mtime`` is part of the cache key rather than the function body: it is what
    makes an overwritten report reload immediately. A TTL would instead choose
    between serving stale data and re-parsing files that never changed.
    """
    try:
        with open(path) as f:
            data: dict = json.load(f)
        return data
    except Exception as e:
        logger.warning("Failed to load run %s: %s", path, e)
        return None


def list_runs() -> list[dict]:
    """List all evaluation runs from the results directory, newest first.

    Also includes the latest in-memory run from session state if not yet
    persisted to disk (happens right after running an evaluation).

    Directory scanning stays uncached (a ``listdir`` plus one ``stat`` per run)
    so a newly written run shows up without anyone clearing a cache; only the
    expensive part — parsing each report, which carries every test result — is
    memoised.
    """
    runs: list[dict] = []
    if os.path.isdir(RESULTS_DIR):
        for run_id in sorted(os.listdir(RESULTS_DIR), reverse=True):
            json_path = os.path.join(RESULTS_DIR, run_id, "report.json")
            try:
                mtime = os.path.getmtime(json_path)
            except OSError:
                continue  # directory without a report.json
            data = _read_report(json_path, mtime)
            if data is not None:
                runs.append({**data, "_run_id": run_id})
    # Include the latest in-memory run if not already in the list
    last = st.session_state.get("last_summary")
    if last and not any(r.get("run_id") == last.get("run_id") for r in runs):
        runs.insert(0, {**last, "_run_id": last.get("run_id", "latest")})
    return runs


# ── Category Constants ────────────────────────────────────────────────────────

CATEGORY_ICONS: dict[str, str] = {
    "functional": "⚡",
    "safety": "🛡️",
    "regression": "🔁",
    "multi_turn": "💬",
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "functional": "Respuestas correctas a preguntas generales",
    "safety": "Intentos de ataque y contenido peligroso",
    "regression": "Estabilidad entre versiones del modelo",
    "multi_turn": "Conversaciones multi-turno coherentes",
}

CATEGORY_LABEL_COLORS: dict[str, str] = {
    "functional": PALETTE["accent"],
    "safety": PALETTE["danger"],
    "regression": PALETTE["success"],
    "multi_turn": PALETTE["info"],
}


# ── Severity Constants ────────────────────────────────────────────────────────

SEVERITY_ICONS: dict[str, str] = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
}

SEVERITY_ORDER: list[str] = ["critical", "high", "medium", "low"]


# ── Score Helpers ─────────────────────────────────────────────────────────────


def pass_rate_color(rate: float) -> str:
    """Return a color hex for a given pass rate value."""
    if rate >= 0.7:
        return PALETTE["success"]
    if rate >= 0.5:
        return PALETTE["warning"]
    return PALETTE["danger"]
