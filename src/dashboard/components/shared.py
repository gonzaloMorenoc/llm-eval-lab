"""Shared utilities for the dashboard — eliminates code duplication across pages.

This module centralizes:
  - Run listing from the results directory
  - Category/severity icon and color mappings
  - Config loading with caching
"""

from __future__ import annotations

import json
import os

import streamlit as st
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS_DIR = os.path.join(_ROOT_DIR, "results")
DATASETS_DIR = os.path.join(_ROOT_DIR, "datasets")
CONFIG_PATH = os.path.join(_ROOT_DIR, "config", "config.yaml")


# ── Config Loading (cached) ──────────────────────────────────────────────────


@st.cache_data(ttl=60)
def load_config() -> dict:
    """Load config.yaml with Streamlit caching (refreshes every 60s)."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Run Listing ───────────────────────────────────────────────────────────────


def list_runs() -> list[dict]:
    """List all evaluation runs from the results directory, newest first.

    Also includes the latest in-memory run from session state if not yet
    persisted to disk (happens right after running an evaluation).
    """
    runs: list[dict] = []
    if os.path.isdir(RESULTS_DIR):
        for run_id in sorted(os.listdir(RESULTS_DIR), reverse=True):
            json_path = os.path.join(RESULTS_DIR, run_id, "report.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    data["_run_id"] = run_id
                    runs.append(data)
                except Exception:
                    pass
    # Include the latest in-memory run if not already in the list
    last = st.session_state.get("last_summary")
    if last and not any(r.get("run_id") == last.get("run_id") for r in runs):
        last["_run_id"] = last.get("run_id", "latest")
        runs.insert(0, last)
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
    "functional": "#6366f1",
    "safety": "#ef4444",
    "regression": "#22c55e",
    "multi_turn": "#38bdf8",
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
        return "#22c55e"
    if rate >= 0.5:
        return "#f59e0b"
    return "#ef4444"
