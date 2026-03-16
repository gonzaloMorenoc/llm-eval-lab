"""Reusable metric card components for the dashboard."""

from __future__ import annotations

import streamlit as st


def metric_card(label: str, value: str, delta: str | None = None, delta_color: str = "normal") -> None:
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        css_class = "metric-delta-up" if delta_color == "normal" else "metric-delta-down"
        delta_html = f'<div class="{css_class}">{delta}</div>'

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pass_fail_badge(passed: bool) -> str:
    """Return HTML for a pass/fail badge."""
    if passed:
        return '<span class="pass-badge">PASS</span>'
    return '<span class="fail-badge">FAIL</span>'


def severity_icon(severity: str) -> str:
    """Return an emoji icon for severity level."""
    icons = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
    }
    return icons.get(severity, "⚪")


def score_color(score: float | None, threshold: float = 0.65) -> str:
    """Return a color string based on score vs threshold."""
    if score is None:
        return "gray"
    if score >= threshold:
        return "#4ade80"  # green
    if score >= threshold * 0.8:
        return "#facc15"  # yellow
    return "#f87171"  # red
