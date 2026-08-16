"""Reusable metric card and badge components for the dashboard."""

from __future__ import annotations

import streamlit as st

from src.dashboard.components.theme import PALETTE


def metric_card(label: str, value: str, delta: str | None = None, delta_color: str = "normal") -> None:
    """Render a styled metric card with optional delta."""
    delta_html = ""
    if delta:
        css_class = "metric-delta-up" if delta_color != "inverse" else "metric-delta-down"
        delta_html = f'<div class="{css_class}">{delta}</div>'

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a row of KPI metrics. Each tuple: (label, value, optional_color_border)."""
    cols = st.columns(len(metrics))
    for col, (label, value, color) in zip(cols, metrics, strict=False):
        with col:
            border = f"border-left: 4px solid {color};" if color else ""
            st.markdown(
                f"""
                <div class="stat-card" style="{border}">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def pass_fail_badge(passed: bool) -> str:
    """Return HTML for a pass/fail badge with accessible text."""
    if passed:
        return '<span class="pass-badge" role="status" aria-label="Test passed">✓ PASS</span>'
    return '<span class="fail-badge" role="status" aria-label="Test failed">✗ FAIL</span>'


def severity_icon(severity: str) -> str:
    """Return an emoji icon for severity level."""
    from src.dashboard.components.shared import SEVERITY_ICONS

    return SEVERITY_ICONS.get(severity, "⚪")


def severity_badge(severity: str) -> str:
    """Return colored HTML badge for severity with icon for accessibility."""
    from src.dashboard.components.shared import SEVERITY_ICONS

    colors = {
        "critical": (f"background:#7f1d1d; color:{PALETTE['danger_pale']};", "CRITICAL"),
        "high": ("background:#7c2d12; color:#fdba74;", "HIGH"),
        "medium": ("background:#713f12; color:#fde047;", "MEDIUM"),
        "low": ("background:#14532d; color:#86efac;", "LOW"),
    }
    style, text = colors.get(severity, (f"background:{PALETTE['text_ghost']}; color:{PALETTE['text_dim']};", severity.upper()))
    icon = SEVERITY_ICONS.get(severity, "⚪")
    return (
        f'<span style="{style} padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem;"'
        f' role="status" aria-label="Severity: {severity}">{icon} {text}</span>'
    )


def score_color(score: float | None, threshold: float = 0.65) -> str:
    """Return a color string based on score vs threshold."""
    if score is None:
        return PALETTE["text_faint"]  # gray
    if score >= threshold:
        return PALETTE["success_bright"]  # green
    if score >= threshold * 0.8:
        return PALETTE["warning_bright"]  # yellow
    return PALETTE["danger_soft"]  # red


def score_bar(score: float | None, max_val: float = 1.0) -> str:
    """Return an inline HTML progress bar for a score."""
    if score is None:
        return f'<span style="color:{PALETTE["text_faint"]};">—</span>'

    pct = min(100, (score / max_val) * 100)
    color = score_color(score, max_val * 0.65)
    return (
        f'<div style="display:flex; align-items:center; gap:6px;">'
        f'<div style="flex:1; background:{PALETTE["border"]}; border-radius:4px; height:8px; overflow:hidden;">'
        f'<div style="width:{pct:.0f}%; background:{color}; height:100%; border-radius:4px;"></div>'
        f"</div>"
        f'<span style="font-size:0.8rem; color:{color}; min-width:40px;">{score:.2f}</span>'
        f"</div>"
    )
