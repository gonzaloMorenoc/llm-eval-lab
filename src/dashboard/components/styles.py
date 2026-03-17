"""Shared design system — CSS and HTML helper functions for all dashboard pages."""

from __future__ import annotations

import streamlit as st

DESIGN_CSS = """
<style>
/* ================================================================
   LLM EVAL LAB — DESIGN SYSTEM v2.0
   Dark-first · Glass morphism · Educational UI
================================================================ */

/* === BASE ======================================================= */
.stApp { background: #0f0f1a; }

/* === CARDS ====================================================== */
.card {
    background: linear-gradient(135deg, #1a1a2e 0%, #22223d 100%);
    border: 1px solid #2d2d44;
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.card-hover:hover {
    border-color: rgba(99,102,241,0.5);
    box-shadow: 0 12px 40px rgba(99,102,241,0.12);
    transform: translateY(-3px);
}

/* Navigation cards */
.nav-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #22223d 100%);
    border: 1px solid #2d2d44;
    border-radius: 16px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    min-height: 168px;
}
.nav-card::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
}
.nav-run::before    { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.nav-results::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
.nav-compare::before { background: linear-gradient(90deg, #38bdf8, #0ea5e9); }
.nav-tests::before   { background: linear-gradient(90deg, #f59e0b, #d97706); }
.nav-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.35);
    border-color: #4d4d6e;
}
.nav-icon  { font-size: 2.4rem; margin-bottom: 0.65rem; display: block; }
.nav-title { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.35rem; }
.nav-desc  { font-size: 0.79rem; color: #94a3b8; line-height: 1.5; }

/* Stat / KPI cards */
.stat-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #22223d 100%);
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 1.25rem 1rem;
}
.stat-value { font-size: 2rem; font-weight: 800; color: #e2e8f0; line-height: 1; }
.stat-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.35rem; }

/* Concept cards */
.concept-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #22223d 100%);
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    height: 100%;
    transition: border-color 0.2s ease;
}
.concept-card:hover { border-color: #4d4d6e; }
.concept-title { font-size: 0.92rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.3rem; }
.concept-desc  { font-size: 0.79rem; color: #94a3b8; line-height: 1.55; margin: 0; }

/* Dataset cards */
.dataset-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #22223d 100%);
    border: 2px solid #2d2d44;
    border-radius: 14px;
    padding: 1.25rem;
    transition: all 0.2s ease;
}
.dataset-card-selected {
    border-color: #6366f1;
    background: linear-gradient(135deg, #1c1c3a 0%, #252553 100%);
    box-shadow: 0 0 20px rgba(99,102,241,0.15);
}

/* === HOW IT WORKS STEPS ======================================== */
.how-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1.25rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #22223d 100%);
    border: 1px solid #2d2d44;
    border-radius: 12px;
    margin-bottom: 0.65rem;
}
.step-num {
    min-width: 38px; height: 38px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.9rem; color: white; flex-shrink: 0;
}
.step-1 { background: linear-gradient(135deg, #6366f1, #8b5cf6); box-shadow: 0 0 16px rgba(99,102,241,0.4); }
.step-2 { background: linear-gradient(135deg, #38bdf8, #0ea5e9); box-shadow: 0 0 16px rgba(56,189,248,0.4); }
.step-3 { background: linear-gradient(135deg, #22c55e, #16a34a); box-shadow: 0 0 16px rgba(34,197,94,0.4); }
.step-4 { background: linear-gradient(135deg, #f59e0b, #d97706); box-shadow: 0 0 16px rgba(245,158,11,0.4); }
.step-title { font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin: 0 0 0.2rem 0; }
.step-desc  { font-size: 0.81rem; color: #94a3b8; line-height: 1.55; margin: 0; }

/* === CALLOUT BOXES ============================================= */
.callout {
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.65rem 0;
    font-size: 0.87rem;
    line-height: 1.65;
}
.callout-info    { background: rgba(56,189,248,0.08);  border-left: 4px solid #38bdf8; color: #bae6fd; }
.callout-success { background: rgba(34,197,94,0.08);   border-left: 4px solid #22c55e; color: #bbf7d0; }
.callout-warning { background: rgba(245,158,11,0.08);  border-left: 4px solid #f59e0b; color: #fde68a; }
.callout-tip     { background: rgba(99,102,241,0.08);  border-left: 4px solid #6366f1; color: #c7d2fe; }
.callout-error   { background: rgba(239,68,68,0.08);   border-left: 4px solid #ef4444; color: #fecaca; }

/* === BADGES ==================================================== */
.badge {
    display: inline-flex; align-items: center; gap: 0.25rem;
    padding: 0.2rem 0.6rem; border-radius: 50px;
    font-size: 0.73rem; font-weight: 600; white-space: nowrap;
}
.badge-pass   { background: rgba(34,197,94,0.12);   color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.badge-fail   { background: rgba(239,68,68,0.12);   color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.badge-warn   { background: rgba(245,158,11,0.12);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
.badge-info   { background: rgba(56,189,248,0.12);  color: #7dd3fc; border: 1px solid rgba(56,189,248,0.25); }
.badge-purple { background: rgba(99,102,241,0.12);  color: #a5b4fc; border: 1px solid rgba(99,102,241,0.25); }
.badge-gray   { background: rgba(107,114,128,0.12); color: #9ca3af; border: 1px solid rgba(107,114,128,0.25); }
.badge-orange { background: rgba(249,115,22,0.12);  color: #fb923c; border: 1px solid rgba(249,115,22,0.25); }

/* === WIZARD PROGRESS BAR ======================================= */
.wizard-bar {
    display: flex; align-items: center; justify-content: center;
    padding: 1rem 1.5rem;
    background: #13132b;
    border-radius: 14px;
    border: 1px solid #2d2d44;
    margin-bottom: 1.5rem;
    gap: 0;
}
.wz-step   { display: flex; align-items: center; gap: 0.5rem; }
.wz-num    { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; flex-shrink: 0; }
.wz-active { background: #6366f1; color: white; box-shadow: 0 0 14px rgba(99,102,241,0.5); }
.wz-done   { background: #22c55e; color: white; }
.wz-pending{ background: #2d2d44; color: #64748b; }
.wz-label  { font-size: 0.82rem; }
.wz-label-active  { color: #e2e8f0; font-weight: 600; }
.wz-label-done    { color: #4ade80; }
.wz-label-pending { color: #64748b; }
.wz-line      { flex: 1; min-width: 30px; max-width: 60px; height: 2px; background: #2d2d44; margin: 0 0.5rem; }
.wz-line-done { background: linear-gradient(90deg, #22c55e, #16a34a); }

/* === SCORE BARS ================================================ */
.score-bar-wrap { display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0; }
.score-bar-bg   { flex: 1; height: 6px; background: #2d2d44; border-radius: 3px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }

/* === PASS/FAIL BADGES (legacy compat) ========================== */
.pass-badge { background: rgba(34,197,94,0.15);  color: #4ade80; padding: 2px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
.fail-badge { background: rgba(239,68,68,0.15);  color: #f87171; padding: 2px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }

/* === EMPTY STATE =============================================== */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    border: 2px dashed #2d2d44;
    border-radius: 16px;
}
.empty-icon  { font-size: 3rem; margin-bottom: 0.75rem; display: block; }
.empty-title { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem; }
.empty-desc  { font-size: 0.87rem; color: #94a3b8; line-height: 1.6; max-width: 400px; margin: 0 auto; }

/* === METRIC EXPLANATION BOX ==================================== */
.metric-explain {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.79rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* === COMPARISON CARD ========================================== */
.cmp-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #22223d 100%);
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.cmp-winner-a { border-left: 4px solid #6366f1 !important; }
.cmp-winner-b { border-left: 4px solid #38bdf8 !important; }
.cmp-tie      { border-left: 4px solid #6b7280 !important; }

/* === PAGE HEADER =============================================== */
.page-header {
    padding: 1.25rem 0 0.75rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid #2d2d44;
}
.page-title { font-size: 1.8rem; font-weight: 800; color: #e2e8f0; margin: 0; }
.page-desc  { font-size: 0.92rem; color: #94a3b8; margin-top: 0.25rem; }

/* === STREAMLIT COMPONENT OVERRIDES ============================ */

/* Primary buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(99,102,241,0.45) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: #13132b !important;
    border-radius: 12px !important;
    padding: 5px !important;
    border: 1px solid #2d2d44 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 18px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #252540 !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: #0d0d1f !important;
    border-right: 1px solid #2d2d44 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #1a1a2e !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* Metric widgets */
[data-testid="stMetric"] {
    background: #1a1a2e !important;
    border: 1px solid #2d2d44 !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
}

/* Dataframe */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }

/* Divider */
hr { border-color: #2d2d44 !important; opacity: 0.5 !important; }

/* Alert boxes */
.stAlert { border-radius: 10px !important; }

/* Selectbox / text input */
div[data-baseweb="select"] > div { border-radius: 8px !important; }
div[data-baseweb="base-input"] { border-radius: 8px !important; }

/* Progress bar */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 4px !important;
}

/* Disabled buttons */
.stButton > button:disabled {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Loading spinner overlay */
.loading-overlay {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    padding: 2rem;
    text-align: center;
    color: #94a3b8;
    font-size: 0.9rem;
}

/* === ACCESSIBILITY ============================================= */

/* Badges: add text-decoration underline for fail badges (colorblind) */
.badge-fail { text-decoration: underline; text-decoration-style: wavy; text-underline-offset: 2px; }
.pass-badge { text-decoration: none; }
.fail-badge { text-decoration: underline; text-decoration-style: wavy; text-underline-offset: 2px; }

/* Focus indicators for keyboard navigation */
.stButton > button:focus-visible,
div[data-baseweb="select"]:focus-within,
.stCheckbox:focus-within {
    outline: 2px solid #6366f1 !important;
    outline-offset: 2px !important;
}

/* Reduced motion preference */
@media (prefers-reduced-motion: reduce) {
    .card, .nav-card, .concept-card, .dataset-card, .stButton > button {
        transition: none !important;
    }
    .nav-card:hover, .card-hover:hover { transform: none !important; }
}

/* === RESPONSIVE ================================================ */

/* Tablet and below */
@media (max-width: 768px) {
    .nav-card { min-height: auto; padding: 1rem 0.75rem; }
    .nav-icon { font-size: 1.8rem; }
    .nav-title { font-size: 0.88rem; }
    .nav-desc { font-size: 0.72rem; }
    .stat-value { font-size: 1.5rem; }
    .stat-label { font-size: 0.65rem; }
    .page-title { font-size: 1.4rem; }
    .page-desc { font-size: 0.82rem; }
    .wizard-bar { padding: 0.6rem 0.75rem; flex-wrap: wrap; gap: 0.3rem; }
    .wz-label { font-size: 0.7rem; }
    .wz-line { min-width: 15px; max-width: 30px; }
    .concept-card { padding: 0.8rem; }
    .how-step { padding: 0.9rem; gap: 0.6rem; }
    .callout { padding: 0.7rem 0.9rem; font-size: 0.8rem; }
}

/* Mobile */
@media (max-width: 480px) {
    .nav-card { min-height: auto; padding: 0.75rem; }
    .nav-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
    .nav-desc { display: none; }
    .stat-value { font-size: 1.25rem; }
    .page-title { font-size: 1.2rem; }
    .wizard-bar { flex-direction: column; align-items: stretch; }
    .wz-line { height: 0; width: 0; margin: 0; }
    .wz-step { justify-content: flex-start; padding: 0.15rem 0; }
}
</style>
"""


def inject_css() -> None:
    """Inject the shared design system CSS into the current page."""
    st.markdown(DESIGN_CSS, unsafe_allow_html=True)


def callout(text: str, kind: str = "info", icon: str = "") -> str:
    """Return a styled callout box HTML string."""
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "tip": "💡", "error": "🚨"}
    _icon = icon or icons.get(kind, "ℹ️")
    return f'<div class="callout callout-{kind}">{_icon} {text}</div>'


def badge(text: str, kind: str = "gray") -> str:
    """Return a styled inline badge HTML string."""
    return f'<span class="badge badge-{kind}">{text}</span>'


def stat_card(label: str, value: str, color: str = "", icon: str = "", extra: str = "") -> str:
    """Return a styled KPI stat card."""
    color_style = f"color:{color};" if color else ""
    icon_html = f"{icon} " if icon else ""
    extra_html = f'<div style="font-size:0.72rem; color:#64748b; margin-top:0.3rem; line-height:1.4;">{extra}</div>' if extra else ""
    return f"""
    <div class="stat-card">
        <div class="stat-value" style="{color_style}">{icon_html}{value}</div>
        <div class="stat-label">{label}</div>
        {extra_html}
    </div>"""


def wizard_bar(steps: list[str], current: int) -> str:
    """Render a wizard progress bar. `current` is 0-indexed active step."""
    parts: list[str] = []
    for i, label in enumerate(steps):
        if i < current:
            cls_num, cls_label, icon = "wz-done", "wz-label-done", "✓"
        elif i == current:
            cls_num, cls_label, icon = "wz-active", "wz-label-active", str(i + 1)
        else:
            cls_num, cls_label, icon = "wz-pending", "wz-label-pending", str(i + 1)

        parts.append(
            f'<div class="wz-step">'
            f'<div class="wz-num {cls_num}">{icon}</div>'
            f'<span class="wz-label {cls_label}">{label}</span>'
            f"</div>"
        )
        if i < len(steps) - 1:
            line_cls = "wz-line-done" if i < current else "wz-line"
            parts.append(f'<div class="wz-line {line_cls}"></div>')

    return f'<div class="wizard-bar">{"".join(parts)}</div>'


def how_step(num: int, title: str, desc: str) -> str:
    """Return a single how-it-works step card HTML."""
    return f"""
    <div class="how-step">
        <div class="step-num step-{num}">{num}</div>
        <div>
            <p class="step-title">{title}</p>
            <p class="step-desc">{desc}</p>
        </div>
    </div>"""


def page_header(icon: str, title: str, desc: str) -> str:
    """Return a styled page header."""
    return f"""
    <div class="page-header">
        <div class="page-title">{icon} {title}</div>
        <div class="page-desc">{desc}</div>
    </div>"""
