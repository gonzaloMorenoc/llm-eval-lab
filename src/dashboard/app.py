"""LLM Eval Lab — Streamlit Dashboard entry point.

Launch: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.charts import (
    pass_rate_bar_chart,
    metrics_radar_chart,
    category_trend_chart,
    COLORS,
)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))


def main() -> None:
    st.set_page_config(
        page_title="LLM Eval Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject global styles
    st.markdown(_global_css(), unsafe_allow_html=True)

    render_sidebar()

    # --- Hero Header ---
    st.markdown(
        """
        <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
            <h1 style="margin:0; font-size:2.5rem;">🔬 LLM Eval Lab</h1>
            <p style="color:#a0a0b0; font-size:1.1rem; margin-top:0.3rem;">
                Framework de evaluación de calidad para chatbots de IA
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Quick Navigation Cards ---
    st.markdown("")
    nav_cols = st.columns(4)
    nav_items = [
        ("🚀", "Run Evaluation", "Lanza evaluaciones contra providers", "pages/1_run.py"),
        ("📊", "Results", "Visualiza métricas y resultados", "pages/2_results.py"),
        ("🔄", "Compare", "Compara ejecuciones side-by-side", "pages/3_compare.py"),
        ("📝", "Test Cases", "Gestiona datasets de test", "pages/4_test_cases.py"),
    ]
    for col, (icon, title, desc, page) in zip(nav_cols, nav_items):
        with col:
            st.markdown(
                f"""
                <div class="nav-card">
                    <div style="font-size:2rem;">{icon}</div>
                    <div style="font-weight:700; font-size:1.1rem; margin:0.3rem 0;">{title}</div>
                    <div style="color:#a0a0b0; font-size:0.85rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.page_link(page, label=f"Ir a {title} →", use_container_width=True)

    st.divider()

    # --- Dataset Overview ---
    from src.runner.runner import load_all_datasets

    cases = load_all_datasets()
    categories: dict[str, int] = {}
    severities: dict[str, int] = {}
    eval_types: dict[str, int] = {}
    for c in cases:
        categories[c.category] = categories.get(c.category, 0) + 1
        severities[c.severity] = severities.get(c.severity, 0) + 1
        for et in c.evaluation_type:
            eval_types[et] = eval_types.get(et, 0) + 1

    st.subheader("📦 Dataset Overview")

    overview_cols = st.columns([2, 1, 1])
    with overview_cols[0]:
        cat_cols = st.columns(len(categories))
        for i, (cat, count) in enumerate(sorted(categories.items())):
            with cat_cols[i]:
                color = {"functional": "#6366f1", "safety": "#f87171", "regression": "#4ade80", "multi_turn": "#38bdf8"}.get(cat, "#888")
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">{cat.replace('_', ' ').title()}</div>
                        <div class="stat-value" style="color:{color};">{count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with overview_cols[1]:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Total Test Cases</div>
                <div class="stat-value">{len(cases)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with overview_cols[2]:
        sev_display = " · ".join(f"{'🔴🟠🟡🟢'['critical high medium low'.split().index(s)]} {count}" for s, count in sorted(severities.items(), key=lambda x: ["critical", "high", "medium", "low"].index(x[0])))
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">By Severity</div>
                <div style="font-size:0.95rem; margin-top:0.3rem;">{sev_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Recent Runs ---
    st.subheader("📈 Recent Runs")

    runs = _list_recent_runs()
    if runs:
        # Show latest run KPIs
        latest = runs[0]
        st.markdown(f"**Latest:** `{latest.get('run_id', '?')}` — {latest.get('chatbot_id', '?')} ({latest.get('chatbot_mode', '?')}) — {latest.get('timestamp', '')[:19]}")

        kpi_cols = st.columns(5)
        kpi_data = [
            ("Pass Rate", f"{latest.get('pass_rate', 0):.1%}", _delta_color(latest.get('pass_rate', 0), 0.7)),
            ("Avg Score", f"{latest.get('avg_score', 0):.3f}", _delta_color(latest.get('avg_score', 0), 0.7)),
            ("Avg Latency", f"{latest.get('avg_latency_ms', 0):.0f}ms", None),
            ("Critical Fails", str(latest.get('critical_failures', 0)), None),
            ("Total", f"{latest.get('passed', 0)}✅ {latest.get('failed', 0)}❌", None),
        ]
        for col, (label, value, color) in zip(kpi_cols, kpi_data):
            with col:
                border = f"border-left: 3px solid {color};" if color else ""
                st.markdown(
                    f"""
                    <div class="stat-card" style="{border}">
                        <div class="stat-label">{label}</div>
                        <div class="stat-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Quick charts for latest run
        if len(runs) >= 1:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                by_cat = latest.get("by_category", {})
                if by_cat:
                    fig = pass_rate_bar_chart(by_cat)
                    st.plotly_chart(fig, use_container_width=True, key="home_bar")

            with chart_col2:
                all_metrics = {}
                all_metrics.update(latest.get("ragas_aggregate", {}))
                all_metrics.update(latest.get("deepeval_aggregate", {}))
                if all_metrics:
                    config = st.session_state.get("config", {})
                    thresholds = {}
                    thresholds.update(config.get("ragas", {}).get("thresholds", {}))
                    thresholds.update(config.get("deepeval", {}).get("thresholds", {}))
                    fig = metrics_radar_chart(all_metrics, thresholds)
                    st.plotly_chart(fig, use_container_width=True, key="home_radar")

        # Runs history table
        if len(runs) > 1:
            st.markdown("**Run History**")
            runs_table = []
            for r in runs[:10]:
                runs_table.append({
                    "Run ID": r.get("run_id", "?"),
                    "Provider": r.get("chatbot_id", "?"),
                    "Mode": r.get("chatbot_mode", "?"),
                    "Pass Rate": f"{r.get('pass_rate', 0):.1%}",
                    "Avg Score": f"{r.get('avg_score', 0):.3f}",
                    "Latency": f"{r.get('avg_latency_ms', 0):.0f}ms",
                    "Tests": r.get("total", 0),
                    "Date": r.get("timestamp", "")[:19],
                })
            st.dataframe(runs_table, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div style="text-align:center; padding:2rem; border:1px dashed #3d3d5c; border-radius:12px;">
                <div style="font-size:2rem;">🚀</div>
                <p style="color:#a0a0b0; margin-top:0.5rem;">
                    No hay ejecuciones todavía.<br>
                    Ve a <strong>Run Evaluation</strong> para lanzar tu primera evaluación.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Footer ---
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color:#6b7280; font-size:0.8rem; padding:0.5rem;">
            LLM Eval Lab v0.3.0 · Built with Streamlit · RAGAS + DeepEval
        </div>
        """,
        unsafe_allow_html=True,
    )


def _list_recent_runs() -> list[dict]:
    runs = []
    if not os.path.isdir(RESULTS_DIR):
        return runs
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
    # Also check session state
    last = st.session_state.get("last_summary")
    if last and not any(r.get("run_id") == last.get("run_id") for r in runs):
        last["_run_id"] = last.get("run_id", "latest")
        runs.insert(0, last)
    return runs


def _delta_color(value: float, threshold: float) -> str:
    if value >= threshold:
        return COLORS["success"]
    if value >= threshold * 0.8:
        return COLORS["warning"]
    return COLORS["danger"]


def _global_css() -> str:
    return """
    <style>
    /* Navigation cards */
    .nav-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #3d3d5c;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        min-height: 120px;
    }
    .nav-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
        border-color: #6366f1;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #3d3d5c;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e0e0e0;
        margin-top: 0.2rem;
    }

    /* Badge styles */
    .pass-badge {
        background: #166534; color: #4ade80;
        padding: 2px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;
    }
    .fail-badge {
        background: #7f1d1d; color: #f87171;
        padding: 2px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* Sidebar tweaks */
    div[data-testid="stSidebar"] {
        min-width: 290px;
    }
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.9rem;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        font-size: 0.95rem !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
    }
    </style>
    """


if __name__ == "__main__":
    main()
