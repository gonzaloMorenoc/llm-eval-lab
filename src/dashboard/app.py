"""LLM Eval Lab — Streamlit Dashboard entry point."""

from __future__ import annotations

import os
import sys

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dashboard.components.sidebar import render_sidebar  # noqa: E402


def main() -> None:
    st.set_page_config(
        page_title="LLM Eval Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global CSS
    st.markdown(
        """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid #3d3d5c;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #e0e0e0;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #a0a0b0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-delta-up { color: #4ade80; font-size: 0.85rem; }
        .metric-delta-down { color: #f87171; font-size: 0.85rem; }
        .pass-badge {
            background: #166534; color: #4ade80;
            padding: 2px 8px; border-radius: 4px; font-weight: 600;
        }
        .fail-badge {
            background: #7f1d1d; color: #f87171;
            padding: 2px 8px; border-radius: 4px; font-weight: 600;
        }
        div[data-testid="stSidebar"] { min-width: 280px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    st.title("🔬 LLM Eval Lab")
    st.markdown("Framework de evaluación de calidad para chatbots de IA.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.page_link("src/dashboard/pages/1_run.py", label="▶ Run Evaluation", icon="🚀")
    with col2:
        st.page_link("src/dashboard/pages/2_results.py", label="📊 Results Dashboard", icon="📊")
    with col3:
        st.page_link("src/dashboard/pages/3_compare.py", label="🔄 Compare Runs", icon="🔄")

    st.divider()

    # Quick overview of available resources
    from src.runner.runner import load_all_datasets

    cases = load_all_datasets()
    categories = {}
    for c in cases:
        categories[c.category] = categories.get(c.category, 0) + 1

    st.subheader("Dataset Overview")
    cols = st.columns(len(categories))
    for i, (cat, count) in enumerate(sorted(categories.items())):
        with cols[i]:
            st.metric(cat.replace("_", " ").title(), count)

    st.metric("Total Test Cases", len(cases))

    # List existing runs
    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    if os.path.isdir(results_dir):
        runs = sorted(os.listdir(results_dir), reverse=True)
        if runs:
            st.subheader("Recent Runs")
            for run_id in runs[:5]:
                json_path = os.path.join(results_dir, run_id, "report.json")
                if os.path.exists(json_path):
                    st.markdown(f"- `{run_id}` — [View results](./results)")
    else:
        st.info("No evaluation runs found yet. Go to **Run Evaluation** to start.")


if __name__ == "__main__":
    main()
