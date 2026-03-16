"""Page 3: Compare Runs — side-by-side comparison of evaluation runs."""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.charts import comparison_bar_chart, metrics_radar_chart, COLORS

st.set_page_config(page_title="Compare Runs — LLM Eval Lab", page_icon="🔄", layout="wide")
render_sidebar()

st.title("🔄 Compare Runs")
st.markdown("Side-by-side comparison of two evaluation runs.")


def _list_runs() -> list[dict]:
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results"))
    runs = []
    if not os.path.isdir(results_dir):
        return runs
    for run_id in sorted(os.listdir(results_dir), reverse=True):
        json_path = os.path.join(results_dir, run_id, "report.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                data["_run_id"] = run_id
                runs.append(data)
            except Exception:
                pass
    return runs


runs = _list_runs()

# Include last session summary if available
last_summary = st.session_state.get("last_summary")
if last_summary and not any(r.get("run_id") == last_summary.get("run_id") for r in runs):
    last_summary["_run_id"] = last_summary.get("run_id", "latest")
    runs.insert(0, last_summary)

if len(runs) < 2:
    st.info("At least 2 completed runs are needed for comparison. Run more evaluations first.")
    st.stop()


def _run_label(r: dict) -> str:
    return f"{r.get('run_id', '?')} — {r.get('chatbot_id', '?')} ({r.get('chatbot_mode', '?')})"


col1, col2 = st.columns(2)
with col1:
    idx_a = st.selectbox("Run A", range(len(runs)), format_func=lambda i: _run_label(runs[i]), key="cmp_a")
with col2:
    default_b = 1 if len(runs) > 1 else 0
    idx_b = st.selectbox("Run B", range(len(runs)), index=default_b, format_func=lambda i: _run_label(runs[i]), key="cmp_b")

run_a = runs[idx_a]
run_b = runs[idx_b]
label_a = f"{run_a.get('chatbot_id', 'Run A')}"
label_b = f"{run_b.get('chatbot_id', 'Run B')}"

st.divider()

# --- KPI Comparison ---
st.subheader("Overview Comparison")

kpis = [
    ("Pass Rate", "pass_rate", True, lambda v: f"{v:.1%}"),
    ("Avg Score", "avg_score", True, lambda v: f"{v:.3f}"),
    ("Avg Latency", "avg_latency_ms", False, lambda v: f"{v:.0f}ms"),
    ("Critical Failures", "critical_failures", False, lambda v: str(int(v))),
    ("Total Tests", "total", None, lambda v: str(int(v))),
]

cols = st.columns(len(kpis))
for i, (name, key, higher_is_better, fmt) in enumerate(kpis):
    with cols[i]:
        val_a = run_a.get(key, 0)
        val_b = run_b.get(key, 0)
        diff = val_b - val_a

        st.markdown(f"**{name}**")
        st.markdown(f"**A:** {fmt(val_a)}")
        st.markdown(f"**B:** {fmt(val_b)}")

        if higher_is_better is not None and diff != 0:
            is_improvement = (diff > 0) == higher_is_better
            arrow = "↑" if diff > 0 else "↓"
            color = "green" if is_improvement else "red"
            st.markdown(f":{color}[{arrow} Δ {fmt(abs(diff))}]")

st.divider()

# --- Metrics Comparison ---
st.subheader("Metric Comparison")

metrics_a = {}
metrics_a.update(run_a.get("ragas_aggregate", {}))
metrics_a.update(run_a.get("deepeval_aggregate", {}))

metrics_b = {}
metrics_b.update(run_b.get("ragas_aggregate", {}))
metrics_b.update(run_b.get("deepeval_aggregate", {}))

if metrics_a or metrics_b:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = comparison_bar_chart(metrics_a, metrics_b, label_a, label_b)
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        # Radar overlay
        import plotly.graph_objects as go

        all_metric_names = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
        if all_metric_names:
            names_closed = all_metric_names + [all_metric_names[0]]
            theta = [n.replace("_", " ").title() for n in names_closed]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[metrics_a.get(m, 0) for m in names_closed],
                theta=theta,
                fill="toself",
                fillcolor="rgba(99, 102, 241, 0.15)",
                line=dict(color=COLORS["primary"], width=2),
                name=label_a,
            ))
            fig.add_trace(go.Scatterpolar(
                r=[metrics_b.get(m, 0) for m in names_closed],
                theta=theta,
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.15)",
                line=dict(color=COLORS["info"], width=2),
                name=label_b,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"]),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border"]),
                    angularaxis=dict(gridcolor=COLORS["border"]),
                ),
                title="Radar Comparison",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    all_metrics = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    table_data = []
    for m in all_metrics:
        val_a = metrics_a.get(m)
        val_b = metrics_b.get(m)
        diff = (val_b or 0) - (val_a or 0)
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        table_data.append({
            "Metric": m.replace("_", " ").title(),
            f"{label_a}": f"{val_a:.3f}" if val_a is not None else "—",
            f"{label_b}": f"{val_b:.3f}" if val_b is not None else "—",
            "Delta": f"{arrow} {abs(diff):.3f}" if diff != 0 else "=",
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)
else:
    st.info("No RAGAS/DeepEval metrics found in selected runs.")

st.divider()

# --- Category Comparison ---
st.subheader("Category Comparison")

cats_a = run_a.get("by_category", {})
cats_b = run_b.get("by_category", {})
all_cats = sorted(set(list(cats_a.keys()) + list(cats_b.keys())))

if all_cats:
    cat_table = []
    for cat in all_cats:
        a_stats = cats_a.get(cat, {})
        b_stats = cats_b.get(cat, {})
        rate_a = a_stats.get("pass_rate", 0)
        rate_b = b_stats.get("pass_rate", 0)
        diff = rate_b - rate_a
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        cat_table.append({
            "Category": cat.replace("_", " ").title(),
            f"Pass Rate ({label_a})": f"{rate_a:.1%}",
            f"Pass Rate ({label_b})": f"{rate_b:.1%}",
            "Delta": f"{arrow} {abs(diff):.1%}" if diff != 0 else "=",
        })
    st.dataframe(cat_table, use_container_width=True, hide_index=True)

st.divider()

# --- Per-Test Comparison ---
st.subheader("Per-Test Differences")

results_a = {r.get("test_case", {}).get("id"): r for r in run_a.get("results", [])}
results_b = {r.get("test_case", {}).get("id"): r for r in run_b.get("results", [])}
common_ids = sorted(set(results_a.keys()) & set(results_b.keys()))

disagreements = []
for tid in common_ids:
    ra = results_a[tid]
    rb = results_b[tid]
    if ra.get("overall_passed") != rb.get("overall_passed"):
        disagreements.append((tid, ra, rb))

if disagreements:
    st.markdown(f"**{len(disagreements)}** test(s) with different pass/fail outcomes:")
    for tid, ra, rb in disagreements:
        icon_a = "✅" if ra.get("overall_passed") else "❌"
        icon_b = "✅" if rb.get("overall_passed") else "❌"
        with st.expander(f"**{tid}** — {label_a}: {icon_a} vs {label_b}: {icon_b}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{label_a}** — Score: {ra.get('overall_score', '—')}")
                st.markdown(f"Response: {ra.get('response', '')[:300]}")
            with col_b:
                st.markdown(f"**{label_b}** — Score: {rb.get('overall_score', '—')}")
                st.markdown(f"Response: {rb.get('response', '')[:300]}")
else:
    st.success("All common test cases have the same pass/fail outcome in both runs.")
