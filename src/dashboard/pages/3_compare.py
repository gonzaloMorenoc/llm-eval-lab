"""Page 3: Compare Runs — interactive side-by-side analysis of evaluation runs."""

from __future__ import annotations

import json
import os
import sys

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import severity_icon, kpi_row
from src.dashboard.components.charts import comparison_bar_chart, COLORS, CATEGORY_COLORS

st.set_page_config(page_title="Compare Runs — LLM Eval Lab", page_icon="🔄", layout="wide")
render_sidebar()

st.markdown(
    """
    <style>
    .stat-card { background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 10px; padding: 1rem; border: 1px solid #3d3d5c; }
    .stat-label { font-size: 0.8rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: #e0e0e0; margin-top: 0.2rem; }
    .winner-a { border-left: 4px solid #6366f1; }
    .winner-b { border-left: 4px solid #38bdf8; }
    .tie { border-left: 4px solid #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔄 Compare Runs")
st.markdown("Side-by-side interactive comparison of evaluation runs.")

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results"))


def _list_runs() -> list[dict]:
    runs = []
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
    last = st.session_state.get("last_summary")
    if last and not any(r.get("run_id") == last.get("run_id") for r in runs):
        last["_run_id"] = last.get("run_id", "latest")
        runs.insert(0, last)
    return runs


runs = _list_runs()

if len(runs) < 2:
    st.info("At least **2 completed runs** are needed for comparison. Run more evaluations first.")
    st.page_link("src/dashboard/pages/1_run.py", label="🚀 Go to Run Evaluation", icon="🚀")
    st.stop()


def _label(r: dict) -> str:
    return f"{r.get('run_id', '?')} — {r.get('chatbot_id', '?')} ({r.get('chatbot_mode', '?')}) — {r.get('timestamp', '')[:16]}"


# --- Run Selection ---
sel_col1, sel_col2 = st.columns(2)
with sel_col1:
    st.markdown(f"<div style='text-align:center; color:{COLORS['primary']}; font-weight:700;'>🅰 Run A</div>", unsafe_allow_html=True)
    idx_a = st.selectbox("Run A", range(len(runs)), format_func=lambda i: _label(runs[i]), key="cmp_a", label_visibility="collapsed")
with sel_col2:
    st.markdown(f"<div style='text-align:center; color:{COLORS['info']}; font-weight:700;'>🅱 Run B</div>", unsafe_allow_html=True)
    idx_b = st.selectbox("Run B", range(len(runs)), index=min(1, len(runs) - 1), format_func=lambda i: _label(runs[i]), key="cmp_b", label_visibility="collapsed")

if idx_a == idx_b:
    st.warning("Select two different runs to compare.")
    st.stop()

run_a, run_b = runs[idx_a], runs[idx_b]
label_a = run_a.get("chatbot_id", "Run A")
label_b = run_b.get("chatbot_id", "Run B")

st.divider()

# --- KPI Comparison ---
st.subheader("📊 Overview Comparison")

kpi_keys = [
    ("Pass Rate", "pass_rate", True, lambda v: f"{v:.1%}"),
    ("Avg Score", "avg_score", True, lambda v: f"{v:.3f}"),
    ("Avg Latency", "avg_latency_ms", False, lambda v: f"{v:.0f}ms"),
    ("Critical Failures", "critical_failures", False, lambda v: str(int(v))),
    ("Total Tests", "total", None, lambda v: str(int(v))),
]

cols = st.columns(len(kpi_keys))
for i, (name, key, higher_is_better, fmt) in enumerate(kpi_keys):
    with cols[i]:
        va = run_a.get(key, 0)
        vb = run_b.get(key, 0)
        diff = vb - va

        if higher_is_better is not None and diff != 0:
            is_b_better = (diff > 0) == higher_is_better
            winner = "B" if is_b_better else "A"
            winner_class = "winner-b" if is_b_better else "winner-a"
            arrow = "↑" if diff > 0 else "↓"
            delta_color = COLORS["success"] if is_b_better else COLORS["danger"]
        else:
            winner = "="
            winner_class = "tie"
            arrow = "="
            delta_color = COLORS["muted"]

        st.markdown(
            f"""
            <div class="stat-card {winner_class}">
                <div class="stat-label">{name}</div>
                <div style="display:flex; justify-content:space-between; margin-top:0.3rem;">
                    <span style="color:{COLORS['primary']}; font-weight:700;">🅰 {fmt(va)}</span>
                    <span style="color:{COLORS['info']}; font-weight:700;">🅱 {fmt(vb)}</span>
                </div>
                <div style="text-align:center; color:{delta_color}; font-size:0.85rem; margin-top:0.3rem;">
                    {arrow} {fmt(abs(diff))} {'→ B wins' if winner == 'B' else ('→ A wins' if winner == 'A' else '')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# --- Metric Charts ---
st.subheader("📐 Metric Comparison")

metrics_a = {}
metrics_a.update(run_a.get("ragas_aggregate", {}))
metrics_a.update(run_a.get("deepeval_aggregate", {}))

metrics_b = {}
metrics_b.update(run_b.get("ragas_aggregate", {}))
metrics_b.update(run_b.get("deepeval_aggregate", {}))

if metrics_a or metrics_b:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = comparison_bar_chart(metrics_a, metrics_b, f"🅰 {label_a}", f"🅱 {label_b}")
        st.plotly_chart(fig, use_container_width=True, key="cmp_bar")

    with chart_col2:
        # Radar overlay
        all_metric_names = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
        if all_metric_names:
            names_closed = all_metric_names + [all_metric_names[0]]
            theta = [n.replace("_", " ").title() for n in names_closed]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[metrics_a.get(m, 0) for m in names_closed], theta=theta,
                fill="toself", fillcolor="rgba(99, 102, 241, 0.15)",
                line=dict(color=COLORS["primary"], width=2.5), name=f"🅰 {label_a}",
                hovertemplate="%{theta}: %{r:.3f}<extra>Run A</extra>",
            ))
            fig.add_trace(go.Scatterpolar(
                r=[metrics_b.get(m, 0) for m in names_closed], theta=theta,
                fill="toself", fillcolor="rgba(56, 189, 248, 0.15)",
                line=dict(color=COLORS["info"], width=2.5), name=f"🅱 {label_b}",
                hovertemplate="%{theta}: %{r:.3f}<extra>Run B</extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"]),
                polar=dict(bgcolor="rgba(0,0,0,0)",
                           radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border"]),
                           angularaxis=dict(gridcolor=COLORS["border"])),
                title=dict(text="Radar Overlay", font=dict(size=14)),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True, key="cmp_radar")

    # Metric details table
    all_metrics = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    table_data = []
    for m in all_metrics:
        va = metrics_a.get(m)
        vb = metrics_b.get(m)
        diff = (vb or 0) - (va or 0)
        arrow = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "=")
        winner = "🅱" if diff > 0.001 else ("🅰" if diff < -0.001 else "=")
        table_data.append({
            "Metric": m.replace("_", " ").title(),
            f"🅰 {label_a}": f"{va:.3f}" if va is not None else "—",
            f"🅱 {label_b}": f"{vb:.3f}" if vb is not None else "—",
            "Delta": f"{arrow} {abs(diff):.3f}",
            "Winner": winner,
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)
else:
    st.info("No RAGAS/DeepEval metrics in selected runs.")

st.divider()

# --- Category Comparison ---
st.subheader("📂 Category Comparison")

cats_a = run_a.get("by_category", {})
cats_b = run_b.get("by_category", {})
all_cats = sorted(set(list(cats_a.keys()) + list(cats_b.keys())))

if all_cats:
    # Visual comparison bar
    for cat in all_cats:
        a_rate = cats_a.get(cat, {}).get("pass_rate", 0) * 100
        b_rate = cats_b.get(cat, {}).get("pass_rate", 0) * 100
        color = CATEGORY_COLORS.get(cat, COLORS["muted"])

        st.markdown(f"**{cat.replace('_', ' ').title()}**")
        bar_col1, bar_col2 = st.columns(2)
        with bar_col1:
            st.progress(a_rate / 100, text=f"🅰 {a_rate:.0f}%")
        with bar_col2:
            st.progress(b_rate / 100, text=f"🅱 {b_rate:.0f}%")

st.divider()

# --- Per-Test Disagreements ---
st.subheader("⚡ Pass/Fail Disagreements")

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
    st.markdown(f"**{len(disagreements)}** test(s) with different outcomes:")
    for tid, ra, rb in disagreements:
        icon_a = "✅" if ra.get("overall_passed") else "❌"
        icon_b = "✅" if rb.get("overall_passed") else "❌"
        sev = severity_icon(ra.get("test_case", {}).get("severity", "medium"))

        with st.expander(f"{sev} **{tid}** — 🅰{icon_a} vs 🅱{icon_b}"):
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown(f"**🅰 {label_a}** — {'✅ Passed' if ra.get('overall_passed') else '❌ Failed'}")
                st.markdown(f"Score: {ra.get('overall_score', '—')}")
                st.text(ra.get("response", "")[:400])
            with d_col2:
                st.markdown(f"**🅱 {label_b}** — {'✅ Passed' if rb.get('overall_passed') else '❌ Failed'}")
                st.markdown(f"Score: {rb.get('overall_score', '—')}")
                st.text(rb.get("response", "")[:400])
else:
    if common_ids:
        st.success("✅ All common test cases have the same pass/fail outcome.")
    else:
        st.info("No common test cases between selected runs.")
