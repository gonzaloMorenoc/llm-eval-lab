"""Page 2: Results Dashboard — rich interactive visualization of evaluation results."""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import severity_icon, kpi_row, severity_badge, score_bar
from src.dashboard.components.charts import (
    COLORS,
    pass_rate_bar_chart,
    metrics_radar_chart,
    latency_histogram,
    severity_pie_chart,
    evaluator_scores_chart,
    score_distribution_chart,
)

st.set_page_config(page_title="Results — LLM Eval Lab", page_icon="📊", layout="wide")
render_sidebar()

st.markdown(
    """
    <style>
    .stat-card { background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 10px; padding: 1rem; border: 1px solid #3d3d5c; }
    .stat-label { font-size: 0.8rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: #e0e0e0; margin-top: 0.2rem; }
    .pass-badge { background: #166534; color: #4ade80; padding: 2px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    .fail-badge { background: #7f1d1d; color: #f87171; padding: 2px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Results Dashboard")

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

if not runs:
    st.info("No evaluation runs found. Go to **Run Evaluation** to create one.")
    st.stop()

# --- Run Selector ---
run_labels = [
    f"{r.get('run_id', '?')} — {r.get('chatbot_id', '?')} ({r.get('chatbot_mode', '?')}) — {r.get('timestamp', '')[:19]}"
    for r in runs
]
selected_idx = st.selectbox("📁 Select Run", range(len(runs)), format_func=lambda i: run_labels[i])
summary = runs[selected_idx]

st.divider()

# --- KPI Row ---
pass_rate = summary.get("pass_rate", 0)
pass_color = COLORS["success"] if pass_rate >= 0.7 else (COLORS["warning"] if pass_rate >= 0.5 else COLORS["danger"])

kpi_row([
    ("Pass Rate", f"{pass_rate:.1%}", pass_color),
    ("Avg Score", f"{summary.get('avg_score', 0):.3f}", COLORS["primary"]),
    ("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f}ms", COLORS["info"]),
    ("Critical Failures", str(summary.get("critical_failures", 0)),
     COLORS["danger"] if summary.get("critical_failures", 0) > 0 else COLORS["success"]),
    ("Total", f"{summary.get('passed', 0)}✅ {summary.get('failed', 0)}❌ / {summary.get('total', 0)}", None),
])

st.divider()

# --- Interactive Charts ---
tab_overview, tab_metrics, tab_latency, tab_evaluators = st.tabs(
    ["📈 Overview", "📐 Metrics", "⏱️ Latency & Performance", "🧪 Evaluators"]
)

with tab_overview:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        by_category = summary.get("by_category", {})
        if by_category:
            fig = pass_rate_bar_chart(by_category)
            st.plotly_chart(fig, use_container_width=True, key="res_bar")
        else:
            st.info("No category data.")

    with chart_col2:
        results = summary.get("results", [])
        failures = [r for r in results if not r.get("overall_passed")]
        if failures:
            severity_counts: dict[str, int] = {}
            for r in failures:
                sev = r.get("test_case", {}).get("severity", "medium")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            fig = severity_pie_chart(severity_counts)
            st.plotly_chart(fig, use_container_width=True, key="res_pie")
        else:
            st.success("🎉 No failures!")

    # Score distribution
    results = summary.get("results", [])
    if any(r.get("overall_score") is not None for r in results):
        fig = score_distribution_chart(results)
        st.plotly_chart(fig, use_container_width=True, key="res_box")

with tab_metrics:
    ragas_col, deepeval_col = st.columns(2)

    with ragas_col:
        st.markdown("#### RAGAS Metrics")
        ragas_agg = summary.get("ragas_aggregate", {})
        if ragas_agg:
            config = st.session_state.get("config", {})
            ragas_thresholds = config.get("ragas", {}).get("thresholds", {})
            fig = metrics_radar_chart(ragas_agg, ragas_thresholds, title="RAGAS Metrics")
            st.plotly_chart(fig, use_container_width=True, key="res_ragas_radar")

            for metric, avg in sorted(ragas_agg.items()):
                t = ragas_thresholds.get(metric, 0.65)
                status = "✅" if avg >= t else "❌"
                st.markdown(
                    f"{status} **{metric.replace('_', ' ').title()}**: {avg:.3f} (threshold: {t})",
                )
        else:
            st.info("RAGAS not enabled for this run.")

    with deepeval_col:
        st.markdown("#### DeepEval Metrics")
        deepeval_agg = summary.get("deepeval_aggregate", {})
        if deepeval_agg:
            config = st.session_state.get("config", {})
            de_thresholds = config.get("deepeval", {}).get("thresholds", {})
            fig = metrics_radar_chart(deepeval_agg, de_thresholds, title="DeepEval Metrics")
            st.plotly_chart(fig, use_container_width=True, key="res_de_radar")

            for metric, avg in sorted(deepeval_agg.items()):
                t = de_thresholds.get(metric, 0.5)
                status = "✅" if avg >= t else "❌"
                st.markdown(
                    f"{status} **{metric.replace('_', ' ').title()}**: {avg:.3f} (threshold: {t})",
                )
        else:
            st.info("DeepEval not enabled for this run.")

with tab_latency:
    results = summary.get("results", [])
    latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms", 0) > 0]

    if latencies:
        fig = latency_histogram(latencies)
        st.plotly_chart(fig, use_container_width=True, key="res_lat_hist")

        lat_cols = st.columns(4)
        sorted_lat = sorted(latencies)
        with lat_cols[0]:
            st.metric("Min", f"{min(latencies):.0f}ms")
        with lat_cols[1]:
            st.metric("Median (P50)", f"{sorted_lat[len(sorted_lat)//2]:.0f}ms")
        with lat_cols[2]:
            p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
            st.metric("P95", f"{sorted_lat[p95_idx]:.0f}ms")
        with lat_cols[3]:
            st.metric("Max", f"{max(latencies):.0f}ms")
    else:
        st.info("No latency data.")

with tab_evaluators:
    results = summary.get("results", [])
    if results:
        fig = evaluator_scores_chart(results)
        st.plotly_chart(fig, use_container_width=True, key="res_eval_chart")

        # Detailed evaluator table
        eval_stats: dict[str, dict] = {}
        for r in results:
            for ev in r.get("evaluations", []):
                name = ev.get("evaluator", "?")
                if name not in eval_stats:
                    eval_stats[name] = {"passed": 0, "failed": 0, "scores": []}
                if ev.get("passed"):
                    eval_stats[name]["passed"] += 1
                else:
                    eval_stats[name]["failed"] += 1
                if ev.get("score") is not None:
                    eval_stats[name]["scores"].append(ev["score"])

        ev_table = []
        for name, stats in sorted(eval_stats.items()):
            total = stats["passed"] + stats["failed"]
            avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else None
            ev_table.append({
                "Evaluator": name.replace("_", " ").title(),
                "Tests": total,
                "Passed": stats["passed"],
                "Failed": stats["failed"],
                "Pass Rate": f"{stats['passed']/total:.1%}" if total else "—",
                "Avg Score": f"{avg:.3f}" if avg else "—",
                "Min Score": f"{min(stats['scores']):.3f}" if stats["scores"] else "—",
                "Max Score": f"{max(stats['scores']):.3f}" if stats["scores"] else "—",
            })
        st.dataframe(ev_table, use_container_width=True, hide_index=True)

st.divider()

# --- Category breakdown ---
st.subheader("📂 Results by Category")
by_cat = summary.get("by_category", {})
if by_cat:
    cat_table = []
    for cat, stats in sorted(by_cat.items()):
        cat_table.append({
            "Category": cat.replace("_", " ").title(),
            "Total": stats.get("total", 0),
            "Passed": stats.get("passed", 0),
            "Failed": stats.get("failed", 0),
            "Pass Rate": f"{stats.get('pass_rate', 0):.1%}",
        })
    st.dataframe(cat_table, use_container_width=True, hide_index=True)

st.divider()

# --- Filterable Results Explorer ---
st.subheader("🔍 Results Explorer")

results = summary.get("results", [])
if results:
    filter_cols = st.columns(4)
    with filter_cols[0]:
        f_cat = st.selectbox("Category", ["All"] + sorted(set(r.get("test_case", {}).get("category", "") for r in results)), key="f_cat")
    with filter_cols[1]:
        f_status = st.selectbox("Status", ["All", "Passed", "Failed"], key="f_status")
    with filter_cols[2]:
        f_sev = st.selectbox("Severity", ["All", "critical", "high", "medium", "low"], key="f_sev")
    with filter_cols[3]:
        f_search = st.text_input("🔍 Search", "", key="f_search")

    filtered = results
    if f_cat != "All":
        filtered = [r for r in filtered if r.get("test_case", {}).get("category") == f_cat]
    if f_status == "Passed":
        filtered = [r for r in filtered if r.get("overall_passed")]
    elif f_status == "Failed":
        filtered = [r for r in filtered if not r.get("overall_passed")]
    if f_sev != "All":
        filtered = [r for r in filtered if r.get("test_case", {}).get("severity") == f_sev]
    if f_search:
        sl = f_search.lower()
        filtered = [r for r in filtered if sl in r.get("test_case", {}).get("id", "").lower() or sl in str(r.get("test_case", {}).get("input", "")).lower()]

    st.caption(f"Showing {len(filtered)} of {len(results)} results")

    for r in filtered:
        tc = r.get("test_case", {})
        passed = r.get("overall_passed", False)
        score = r.get("overall_score")
        icon = "✅" if passed else "❌"
        sev = severity_icon(tc.get("severity", "medium"))

        label = f"{icon} {sev} **{tc.get('id', '?')}** — {tc.get('category', '?')} — Score: {f'{score:.3f}' if score is not None else '—'} — {r.get('latency_ms', 0):.0f}ms"

        with st.expander(label):
            detail_col1, detail_col2 = st.columns([3, 2])

            with detail_col1:
                input_val = tc.get("input", "")
                if isinstance(input_val, list):
                    st.markdown("**Conversation:**")
                    for msg in input_val:
                        role = msg.get("role", "?")
                        emoji = "👤" if role == "user" else "🤖"
                        st.markdown(f"{emoji} **{role}:** {msg.get('content', '')}")
                else:
                    st.markdown(f"**Input:** {input_val}")

                st.markdown(f"**Expected:** {tc.get('expected_behavior', '')}")
                st.markdown(f"**Response:**")
                st.text(r.get("response", "")[:600])

            with detail_col2:
                st.markdown("**Evaluations:**")
                for ev in r.get("evaluations", []):
                    ev_icon = "✅" if ev.get("passed") else "❌"
                    ev_score = ev.get("score")
                    st.markdown(f"{ev_icon} **{ev.get('evaluator', '?')}**")
                    if ev_score is not None:
                        st.markdown(score_bar(ev_score), unsafe_allow_html=True)
                    reason = ev.get("reason", "")
                    if reason:
                        st.caption(reason[:200])

                if r.get("retrieved_contexts"):
                    st.markdown(f"**Retrieved Contexts:** {len(r['retrieved_contexts'])} docs")
