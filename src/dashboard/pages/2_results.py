"""Page 2: Results Dashboard — visualize evaluation results."""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import severity_icon, pass_fail_badge
from src.dashboard.components.charts import (
    pass_rate_bar_chart,
    metrics_radar_chart,
    latency_histogram,
    severity_pie_chart,
)

st.set_page_config(page_title="Results — LLM Eval Lab", page_icon="📊", layout="wide")
render_sidebar()

st.title("📊 Results Dashboard")


def _list_runs() -> list[dict]:
    """List available run summaries from the results directory."""
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


def _get_summary(runs: list[dict], run_id: str) -> dict | None:
    for r in runs:
        if r.get("_run_id") == run_id or r.get("run_id") == run_id:
            return r
    return None


# Load available runs
runs = _list_runs()

# Also check session state for just-completed run
last_summary = st.session_state.get("last_summary")
if last_summary and not any(r.get("run_id") == last_summary.get("run_id") for r in runs):
    last_summary["_run_id"] = last_summary.get("run_id", "latest")
    runs.insert(0, last_summary)

if not runs:
    st.info("No evaluation runs found. Go to **Run Evaluation** to create one, or use mock mode for testing.")
    st.stop()

# Run selector
run_labels = [
    f"{r.get('run_id', 'unknown')} — {r.get('chatbot_id', '?')} ({r.get('chatbot_mode', '?')}) — {r.get('timestamp', '')[:19]}"
    for r in runs
]
selected_idx = st.selectbox("Select Run", range(len(runs)), format_func=lambda i: run_labels[i])
summary = runs[selected_idx]

st.divider()

# --- KPI Row ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Pass Rate", f"{summary.get('pass_rate', 0):.1%}")
with col2:
    st.metric("Avg Score", f"{summary.get('avg_score', 0):.3f}")
with col3:
    st.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f}ms")
with col4:
    st.metric("Critical Failures", summary.get("critical_failures", 0))
with col5:
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    st.metric("Total", f"{passed}✅ / {failed}❌ / {total}")

st.divider()

# --- Charts Row ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Pass rate by category
    by_category = summary.get("by_category", {})
    if by_category:
        fig = pass_rate_bar_chart(by_category)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No category data available.")

with chart_col2:
    # Metrics radar chart (RAGAS + DeepEval combined)
    all_metrics = {}
    all_metrics.update(summary.get("ragas_aggregate", {}))
    all_metrics.update(summary.get("deepeval_aggregate", {}))

    if all_metrics:
        config = st.session_state.get("config", {})
        thresholds = {}
        thresholds.update(config.get("ragas", {}).get("thresholds", {}))
        thresholds.update(config.get("deepeval", {}).get("thresholds", {}))
        fig = metrics_radar_chart(all_metrics, thresholds, title="Evaluation Metrics")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No RAGAS/DeepEval metrics available for this run.")

st.divider()

# --- Metrics Tables ---
tab_ragas, tab_deepeval, tab_latency, tab_severity = st.tabs(
    ["RAGAS Metrics", "DeepEval Metrics", "Latency", "Severity"]
)

with tab_ragas:
    ragas_agg = summary.get("ragas_aggregate", {})
    if ragas_agg:
        config = st.session_state.get("config", {})
        ragas_thresholds = config.get("ragas", {}).get("thresholds", {})
        table_data = []
        for metric, avg in sorted(ragas_agg.items()):
            t = ragas_thresholds.get(metric, 0.65)
            status = "✅ PASS" if avg >= t else "❌ FAIL"
            table_data.append({
                "Metric": metric.replace("_", " ").title(),
                "Avg Score": f"{avg:.3f}",
                "Threshold": f"{t}",
                "Status": status,
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("RAGAS evaluator was not enabled for this run.")

with tab_deepeval:
    deepeval_agg = summary.get("deepeval_aggregate", {})
    if deepeval_agg:
        config = st.session_state.get("config", {})
        de_thresholds = config.get("deepeval", {}).get("thresholds", {})
        table_data = []
        for metric, avg in sorted(deepeval_agg.items()):
            t = de_thresholds.get(metric, 0.5)
            status = "✅ PASS" if avg >= t else "❌ FAIL"
            table_data.append({
                "Metric": metric.replace("_", " ").title(),
                "Avg Score": f"{avg:.3f}",
                "Threshold": f"{t}",
                "Status": status,
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("DeepEval evaluator was not enabled for this run.")

with tab_latency:
    results = summary.get("results", [])
    latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms", 0) > 0]
    if latencies:
        fig = latency_histogram(latencies)
        st.plotly_chart(fig, use_container_width=True)

        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            st.metric("Min", f"{min(latencies):.0f}ms")
        with lcol2:
            st.metric("Median", f"{sorted(latencies)[len(latencies)//2]:.0f}ms")
        with lcol3:
            st.metric("Max", f"{max(latencies):.0f}ms")
    else:
        st.info("No latency data available.")

with tab_severity:
    results = summary.get("results", [])
    failures = [r for r in results if not r.get("overall_passed", False)]
    if failures:
        severity_counts: dict[str, int] = {}
        for r in failures:
            sev = r.get("test_case", {}).get("severity", "medium")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        fig = severity_pie_chart(severity_counts)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No failures in this run!")

st.divider()

# --- Results by Category ---
st.subheader("Results by Category")
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

# --- Detailed Results Table ---
st.subheader("All Test Results")

results = summary.get("results", [])
if results:
    # Filter controls
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        filter_category = st.selectbox(
            "Category", ["All"] + sorted(set(r.get("test_case", {}).get("category", "") for r in results))
        )
    with filter_col2:
        filter_status = st.selectbox("Status", ["All", "Passed", "Failed"])
    with filter_col3:
        filter_severity = st.selectbox(
            "Severity", ["All", "critical", "high", "medium", "low"]
        )

    filtered = results
    if filter_category != "All":
        filtered = [r for r in filtered if r.get("test_case", {}).get("category") == filter_category]
    if filter_status == "Passed":
        filtered = [r for r in filtered if r.get("overall_passed")]
    elif filter_status == "Failed":
        filtered = [r for r in filtered if not r.get("overall_passed")]
    if filter_severity != "All":
        filtered = [r for r in filtered if r.get("test_case", {}).get("severity") == filter_severity]

    for r in filtered:
        tc = r.get("test_case", {})
        passed = r.get("overall_passed", False)
        score = r.get("overall_score")
        icon = "✅" if passed else "❌"
        sev = severity_icon(tc.get("severity", "medium"))

        with st.expander(f"{icon} {sev} **{tc.get('id', '?')}** — {tc.get('category', '?')} — Score: {f'{score:.3f}' if score is not None else '—'}"):
            input_display = tc.get("input", "")
            if isinstance(input_display, list):
                for msg in input_display:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    st.markdown(f"**{role}:** {content}")
            else:
                st.markdown(f"**Input:** {input_display}")

            st.markdown(f"**Expected:** {tc.get('expected_behavior', '')}")
            st.markdown(f"**Response:** {r.get('response', '')[:500]}")

            if r.get("retrieved_contexts"):
                with st.expander("Retrieved Contexts"):
                    for i, ctx in enumerate(r["retrieved_contexts"][:5]):
                        st.markdown(f"**Context {i+1}:** {ctx[:300]}")

            st.markdown("**Evaluations:**")
            for ev in r.get("evaluations", []):
                ev_icon = "✅" if ev.get("passed") else "❌"
                ev_score = ev.get("score")
                score_str = f"{ev_score:.3f}" if ev_score is not None else "—"
                st.markdown(f"- {ev_icon} **{ev.get('evaluator', '?')}** (score: {score_str}): {ev.get('reason', '')[:200]}")

    st.caption(f"Showing {len(filtered)} of {len(results)} results.")
