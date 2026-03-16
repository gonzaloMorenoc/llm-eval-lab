"""Page 4: Test Cases Manager — browse, filter, search, add, and analyze test cases."""

from __future__ import annotations

import json
import os
import sys

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import severity_icon, severity_badge, kpi_row
from src.dashboard.components.charts import COLORS, CATEGORY_COLORS, SEVERITY_COLORS

st.set_page_config(page_title="Test Cases — LLM Eval Lab", page_icon="📝", layout="wide")
render_sidebar()

st.markdown(
    """
    <style>
    .stat-card { background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 10px; padding: 1rem; border: 1px solid #3d3d5c; }
    .stat-label { font-size: 0.8rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: #e0e0e0; margin-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📝 Test Cases Manager")

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"))

from src.runner.runner import load_dataset

# --- Load all datasets ---
dataset_files = {}
for fname in sorted(os.listdir(DATASETS_DIR)):
    if fname.endswith(".jsonl") and fname != "rag_knowledge_base.jsonl":
        name = fname.replace(".jsonl", "")
        cases = load_dataset(os.path.join(DATASETS_DIR, fname))
        dataset_files[name] = cases

all_cases = []
for cases in dataset_files.values():
    all_cases.extend(cases)
total_cases = len(all_cases)

# --- Dataset Analytics ---
st.subheader("📊 Dataset Analytics")

# Category distribution
categories = {}
severities = {}
eval_types = {}
topics = {}

for c in all_cases:
    categories[c.category] = categories.get(c.category, 0) + 1
    severities[c.severity] = severities.get(c.severity, 0) + 1
    for et in c.evaluation_type:
        eval_types[et] = eval_types.get(et, 0) + 1
    topic = c.metadata.get("topic", "untagged")
    topics[topic] = topics.get(topic, 0) + 1

# KPI row
kpi_row([
    ("Total Cases", str(total_cases), COLORS["primary"]),
    ("Categories", str(len(categories)), COLORS["info"]),
    ("Unique Topics", str(len(topics)), COLORS["success"]),
    ("With Reference", str(sum(1 for c in all_cases if c.reference)), COLORS["warning"]),
])

st.markdown("")

# Charts
chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    # Category donut
    fig = go.Figure(go.Pie(
        labels=[c.replace("_", " ").title() for c in categories.keys()],
        values=list(categories.values()),
        hole=0.45,
        marker=dict(colors=[CATEGORY_COLORS.get(c, COLORS["muted"]) for c in categories.keys()]),
        textinfo="label+value",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"]),
        title=dict(text="By Category", font=dict(size=13)),
        height=280, margin=dict(l=20, r=20, t=40, b=20), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="tc_cat")

with chart_col2:
    # Severity donut
    sev_order = ["critical", "high", "medium", "low"]
    sev_labels = [s for s in sev_order if s in severities]
    fig = go.Figure(go.Pie(
        labels=[s.title() for s in sev_labels],
        values=[severities[s] for s in sev_labels],
        hole=0.45,
        marker=dict(colors=[SEVERITY_COLORS.get(s, COLORS["muted"]) for s in sev_labels]),
        textinfo="label+value",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"]),
        title=dict(text="By Severity", font=dict(size=13)),
        height=280, margin=dict(l=20, r=20, t=40, b=20), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="tc_sev")

with chart_col3:
    # Evaluator coverage bar
    fig = go.Figure(go.Bar(
        x=list(eval_types.values()),
        y=[e.replace("_", " ").title() for e in eval_types.keys()],
        orientation="h",
        marker_color=COLORS["primary"],
        text=list(eval_types.values()),
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        title=dict(text="Evaluator Coverage", font=dict(size=13)),
        height=280, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["border"]),
    )
    st.plotly_chart(fig, use_container_width=True, key="tc_eval")

st.divider()

# --- Browse & Filter ---
st.subheader("🔍 Browse Test Cases")

filter_cols = st.columns(5)
with filter_cols[0]:
    f_cat = st.selectbox("Category", ["All"] + sorted(set(c.category for c in all_cases)), key="tc_f_cat")
with filter_cols[1]:
    f_sev = st.selectbox("Severity", ["All", "critical", "high", "medium", "low"], key="tc_f_sev")
with filter_cols[2]:
    all_eval_types_list = sorted(set(et for c in all_cases for et in c.evaluation_type))
    f_eval = st.selectbox("Evaluator", ["All"] + all_eval_types_list, key="tc_f_eval")
with filter_cols[3]:
    f_ref = st.selectbox("Has Reference", ["All", "Yes", "No"], key="tc_f_ref")
with filter_cols[4]:
    f_search = st.text_input("🔍 Search", "", key="tc_search", placeholder="ID, input, topic...")

# Apply filters
filtered = all_cases
if f_cat != "All":
    filtered = [c for c in filtered if c.category == f_cat]
if f_sev != "All":
    filtered = [c for c in filtered if c.severity == f_sev]
if f_eval != "All":
    filtered = [c for c in filtered if f_eval in c.evaluation_type]
if f_ref == "Yes":
    filtered = [c for c in filtered if c.reference]
elif f_ref == "No":
    filtered = [c for c in filtered if not c.reference]
if f_search:
    sl = f_search.lower()
    filtered = [
        c for c in filtered
        if sl in c.id.lower()
        or sl in str(c.input).lower()
        or sl in c.expected_behavior.lower()
        or sl in str(c.metadata).lower()
    ]

st.caption(f"Showing **{len(filtered)}** of {total_cases} test cases")

# Summary table
table_data = []
for c in filtered:
    input_preview = c.input if isinstance(c.input, str) else f"[{len(c.input)} messages]"
    table_data.append({
        "ID": c.id,
        "Category": c.category,
        "Severity": f"{severity_icon(c.severity)} {c.severity}",
        "Input": input_preview[:70] + ("..." if len(str(input_preview)) > 70 else ""),
        "Evaluators": ", ".join(c.evaluation_type),
        "Ref": "✅" if c.reference else "—",
        "Topic": c.metadata.get("topic", "—"),
    })

st.dataframe(table_data, use_container_width=True, hide_index=True, height=min(400, 40 + 35 * len(table_data)))

# --- Detailed Expandable View ---
if filtered:
    st.divider()
    st.subheader("📋 Detailed View")

    for c in filtered:
        sev = severity_icon(c.severity)
        cat_icon = {"functional": "⚡", "safety": "🛡️", "regression": "🔁", "multi_turn": "💬"}.get(c.category, "📋")

        with st.expander(f"{cat_icon} {sev} **{c.id}** — {c.category} — {c.severity}"):
            d_col1, d_col2 = st.columns([3, 2])

            with d_col1:
                if isinstance(c.input, list):
                    st.markdown("**Multi-turn Conversation:**")
                    for msg in c.input:
                        role = msg.get("role", "?")
                        emoji = "👤" if role == "user" else "🤖"
                        st.markdown(f"{emoji} **{role}:** {msg.get('content', '')}")
                else:
                    st.markdown(f"**Input:** {c.input}")

                st.markdown(f"**Expected Behavior:** {c.expected_behavior}")

                if c.reference:
                    st.markdown(f"**Reference Answer:** {c.reference}")
                else:
                    st.caption("No reference answer")

            with d_col2:
                st.markdown(f"**Evaluation Types:** `{', '.join(c.evaluation_type)}`")
                if c.ragas_metrics:
                    st.markdown(f"**RAGAS Metrics:** `{', '.join(c.ragas_metrics)}`")
                st.markdown(f"**Severity:** {severity_icon(c.severity)} {c.severity}")

                if c.metadata:
                    st.markdown("**Metadata:**")
                    for k, v in c.metadata.items():
                        st.markdown(f"- `{k}`: {v}")

st.divider()

# --- Add New Test Case ---
st.subheader("➕ Add New Test Case")

with st.form("add_test_case", clear_on_submit=True):
    form_col1, form_col2 = st.columns(2)

    with form_col1:
        new_id = st.text_input("Test Case ID *", placeholder="func_014")
        new_category = st.selectbox("Category *", ["functional", "safety", "regression", "multi_turn"], key="new_cat")
        new_severity = st.selectbox("Severity *", ["low", "medium", "high", "critical"], index=1, key="new_sev")
        new_input = st.text_area("Input (single-turn) *", placeholder="What is Kubernetes?", height=100)
        new_expected = st.text_area("Expected Behavior *", placeholder="Explains Kubernetes as a container orchestration platform.", height=80)

    with form_col2:
        new_reference = st.text_area("Reference Answer (optional)", placeholder="Kubernetes is an open-source...", height=100)
        new_eval_types = st.multiselect(
            "Evaluation Types *",
            ["rule_based", "safety", "ragas", "deepeval", "consistency", "llm_judge"],
            default=["rule_based"],
        )
        new_topic = st.text_input("Topic", placeholder="devops")

        # Preview
        if new_id and new_input:
            st.markdown("**Preview:**")
            st.json({
                "id": new_id,
                "category": new_category,
                "severity": new_severity,
                "input": new_input[:80] + "...",
                "evaluation_type": new_eval_types,
            })

    submitted = st.form_submit_button("➕ Add Test Case", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not new_id:
            errors.append("ID is required")
        if not new_input:
            errors.append("Input is required")
        if not new_expected:
            errors.append("Expected behavior is required")
        if not new_eval_types:
            errors.append("At least one evaluator is required")
        if any(c.id == new_id for c in all_cases):
            errors.append(f"ID `{new_id}` already exists")

        if errors:
            for e in errors:
                st.error(e)
        else:
            new_case = {
                "id": new_id,
                "category": new_category,
                "input": new_input,
                "expected_behavior": new_expected,
                "reference": new_reference or None,
                "evaluation_type": new_eval_types,
                "ragas_metrics": None,
                "severity": new_severity,
                "metadata": {},
            }
            if new_topic:
                new_case["metadata"]["topic"] = new_topic
            new_case["metadata"]["dataset_version"] = "2.0"

            target_file = os.path.join(DATASETS_DIR, f"{new_category}.jsonl")
            with open(target_file, "a") as f:
                f.write("\n" + json.dumps(new_case, ensure_ascii=False))

            st.success(f"✅ Test case `{new_id}` added to `{new_category}.jsonl`!")
            st.rerun()
