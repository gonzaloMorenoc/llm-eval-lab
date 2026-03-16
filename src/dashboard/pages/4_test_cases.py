"""Page 4: Test Cases Manager — browse, filter, and add test cases."""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import severity_icon

st.set_page_config(page_title="Test Cases — LLM Eval Lab", page_icon="📝", layout="wide")
render_sidebar()

st.title("📝 Test Cases Manager")
st.markdown("Browse, filter, and manage evaluation test cases.")

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"))

# --- Dataset overview ---
st.subheader("Dataset Overview")

from src.runner.runner import load_dataset

dataset_files = {}
for fname in sorted(os.listdir(DATASETS_DIR)):
    if fname.endswith(".jsonl") and fname != "rag_knowledge_base.jsonl":
        name = fname.replace(".jsonl", "")
        cases = load_dataset(os.path.join(DATASETS_DIR, fname))
        dataset_files[name] = cases

total_cases = sum(len(c) for c in dataset_files.values())

cols = st.columns(len(dataset_files) + 1)
for i, (name, cases) in enumerate(dataset_files.items()):
    with cols[i]:
        st.metric(name.replace("_", " ").title(), len(cases))
with cols[-1]:
    st.metric("Total", total_cases)

st.divider()

# --- Filters ---
st.subheader("Browse Test Cases")

filter_col1, filter_col2, filter_col3 = st.columns(3)

all_cases = []
for cases in dataset_files.values():
    all_cases.extend(cases)

with filter_col1:
    categories = ["All"] + sorted(set(c.category for c in all_cases))
    filter_cat = st.selectbox("Category", categories)

with filter_col2:
    severities = ["All", "critical", "high", "medium", "low"]
    filter_sev = st.selectbox("Severity", severities)

with filter_col3:
    all_eval_types = set()
    for c in all_cases:
        all_eval_types.update(c.evaluation_type)
    eval_types = ["All"] + sorted(all_eval_types)
    filter_eval = st.selectbox("Evaluator", eval_types)

# Search
search = st.text_input("🔍 Search (ID, input, expected behavior)", "")

# Apply filters
filtered = all_cases
if filter_cat != "All":
    filtered = [c for c in filtered if c.category == filter_cat]
if filter_sev != "All":
    filtered = [c for c in filtered if c.severity == filter_sev]
if filter_eval != "All":
    filtered = [c for c in filtered if filter_eval in c.evaluation_type]
if search:
    search_lower = search.lower()
    filtered = [
        c for c in filtered
        if search_lower in c.id.lower()
        or search_lower in str(c.input).lower()
        or search_lower in c.expected_behavior.lower()
    ]

st.caption(f"Showing {len(filtered)} of {total_cases} test cases.")

# --- Results table ---
table_data = []
for c in filtered:
    input_preview = c.input if isinstance(c.input, str) else str(c.input)
    table_data.append({
        "ID": c.id,
        "Category": c.category,
        "Severity": f"{severity_icon(c.severity)} {c.severity}",
        "Input": input_preview[:80] + ("..." if len(input_preview) > 80 else ""),
        "Evaluators": ", ".join(c.evaluation_type),
        "Has Reference": "✅" if c.reference else "❌",
    })

st.dataframe(table_data, use_container_width=True, hide_index=True)

# --- Detailed view ---
if filtered:
    st.divider()
    st.subheader("Detailed View")

    for c in filtered:
        sev = severity_icon(c.severity)
        with st.expander(f"{sev} **{c.id}** — {c.category} — {c.severity}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input:**")
                if isinstance(c.input, list):
                    for msg in c.input:
                        role = msg.get("role", "?")
                        content = msg.get("content", "")
                        st.markdown(f"- **{role}:** {content}")
                else:
                    st.markdown(c.input)

                st.markdown(f"**Expected Behavior:** {c.expected_behavior}")

            with col2:
                st.markdown(f"**Reference:** {c.reference or '—'}")
                st.markdown(f"**Evaluation Types:** `{', '.join(c.evaluation_type)}`")
                if c.ragas_metrics:
                    st.markdown(f"**RAGAS Metrics:** `{', '.join(c.ragas_metrics)}`")
                st.markdown(f"**Metadata:** `{json.dumps(c.metadata)}`")

st.divider()

# --- Add new test case ---
st.subheader("➕ Add New Test Case")
st.markdown("Add a new test case to an existing dataset.")

with st.form("add_test_case", clear_on_submit=True):
    form_col1, form_col2 = st.columns(2)

    with form_col1:
        new_id = st.text_input("Test Case ID", placeholder="func_014")
        new_category = st.selectbox("Category", ["functional", "safety", "regression", "multi_turn"], key="new_cat")
        new_input = st.text_area("Input (single-turn text)", placeholder="What is Kubernetes?")
        new_expected = st.text_area("Expected Behavior", placeholder="Explains Kubernetes as a container orchestration platform.")

    with form_col2:
        new_reference = st.text_area("Reference (optional)", placeholder="Kubernetes is an open-source container orchestration system...")
        new_severity = st.selectbox("Severity", ["low", "medium", "high", "critical"], index=1, key="new_sev")
        new_eval_types = st.multiselect(
            "Evaluation Types",
            ["rule_based", "safety", "ragas", "deepeval", "consistency", "llm_judge"],
            default=["rule_based"],
        )
        new_topic = st.text_input("Topic (metadata)", placeholder="devops")

    submitted = st.form_submit_button("Add Test Case", type="primary", use_container_width=True)

    if submitted:
        if not new_id or not new_input or not new_expected:
            st.error("ID, Input, and Expected Behavior are required.")
        elif any(c.id == new_id for c in all_cases):
            st.error(f"Test case ID `{new_id}` already exists.")
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
                "metadata": {"topic": new_topic, "dataset_version": "2.0"} if new_topic else {"dataset_version": "2.0"},
            }

            # Append to the appropriate dataset file
            target_file = os.path.join(DATASETS_DIR, f"{new_category}.jsonl")
            with open(target_file, "a") as f:
                f.write("\n" + json.dumps(new_case, ensure_ascii=False))

            st.success(f"Test case `{new_id}` added to `{new_category}.jsonl`!")
            st.rerun()
