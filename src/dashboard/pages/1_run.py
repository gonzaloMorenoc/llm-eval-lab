"""Page 1: Run Evaluation — configure and launch evaluation runs."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import severity_icon

st.set_page_config(page_title="Run Evaluation — LLM Eval Lab", page_icon="🚀", layout="wide")
render_sidebar()

st.title("🚀 Run Evaluation")
st.markdown("Configure and launch an evaluation run against a chatbot provider.")


# --- Dataset selection ---
st.subheader("1. Select Datasets")

from src.runner.runner import load_all_datasets, load_dataset

datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"))
available_datasets = {}
for fname in sorted(os.listdir(datasets_dir)):
    if fname.endswith(".jsonl") and fname != "rag_knowledge_base.jsonl":
        name = fname.replace(".jsonl", "")
        cases = load_dataset(os.path.join(datasets_dir, fname))
        available_datasets[name] = cases

cols = st.columns(len(available_datasets))
selected_datasets: list[str] = []
for i, (name, cases) in enumerate(available_datasets.items()):
    with cols[i]:
        checked = st.checkbox(
            f"{name.replace('_', ' ').title()} ({len(cases)})",
            value=True,
            key=f"ds_{name}",
        )
        if checked:
            selected_datasets.append(name)

# Gather selected test cases
all_selected_cases = []
for name in selected_datasets:
    all_selected_cases.extend(available_datasets[name])

st.info(f"**{len(all_selected_cases)}** test cases selected across **{len(selected_datasets)}** datasets.")


# --- Configuration summary ---
st.subheader("2. Configuration Summary")

provider = st.session_state.get("selected_provider", "mock")
mode = st.session_state.get("selected_mode", "plain")
active_evals = st.session_state.get("active_evaluators", ["rule_based", "safety"])
max_concurrent = st.session_state.get("max_concurrent", 5)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Provider:** `{provider}`")
    st.markdown(f"**Mode:** `{mode}`")
with col2:
    st.markdown(f"**Evaluators:** {', '.join(active_evals)}")
    st.markdown(f"**Concurrency:** {max_concurrent}")
with col3:
    st.markdown(f"**Test cases:** {len(all_selected_cases)}")
    st.markdown(f"**Datasets:** {', '.join(selected_datasets)}")

st.divider()

# --- Run button ---
st.subheader("3. Execute")


def _build_chatbot(provider_name: str, mode: str):
    """Build a chatbot instance based on config."""
    if provider_name == "mock":
        from src.chatbots.mock_adapter import MockChatbot, MockRAGChatbot
        return MockRAGChatbot() if mode == "rag" else MockChatbot()

    if mode == "rag":
        from src.chatbots.rag_chatbot import DemoRAGChatbot
        return DemoRAGChatbot(provider_name=provider_name)

    from src.chatbots.openai_compatible import OpenAICompatibleChatbot
    return OpenAICompatibleChatbot(provider_name=provider_name)


def _build_evaluators(active_evals: list[str]) -> dict:
    """Build evaluator instances from selected names."""
    evaluators = {}

    if "rule_based" in active_evals:
        from src.evaluators.rule_based import RuleBasedEvaluator
        evaluators["rule_based"] = RuleBasedEvaluator()

    if "safety" in active_evals:
        from src.evaluators.safety import SafetyEvaluator
        evaluators["safety"] = SafetyEvaluator()

    if "ragas" in active_evals:
        try:
            from src.evaluators.ragas_evaluator import RagasEvaluator
            evaluators["ragas"] = RagasEvaluator()
        except Exception as e:
            st.warning(f"Could not init RAGAS: {e}")

    if "deepeval" in active_evals:
        try:
            from src.evaluators.deepeval_evaluator import DeepEvalEvaluator
            evaluators["deepeval"] = DeepEvalEvaluator()
        except Exception as e:
            st.warning(f"Could not init DeepEval: {e}")

    if "consistency" in active_evals:
        from src.evaluators.consistency import ConsistencyEvaluator
        evaluators["consistency"] = ConsistencyEvaluator()

    if "llm_judge" in active_evals:
        try:
            from src.evaluators.llm_judge import LLMJudgeEvaluator
            evaluators["llm_judge"] = LLMJudgeEvaluator()
        except Exception as e:
            st.warning(f"Could not init LLM Judge: {e}")

    return evaluators


if st.button("🚀 Start Evaluation", type="primary", use_container_width=True):
    if not all_selected_cases:
        st.error("No test cases selected. Please select at least one dataset.")
    elif not active_evals:
        st.error("No evaluators selected. Please enable at least one evaluator in the sidebar.")
    else:
        with st.status("Running evaluation...", expanded=True) as status:
            st.write(f"Building chatbot: **{provider}** ({mode} mode)")
            chatbot = _build_chatbot(provider, mode)

            st.write(f"Initializing evaluators: **{', '.join(active_evals)}**")
            evaluators = _build_evaluators(active_evals)

            st.write(f"Running **{len(all_selected_cases)}** test cases with concurrency={max_concurrent}...")

            from src.runner.runner import EvalRunner

            config = st.session_state.get("config", {})
            config.setdefault("runner", {})["max_concurrent"] = max_concurrent
            runner = EvalRunner(chatbot=chatbot, evaluators=evaluators, config=config)

            progress_bar = st.progress(0, text="Starting...")

            # Run async evaluation
            loop = asyncio.new_event_loop()
            summary = loop.run_until_complete(runner.run(all_selected_cases))
            loop.close()

            progress_bar.progress(100, text="Complete!")

            # Save results
            results_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", summary.run_id)
            )
            from src.reporting.json_reporter import generate_json_report
            from src.reporting.markdown_reporter import generate_markdown_report

            json_path = generate_json_report(summary, results_dir)
            md_path = generate_markdown_report(summary, results_dir)

            status.update(label="Evaluation complete!", state="complete")

        # Store summary for immediate viewing
        st.session_state["last_summary"] = summary.model_dump()
        st.session_state["last_run_id"] = summary.run_id

        # Results preview
        st.success(f"Run **{summary.run_id}** completed!")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pass Rate", f"{summary.pass_rate:.1%}")
        with col2:
            st.metric("Avg Score", f"{summary.avg_score:.3f}")
        with col3:
            st.metric("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
        with col4:
            st.metric("Critical Failures", summary.critical_failures)

        st.divider()

        # Results table
        st.subheader("Results")
        results_data = []
        for r in summary.results:
            results_data.append({
                "ID": r.test_case.id,
                "Category": r.test_case.category,
                "Severity": f"{severity_icon(r.test_case.severity)} {r.test_case.severity}",
                "Passed": "✅" if r.overall_passed else "❌",
                "Score": f"{r.overall_score:.3f}" if r.overall_score is not None else "—",
                "Latency": f"{r.latency_ms:.0f}ms",
                "Error": r.error or "",
            })

        st.dataframe(results_data, use_container_width=True, hide_index=True)

        # Detailed failures
        failures = [r for r in summary.results if not r.overall_passed]
        if failures:
            st.subheader(f"Failed Tests ({len(failures)})")
            for r in failures:
                with st.expander(f"{severity_icon(r.test_case.severity)} {r.test_case.id} — {r.test_case.category}"):
                    input_display = r.test_case.input if isinstance(r.test_case.input, str) else str(r.test_case.input)
                    st.markdown(f"**Input:** {input_display[:300]}")
                    st.markdown(f"**Expected:** {r.test_case.expected_behavior}")
                    st.markdown(f"**Response:** {r.response[:500]}")
                    if r.error:
                        st.error(f"Error: {r.error}")
                    for ev in r.evaluations:
                        icon = "✅" if ev.passed else "❌"
                        st.markdown(f"- {icon} **{ev.evaluator}**: score={ev.score} — {ev.reason[:200]}")

        st.divider()
        st.markdown(f"📁 Reports saved to `results/{summary.run_id}/`")
        st.page_link("src/dashboard/pages/2_results.py", label="📊 View Full Results Dashboard →")
