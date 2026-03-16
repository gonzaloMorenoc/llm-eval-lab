"""Page 1: Run Evaluation — configure and launch evaluation runs with live feedback."""

from __future__ import annotations

import asyncio
import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import severity_icon, kpi_row, score_bar
from src.dashboard.components.charts import COLORS

st.set_page_config(page_title="Run Evaluation — LLM Eval Lab", page_icon="🚀", layout="wide")
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

st.title("🚀 Run Evaluation")

# --- Step 1: Dataset Selection ---
st.markdown("### 1. Select Datasets")

from src.runner.runner import load_all_datasets, load_dataset

datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"))
available_datasets = {}
for fname in sorted(os.listdir(datasets_dir)):
    if fname.endswith(".jsonl") and fname != "rag_knowledge_base.jsonl":
        name = fname.replace(".jsonl", "")
        cases = load_dataset(os.path.join(datasets_dir, fname))
        available_datasets[name] = cases

# Visual dataset cards
cols = st.columns(len(available_datasets))
selected_datasets: list[str] = []
cat_icons = {"functional": "⚡", "safety": "🛡️", "regression": "🔁", "multi_turn": "💬"}

for i, (name, cases) in enumerate(available_datasets.items()):
    with cols[i]:
        icon = cat_icons.get(name, "📋")
        checked = st.checkbox(
            f"{icon} {name.replace('_', ' ').title()} ({len(cases)})",
            value=True,
            key=f"ds_{name}",
        )
        if checked:
            selected_datasets.append(name)
        # Show severity breakdown
        sevs = {}
        for c in cases:
            sevs[c.severity] = sevs.get(c.severity, 0) + 1
        sev_str = " ".join(f"{severity_icon(s)}{n}" for s, n in sorted(sevs.items()))
        st.caption(sev_str)

all_selected_cases = []
for name in selected_datasets:
    all_selected_cases.extend(available_datasets[name])

st.info(f"📦 **{len(all_selected_cases)}** test cases across **{len(selected_datasets)}** datasets")

st.divider()

# --- Step 2: Configuration Summary ---
st.markdown("### 2. Configuration")

provider = st.session_state.get("selected_provider", "mock")
mode = st.session_state.get("selected_mode", "plain")
active_evals = st.session_state.get("active_evaluators", ["rule_based", "safety"])
max_concurrent = st.session_state.get("max_concurrent", 5)

config_cols = st.columns(4)
with config_cols[0]:
    st.markdown(f"**🔌 Provider**")
    st.code(provider, language=None)
with config_cols[1]:
    st.markdown(f"**🔀 Mode**")
    st.code(mode, language=None)
with config_cols[2]:
    st.markdown(f"**🧪 Evaluators**")
    st.code(", ".join(active_evals), language=None)
with config_cols[3]:
    st.markdown(f"**⚙️ Concurrency**")
    st.code(str(max_concurrent), language=None)

st.divider()

# --- Step 3: Execute ---
st.markdown("### 3. Execute")


def _build_chatbot(provider_name: str, run_mode: str):
    if provider_name == "mock":
        from src.chatbots.mock_adapter import MockChatbot, MockRAGChatbot
        return MockRAGChatbot() if run_mode == "rag" else MockChatbot()
    if run_mode == "rag":
        from src.chatbots.rag_chatbot import DemoRAGChatbot
        return DemoRAGChatbot(provider_name=provider_name)
    from src.chatbots.openai_compatible import OpenAICompatibleChatbot
    return OpenAICompatibleChatbot(provider_name=provider_name)


def _build_evaluators(eval_names: list[str]) -> dict:
    evaluators = {}
    if "rule_based" in eval_names:
        from src.evaluators.rule_based import RuleBasedEvaluator
        evaluators["rule_based"] = RuleBasedEvaluator()
    if "safety" in eval_names:
        from src.evaluators.safety import SafetyEvaluator
        evaluators["safety"] = SafetyEvaluator()
    if "ragas" in eval_names:
        try:
            from src.evaluators.ragas_evaluator import RagasEvaluator
            evaluators["ragas"] = RagasEvaluator()
        except Exception as e:
            st.warning(f"Could not init RAGAS: {e}")
    if "deepeval" in eval_names:
        try:
            from src.evaluators.deepeval_evaluator import DeepEvalEvaluator
            evaluators["deepeval"] = DeepEvalEvaluator()
        except Exception as e:
            st.warning(f"Could not init DeepEval: {e}")
    if "consistency" in eval_names:
        from src.evaluators.consistency import ConsistencyEvaluator
        evaluators["consistency"] = ConsistencyEvaluator()
    if "llm_judge" in eval_names:
        try:
            from src.evaluators.llm_judge import LLMJudgeEvaluator
            evaluators["llm_judge"] = LLMJudgeEvaluator()
        except Exception as e:
            st.warning(f"Could not init LLM Judge: {e}")
    return evaluators


# Run button
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    run_clicked = st.button("🚀 Start Evaluation", type="primary", use_container_width=True, disabled=not all_selected_cases or not active_evals)
with col_btn2:
    if not all_selected_cases:
        st.warning("No test cases")
    elif not active_evals:
        st.warning("No evaluators")

if run_clicked:
    start_time = time.time()

    with st.status("🔬 Running evaluation...", expanded=True) as status:
        st.write(f"🔌 Building chatbot: **{provider}** ({mode} mode)")
        chatbot = _build_chatbot(provider, mode)
        st.write(f"✅ Chatbot ready: `{chatbot.get_id()}`")

        st.write(f"🧪 Initializing evaluators: **{', '.join(active_evals)}**")
        evaluators = _build_evaluators(active_evals)
        st.write(f"✅ {len(evaluators)} evaluators ready")

        st.write(f"⏳ Running **{len(all_selected_cases)}** test cases (concurrency={max_concurrent})...")

        from src.runner.runner import EvalRunner

        config = st.session_state.get("config", {})
        config.setdefault("runner", {})["max_concurrent"] = max_concurrent
        runner = EvalRunner(chatbot=chatbot, evaluators=evaluators, config=config)

        progress = st.progress(0, text="Starting evaluation...")

        loop = asyncio.new_event_loop()
        summary = loop.run_until_complete(runner.run(all_selected_cases))
        loop.close()

        elapsed = time.time() - start_time
        progress.progress(100, text=f"Complete in {elapsed:.1f}s!")

        # Save results
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", summary.run_id)
        )
        from src.reporting.json_reporter import generate_json_report
        from src.reporting.markdown_reporter import generate_markdown_report

        generate_json_report(summary, results_dir)
        generate_markdown_report(summary, results_dir)

        status.update(label=f"✅ Evaluation complete in {elapsed:.1f}s", state="complete")

    st.session_state["last_summary"] = summary.model_dump()
    st.session_state["last_run_id"] = summary.run_id

    # --- Results Display ---
    st.success(f"Run **{summary.run_id}** completed — {summary.total} tests in {elapsed:.1f}s")

    # KPI row
    pass_color = COLORS["success"] if summary.pass_rate >= 0.7 else (COLORS["warning"] if summary.pass_rate >= 0.5 else COLORS["danger"])
    kpi_row([
        ("Pass Rate", f"{summary.pass_rate:.1%}", pass_color),
        ("Avg Score", f"{summary.avg_score:.3f}", COLORS["primary"]),
        ("Avg Latency", f"{summary.avg_latency_ms:.0f}ms", COLORS["info"]),
        ("Critical Failures", str(summary.critical_failures), COLORS["danger"] if summary.critical_failures > 0 else COLORS["success"]),
        ("Passed / Failed", f"{summary.passed} / {summary.failed}", None),
    ])

    st.divider()

    # Results tabs
    tab_table, tab_failures, tab_details = st.tabs(["📋 All Results", "❌ Failures", "🔍 Details"])

    with tab_table:
        results_data = []
        for r in summary.results:
            results_data.append({
                "": "✅" if r.overall_passed else "❌",
                "ID": r.test_case.id,
                "Category": r.test_case.category,
                "Severity": f"{severity_icon(r.test_case.severity)} {r.test_case.severity}",
                "Score": f"{r.overall_score:.3f}" if r.overall_score is not None else "—",
                "Latency": f"{r.latency_ms:.0f}ms",
                "Evaluators": ", ".join(e.evaluator for e in r.evaluations),
            })
        st.dataframe(results_data, use_container_width=True, hide_index=True)

    with tab_failures:
        failures = [r for r in summary.results if not r.overall_passed]
        if not failures:
            st.success("🎉 All tests passed!")
        else:
            st.markdown(f"**{len(failures)} failed tests:**")
            for r in sorted(failures, key=lambda x: ["critical", "high", "medium", "low"].index(x.test_case.severity)):
                sev = severity_icon(r.test_case.severity)
                with st.expander(f"{sev} **{r.test_case.id}** — {r.test_case.category} — {r.test_case.severity}"):
                    input_display = r.test_case.input if isinstance(r.test_case.input, str) else str(r.test_case.input)
                    st.markdown(f"**Input:** {input_display[:400]}")
                    st.markdown(f"**Expected:** {r.test_case.expected_behavior}")
                    st.markdown(f"**Response:** {r.response[:500]}")
                    if r.error:
                        st.error(f"Error: {r.error}")
                    st.markdown("---")
                    for ev in r.evaluations:
                        icon = "✅" if ev.passed else "❌"
                        score_str = f"{ev.score:.3f}" if ev.score is not None else "—"
                        st.markdown(f"{icon} **{ev.evaluator}** (score: {score_str}) — {ev.reason[:250]}")

    with tab_details:
        # Per-evaluator breakdown
        evaluator_stats: dict[str, dict] = {}
        for r in summary.results:
            for ev in r.evaluations:
                if ev.evaluator not in evaluator_stats:
                    evaluator_stats[ev.evaluator] = {"passed": 0, "failed": 0, "scores": []}
                if ev.passed:
                    evaluator_stats[ev.evaluator]["passed"] += 1
                else:
                    evaluator_stats[ev.evaluator]["failed"] += 1
                if ev.score is not None:
                    evaluator_stats[ev.evaluator]["scores"].append(ev.score)

        st.markdown("**Per-Evaluator Breakdown:**")
        ev_table = []
        for name, stats in sorted(evaluator_stats.items()):
            total = stats["passed"] + stats["failed"]
            avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else None
            ev_table.append({
                "Evaluator": name.replace("_", " ").title(),
                "Passed": stats["passed"],
                "Failed": stats["failed"],
                "Pass Rate": f"{stats['passed']/total:.1%}" if total > 0 else "—",
                "Avg Score": f"{avg:.3f}" if avg is not None else "—",
            })
        st.dataframe(ev_table, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(f"📁 Reports saved to `results/{summary.run_id}/`")
    st.page_link("src/dashboard/pages/2_results.py", label="📊 View Full Results Dashboard →", icon="📊")
