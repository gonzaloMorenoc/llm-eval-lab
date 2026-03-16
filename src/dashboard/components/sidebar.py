"""Sidebar — global configuration with API key status, tooltips, and provider info."""

from __future__ import annotations

import os
import sys

import streamlit as st
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


def render_sidebar() -> dict:
    """Render the sidebar and return the current configuration state."""
    config = _load_config()
    providers = list(config.get("providers", {}).keys())

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding:0.5rem 0;">
                <span style="font-size:2rem;">🔬</span>
                <h2 style="margin:0; font-size:1.3rem;">LLM Eval Lab</h2>
                <span style="color:#a0a0b0; font-size:0.75rem;">QA Framework for AI Chatbots</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # --- Provider Section ---
        st.markdown("##### 🔌 Provider")
        active = config.get("active_provider", "groq")
        all_providers = ["mock"] + providers
        default_idx = all_providers.index(active) if active in all_providers else 0

        provider = st.selectbox(
            "Provider",
            options=all_providers,
            index=default_idx,
            key="sidebar_provider",
            label_visibility="collapsed",
        )

        # Show provider info
        if provider != "mock" and provider in config.get("providers", {}):
            p_cfg = config["providers"][provider]
            model = p_cfg.get("model", "?")
            limits = p_cfg.get("free_limits", "?")
            api_key_env = p_cfg.get("api_key_env", "")
            has_key = bool(os.getenv(api_key_env, ""))

            key_status = "🟢 Configured" if has_key else "🔴 Missing"
            st.markdown(
                f"""
                <div style="background:#1e1e2e; border-radius:8px; padding:0.6rem; border:1px solid #3d3d5c; font-size:0.8rem;">
                    <div><strong>Model:</strong> <code>{model}</code></div>
                    <div><strong>Limits:</strong> {limits}</div>
                    <div><strong>API Key:</strong> {key_status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif provider == "mock":
            st.markdown(
                """
                <div style="background:#1e1e2e; border-radius:8px; padding:0.6rem; border:1px solid #3d3d5c; font-size:0.8rem;">
                    <div>🧪 <strong>Mock mode</strong> — no API key needed</div>
                    <div style="color:#a0a0b0;">Uses keyword matching for deterministic testing</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # --- Mode ---
        st.markdown("##### 🔀 Mode")
        mode = st.radio(
            "Mode",
            options=["plain", "rag"],
            horizontal=True,
            key="sidebar_mode",
            label_visibility="collapsed",
            help="**Plain**: Direct LLM calls. **RAG**: Retrieval-Augmented Generation with ChromaDB.",
        )

        st.divider()

        # --- Evaluators ---
        st.markdown("##### 🧪 Evaluators")

        evaluator_info = {
            "rule_based": ("📏", "Deterministic checks: non-empty, length, keywords, latency"),
            "safety": ("🛡️", "Prompt injection, system leak, unsafe content detection"),
            "ragas": ("📐", "RAGAS metrics: relevancy, factual correctness, faithfulness"),
            "deepeval": ("🔍", "DeepEval: hallucination, bias, toxicity, GEval"),
            "consistency": ("🔄", "Response stability across multiple runs"),
            "llm_judge": ("⚖️", "LLM-as-judge with rubric-based scoring"),
        }

        evals = {}
        for eval_name, (icon, description) in evaluator_info.items():
            default = eval_name in ("rule_based", "safety")
            evals[eval_name] = st.checkbox(
                f"{icon} {eval_name.replace('_', ' ').title()}",
                value=default,
                key=f"eval_{eval_name}",
                help=description,
            )

        active_evals = [k for k, v in evals.items() if v]

        # API key warnings
        needs_openai = any(e in active_evals for e in ("ragas", "deepeval"))
        if needs_openai and not os.getenv("OPENAI_API_KEY"):
            st.warning("⚠️ RAGAS/DeepEval requieren `OPENAI_API_KEY`", icon="🔑")

        st.divider()

        # --- Runner Settings ---
        st.markdown("##### ⚙️ Runner Settings")
        runner_cfg = config.get("runner", {})

        max_concurrent = st.slider(
            "Concurrency",
            min_value=1,
            max_value=20,
            value=runner_cfg.get("max_concurrent", 5),
            key="runner_concurrent",
            help="Maximum number of test cases running in parallel.",
        )

        timeout = st.number_input(
            "Timeout (ms)",
            min_value=5000,
            max_value=120000,
            value=runner_cfg.get("timeout_ms", 30000),
            step=5000,
            key="runner_timeout",
            help="Maximum time to wait for a single chatbot response.",
        )

        st.divider()

        # --- Footer ---
        st.markdown(
            """
            <div style="text-align:center; color:#6b7280; font-size:0.7rem; padding:0.3rem;">
                v0.3.0 · <a href="https://github.com/gonzaloMorenoc/llm-eval-lab" style="color:#6366f1;">GitHub</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Store in session state
    st.session_state["config"] = config
    st.session_state["selected_provider"] = provider
    st.session_state["selected_mode"] = mode
    st.session_state["active_evaluators"] = active_evals
    st.session_state["max_concurrent"] = max_concurrent
    st.session_state["timeout_ms"] = timeout

    return {
        "provider": provider,
        "mode": mode,
        "evaluators": active_evals,
        "max_concurrent": max_concurrent,
        "timeout_ms": timeout,
        "config": config,
    }
