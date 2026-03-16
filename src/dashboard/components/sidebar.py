"""Sidebar — global configuration for provider, mode, evaluators."""

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
        st.image("https://img.icons8.com/fluency/96/test-tube.png", width=48)
        st.title("LLM Eval Lab")
        st.caption("QA Framework for AI Chatbots")

        st.divider()

        # Provider selection
        st.subheader("Configuration")
        active = config.get("active_provider", "groq")
        provider = st.selectbox(
            "Provider",
            options=["mock"] + providers,
            index=(["mock"] + providers).index(active) if active in providers else 0,
            key="sidebar_provider",
        )

        # Mode
        mode = st.radio("Mode", options=["plain", "rag"], horizontal=True, key="sidebar_mode")

        st.divider()

        # Evaluators
        st.subheader("Evaluators")
        evals = {
            "rule_based": st.checkbox("Rule-Based", value=True, key="eval_rb"),
            "safety": st.checkbox("Safety", value=True, key="eval_safety"),
            "ragas": st.checkbox("RAGAS", value=False, key="eval_ragas"),
            "deepeval": st.checkbox("DeepEval", value=False, key="eval_deepeval"),
            "consistency": st.checkbox("Consistency", value=False, key="eval_consistency"),
            "llm_judge": st.checkbox("LLM Judge", value=False, key="eval_judge"),
        }

        active_evals = [k for k, v in evals.items() if v]

        st.divider()

        # Runner settings
        st.subheader("Runner")
        max_concurrent = st.slider("Max Concurrent", 1, 20, config.get("runner", {}).get("max_concurrent", 5), key="runner_concurrent")

        st.divider()
        st.caption("v0.2.0 · [GitHub](https://github.com/gonzaloMorenoc/llm-eval-lab)")

    # Store in session state for use by pages
    st.session_state["config"] = config
    st.session_state["selected_provider"] = provider
    st.session_state["selected_mode"] = mode
    st.session_state["active_evaluators"] = active_evals
    st.session_state["max_concurrent"] = max_concurrent

    return {
        "provider": provider,
        "mode": mode,
        "evaluators": active_evals,
        "max_concurrent": max_concurrent,
        "config": config,
    }
