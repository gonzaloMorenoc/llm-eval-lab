"""Sidebar — global configuration with onboarding hints, API key status, and tooltips."""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.theme import PALETTE


def render_sidebar() -> dict:
    """Render the sidebar and return the current configuration state."""
    from src.dashboard.components.shared import load_config

    config = load_config()
    providers = list(config.get("providers", {}).keys())

    with st.sidebar:
        # ── Logo & Title ──────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="text-align:center; padding:0.75rem 0 0.5rem;">
                <div style="font-size:2.2rem; filter:drop-shadow(0 0 12px rgba(99,102,241,0.5));">🔬</div>
                <div style="font-size:1.15rem; font-weight:800; color:{PALETTE["text"]}; margin-top:0.1rem;">LLM Eval Lab</div>
                <div style="font-size:0.72rem; color:#4b5563; margin-top:0.1rem; letter-spacing:0.05em;">QA Framework para IA · v0.3.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Provider ──────────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="font-size:0.68rem; color:{PALETTE["accent"]}; text-transform:uppercase;
                 letter-spacing:0.1em; font-weight:700; margin-bottom:0.4rem;">
                🔌 Provider (¿Qué LLM evaluar?)
            </div>
            """,
            unsafe_allow_html=True,
        )

        active = config.get("active_provider", "groq")
        all_providers = ["mock", *providers]
        default_idx = all_providers.index(active) if active in all_providers else 0

        provider = st.selectbox(
            "Provider",
            options=all_providers,
            index=default_idx,
            key="sidebar_provider",
            label_visibility="collapsed",
        )

        # Provider info card
        if provider == "mock":
            st.markdown(
                f"""
                <div style="background:{PALETTE["bg_sunken"]}; border-radius:8px; padding:0.65rem 0.75rem;
                     border:1px solid {PALETTE["border"]}; font-size:0.78rem; margin-top:0.25rem;">
                    <div style="color:{PALETTE["success"]}; font-weight:700; margin-bottom:0.2rem;">
                        🧪 Mock Mode — Sin API key
                    </div>
                    <div style="color:{PALETTE["text_muted"]}; line-height:1.45;">
                        Usa keyword matching para tests deterministas.
                        <strong>Perfecto para aprender</strong> cómo funciona el sistema.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif provider in config.get("providers", {}):
            p_cfg = config["providers"][provider]
            model = p_cfg.get("model", "?")
            limits = p_cfg.get("free_limits", "Sin info")
            api_key_env = p_cfg.get("api_key_env", "")
            has_key = bool(os.getenv(api_key_env, ""))

            key_color = PALETTE["success"] if has_key else PALETTE["danger"]
            key_status = "🟢 Configurada" if has_key else "🔴 Faltante"
            key_hint = (
                "" if has_key else f"<br><span style='color:{PALETTE['danger']};'>Configura <code>{api_key_env}</code> en tu .env</span>"
            )

            st.markdown(
                f"""
                <div style="background:{PALETTE["bg_sunken"]}; border-radius:8px; padding:0.65rem 0.75rem;
                     border:1px solid {PALETTE["border"]}; font-size:0.78rem; margin-top:0.25rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span style="color:{PALETTE["text_soft"]};">Modelo</span>
                        <code style="color:{PALETTE["accent_pale"]};">{model}</code>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span style="color:{PALETTE["text_soft"]};">Free tier</span>
                        <span style="color:{PALETTE["text"]};">{limits}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{PALETTE["text_soft"]};">API Key</span>
                        <span style="color:{key_color}; font-weight:600;">{key_status}</span>
                    </div>
                    {key_hint}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Mode ──────────────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="font-size:0.68rem; color:{PALETTE["accent"]}; text-transform:uppercase;
                 letter-spacing:0.1em; font-weight:700; margin-bottom:0.4rem;">
                🔀 Modo de Evaluación
            </div>
            """,
            unsafe_allow_html=True,
        )

        mode = st.radio(
            "Modo",
            options=["plain", "rag"],
            horizontal=True,
            key="sidebar_mode",
            label_visibility="collapsed",
            help="**Plain**: LLM directo sin contexto externo. **RAG**: Retrieval-Augmented Generation con ChromaDB.",
        )

        mode_desc = {
            "plain": "Llamadas directas al LLM sin contexto recuperado. Evalúa el conocimiento base del modelo.",
            "rag": "Recupera documentos de ChromaDB antes de responder. Evalúa faithfulness y context precision.",
        }
        st.markdown(
            f"""
            <div style="font-size:0.75rem; color:{PALETTE["text_muted"]}; background:{PALETTE["bg_sunken"]}; border-radius:6px;
                 padding:0.5rem 0.65rem; border:1px solid {PALETTE["border"]}; margin-top:0.25rem;">
                {mode_desc[mode]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Evaluators ────────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="font-size:0.68rem; color:{PALETTE["accent"]}; text-transform:uppercase;
                 letter-spacing:0.1em; font-weight:700; margin-bottom:0.4rem;">
                🧪 Evaluadores
            </div>
            """,
            unsafe_allow_html=True,
        )

        evaluator_info = {
            "rule_based": ("📏", "Rule-Based", "Checks deterministas: vacío, longitud, keywords, latencia."),
            "safety": ("🛡️", "Safety", "Detecta prompt injection, filtración del system prompt, contenido dañino."),
            "ragas": ("📐", "RAGAS", "Relevancy, Faithfulness, Context Precision. ⚠️ Requiere OPENAI_API_KEY."),
            "deepeval": ("🔍", "DeepEval", "Hallucination, Bias, Toxicity, GEval. ⚠️ Requiere OPENAI_API_KEY."),
            "consistency": ("🔄", "Consistency", "Estabilidad de respuestas: mide variabilidad mediante similitud coseno."),
            "llm_judge": ("⚖️", "LLM Judge", "GPT-4 evalúa según rúbricas personalizadas. ⚠️ Requiere OPENAI_API_KEY."),
        }

        evals: dict[str, bool] = {}
        for eval_name, (icon, label, description) in evaluator_info.items():
            default = eval_name in ("rule_based", "safety")
            evals[eval_name] = st.checkbox(
                f"{icon} {label}",
                value=default,
                key=f"eval_{eval_name}",
                help=description,
            )

        active_evals = [k for k, v in evals.items() if v]

        # API key warnings
        needs_openai = any(e in active_evals for e in ("ragas", "deepeval", "llm_judge"))
        if needs_openai and not os.getenv("OPENAI_API_KEY"):
            st.markdown(
                f"""
                <div style="background:rgba(239,68,68,0.08); border-left:3px solid {PALETTE["danger"]};
                     border-radius:0 6px 6px 0; padding:0.5rem 0.65rem; font-size:0.76rem;
                     color:{PALETTE["danger_pale"]}; margin-top:0.25rem;">
                    ⚠️ RAGAS / DeepEval / LLM Judge requieren <code>OPENAI_API_KEY</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not active_evals:
            st.markdown(
                f"""
                <div style="background:rgba(245,158,11,0.08); border-left:3px solid {PALETTE["warning"]};
                     border-radius:0 6px 6px 0; padding:0.5rem 0.65rem; font-size:0.76rem;
                     color:{PALETTE["warning_pale"]}; margin-top:0.25rem;">
                    ⚠️ Activa al menos un evaluador
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Runner Settings ────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="font-size:0.68rem; color:{PALETTE["accent"]}; text-transform:uppercase;
                 letter-spacing:0.1em; font-weight:700; margin-bottom:0.4rem;">
                ⚙️ Configuración del Runner
            </div>
            """,
            unsafe_allow_html=True,
        )

        runner_cfg = config.get("runner", {})

        max_concurrent = st.slider(
            "Concurrencia",
            min_value=1,
            max_value=20,
            value=runner_cfg.get("max_concurrent", 5),
            key="runner_concurrent",
            help="Tests ejecutándose en paralelo. Más = más rápido pero más carga en la API.",
        )

        timeout = st.number_input(
            "Timeout (ms)",
            min_value=5000,
            max_value=120000,
            value=runner_cfg.get("timeout_ms", 30000),
            step=5000,
            key="runner_timeout",
            help="Tiempo máximo de espera por respuesta. Aumenta si el proveedor es lento.",
        )

        samples = st.slider(
            "Muestras",
            min_value=1,
            max_value=10,
            value=1,
            key="runner_samples",
            help=(
                "Cuántas veces se ejecuta la suite completa. Más de una revela qué casos "
                "son inestables (pasan unas veces y otras no). Multiplica las llamadas a la API."
            ),
        )

        samples_note = f" · {samples} muestras" if samples > 1 else ""
        st.markdown(
            f"""
            <div style="font-size:0.75rem; color:{PALETTE["text_muted"]}; margin-top:0.25rem;">
                {max_concurrent} tests paralelos · {timeout // 1000}s timeout{samples_note}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Quick Tips ─────────────────────────────────────────────────────────
        with st.expander("💡 Tips rápidos"):
            st.markdown(
                """
                **Empezar sin API key:**
                Usa **Mock** + **Rule-Based** + **Safety**

                **Evaluar RAG:**
                Activa **RAGAS** y usa modo **RAG**

                **Benchmarking:**
                Haz varios runs cambiando el provider y usa **Compare** para ver cuál gana

                **Severidades:**
                - 🔴 Critical = bloquea producción
                - 🟠 High = importante corregir
                - 🟡 Medium = mejorar cuando sea posible
                - 🟢 Low = mejora menor
                """
            )

        # ── Footer ─────────────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="text-align:center; color:{PALETTE["text_ghost"]}; font-size:0.7rem; padding:0.5rem 0 0.25rem;">
                v0.3.0 · <a href="https://github.com/gonzaloMorenoc/llm-eval-lab" style="color:{PALETTE["accent"]}; text-decoration:none;">GitHub ↗</a>
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
    st.session_state["samples"] = samples

    return {
        "provider": provider,
        "mode": mode,
        "evaluators": active_evals,
        "max_concurrent": max_concurrent,
        "timeout_ms": timeout,
        "samples": samples,
        "config": config,
    }
