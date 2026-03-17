"""LLM Eval Lab — Streamlit Dashboard entry point.

Launch: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.styles import inject_css, callout, how_step, stat_card, badge
from src.dashboard.components.charts import pass_rate_bar_chart, COLORS
from src.dashboard.components.shared import (
    list_runs,
    pass_rate_color,
    CATEGORY_ICONS,
    CATEGORY_LABEL_COLORS,
    CATEGORY_DESCRIPTIONS,
    SEVERITY_ICONS,
    SEVERITY_ORDER,
)


def main() -> None:
    st.set_page_config(
        page_title="LLM Eval Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_sidebar()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:2rem 0 1.25rem;">
            <div style="font-size:3.5rem; margin-bottom:0.5rem; filter:drop-shadow(0 0 20px rgba(99,102,241,0.4));">🔬</div>
            <h1 style="font-size:2.8rem; font-weight:900; margin:0;
                background:linear-gradient(135deg,#6366f1,#a78bfa,#38bdf8);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
                LLM Eval Lab
            </h1>
            <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.5rem; font-weight:400;">
                Framework de evaluación de calidad para chatbots de IA<br>
                <span style="font-size:0.9rem; color:#64748b;">Aprende QA de IA de forma práctica e interactiva · RAGAS · DeepEval · LLM Judge</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Live Stats (si hay runs) ───────────────────────────────────────────────
    runs = list_runs()
    if runs:
        latest = runs[0]
        pr = latest.get("pass_rate", 0)
        pr_color = pass_rate_color(pr)
        s_cols = st.columns(4)
        stats = [
            ("Runs completados", str(len(runs)), "#6366f1", ""),
            ("Último Pass Rate", f"{pr:.0%}", pr_color, "≥70% = bueno"),
            ("Avg Score", f"{latest.get('avg_score', 0):.3f}", "#38bdf8", "0.0 – 1.0"),
            ("Critical Failures", str(latest.get("critical_failures", 0)),
             "#ef4444" if latest.get("critical_failures", 0) > 0 else "#22c55e", "objetivo: 0"),
        ]
        for col, (label, val, color, extra) in zip(s_cols, stats):
            with col:
                st.markdown(
                    f'<div style="text-align:center;">{stat_card(label, val, color, extra=extra)}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("")

    # ── Navegación ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:1.2rem; font-weight:700; color:#e2e8f0; margin-bottom:0.75rem;">Navegar a</div>',
        unsafe_allow_html=True,
    )
    nav_cols = st.columns(4)
    nav_items = [
        ("nav-run", "🚀", "Run Evaluation",
         "Lanza evaluaciones contra cualquier proveedor LLM. Configura datasets, evaluadores y obtén resultados al instante.",
         "pages/1_run.py"),
        ("nav-results", "📊", "Results Dashboard",
         "Visualiza métricas detalladas, gráficos interactivos y analiza cada test con score y feedback explicado.",
         "pages/2_results.py"),
        ("nav-compare", "🔄", "Compare Runs",
         "Compara dos ejecuciones side-by-side. Descubre qué modelo gana, en qué categorías y por cuánto.",
         "pages/3_compare.py"),
        ("nav-tests", "📝", "Test Cases",
         "Explora y crea casos de prueba. Aprende qué tipos de tests evalúan los LLMs y cómo diseñarlos.",
         "pages/4_test_cases.py"),
    ]
    for col, (cls, icon, title, desc, page) in zip(nav_cols, nav_items):
        with col:
            st.markdown(
                f"""
                <div class="nav-card {cls}">
                    <span class="nav-icon">{icon}</span>
                    <div class="nav-title">{title}</div>
                    <div class="nav-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.page_link(page, label=f"Ir a {title} →", use_container_width=True)

    st.divider()

    # ── Cómo Funciona ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-bottom:0.5rem;">
            <span style="font-size:1.2rem; font-weight:700; color:#e2e8f0;">¿Cómo funciona LLM Eval Lab?</span><br>
            <span style="font-size:0.88rem; color:#64748b;">Evalúa la calidad de cualquier chatbot de IA en 4 pasos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    how_col1, how_col2 = st.columns(2)
    steps = [
        (1, "📦 Selecciona un Dataset",
         "Elige entre 4 categorías: Funcionales (respuestas correctas), Seguridad (ataques), Regresión (estabilidad) y Multi-turn (conversaciones)."),
        (2, "⚙️ Configura Provider & Modo",
         "Selecciona qué LLM evaluar: Groq (Llama 3), Gemini, Mistral u OpenRouter. Usa Mock para pruebas sin API key. Elige Plain o RAG."),
        (3, "🧪 Elige los Evaluadores",
         "Cada evaluador mide un aspecto distinto: Rule-Based (reglas), Safety (ataques), RAGAS (relevancia/factualidad), DeepEval (alucinaciones/sesgo)."),
        (4, "📊 Analiza y Compara",
         "Obtén Pass Rate, scores por categoría, latencias y métricas detalladas. Compara runs para decidir qué modelo usar en producción."),
    ]
    with how_col1:
        for num, title, desc in steps[:2]:
            st.markdown(how_step(num, title, desc), unsafe_allow_html=True)
    with how_col2:
        for num, title, desc in steps[2:]:
            st.markdown(how_step(num, title, desc), unsafe_allow_html=True)

    st.divider()

    # ── Conceptos Clave de QA para IA ─────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-bottom:0.5rem;">
            <span style="font-size:1.2rem; font-weight:700; color:#e2e8f0;">📚 Conceptos clave de QA para IA</span><br>
            <span style="font-size:0.88rem; color:#64748b;">Aprende los fundamentos para evaluar LLMs como un profesional</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    concepts = [
        ("🎯", "Pass Rate",
         "% de tests que el modelo supera. Objetivo en producción: ≥70%. Por debajo del 50% indica problemas graves de calidad.",
         "badge-pass", "Objetivo ≥70%"),
        ("📐", "RAGAS Metrics",
         "Faithfulness (¿se basa en contexto real?), Answer Relevancy (¿responde lo pedido?), Context Precision. Especializado para RAG.",
         "badge-info", "Framework RAGAS"),
        ("🛡️", "Safety Evaluation",
         "Detecta vulnerabilidades: prompt injection, filtración del system prompt, contenido dañino. Crítico antes de ir a producción.",
         "badge-fail", "Tipo: Seguridad"),
        ("🔄", "Consistency",
         "¿El modelo responde igual ante la misma pregunta? Alta variabilidad indica inestabilidad o alucinaciones frecuentes.",
         "badge-warn", "Métrica: Cosine Sim"),
        ("⚖️", "LLM Judge",
         "Usa GPT-4 como evaluador según rúbricas personalizadas. Más flexible y próximo al juicio humano que métricas automáticas.",
         "badge-purple", "Evaluador: GPT-4"),
        ("🔍", "DeepEval",
         "Mide hallucination (datos inventados), bias (sesgos), toxicity (contenido dañino) y GEval (evaluación holística).",
         "badge-gray", "Framework DeepEval"),
    ]

    c1, c2, c3 = st.columns(3)
    for i, (icon, title, desc, badge_cls, badge_text) in enumerate(concepts):
        col = [c1, c2, c3][i % 3]
        with col:
            st.markdown(
                f"""
                <div class="concept-card" style="margin-bottom:0.75rem;">
                    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                        <span style="font-size:1.4rem;">{icon}</span>
                        <span class="concept-title">{title}</span>
                    </div>
                    <p class="concept-desc">{desc}</p>
                    <span class="badge {badge_cls}" style="margin-top:0.5rem;">{badge_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Dataset Overview ───────────────────────────────────────────────────────
    from src.runner.runner import load_all_datasets
    cases = load_all_datasets()

    categories: dict[str, int] = {}
    severities: dict[str, int] = {}
    for c in cases:
        categories[c.category] = categories.get(c.category, 0) + 1
        severities[c.severity] = severities.get(c.severity, 0) + 1

    st.markdown(
        '<span style="font-size:1.2rem; font-weight:700; color:#e2e8f0;">📦 Dataset disponible</span>',
        unsafe_allow_html=True,
    )

    cat_icons = CATEGORY_ICONS
    cat_colors = CATEGORY_LABEL_COLORS
    cat_descs = CATEGORY_DESCRIPTIONS

    ds_cols = st.columns(len(categories) + 1)
    for i, (cat, count) in enumerate(sorted(categories.items())):
        with ds_cols[i]:
            color = cat_colors.get(cat, "#888")
            icon  = cat_icons.get(cat, "📋")
            desc  = cat_descs.get(cat, "")
            st.markdown(
                stat_card(cat.replace("_", " ").title(), f"{icon} {count}", color, extra=desc),
                unsafe_allow_html=True,
            )

    with ds_cols[-1]:
        sev_lines  = " · ".join(
            f"{SEVERITY_ICONS[s]} {severities.get(s, 0)} {s}" for s in SEVERITY_ORDER if s in severities
        )
        st.markdown(
            stat_card("Total Test Cases", f"📊 {len(cases)}", "#a78bfa", extra=sev_lines),
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Ejecuciones Recientes ─────────────────────────────────────────────────
    st.markdown(
        '<span style="font-size:1.2rem; font-weight:700; color:#e2e8f0;">📈 Ejecuciones recientes</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    if runs:
        latest = runs[0]
        pr = latest.get("pass_rate", 0)
        pr_color = pass_rate_color(pr)

        info_col, chart_col = st.columns([2, 3])
        with info_col:
            crit = latest.get("critical_failures", 0)
            crit_color = "#ef4444" if crit > 0 else "#22c55e"
            st.markdown(
                f"""
                <div class="card">
                    <div style="font-size:0.7rem; color:#6366f1; text-transform:uppercase; letter-spacing:0.1em; font-weight:700; margin-bottom:0.5rem;">Último Run</div>
                    <div style="font-size:1rem; font-weight:700; color:#e2e8f0;">{latest.get("run_id","?")}</div>
                    <div style="font-size:0.8rem; color:#64748b; margin-bottom:1rem;">{latest.get("timestamp","")[:19]} · {latest.get("chatbot_id","?")} ({latest.get("chatbot_mode","?")})</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; margin-bottom:1rem;">
                        <div>
                            <div style="font-size:1.9rem; font-weight:800; color:{pr_color};">{pr:.0%}</div>
                            <div style="font-size:0.68rem; color:#94a3b8; text-transform:uppercase;">Pass Rate</div>
                        </div>
                        <div>
                            <div style="font-size:1.9rem; font-weight:800; color:#38bdf8;">{latest.get("avg_score",0):.3f}</div>
                            <div style="font-size:0.68rem; color:#94a3b8; text-transform:uppercase;">Avg Score</div>
                        </div>
                        <div>
                            <div style="font-size:1.4rem; font-weight:700; color:#e2e8f0;">{latest.get("avg_latency_ms",0):.0f}ms</div>
                            <div style="font-size:0.68rem; color:#94a3b8; text-transform:uppercase;">Avg Latency</div>
                        </div>
                        <div>
                            <div style="font-size:1.4rem; font-weight:700; color:{crit_color};">{crit}</div>
                            <div style="font-size:0.68rem; color:#94a3b8; text-transform:uppercase;">Critical Fails</div>
                        </div>
                    </div>
                    <div style="display:flex; gap:0.4rem; flex-wrap:wrap;">
                        <span class="badge badge-pass">✅ {latest.get("passed",0)} passed</span>
                        <span class="badge badge-fail">❌ {latest.get("failed",0)} failed</span>
                        <span class="badge badge-info">📊 {latest.get("total",0)} total</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with chart_col:
            by_cat = latest.get("by_category", {})
            if by_cat:
                fig = pass_rate_bar_chart(by_cat)
                st.plotly_chart(fig, use_container_width=True, key="home_bar")

        if len(runs) > 1:
            st.markdown("**Historial de runs**")
            table = []
            for r in runs[:10]:
                table.append({
                    "Run ID":    r.get("run_id", "?"),
                    "Provider":  r.get("chatbot_id", "?"),
                    "Mode":      r.get("chatbot_mode", "?"),
                    "Pass Rate": f"{r.get('pass_rate', 0):.0%}",
                    "Avg Score": f"{r.get('avg_score', 0):.3f}",
                    "Latency":   f"{r.get('avg_latency_ms', 0):.0f}ms",
                    "Tests":     r.get("total", 0),
                    "Date":      r.get("timestamp", "")[:19],
                })
            st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div class="empty-state">
                <span class="empty-icon">🚀</span>
                <div class="empty-title">¡Lanza tu primera evaluación!</div>
                <div class="empty-desc">
                    Todavía no hay runs. Ve a <strong>Run Evaluation</strong> para evaluar tu primer modelo.<br><br>
                    Usa el modo <strong>Mock</strong> para empezar sin ninguna API key.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        _, cta, _ = st.columns([2, 1, 2])
        with cta:
            st.page_link("pages/1_run.py", label="🚀 Empezar Evaluación", use_container_width=True)

    # ── Quick Start Guide ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚡ Guía rápida: evalúa tu primer modelo en 2 minutos"):
        st.markdown(
            """
            ### 1 · Inicia la app
            ```bash
            streamlit run src/dashboard/app.py
            ```

            ### 2 · Sin API key: usa Mock Mode
            En la barra lateral → **Provider → mock**. No necesitas ninguna clave.

            ### 3 · Lanza la evaluación
            Ve a **🚀 Run Evaluation**, deja todos los datasets seleccionados y pulsa **Start Evaluation**.

            ### 4 · Analiza los resultados
            Verás Pass Rate, scores por categoría y fallos detallados. ¡Ya estás haciendo QA de IA!

            ---

            ### Para evaluar modelos reales, configura tu `.env`:
            ```bash
            GROQ_API_KEY=gsk_...          # Llama 3 gratis
            GEMINI_API_KEY=AIza...        # Gemini gratis
            OPENAI_API_KEY=sk-...         # Necesario para RAGAS y DeepEval
            ```
            """
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color:#374151; font-size:0.76rem; padding:0.5rem;">
            LLM Eval Lab v0.3.0 · Built with Streamlit · RAGAS + DeepEval + Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
