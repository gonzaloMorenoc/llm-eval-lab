"""Page 1: Run Evaluation — wizard-style UX with step indicator and live feedback."""

from __future__ import annotations

import asyncio
import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.styles import inject_css, callout, badge, wizard_bar, stat_card, page_header
from src.dashboard.components.metrics import severity_icon, kpi_row, score_bar
from src.dashboard.components.charts import COLORS
from src.dashboard.components.shared import (
    CATEGORY_ICONS,
    CATEGORY_LABEL_COLORS,
    SEVERITY_ICONS,
    SEVERITY_ORDER,
    pass_rate_color,
)

st.set_page_config(page_title="Run Evaluation — LLM Eval Lab", page_icon="🚀", layout="wide")
inject_css()
render_sidebar()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    page_header("🚀", "Run Evaluation", "Evalúa un modelo LLM contra tu dataset de tests en tres pasos"),
    unsafe_allow_html=True,
)

# ── Wizard progress bar ───────────────────────────────────────────────────────
st.markdown(
    wizard_bar(["Seleccionar Datasets", "Revisar Configuración", "Ejecutar & Resultados"], 0),
    unsafe_allow_html=True,
)

# ── Imports ───────────────────────────────────────────────────────────────────
from src.runner.runner import load_all_datasets, load_dataset

datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"))
available_datasets: dict = {}
for fname in sorted(os.listdir(datasets_dir)):
    if fname.endswith(".jsonl") and fname != "rag_knowledge_base.jsonl":
        name = fname.replace(".jsonl", "")
        cases = load_dataset(os.path.join(datasets_dir, fname))
        available_datasets[name] = cases

# ── Step 1 · Dataset Selection ────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
        <div style="width:28px; height:28px; border-radius:50%; background:linear-gradient(135deg,#6366f1,#8b5cf6);
            display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.85rem; color:white;
            box-shadow:0 0 12px rgba(99,102,241,0.4); flex-shrink:0;">1</div>
        <span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">Selecciona los Datasets</span>
    </div>
    <div style="font-size:0.82rem; color:#64748b; margin-bottom:1rem; margin-left:2.25rem;">
        Cada categoría evalúa un aspecto diferente de la calidad del LLM
    </div>
    """,
    unsafe_allow_html=True,
)

cat_icons = CATEGORY_ICONS
cat_colors = CATEGORY_LABEL_COLORS
cat_descs = {
    "functional":  "Evalúa si el modelo responde correctamente a preguntas generales, de dominio y de razonamiento.",
    "safety":      "Detecta vulnerabilidades: prompt injection, filtración del system prompt y contenido dañino.",
    "regression":  "Verifica que el modelo mantiene la calidad entre versiones. Detecta regresiones y cambios de comportamiento.",
    "multi_turn":  "Mide la coherencia y memoria en conversaciones con múltiples turnos usuario-asistente.",
}

ds_cols = st.columns(len(available_datasets))
selected_datasets: list[str] = []

for i, (name, cases) in enumerate(available_datasets.items()):
    with ds_cols[i]:
        icon   = cat_icons.get(name, "📋")
        color  = cat_colors.get(name, "#888")
        desc   = cat_descs.get(name, "")

        # Severity breakdown
        sevs: dict[str, int] = {}
        for c in cases:
            sevs[c.severity] = sevs.get(c.severity, 0) + 1
        sev_str = " · ".join(f"{SEVERITY_ICONS.get(s, '')} {n}" for s, n in sorted(sevs.items()))

        checked = st.checkbox(
            f"{icon} **{name.replace('_', ' ').title()}**",
            value=True,
            key=f"ds_{name}",
        )
        if checked:
            selected_datasets.append(name)

        border_color = color if checked else "#2d2d44"
        st.markdown(
            f"""
            <div class="dataset-card {'dataset-card-selected' if checked else ''}"
                 style="border-color:{border_color}; margin-top:-0.25rem;">
                <div style="font-size:1.6rem; text-align:center; margin-bottom:0.4rem;">{icon}</div>
                <div style="font-size:0.78rem; font-weight:700; color:{color}; text-align:center; margin-bottom:0.3rem;">
                    {len(cases)} tests
                </div>
                <div style="font-size:0.73rem; color:#94a3b8; line-height:1.4; margin-bottom:0.5rem;">{desc}</div>
                <div style="font-size:0.7rem; color:#64748b;">{sev_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

all_selected_cases = []
for name in selected_datasets:
    all_selected_cases.extend(available_datasets[name])

st.markdown("")
if all_selected_cases:
    st.markdown(
        callout(
            f"<strong>{len(all_selected_cases)} test cases</strong> seleccionados de "
            f"<strong>{len(selected_datasets)}</strong> datasets · Tiempo estimado con Mock: ~{len(all_selected_cases) * 2}s",
            kind="info",
            icon="📦",
        ),
        unsafe_allow_html=True,
    )
else:
    st.markdown(callout("Selecciona al menos un dataset para continuar.", kind="warning"), unsafe_allow_html=True)

st.divider()

# ── Step 2 · Configuration ────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
        <div style="width:28px; height:28px; border-radius:50%; background:linear-gradient(135deg,#38bdf8,#0ea5e9);
            display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.85rem; color:white;
            box-shadow:0 0 12px rgba(56,189,248,0.4); flex-shrink:0;">2</div>
        <span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">Configuración Actual</span>
    </div>
    <div style="font-size:0.82rem; color:#64748b; margin-bottom:1rem; margin-left:2.25rem;">
        Ajusta el provider, modo y evaluadores en la barra lateral izquierda
    </div>
    """,
    unsafe_allow_html=True,
)

provider      = st.session_state.get("selected_provider", "mock")
mode          = st.session_state.get("selected_mode", "plain")
active_evals  = st.session_state.get("active_evaluators", ["rule_based", "safety"])
max_concurrent = st.session_state.get("max_concurrent", 5)

eval_descriptions = {
    "rule_based":  ("📏", "Rule-Based",  "Verificaciones deterministas: longitud, keywords, latencia."),
    "safety":      ("🛡️", "Safety",      "Detecta prompt injection, filtración de system prompt y contenido dañino."),
    "ragas":       ("📐", "RAGAS",       "Relevancy, Faithfulness, Context Precision. Requiere OpenAI API key."),
    "deepeval":    ("🔍", "DeepEval",    "Hallucination, Bias, Toxicity, GEval. Requiere OpenAI API key."),
    "consistency": ("🔄", "Consistency", "Estabilidad entre ejecuciones. Mide variabilidad de respuestas."),
    "llm_judge":   ("⚖️", "LLM Judge",  "GPT-4 evalúa según rúbricas personalizadas."),
}

cfg_cols = st.columns([1, 1, 2, 1])
with cfg_cols[0]:
    provider_color = "#22c55e" if provider == "mock" else "#6366f1"
    st.markdown(
        f"""
        <div class="stat-card" style="border-left:3px solid {provider_color};">
            <div class="stat-label">🔌 Provider</div>
            <div style="font-size:1.2rem; font-weight:700; color:{provider_color}; margin-top:0.3rem;">{provider}</div>
            <div style="font-size:0.72rem; color:#64748b; margin-top:0.2rem;">
                {"Sin API key · Ideal para pruebas" if provider == "mock" else "LLM real · Necesita API key"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cfg_cols[1]:
    mode_color = "#38bdf8" if mode == "plain" else "#f59e0b"
    st.markdown(
        f"""
        <div class="stat-card" style="border-left:3px solid {mode_color};">
            <div class="stat-label">🔀 Modo</div>
            <div style="font-size:1.2rem; font-weight:700; color:{mode_color}; margin-top:0.3rem;">{mode.title()}</div>
            <div style="font-size:0.72rem; color:#64748b; margin-top:0.2rem;">
                {"LLM directo sin contexto externo" if mode == "plain" else "Retrieval Augmented Generation con ChromaDB"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cfg_cols[2]:
    evals_html = " ".join(
        f'<span class="badge badge-purple">{eval_descriptions.get(e, ("", e, ""))[0]} {eval_descriptions.get(e, ("", e, ""))[1]}</span>'
        for e in active_evals
    )
    st.markdown(
        f"""
        <div class="stat-card" style="border-left:3px solid #a78bfa;">
            <div class="stat-label">🧪 Evaluadores activos ({len(active_evals)})</div>
            <div style="margin-top:0.5rem; display:flex; flex-wrap:wrap; gap:0.3rem;">{evals_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cfg_cols[3]:
    st.markdown(
        f"""
        <div class="stat-card" style="border-left:3px solid #64748b;">
            <div class="stat-label">⚙️ Concurrencia</div>
            <div style="font-size:1.5rem; font-weight:700; color:#e2e8f0; margin-top:0.3rem;">{max_concurrent}</div>
            <div style="font-size:0.72rem; color:#64748b; margin-top:0.2rem;">tests paralelos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Evaluator descriptions
with st.expander("💡 ¿Qué hace cada evaluador?"):
    ev_cols = st.columns(3)
    for i, (ev_key, (icon, name, desc)) in enumerate(eval_descriptions.items()):
        is_active = ev_key in active_evals
        with ev_cols[i % 3]:
            border = "border-color:#6366f1;" if is_active else ""
            status_badge = badge("Activo", "purple") if is_active else badge("Inactivo", "gray")
            st.markdown(
                f"""
                <div class="concept-card" style="{border} margin-bottom:0.5rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                        <span class="concept-title">{icon} {name}</span>
                        {status_badge}
                    </div>
                    <p class="concept-desc">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()


# ── Step 3 · Execute ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
        <div style="width:28px; height:28px; border-radius:50%; background:linear-gradient(135deg,#22c55e,#16a34a);
            display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.85rem; color:white;
            box-shadow:0 0 12px rgba(34,197,94,0.4); flex-shrink:0;">3</div>
        <span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">Ejecutar Evaluación</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Validation hints
if not all_selected_cases:
    st.markdown(callout("Selecciona al menos un dataset en el Paso 1.", kind="warning"), unsafe_allow_html=True)
elif not active_evals:
    st.markdown(callout("Activa al menos un evaluador en la barra lateral.", kind="warning"), unsafe_allow_html=True)
else:
    st.markdown(
        callout(
            f"Todo listo · <strong>{len(all_selected_cases)} tests</strong> con "
            f"<strong>{len(active_evals)} evaluadores</strong> en modo <strong>{provider}/{mode}</strong>",
            kind="success",
            icon="✅",
        ),
        unsafe_allow_html=True,
    )

run_col, _ = st.columns([3, 1])
with run_col:
    run_clicked = st.button(
        "🚀 Iniciar Evaluación",
        type="primary",
        use_container_width=True,
        disabled=not all_selected_cases or not active_evals,
    )


# ── Builder functions ─────────────────────────────────────────────────────────
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
            st.warning(f"No se pudo inicializar RAGAS: {e}")
    if "deepeval" in eval_names:
        try:
            from src.evaluators.deepeval_evaluator import DeepEvalEvaluator
            evaluators["deepeval"] = DeepEvalEvaluator()
        except Exception as e:
            st.warning(f"No se pudo inicializar DeepEval: {e}")
    if "consistency" in eval_names:
        from src.evaluators.consistency import ConsistencyEvaluator
        evaluators["consistency"] = ConsistencyEvaluator()
    if "llm_judge" in eval_names:
        try:
            from src.evaluators.llm_judge import LLMJudgeEvaluator
            evaluators["llm_judge"] = LLMJudgeEvaluator()
        except Exception as e:
            st.warning(f"No se pudo inicializar LLM Judge: {e}")
    return evaluators


# ── Execution ─────────────────────────────────────────────────────────────────
if run_clicked:
    start_time = time.time()

    with st.status("🔬 Ejecutando evaluación...", expanded=True) as status:
        st.write(f"🔌 Construyendo chatbot: **{provider}** (modo {mode})")
        chatbot = _build_chatbot(provider, mode)
        st.write(f"✅ Chatbot listo: `{chatbot.get_id()}`")

        st.write(f"🧪 Inicializando evaluadores: **{', '.join(active_evals)}**")
        evaluators = _build_evaluators(active_evals)
        st.write(f"✅ {len(evaluators)} evaluadores listos")

        st.write(f"⏳ Ejecutando **{len(all_selected_cases)}** test cases (concurrencia={max_concurrent})...")

        from src.runner.runner import EvalRunner
        config = st.session_state.get("config", {})
        config.setdefault("runner", {})["max_concurrent"] = max_concurrent
        runner = EvalRunner(chatbot=chatbot, evaluators=evaluators, config=config)

        progress = st.progress(0, text="Iniciando evaluación...")
        loop = asyncio.new_event_loop()
        summary = loop.run_until_complete(runner.run(all_selected_cases))
        loop.close()

        elapsed = time.time() - start_time
        progress.progress(100, text=f"¡Completado en {elapsed:.1f}s!")

        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", summary.run_id)
        )
        from src.reporting.json_reporter import generate_json_report
        from src.reporting.markdown_reporter import generate_markdown_report
        generate_json_report(summary, results_dir)
        generate_markdown_report(summary, results_dir)

        status.update(label=f"✅ Evaluación completada en {elapsed:.1f}s", state="complete")

    st.session_state["last_summary"] = summary.model_dump()
    st.session_state["last_run_id"]  = summary.run_id

    # ── Results ───────────────────────────────────────────────────────────────
    pr = summary.pass_rate
    pr_color = pass_rate_color(pr)
    pr_msg = "¡Excelente!" if pr >= 0.7 else ("Mejorable" if pr >= 0.5 else "Necesita atención")

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1a1a2e,#22223d); border:1px solid #2d2d44;
             border-left:4px solid {pr_color}; border-radius:12px; padding:1.25rem; margin:1rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:0.7rem; color:#6366f1; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Run completado</div>
                    <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-top:0.2rem;">{summary.run_id}</div>
                    <div style="font-size:0.8rem; color:#64748b;">{summary.total} tests · {elapsed:.1f}s · {provider}/{mode}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:2.5rem; font-weight:900; color:{pr_color};">{pr:.0%}</div>
                    <div style="font-size:0.8rem; color:#94a3b8;">Pass Rate · {pr_msg}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    pass_color = COLORS["success"] if pr >= 0.7 else (COLORS["warning"] if pr >= 0.5 else COLORS["danger"])
    kpi_row([
        ("Pass Rate",        f"{pr:.1%}",                    pass_color),
        ("Avg Score",        f"{summary.avg_score:.3f}",     COLORS["primary"]),
        ("Avg Latency",      f"{summary.avg_latency_ms:.0f}ms", COLORS["info"]),
        ("Critical Failures", str(summary.critical_failures), COLORS["danger"] if summary.critical_failures > 0 else COLORS["success"]),
        ("Passed / Failed",  f"{summary.passed} / {summary.failed}", None),
    ])

    # Insight callout
    if summary.critical_failures > 0:
        st.markdown(
            callout(
                f"⚠️ Hay <strong>{summary.critical_failures} fallos críticos</strong>. Revisa la pestaña Failures — estos tests deben resolverse antes de producción.",
                kind="error",
            ),
            unsafe_allow_html=True,
        )
    elif pr < 0.5:
        st.markdown(
            callout("El Pass Rate está por debajo del 50%. Revisa la configuración del modelo o amplía los datos de entrenamiento.", kind="warning"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            callout(f"Pass Rate de {pr:.0%}. Revisa los fallos para identificar áreas de mejora.", kind="tip"),
            unsafe_allow_html=True,
        )

    st.divider()

    # Results tabs
    tab_table, tab_failures, tab_details = st.tabs(["📋 Todos los Resultados", "❌ Fallos", "🔍 Desglose por Evaluador"])

    with tab_table:
        results_data = []
        for r in summary.results:
            results_data.append({
                "":          "✅" if r.overall_passed else "❌",
                "ID":        r.test_case.id,
                "Categoría": r.test_case.category,
                "Severidad": f"{severity_icon(r.test_case.severity)} {r.test_case.severity}",
                "Score":     f"{r.overall_score:.3f}" if r.overall_score is not None else "—",
                "Latencia":  f"{r.latency_ms:.0f}ms",
                "Evaluadores": ", ".join(e.evaluator for e in r.evaluations),
            })
        st.dataframe(results_data, use_container_width=True, hide_index=True)

    with tab_failures:
        failures = [r for r in summary.results if not r.overall_passed]
        if not failures:
            st.markdown(
                callout("🎉 ¡Todos los tests pasaron! El modelo superó todos los casos de prueba.", kind="success"),
                unsafe_allow_html=True,
            )
        else:
            sorted_failures = sorted(failures, key=lambda x: SEVERITY_ORDER.index(x.test_case.severity))

            st.markdown(f"**{len(failures)} tests fallidos** (ordenados por severidad):")
            for r in sorted_failures:
                sev = severity_icon(r.test_case.severity)
                sev_badge = {"critical": "badge-fail", "high": "badge-warn", "medium": "badge-warn", "low": "badge-gray"}.get(r.test_case.severity, "badge-gray")
                with st.expander(f"{sev} **{r.test_case.id}** — {r.test_case.category}"):
                    f_col1, f_col2 = st.columns([3, 2])
                    with f_col1:
                        input_display = r.test_case.input if isinstance(r.test_case.input, str) else str(r.test_case.input)
                        st.markdown(f"**📥 Input:** {input_display[:400]}")
                        st.markdown(f"**🎯 Expected:** {r.test_case.expected_behavior}")
                        st.markdown("**🤖 Response:**")
                        st.text(r.response[:500])
                        if r.error:
                            st.error(f"Error: {r.error}")
                    with f_col2:
                        st.markdown(
                            f'<span class="badge {sev_badge}">{r.test_case.severity.upper()}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("")
                        st.markdown("**Evaluaciones:**")
                        for ev in r.evaluations:
                            icon = "✅" if ev.passed else "❌"
                            score_str = f"{ev.score:.3f}" if ev.score is not None else "—"
                            st.markdown(f"{icon} **{ev.evaluator}** ({score_str})")
                            if ev.reason:
                                st.caption(ev.reason[:200])

    with tab_details:
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

        ev_table = []
        for name, stats in sorted(evaluator_stats.items()):
            total = stats["passed"] + stats["failed"]
            avg   = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else None
            ev_table.append({
                "Evaluador":  name.replace("_", " ").title(),
                "Tests":      total,
                "Pasados":    stats["passed"],
                "Fallados":   stats["failed"],
                "Pass Rate":  f"{stats['passed']/total:.1%}" if total > 0 else "—",
                "Avg Score":  f"{avg:.3f}" if avg is not None else "—",
            })
        st.dataframe(ev_table, use_container_width=True, hide_index=True)

        st.markdown(
            callout(
                "Tip: Si Rule-Based falla con alta frecuencia, revisa los criterios de evaluación. "
                "Si Safety falla, el modelo puede ser vulnerable a prompt injection.",
                kind="tip",
            ),
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(f"📁 Reportes guardados en `results/{summary.run_id}/`")
    st.page_link("pages/2_results.py", label="📊 Ver Dashboard Completo de Resultados →", use_container_width=False)
