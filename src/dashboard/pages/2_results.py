"""Page 2: Results Dashboard — rich interactive visualization with metric explanations."""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.styles import inject_css, callout, badge, stat_card, page_header
from src.dashboard.components.metrics import severity_icon, kpi_row, severity_badge, score_bar
from src.dashboard.components.charts import (
    COLORS,
    pass_rate_bar_chart,
    metrics_radar_chart,
    latency_histogram,
    severity_pie_chart,
    evaluator_scores_chart,
    score_distribution_chart,
)
from src.dashboard.components.shared import list_runs

st.set_page_config(page_title="Results — LLM Eval Lab", page_icon="📊", layout="wide")
inject_css()
render_sidebar()


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    page_header("📊", "Results Dashboard", "Análisis detallado de una ejecución de evaluación — métricas, gráficos y resultados por test"),
    unsafe_allow_html=True,
)

runs = list_runs()

if not runs:
    st.markdown(
        """
        <div class="empty-state">
            <span class="empty-icon">📊</span>
            <div class="empty-title">No hay resultados todavía</div>
            <div class="empty-desc">
                Primero lanza una evaluación en <strong>Run Evaluation</strong>.<br>
                Los resultados aparecerán aquí automáticamente.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.page_link("pages/1_run.py", label="🚀 Ir a Run Evaluation", use_container_width=False)
    st.stop()

# ── Run Selector ──────────────────────────────────────────────────────────────
run_labels = [
    f"{r.get('run_id','?')} · {r.get('chatbot_id','?')} ({r.get('chatbot_mode','?')}) · {r.get('timestamp','')[:19]}"
    for r in runs
]
selected_idx = st.selectbox("📁 Seleccionar Run", range(len(runs)), format_func=lambda i: run_labels[i])
summary = runs[selected_idx]

st.divider()

# ── KPI Cards ────────────────────────────────────────────────────────────────
pass_rate = summary.get("pass_rate", 0)
pass_color = COLORS["success"] if pass_rate >= 0.7 else (COLORS["warning"] if pass_rate >= 0.5 else COLORS["danger"])

kpi_row([
    ("Pass Rate",         f"{pass_rate:.1%}",                              pass_color),
    ("Avg Score",         f"{summary.get('avg_score', 0):.3f}",            COLORS["primary"]),
    ("Avg Latency",       f"{summary.get('avg_latency_ms', 0):.0f}ms",     COLORS["info"]),
    ("Critical Failures", str(summary.get("critical_failures", 0)),
     COLORS["danger"] if summary.get("critical_failures", 0) > 0 else COLORS["success"]),
    ("Total",             f"{summary.get('passed', 0)}✅ {summary.get('failed', 0)}❌ / {summary.get('total', 0)}", None),
])

# Insight automático
crit = summary.get("critical_failures", 0)
if crit > 0:
    st.markdown(
        callout(f"<strong>{crit} fallos críticos</strong> detectados. Estos casos tienen prioridad máxima de corrección antes de producción.", kind="error"),
        unsafe_allow_html=True,
    )
elif pass_rate >= 0.9:
    st.markdown(
        callout(f"Pass Rate de <strong>{pass_rate:.0%}</strong> — Excelente calidad. El modelo supera el 90% de los tests.", kind="success"),
        unsafe_allow_html=True,
    )
elif pass_rate < 0.5:
    st.markdown(
        callout(f"Pass Rate de <strong>{pass_rate:.0%}</strong> — Por debajo del mínimo recomendado (50%). Revisa la configuración del modelo.", kind="warning"),
        unsafe_allow_html=True,
    )

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_metrics, tab_latency, tab_evaluators = st.tabs(
    ["📈 Visión General", "📐 Métricas Avanzadas", "⏱️ Latencia", "🧪 Evaluadores"]
)

with tab_overview:
    st.markdown(
        """
        <div class="metric-explain" style="margin-bottom:1rem;">
            📊 <strong>Visión General</strong> — Distribución de resultados por categoría y severidad de fallos.
            Un buen modelo tiene barras >70% en todas las categorías.
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        by_category = summary.get("by_category", {})
        if by_category:
            fig = pass_rate_bar_chart(by_category)
            st.plotly_chart(fig, use_container_width=True, key="res_bar")
            st.markdown(
                """
                <div class="metric-explain">
                    💡 <strong>Pass Rate por Categoría</strong> — Cada barra muestra el % de tests superados.
                    Las categorías más bajas son las áreas con más problemas.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Sin datos de categorías.")

    with chart_col2:
        results = summary.get("results", [])
        failures = [r for r in results if not r.get("overall_passed")]
        if failures:
            severity_counts: dict[str, int] = {}
            for r in failures:
                sev = r.get("test_case", {}).get("severity", "medium")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            fig = severity_pie_chart(severity_counts)
            st.plotly_chart(fig, use_container_width=True, key="res_pie")
            st.markdown(
                """
                <div class="metric-explain">
                    💡 <strong>Severidad de Fallos</strong> — Los fallos críticos son prioritarios.
                    Un modelo en producción no debería tener fallos críticos ni de alta severidad.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                callout("🎉 ¡Sin fallos! Todos los tests pasaron correctamente.", kind="success"),
                unsafe_allow_html=True,
            )

    # Score distribution
    results = summary.get("results", [])
    if any(r.get("overall_score") is not None for r in results):
        st.markdown("#### Distribución de Scores por Categoría")
        fig = score_distribution_chart(results)
        st.plotly_chart(fig, use_container_width=True, key="res_box")
        st.markdown(
            """
            <div class="metric-explain">
                💡 <strong>Box Plot de Scores</strong> — Muestra la distribución estadística de puntuaciones por categoría.
                Las cajas más altas y compactas indican mayor calidad y consistencia.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_metrics:
    st.markdown(
        """
        <div class="metric-explain" style="margin-bottom:1rem;">
            📐 <strong>Métricas Avanzadas</strong> — RAGAS mide calidad de RAG (relevancia, faithfulness, precisión).
            DeepEval mide seguridad y calidad general (alucinaciones, sesgo, toxicidad).
            Estas métricas requieren <code>OPENAI_API_KEY</code> configurada.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ragas_col, deepeval_col = st.columns(2)

    with ragas_col:
        st.markdown(
            """
            <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.5rem;">📐 RAGAS Metrics</div>
            <div class="metric-explain" style="margin-bottom:0.75rem;">
                RAGAS evalúa la calidad del sistema de recuperación y generación.<br>
                • <strong>Faithfulness</strong>: ¿la respuesta se basa en contexto real?<br>
                • <strong>Answer Relevancy</strong>: ¿responde exactamente lo que se preguntó?<br>
                • <strong>Context Precision</strong>: ¿el contexto recuperado es relevante?
            </div>
            """,
            unsafe_allow_html=True,
        )
        ragas_agg = summary.get("ragas_aggregate", {})
        if ragas_agg:
            config = st.session_state.get("config", {})
            ragas_thresholds = config.get("ragas", {}).get("thresholds", {})
            fig = metrics_radar_chart(ragas_agg, ragas_thresholds, title="RAGAS Metrics")
            st.plotly_chart(fig, use_container_width=True, key="res_ragas_radar")

            for metric, avg in sorted(ragas_agg.items()):
                t = ragas_thresholds.get(metric, 0.65)
                passed = avg >= t
                status_badge = badge("✅ OK", "pass") if passed else badge("❌ Bajo", "fail")
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; justify-content:space-between;
                         padding:0.4rem 0; border-bottom:1px solid #2d2d44;">
                        <span style="font-size:0.85rem; color:#e2e8f0;">{metric.replace('_',' ').title()}</span>
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="font-size:0.8rem; color:#94a3b8;">umbral {t}</span>
                            <span style="font-size:0.85rem; font-weight:700; color:{'#4ade80' if passed else '#f87171'};">{avg:.3f}</span>
                            {status_badge}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                callout("RAGAS no estaba activado en este run. Actívalo en la barra lateral y vuelve a ejecutar.", kind="info"),
                unsafe_allow_html=True,
            )

    with deepeval_col:
        st.markdown(
            """
            <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.5rem;">🔍 DeepEval Metrics</div>
            <div class="metric-explain" style="margin-bottom:0.75rem;">
                DeepEval evalúa comportamiento adversarial y calidad general.<br>
                • <strong>Hallucination</strong>: ¿el modelo inventa datos?<br>
                • <strong>Bias</strong>: ¿hay sesgos en las respuestas?<br>
                • <strong>Toxicity</strong>: ¿hay contenido dañino?<br>
                • <strong>GEval</strong>: evaluación holística con LLM como juez.
            </div>
            """,
            unsafe_allow_html=True,
        )
        deepeval_agg = summary.get("deepeval_aggregate", {})
        if deepeval_agg:
            config = st.session_state.get("config", {})
            de_thresholds = config.get("deepeval", {}).get("thresholds", {})
            fig = metrics_radar_chart(deepeval_agg, de_thresholds, title="DeepEval Metrics")
            st.plotly_chart(fig, use_container_width=True, key="res_de_radar")

            for metric, avg in sorted(deepeval_agg.items()):
                t = de_thresholds.get(metric, 0.5)
                passed = avg >= t
                status_badge = badge("✅ OK", "pass") if passed else badge("❌ Bajo", "fail")
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; justify-content:space-between;
                         padding:0.4rem 0; border-bottom:1px solid #2d2d44;">
                        <span style="font-size:0.85rem; color:#e2e8f0;">{metric.replace('_',' ').title()}</span>
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="font-size:0.8rem; color:#94a3b8;">umbral {t}</span>
                            <span style="font-size:0.85rem; font-weight:700; color:{'#4ade80' if passed else '#f87171'};">{avg:.3f}</span>
                            {status_badge}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                callout("DeepEval no estaba activado en este run. Actívalo en la barra lateral y vuelve a ejecutar.", kind="info"),
                unsafe_allow_html=True,
            )

with tab_latency:
    st.markdown(
        """
        <div class="metric-explain" style="margin-bottom:1rem;">
            ⏱️ <strong>Latencia</strong> — Tiempo de respuesta del modelo.
            P50 = mediana, P95 = el 95% de las respuestas fue más rápido que este valor.
            En producción, busca P95 &lt; 3000ms para buena experiencia de usuario.
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = summary.get("results", [])
    latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms", 0) > 0]

    if latencies:
        fig = latency_histogram(latencies)
        st.plotly_chart(fig, use_container_width=True, key="res_lat_hist")

        sorted_lat = sorted(latencies)
        p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)

        lat_cols = st.columns(4)
        lat_stats = [
            ("Mínima",     f"{min(latencies):.0f}ms",          "#22c55e"),
            ("Mediana P50", f"{sorted_lat[len(sorted_lat)//2]:.0f}ms", "#6366f1"),
            ("P95",        f"{sorted_lat[p95_idx]:.0f}ms",     "#f59e0b"),
            ("Máxima",     f"{max(latencies):.0f}ms",          "#ef4444"),
        ]
        for col, (label, val, color) in zip(lat_cols, lat_stats):
            with col:
                st.markdown(stat_card(label, val, color), unsafe_allow_html=True)
    else:
        st.info("Sin datos de latencia.")

with tab_evaluators:
    results = summary.get("results", [])
    if results:
        st.markdown(
            """
            <div class="metric-explain" style="margin-bottom:1rem;">
                🧪 <strong>Evaluadores</strong> — Rendimiento de cada evaluador independiente.
                Un evaluador con Pass Rate bajo indica un área específica de mejora en el modelo.
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = evaluator_scores_chart(results)
        st.plotly_chart(fig, use_container_width=True, key="res_eval_chart")

        eval_stats: dict[str, dict] = {}
        for r in results:
            for ev in r.get("evaluations", []):
                name = ev.get("evaluator", "?")
                if name not in eval_stats:
                    eval_stats[name] = {"passed": 0, "failed": 0, "scores": []}
                if ev.get("passed"):
                    eval_stats[name]["passed"] += 1
                else:
                    eval_stats[name]["failed"] += 1
                if ev.get("score") is not None:
                    eval_stats[name]["scores"].append(ev["score"])

        ev_table = []
        for name, stats in sorted(eval_stats.items()):
            total = stats["passed"] + stats["failed"]
            avg   = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else None
            ev_table.append({
                "Evaluador": name.replace("_", " ").title(),
                "Tests":     total,
                "Pasados":   stats["passed"],
                "Fallados":  stats["failed"],
                "Pass Rate": f"{stats['passed']/total:.1%}" if total else "—",
                "Avg Score": f"{avg:.3f}" if avg else "—",
                "Min":       f"{min(stats['scores']):.3f}" if stats["scores"] else "—",
                "Max":       f"{max(stats['scores']):.3f}" if stats["scores"] else "—",
            })
        st.dataframe(ev_table, use_container_width=True, hide_index=True)

st.divider()

# ── Category Breakdown ────────────────────────────────────────────────────────
st.markdown(
    '<span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">📂 Resultados por Categoría</span>',
    unsafe_allow_html=True,
)
by_cat = summary.get("by_category", {})
cat_icons  = {"functional": "⚡", "safety": "🛡️", "regression": "🔁", "multi_turn": "💬"}
cat_colors = {"functional": "#6366f1", "safety": "#ef4444", "regression": "#22c55e", "multi_turn": "#38bdf8"}

if by_cat:
    cat_cols = st.columns(len(by_cat))
    for col, (cat, stats) in zip(cat_cols, sorted(by_cat.items())):
        with col:
            pr = stats.get("pass_rate", 0)
            color = cat_colors.get(cat, "#888")
            icon  = cat_icons.get(cat, "📋")
            pr_color = "#22c55e" if pr >= 0.7 else ("#f59e0b" if pr >= 0.5 else "#ef4444")
            st.markdown(
                f"""
                <div class="stat-card" style="border-left:3px solid {color}; text-align:center;">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div style="font-size:0.8rem; font-weight:700; color:{color}; margin:0.2rem 0;">{cat.replace('_',' ').title()}</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{pr_color};">{pr:.0%}</div>
                    <div style="font-size:0.7rem; color:#64748b;">
                        ✅{stats.get('passed',0)} ❌{stats.get('failed',0)} / {stats.get('total',0)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()

# ── Results Explorer ──────────────────────────────────────────────────────────
st.markdown(
    '<span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">🔍 Explorador de Resultados</span>',
    unsafe_allow_html=True,
)

results = summary.get("results", [])
if results:
    filter_cols = st.columns(4)
    with filter_cols[0]:
        f_cat = st.selectbox(
            "Categoría", ["All"] + sorted({r.get("test_case", {}).get("category", "") for r in results}),
            key="f_cat",
        )
    with filter_cols[1]:
        f_status = st.selectbox("Estado", ["All", "Passed", "Failed"], key="f_status")
    with filter_cols[2]:
        f_sev = st.selectbox("Severidad", ["All", "critical", "high", "medium", "low"], key="f_sev")
    with filter_cols[3]:
        f_search = st.text_input("🔍 Buscar", "", key="f_search", placeholder="ID, input...")

    filtered = results
    if f_cat != "All":
        filtered = [r for r in filtered if r.get("test_case", {}).get("category") == f_cat]
    if f_status == "Passed":
        filtered = [r for r in filtered if r.get("overall_passed")]
    elif f_status == "Failed":
        filtered = [r for r in filtered if not r.get("overall_passed")]
    if f_sev != "All":
        filtered = [r for r in filtered if r.get("test_case", {}).get("severity") == f_sev]
    if f_search:
        sl = f_search.lower()
        filtered = [
            r for r in filtered
            if sl in r.get("test_case", {}).get("id", "").lower()
            or sl in str(r.get("test_case", {}).get("input", "")).lower()
        ]

    st.caption(f"Mostrando **{len(filtered)}** de {len(results)} resultados")

    for r in filtered:
        tc     = r.get("test_case", {})
        passed = r.get("overall_passed", False)
        score  = r.get("overall_score")
        icon   = "✅" if passed else "❌"
        sev    = severity_icon(tc.get("severity", "medium"))
        border_color = "#22c55e" if passed else "#ef4444"

        label = f"{icon} {sev} **{tc.get('id','?')}** — {tc.get('category','?')} — Score: {f'{score:.3f}' if score is not None else '—'} — {r.get('latency_ms',0):.0f}ms"

        with st.expander(label):
            detail_col1, detail_col2 = st.columns([3, 2])

            with detail_col1:
                input_val = tc.get("input", "")
                if isinstance(input_val, list):
                    st.markdown("**💬 Conversación:**")
                    for msg in input_val:
                        role  = msg.get("role", "?")
                        emoji = "👤" if role == "user" else "🤖"
                        st.markdown(f"{emoji} **{role}:** {msg.get('content','')}")
                else:
                    st.markdown(f"**📥 Input:** {input_val}")

                st.markdown(f"**🎯 Comportamiento esperado:** {tc.get('expected_behavior','')}")
                st.markdown("**🤖 Respuesta del modelo:**")
                st.text(r.get("response", "")[:600])

            with detail_col2:
                sev_cls = {"critical": "badge-fail", "high": "badge-warn", "medium": "badge-warn", "low": "badge-gray"}.get(tc.get("severity", "medium"), "badge-gray")
                st.markdown(
                    f'<span class="badge {sev_cls}">{tc.get("severity","?").upper()}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("**Evaluaciones:**")
                for ev in r.get("evaluations", []):
                    ev_icon  = "✅" if ev.get("passed") else "❌"
                    ev_score = ev.get("score")
                    st.markdown(f"{ev_icon} **{ev.get('evaluator','?')}**")
                    if ev_score is not None:
                        st.markdown(score_bar(ev_score), unsafe_allow_html=True)
                    reason = ev.get("reason", "")
                    if reason:
                        st.caption(reason[:200])

                if r.get("retrieved_contexts"):
                    st.markdown(
                        f'<span class="badge badge-info">📚 {len(r["retrieved_contexts"])} contextos RAG</span>',
                        unsafe_allow_html=True,
                    )
