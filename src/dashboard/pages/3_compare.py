"""Page 3: Compare Runs — interactive side-by-side analysis of evaluation runs."""

from __future__ import annotations

import os
import sys

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.styles import inject_css, callout, badge, stat_card, page_header
from src.dashboard.components.metrics import severity_icon
from src.dashboard.components.charts import comparison_bar_chart, COLORS, CATEGORY_COLORS
from src.dashboard.components.shared import list_runs

st.set_page_config(page_title="Compare Runs — LLM Eval Lab", page_icon="🔄", layout="wide")
inject_css()
render_sidebar()


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    page_header("🔄", "Compare Runs", "Comparación side-by-side de dos evaluaciones — descubre qué modelo rinde mejor y en qué categorías"),
    unsafe_allow_html=True,
)

runs = list_runs()

if len(runs) < 2:
    st.markdown(
        """
        <div class="empty-state">
            <span class="empty-icon">🔄</span>
            <div class="empty-title">Necesitas al menos 2 runs para comparar</div>
            <div class="empty-desc">
                Lanza más evaluaciones en <strong>Run Evaluation</strong>.<br>
                Prueba diferentes providers (Groq vs Gemini) o modos (Plain vs RAG) para comparar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.page_link("pages/1_run.py", label="🚀 Ir a Run Evaluation", use_container_width=False)
    st.stop()


def _label(r: dict) -> str:
    return f"{r.get('run_id','?')} · {r.get('chatbot_id','?')} ({r.get('chatbot_mode','?')}) · {r.get('timestamp','')[:16]}"


# ── Run Selection ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="metric-explain" style="margin-bottom:1rem;">
        💡 Selecciona dos runs para comparar. Intenta comparar el mismo dataset con diferentes providers o modos.
    </div>
    """,
    unsafe_allow_html=True,
)

sel_col1, sel_col2 = st.columns(2)
with sel_col1:
    st.markdown(
        f"<div style='text-align:center; color:{COLORS['primary']}; font-weight:700; margin-bottom:0.4rem;'>🅰 Run A</div>",
        unsafe_allow_html=True,
    )
    idx_a = st.selectbox("Run A", range(len(runs)), format_func=lambda i: _label(runs[i]), key="cmp_a", label_visibility="collapsed")

with sel_col2:
    st.markdown(
        f"<div style='text-align:center; color:{COLORS['info']}; font-weight:700; margin-bottom:0.4rem;'>🅱 Run B</div>",
        unsafe_allow_html=True,
    )
    idx_b = st.selectbox("Run B", range(len(runs)), index=min(1, len(runs) - 1), format_func=lambda i: _label(runs[i]), key="cmp_b", label_visibility="collapsed")

if idx_a == idx_b:
    st.markdown(callout("Selecciona dos runs distintos para comparar.", kind="warning"), unsafe_allow_html=True)
    st.stop()

run_a, run_b = runs[idx_a], runs[idx_b]
label_a = run_a.get("chatbot_id", "Run A")
label_b = run_b.get("chatbot_id", "Run B")

st.divider()

# ── KPI Comparison ────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.5rem;">
        📊 Comparación de Indicadores Clave
    </div>
    <div style="font-size:0.82rem; color:#64748b; margin-bottom:1rem;">
        Las tarjetas muestran el delta entre A y B. El borde indica qué run ganó cada métrica.
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_keys = [
    ("Pass Rate",         "pass_rate",         True,  lambda v: f"{v:.1%}"),
    ("Avg Score",         "avg_score",          True,  lambda v: f"{v:.3f}"),
    ("Avg Latency",       "avg_latency_ms",     False, lambda v: f"{v:.0f}ms"),
    ("Critical Failures", "critical_failures",  False, lambda v: str(int(v))),
    ("Total Tests",       "total",              None,  lambda v: str(int(v))),
]

kpi_cols = st.columns(len(kpi_keys))
a_wins = 0
b_wins = 0

for i, (name, key, higher_is_better, fmt) in enumerate(kpi_keys):
    with kpi_cols[i]:
        va = run_a.get(key, 0)
        vb = run_b.get(key, 0)
        diff = vb - va

        if higher_is_better is not None and abs(diff) > 0.001:
            is_b_better  = (diff > 0) == higher_is_better
            winner       = "B" if is_b_better else "A"
            arrow        = "↑" if diff > 0 else "↓"
            border_color = COLORS["info"] if is_b_better else COLORS["primary"]
            delta_color  = COLORS["success"] if is_b_better else COLORS["danger"]
            if is_b_better:
                b_wins += 1
            else:
                a_wins += 1
        else:
            winner       = "="
            arrow        = "="
            border_color = "#6b7280"
            delta_color  = COLORS["muted"]

        winner_text = f"→ 🅱 gana" if winner == "B" else ("→ 🅰 gana" if winner == "A" else "Empate")
        st.markdown(
            f"""
            <div class="cmp-card" style="border-left:4px solid {border_color};">
                <div style="font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;">{name}</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                    <div style="text-align:center;">
                        <div style="font-size:0.65rem; color:{COLORS['primary']}; font-weight:700;">🅰 A</div>
                        <div style="font-size:1.1rem; font-weight:800; color:{COLORS['primary']};">{fmt(va)}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:0.65rem; color:{COLORS['info']}; font-weight:700;">🅱 B</div>
                        <div style="font-size:1.1rem; font-weight:800; color:{COLORS['info']};">{fmt(vb)}</div>
                    </div>
                </div>
                <div style="text-align:center; color:{delta_color}; font-size:0.78rem; font-weight:600;">
                    {arrow} {fmt(abs(diff))} · {winner_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Winner summary
st.markdown("")
if a_wins > b_wins:
    st.markdown(
        callout(f"🅰 <strong>{label_a}</strong> gana {a_wins} de {a_wins+b_wins} métricas comparadas.", kind="tip"),
        unsafe_allow_html=True,
    )
elif b_wins > a_wins:
    st.markdown(
        callout(f"🅱 <strong>{label_b}</strong> gana {b_wins} de {a_wins+b_wins} métricas comparadas.", kind="info"),
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        callout("Empate técnico — ambos runs tienen rendimiento similar en las métricas principales.", kind="warning"),
        unsafe_allow_html=True,
    )

st.divider()

# ── Metric Charts ─────────────────────────────────────────────────────────────
metrics_a: dict = {}
metrics_a.update(run_a.get("ragas_aggregate", {}))
metrics_a.update(run_a.get("deepeval_aggregate", {}))

metrics_b: dict = {}
metrics_b.update(run_b.get("ragas_aggregate", {}))
metrics_b.update(run_b.get("deepeval_aggregate", {}))

if metrics_a or metrics_b:
    st.markdown(
        f"""
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.25rem;">📐 Comparación de Métricas RAGAS / DeepEval</div>
        <div class="metric-explain" style="margin-bottom:1rem;">
            Compara las métricas avanzadas entre los dos runs. Las diferencias en Faithfulness y Relevancy
            indican cambios significativos en la calidad del RAG.
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = comparison_bar_chart(metrics_a, metrics_b, f"🅰 {label_a}", f"🅱 {label_b}")
        st.plotly_chart(fig, use_container_width=True, key="cmp_bar")

    with chart_col2:
        all_metric_names = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
        if all_metric_names:
            names_closed = all_metric_names + [all_metric_names[0]]
            theta = [n.replace("_", " ").title() for n in names_closed]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[metrics_a.get(m, 0) for m in names_closed], theta=theta,
                fill="toself", fillcolor="rgba(99, 102, 241, 0.15)",
                line=dict(color=COLORS["primary"], width=2.5), name=f"🅰 {label_a}",
                hovertemplate="%{theta}: %{r:.3f}<extra>Run A</extra>",
            ))
            fig.add_trace(go.Scatterpolar(
                r=[metrics_b.get(m, 0) for m in names_closed], theta=theta,
                fill="toself", fillcolor="rgba(56, 189, 248, 0.15)",
                line=dict(color=COLORS["info"], width=2.5), name=f"🅱 {label_b}",
                hovertemplate="%{theta}: %{r:.3f}<extra>Run B</extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"]),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border"]),
                    angularaxis=dict(gridcolor=COLORS["border"]),
                ),
                title=dict(text="Radar Overlay — A vs B", font=dict(size=13)),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True, key="cmp_radar")

    # Metric detail table
    all_metrics = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    table_data = []
    for m in all_metrics:
        va = metrics_a.get(m)
        vb = metrics_b.get(m)
        diff = (vb or 0) - (va or 0)
        arrow  = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "=")
        winner = "🅱 B" if diff > 0.001 else ("🅰 A" if diff < -0.001 else "=")
        table_data.append({
            "Métrica":         m.replace("_", " ").title(),
            f"🅰 {label_a}":  f"{va:.3f}" if va is not None else "—",
            f"🅱 {label_b}":  f"{vb:.3f}" if vb is not None else "—",
            "Delta":           f"{arrow} {abs(diff):.3f}",
            "Ganador":         winner,
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)
else:
    st.markdown(
        callout("Ninguno de los dos runs tiene métricas RAGAS/DeepEval. Actívalas en la barra lateral.", kind="info"),
        unsafe_allow_html=True,
    )

st.divider()

# ── Category Comparison ───────────────────────────────────────────────────────
st.markdown(
    """
    <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.25rem;">📂 Comparación por Categoría</div>
    <div class="metric-explain" style="margin-bottom:1rem;">
        Las barras de progreso muestran el Pass Rate de cada modelo por categoría.
        Diferencias grandes indican que un modelo es mejor en esa área específica.
    </div>
    """,
    unsafe_allow_html=True,
)

cats_a = run_a.get("by_category", {})
cats_b = run_b.get("by_category", {})
all_cats = sorted(set(list(cats_a.keys()) + list(cats_b.keys())))
cat_icons = {"functional": "⚡", "safety": "🛡️", "regression": "🔁", "multi_turn": "💬"}

if all_cats:
    for cat in all_cats:
        a_rate = cats_a.get(cat, {}).get("pass_rate", 0)
        b_rate = cats_b.get(cat, {}).get("pass_rate", 0)
        icon   = cat_icons.get(cat, "📋")
        winner_hint = "🅰 Gana A" if a_rate > b_rate + 0.02 else ("🅱 Gana B" if b_rate > a_rate + 0.02 else "Empate")
        winner_color = COLORS["primary"] if a_rate > b_rate + 0.02 else (COLORS["info"] if b_rate > a_rate + 0.02 else "#6b7280")

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
                <span style="font-size:1rem;">{icon}</span>
                <span style="font-size:0.9rem; font-weight:600; color:#e2e8f0;">{cat.replace('_',' ').title()}</span>
                <span class="badge" style="background:rgba(99,102,241,0.1); color:{winner_color}; border-color:{winner_color};">{winner_hint}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bar_col1, bar_col2 = st.columns(2)
        with bar_col1:
            st.progress(a_rate, text=f"🅰 {label_a}: {a_rate:.0%}")
        with bar_col2:
            st.progress(b_rate, text=f"🅱 {label_b}: {b_rate:.0%}")
        st.markdown("")

st.divider()

# ── Disagreements ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.25rem;">⚡ Desacuerdos Pass/Fail</div>
    <div class="metric-explain" style="margin-bottom:1rem;">
        Tests donde los dos runs difieren en el resultado (uno pasa, el otro falla).
        Estos casos revelan diferencias cualitativas entre los modelos.
    </div>
    """,
    unsafe_allow_html=True,
)

results_a = {r.get("test_case", {}).get("id"): r for r in run_a.get("results", [])}
results_b = {r.get("test_case", {}).get("id"): r for r in run_b.get("results", [])}
common_ids = sorted(set(results_a.keys()) & set(results_b.keys()))

disagreements = [
    (tid, results_a[tid], results_b[tid])
    for tid in common_ids
    if results_a[tid].get("overall_passed") != results_b[tid].get("overall_passed")
]

if disagreements:
    st.markdown(
        callout(
            f"<strong>{len(disagreements)} tests</strong> con resultado diferente entre A y B. "
            "Expande cada uno para ver las respuestas y entender por qué difieren.",
            kind="warning",
        ),
        unsafe_allow_html=True,
    )
    for tid, ra, rb in disagreements:
        icon_a = "✅" if ra.get("overall_passed") else "❌"
        icon_b = "✅" if rb.get("overall_passed") else "❌"
        sev    = severity_icon(ra.get("test_case", {}).get("severity", "medium"))

        with st.expander(f"{sev} **{tid}** — 🅰{icon_a} vs 🅱{icon_b}"):
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                outcome_color = "#22c55e" if ra.get("overall_passed") else "#ef4444"
                st.markdown(
                    f"""
                    <div style="border-left:3px solid {outcome_color}; padding-left:0.75rem; margin-bottom:0.5rem;">
                        <div style="font-weight:700; color:{COLORS['primary']};">🅰 {label_a}</div>
                        <div style="color:{'#22c55e' if ra.get('overall_passed') else '#ef4444'}; font-weight:600;">
                            {'✅ Passed' if ra.get('overall_passed') else '❌ Failed'}
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8;">Score: {ra.get('overall_score','—')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.text(ra.get("response", "")[:400])
            with d_col2:
                outcome_color = "#22c55e" if rb.get("overall_passed") else "#ef4444"
                st.markdown(
                    f"""
                    <div style="border-left:3px solid {outcome_color}; padding-left:0.75rem; margin-bottom:0.5rem;">
                        <div style="font-weight:700; color:{COLORS['info']};">🅱 {label_b}</div>
                        <div style="color:{'#22c55e' if rb.get('overall_passed') else '#ef4444'}; font-weight:600;">
                            {'✅ Passed' if rb.get('overall_passed') else '❌ Failed'}
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8;">Score: {rb.get('overall_score','—')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.text(rb.get("response", "")[:400])
else:
    if common_ids:
        st.markdown(
            callout("✅ Todos los tests comunes tienen el mismo resultado en ambos runs.", kind="success"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            callout("No hay tests comunes entre los runs seleccionados.", kind="info"),
            unsafe_allow_html=True,
        )
