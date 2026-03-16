"""Page 4: Test Cases Manager — browse, filter, search, add, and analyze test cases."""

from __future__ import annotations

import json
import os
import sys

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.styles import inject_css, callout, badge, stat_card
from src.dashboard.components.metrics import severity_icon, severity_badge
from src.dashboard.components.charts import COLORS, CATEGORY_COLORS, SEVERITY_COLORS

st.set_page_config(page_title="Test Cases — LLM Eval Lab", page_icon="📝", layout="wide")
inject_css()
render_sidebar()

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"))

from src.runner.runner import load_dataset

# ── Load all datasets ─────────────────────────────────────────────────────────
dataset_files: dict = {}
for fname in sorted(os.listdir(DATASETS_DIR)):
    if fname.endswith(".jsonl") and fname != "rag_knowledge_base.jsonl":
        name = fname.replace(".jsonl", "")
        cases = load_dataset(os.path.join(DATASETS_DIR, fname))
        dataset_files[name] = cases

all_cases = [c for cases in dataset_files.values() for c in cases]
total_cases = len(all_cases)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <div class="page-title">📝 Test Cases Manager</div>
        <div class="page-desc">Explora, filtra y crea casos de prueba · Aprende a diseñar tests para evaluar LLMs</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Educational callout ───────────────────────────────────────────────────────
with st.expander("📚 ¿Cómo están estructurados los test cases?"):
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.markdown(
            """
            Un **test case** tiene estos campos clave:
            - **`id`** — Identificador único (ej: `func_001`)
            - **`category`** — Tipo de test: `functional`, `safety`, `regression`, `multi_turn`
            - **`severity`** — Impacto: `critical`, `high`, `medium`, `low`
            - **`input`** — Pregunta o conversación a enviar al chatbot
            - **`expected_behavior`** — Qué debería hacer el modelo (no la respuesta exacta)
            - **`evaluation_type`** — Qué evaluadores aplican: `rule_based`, `safety`, etc.
            """
        )
    with exp_col2:
        st.markdown("**Ejemplo de test case:**")
        st.json({
            "id": "func_001",
            "category": "functional",
            "severity": "high",
            "input": "¿Qué es Kubernetes?",
            "expected_behavior": "Explica Kubernetes como plataforma de orquestación de contenedores",
            "evaluation_type": ["rule_based", "ragas"],
        })

st.divider()

# ── Analytics ─────────────────────────────────────────────────────────────────
st.markdown(
    '<span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">📊 Análisis del Dataset</span>',
    unsafe_allow_html=True,
)
st.markdown("")

categories: dict[str, int] = {}
severities: dict[str, int] = {}
eval_types: dict[str, int] = {}
topics:     dict[str, int] = {}

for c in all_cases:
    categories[c.category] = categories.get(c.category, 0) + 1
    severities[c.severity] = severities.get(c.severity, 0) + 1
    for et in c.evaluation_type:
        eval_types[et] = eval_types.get(et, 0) + 1
    topic = c.metadata.get("topic", "untagged")
    topics[topic] = topics.get(topic, 0) + 1

# KPI row
kpi_cols = st.columns(4)
kpi_data = [
    ("Total Test Cases", str(total_cases), "#6366f1"),
    ("Categorías",       str(len(categories)), "#38bdf8"),
    ("Topics únicos",    str(len(topics)), "#22c55e"),
    ("Con Reference",    str(sum(1 for c in all_cases if c.reference)), "#f59e0b"),
]
for col, (label, val, color) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(stat_card(label, val, color), unsafe_allow_html=True)

st.markdown("")

# Charts
chart_col1, chart_col2, chart_col3 = st.columns(3)

cat_descriptions = {
    "functional":  "Tests de respuestas correctas",
    "safety":      "Tests de seguridad y ataques",
    "regression":  "Tests de estabilidad",
    "multi_turn":  "Tests de conversación",
}

with chart_col1:
    fig = go.Figure(go.Pie(
        labels=[f"{c.replace('_',' ').title()}" for c in categories.keys()],
        values=list(categories.values()),
        hole=0.45,
        marker=dict(colors=[CATEGORY_COLORS.get(c, COLORS["muted"]) for c in categories.keys()]),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>%{value} tests<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"]),
        title=dict(text="Por Categoría", font=dict(size=13)),
        height=280, margin=dict(l=20, r=20, t=40, b=20), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="tc_cat")

with chart_col2:
    sev_order = ["critical", "high", "medium", "low"]
    sev_labels = [s for s in sev_order if s in severities]
    fig = go.Figure(go.Pie(
        labels=[s.title() for s in sev_labels],
        values=[severities[s] for s in sev_labels],
        hole=0.45,
        marker=dict(colors=[SEVERITY_COLORS.get(s, COLORS["muted"]) for s in sev_labels]),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>%{value} tests<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"]),
        title=dict(text="Por Severidad", font=dict(size=13)),
        height=280, margin=dict(l=20, r=20, t=40, b=20), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="tc_sev")

with chart_col3:
    fig = go.Figure(go.Bar(
        x=list(eval_types.values()),
        y=[e.replace("_", " ").title() for e in eval_types.keys()],
        orientation="h",
        marker=dict(
            color=[COLORS["primary"]] * len(eval_types),
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=list(eval_types.values()),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} tests<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        title=dict(text="Cobertura por Evaluador", font=dict(size=13)),
        height=280, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, key="tc_eval")

# Category descriptions
cat_icons   = {"functional": "⚡", "safety": "🛡️", "regression": "🔁", "multi_turn": "💬"}
cat_colors  = {"functional": "#6366f1", "safety": "#ef4444", "regression": "#22c55e", "multi_turn": "#38bdf8"}
cat_descs_full = {
    "functional":  "Evalúa si el modelo responde correctamente a preguntas generales, de dominio y de razonamiento lógico. Son los tests más básicos y cubren el comportamiento esperado del chatbot.",
    "safety":      "Detecta vulnerabilidades de seguridad: prompt injection (intentos de hackear el system prompt), filtración de información confidencial y generación de contenido dañino.",
    "regression":  "Verifica que el comportamiento del modelo no cambia entre versiones. Importante para detectar regresiones cuando se actualiza el modelo base o el sistema de prompts.",
    "multi_turn":  "Evalúa la coherencia y memoria en conversaciones de múltiples turnos. El modelo debe recordar el contexto y mantener consistencia a lo largo de la conversación.",
}

cat_detail_cols = st.columns(len(categories))
for col, (cat, count) in zip(cat_detail_cols, sorted(categories.items())):
    with col:
        color = cat_colors.get(cat, "#888")
        icon  = cat_icons.get(cat, "📋")
        desc  = cat_descs_full.get(cat, "")
        st.markdown(
            f"""
            <div class="concept-card" style="border-left:3px solid {color};">
                <div class="concept-title">{icon} {cat.replace('_',' ').title()}</div>
                <div style="font-size:1.4rem; font-weight:800; color:{color}; margin:0.3rem 0;">{count} tests</div>
                <p class="concept-desc">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Browse & Filter ───────────────────────────────────────────────────────────
st.markdown(
    '<span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">🔍 Explorar Test Cases</span>',
    unsafe_allow_html=True,
)

filter_cols = st.columns(5)
with filter_cols[0]:
    f_cat = st.selectbox("Categoría", ["All"] + sorted({c.category for c in all_cases}), key="tc_f_cat")
with filter_cols[1]:
    f_sev = st.selectbox("Severidad", ["All", "critical", "high", "medium", "low"], key="tc_f_sev")
with filter_cols[2]:
    all_eval_list = sorted({et for c in all_cases for et in c.evaluation_type})
    f_eval = st.selectbox("Evaluador", ["All"] + all_eval_list, key="tc_f_eval")
with filter_cols[3]:
    f_ref = st.selectbox("Tiene Reference", ["All", "Sí", "No"], key="tc_f_ref")
with filter_cols[4]:
    f_search = st.text_input("🔍 Buscar", "", key="tc_search", placeholder="ID, input, topic...")

filtered = all_cases
if f_cat != "All":
    filtered = [c for c in filtered if c.category == f_cat]
if f_sev != "All":
    filtered = [c for c in filtered if c.severity == f_sev]
if f_eval != "All":
    filtered = [c for c in filtered if f_eval in c.evaluation_type]
if f_ref == "Sí":
    filtered = [c for c in filtered if c.reference]
elif f_ref == "No":
    filtered = [c for c in filtered if not c.reference]
if f_search:
    sl = f_search.lower()
    filtered = [
        c for c in filtered
        if sl in c.id.lower()
        or sl in str(c.input).lower()
        or sl in c.expected_behavior.lower()
        or sl in str(c.metadata).lower()
    ]

pct = len(filtered) / total_cases * 100 if total_cases > 0 else 0
st.caption(f"Mostrando **{len(filtered)}** de {total_cases} test cases ({pct:.0f}%)")

# Summary table
table_data = []
for c in filtered:
    input_preview = c.input if isinstance(c.input, str) else f"[{len(c.input)} mensajes]"
    table_data.append({
        "ID":          c.id,
        "Categoría":   f"{cat_icons.get(c.category,'📋')} {c.category}",
        "Severidad":   f"{severity_icon(c.severity)} {c.severity}",
        "Input":       str(input_preview)[:70] + ("…" if len(str(input_preview)) > 70 else ""),
        "Evaluadores": ", ".join(c.evaluation_type),
        "Reference":   "✅" if c.reference else "—",
        "Topic":       c.metadata.get("topic", "—"),
    })
st.dataframe(table_data, use_container_width=True, hide_index=True, height=min(420, 40 + 35 * len(table_data)))

# ── Detailed Expandable View ──────────────────────────────────────────────────
if filtered:
    st.divider()
    st.markdown(
        '<span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">📋 Vista Detallada</span>',
        unsafe_allow_html=True,
    )

    show_max = min(len(filtered), 20)
    if len(filtered) > 20:
        st.caption(f"Mostrando los primeros {show_max} de {len(filtered)} resultados. Filtra para ver más.")

    eval_badges = {
        "rule_based":  ("badge-purple", "📏"),
        "safety":      ("badge-fail",   "🛡️"),
        "ragas":       ("badge-info",   "📐"),
        "deepeval":    ("badge-gray",   "🔍"),
        "consistency": ("badge-warn",   "🔄"),
        "llm_judge":   ("badge-purple", "⚖️"),
    }

    for c in filtered[:show_max]:
        sev     = severity_icon(c.severity)
        cat_icon = cat_icons.get(c.category, "📋")
        sev_cls  = {"critical": "badge-fail", "high": "badge-warn", "medium": "badge-warn", "low": "badge-gray"}.get(c.severity, "badge-gray")

        with st.expander(f"{cat_icon} {sev} **{c.id}** — {c.category} · {c.severity}"):
            d_col1, d_col2 = st.columns([3, 2])

            with d_col1:
                if isinstance(c.input, list):
                    st.markdown("**💬 Conversación multi-turno:**")
                    for msg in c.input:
                        role  = msg.get("role", "?")
                        emoji = "👤" if role == "user" else "🤖"
                        st.markdown(f"{emoji} **{role.title()}:** {msg.get('content','')}")
                else:
                    st.markdown(f"**📥 Input:** {c.input}")

                st.markdown(f"**🎯 Comportamiento esperado:** {c.expected_behavior}")

                if c.reference:
                    with st.container():
                        st.markdown(
                            callout(f"<strong>Reference Answer:</strong> {c.reference}", kind="info"),
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown('<span style="font-size:0.8rem; color:#64748b;">Sin reference answer</span>', unsafe_allow_html=True)

            with d_col2:
                # Severity badge
                st.markdown(
                    f'<span class="badge {sev_cls}">{sev} {c.severity.upper()}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                # Evaluator badges
                st.markdown('<div style="font-size:0.82rem; font-weight:600; color:#94a3b8; margin-bottom:0.4rem;">Evaluadores</div>', unsafe_allow_html=True)
                badges_html = " ".join(
                    f'<span class="badge {eval_badges.get(et, ("badge-gray",""))[0]}">{eval_badges.get(et, ("",""))[1]} {et.replace("_"," ").title()}</span>'
                    for et in c.evaluation_type
                )
                st.markdown(badges_html, unsafe_allow_html=True)

                if c.ragas_metrics:
                    st.markdown("")
                    st.markdown('<div style="font-size:0.82rem; font-weight:600; color:#94a3b8; margin-bottom:0.4rem;">RAGAS Metrics</div>', unsafe_allow_html=True)
                    ragas_html = " ".join(
                        f'<span class="badge badge-info">{m.replace("_"," ").title()}</span>' for m in c.ragas_metrics
                    )
                    st.markdown(ragas_html, unsafe_allow_html=True)

                if c.metadata:
                    st.markdown("")
                    st.markdown('<div style="font-size:0.82rem; font-weight:600; color:#94a3b8; margin-bottom:0.4rem;">Metadata</div>', unsafe_allow_html=True)
                    for k, v in c.metadata.items():
                        st.markdown(
                            f'<div style="font-size:0.78rem; color:#64748b;"><code>{k}</code>: {v}</div>',
                            unsafe_allow_html=True,
                        )

st.divider()

# ── Add New Test Case ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.25rem;">➕ Crear Nuevo Test Case</div>
    <div class="metric-explain" style="margin-bottom:1rem;">
        💡 Tip: Comienza por definir el <strong>comportamiento esperado</strong> antes que el input.
        Un buen test case describe <em>qué debería hacer</em> el modelo, no <em>qué debería decir</em> exactamente.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("add_test_case", clear_on_submit=True):
    form_col1, form_col2 = st.columns(2)

    with form_col1:
        new_id = st.text_input(
            "ID del Test Case *",
            placeholder="func_014",
            help="Formato sugerido: {categoria}_{número} · ej: func_014, safety_013",
        )
        new_category = st.selectbox(
            "Categoría *",
            ["functional", "safety", "regression", "multi_turn"],
            key="new_cat",
            help="functional=respuestas generales · safety=ataques · regression=estabilidad · multi_turn=conversación",
        )
        new_severity = st.selectbox(
            "Severidad *",
            ["low", "medium", "high", "critical"],
            index=1,
            key="new_sev",
            help="critical=bloqueante para producción · high=importante · medium=moderado · low=menor",
        )
        new_input = st.text_area(
            "Input (pregunta al chatbot) *",
            placeholder="¿Qué es Kubernetes?",
            height=100,
            help="La pregunta o instrucción que se enviará al chatbot. Para multi-turn, usa el formato JSON de lista.",
        )
        new_expected = st.text_area(
            "Comportamiento esperado *",
            placeholder="Explica Kubernetes como una plataforma de orquestación de contenedores de código abierto.",
            height=80,
            help="Describe QUÉ debería hacer el modelo, no la respuesta exacta. Se usa para evaluar la relevancia.",
        )

    with form_col2:
        new_reference = st.text_area(
            "Reference Answer (opcional)",
            placeholder="Kubernetes es una plataforma open-source para automatizar el despliegue, escalado y gestión de aplicaciones en contenedores...",
            height=100,
            help="Respuesta ideal de referencia. Usada por RAGAS para medir Faithfulness y Answer Relevancy.",
        )
        new_eval_types = st.multiselect(
            "Tipos de Evaluación *",
            ["rule_based", "safety", "ragas", "deepeval", "consistency", "llm_judge"],
            default=["rule_based"],
            help="Selecciona qué evaluadores aplicarán a este test case.",
        )
        new_topic = st.text_input(
            "Topic (opcional)",
            placeholder="devops",
            help="Categoría temática para agrupar tests similares.",
        )

        # Live preview
        if new_id and new_input:
            st.markdown(
                '<div style="font-size:0.82rem; font-weight:600; color:#94a3b8; margin-bottom:0.4rem;">Preview JSON</div>',
                unsafe_allow_html=True,
            )
            st.json({
                "id":              new_id,
                "category":        new_category,
                "severity":        new_severity,
                "input":           new_input[:80] + ("..." if len(new_input) > 80 else ""),
                "evaluation_type": new_eval_types,
            })

    submitted = st.form_submit_button("➕ Añadir Test Case", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not new_id:          errors.append("El ID es obligatorio")
        if not new_input:       errors.append("El input es obligatorio")
        if not new_expected:    errors.append("El comportamiento esperado es obligatorio")
        if not new_eval_types:  errors.append("Selecciona al menos un tipo de evaluación")
        if any(c.id == new_id for c in all_cases):
            errors.append(f"Ya existe un test case con ID `{new_id}`")

        if errors:
            for e in errors:
                st.error(e)
        else:
            new_case = {
                "id":                new_id,
                "category":          new_category,
                "input":             new_input,
                "expected_behavior": new_expected,
                "reference":         new_reference or None,
                "evaluation_type":   new_eval_types,
                "ragas_metrics":     None,
                "severity":          new_severity,
                "metadata":          {},
            }
            if new_topic:
                new_case["metadata"]["topic"] = new_topic
            new_case["metadata"]["dataset_version"] = "2.0"

            target_file = os.path.join(DATASETS_DIR, f"{new_category}.jsonl")
            with open(target_file, "a") as f:
                f.write("\n" + json.dumps(new_case, ensure_ascii=False))

            st.success(f"✅ Test case `{new_id}` añadido a `{new_category}.jsonl`!")
            st.rerun()
