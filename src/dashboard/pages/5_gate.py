"""Page 5: Quality Gate — verdict for a stored run against a committed baseline.

This page never runs an evaluation (Run Evaluation owns that) and never writes
config/gate.yaml. It reads what exists and applies the gate engine unchanged.

The verdict lives in a function rather than at module level so that an
unreadable baseline can `return` instead of `st.stop()`: stopping the script
would also hide the "create a baseline" section below, leaving a user with a
corrupt baseline no way out of it from the UI.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.gate_view import BaselineSummary, blocking_reasons, dataset_drift, list_baselines, verdict_rows
from src.dashboard.components.shared import BASELINES_DIR, list_runs
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.styles import callout, inject_css, page_header
from src.dashboard.components.theme import PALETTE
from src.gate.baseline import BaselineError, build_baseline, load_baseline, save_baseline
from src.gate.comparison import CompatibilityError
from src.gate.models import GatePolicy, MetricPolicy
from src.gate.policy import PolicyError, evaluate_gate, load_policy
from src.runner.models import RunSummary

st.set_page_config(page_title="Quality Gate — LLM Eval Lab", page_icon="🎯", layout="wide")
inject_css()
render_sidebar()

_POLICY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "gate.yaml"))

st.markdown(
    page_header(
        "🎯",
        "Quality Gate",
        "¿Este run pasaría el gate de CI? Compara un run guardado contra un baseline y explica el veredicto",
    ),
    unsafe_allow_html=True,
)


def _load_gate_policy() -> GatePolicy:
    """The policy in force, falling back to the built-in defaults."""
    if not os.path.exists(_POLICY_PATH):
        return GatePolicy()
    try:
        return load_policy(_POLICY_PATH)
    except PolicyError as e:
        st.markdown(
            callout(f"La política <code>gate.yaml</code> no es válida ({e}). Se usan los valores por defecto.", kind="warning"),
            unsafe_allow_html=True,
        )
        return GatePolicy()


def _render_drift(baseline_file, summary: RunSummary) -> None:
    """Informational only — drift never changes the verdict."""
    drift = dataset_drift(baseline_file, [r.test_case for r in summary.results])
    if not drift.comparable:
        st.markdown(
            callout(
                f"El run no incluye {len(drift.missing_ids)} caso(s) del baseline "
                f"(<code>{', '.join(drift.missing_ids[:5])}</code>), así que no se puede comprobar si los "
                "casos han cambiado. El veredicto sigue calculándose sobre los casos comunes.",
                kind="info",
            ),
            unsafe_allow_html=True,
        )
    elif drift.drifted:
        st.markdown(
            callout(
                "<strong>Este baseline se creó con otra versión de los casos de prueba.</strong> "
                "Algún caso cambió de contenido conservando su id, así que baseline y run no están "
                "midiendo exactamente lo mismo. Considera regenerar el baseline.",
                kind="warning",
            ),
            unsafe_allow_html=True,
        )


def _render_verdict_card(verdict, reasons: list[str]) -> None:
    color = PALETTE["success"] if verdict.passed else PALETTE["danger"]
    label = "✅ PASS" if verdict.passed else "❌ FAIL"
    reasons_html = (
        "".join(f'<li style="margin-bottom:0.25rem;">{r}</li>' for r in reasons)
        if reasons
        else f'<li style="color:{PALETTE["text_soft"]};">Ninguna métrica gateada empeoró de forma significativa.</li>'
    )
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{PALETTE["bg"]},{PALETTE["bg_raised"]}); border:1px solid {PALETTE["border"]};
             border-left:4px solid {color}; border-radius:12px; padding:1.25rem; margin:1rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:2rem;">
                <div style="flex:1;">
                    <div style="font-size:0.7rem; color:{PALETTE["accent"]}; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">
                        Veredicto del gate
                    </div>
                    <ul style="margin:0.5rem 0 0 1rem; padding:0; color:{PALETTE["text"]}; font-size:0.9rem;">{reasons_html}</ul>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:2.2rem; font-weight:900; color:{color};">{label}</div>
                    <div style="font-size:0.8rem; color:{PALETTE["text_soft"]};">
                        {verdict.samples} muestra(s) · flakiness {verdict.mean_flakiness:.2f}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_policy_panel(baseline_file, current, verdict, policy: GatePolicy) -> None:
    """Show the policy in force and let its thresholds be simulated.

    Nothing here touches disk: the sliders build a fresh GatePolicy and the gate
    is re-evaluated in memory. Changing what CI does still means editing the YAML.
    """
    with st.expander("⚖️ Política vigente · simular otros umbrales"):
        source = "config/gate.yaml" if os.path.exists(_POLICY_PATH) else "valores por defecto integrados"
        st.markdown(
            f"""
            <div class="metric-explain" style="margin-bottom:1rem;">
                Reglas en vigor, leídas de <code>{source}</code>. Los controles de abajo <strong>no modifican
                el fichero</strong>: solo recalculan el veredicto para que veas qué efecto tendría cambiarlas.
                Para que un cambio afecte al CI, edítalo en <code>config/gate.yaml</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        gated_metric = next(iter(policy.metrics), None)
        pol_cols = st.columns(3)
        with pol_cols[0]:
            sim_alpha = st.slider(
                "Nivel de significancia (p)",
                0.01,
                0.20,
                float(policy.significance_level),
                0.01,
                help="Un p-valor por debajo de este umbral se considera una diferencia real, no ruido.",
            )
        with pol_cols[1]:
            sim_effect = st.slider(
                "Efecto mínimo",
                0.0,
                0.30,
                float(policy.min_effect_size),
                0.01,
                help="Regresiones más pequeñas que esto nunca rompen la build, aunque sean significativas.",
            )
        with pol_cols[2]:
            sim_max_regression = st.slider(
                f"Regresión máxima · {gated_metric or 'sin métricas gateadas'}",
                0.0,
                0.50,
                float(policy.metrics[gated_metric].max_regression) if gated_metric else 0.05,
                0.01,
                disabled=gated_metric is None,
                help="Cuánto puede empeorar la métrica gateada antes de romper la build.",
            )

        simulated = policy.model_copy(
            update={
                "significance_level": sim_alpha,
                "min_effect_size": sim_effect,
                "metrics": (
                    {**policy.metrics, gated_metric: MetricPolicy(max_regression=sim_max_regression)} if gated_metric else policy.metrics
                ),
            }
        )

        try:
            sim_verdict = evaluate_gate(baseline_file, current, simulated)
        except CompatibilityError as e:
            st.markdown(callout(f"No se puede simular: {e}", kind="error"), unsafe_allow_html=True)
            return

        changed = sim_verdict.passed != verdict.passed
        sim_label = "✅ PASS" if sim_verdict.passed else "❌ FAIL"
        real_label = "✅ PASS" if verdict.passed else "❌ FAIL"
        st.markdown(
            callout(
                f"Con estos umbrales el veredicto sería <strong>{sim_label}</strong> "
                f"(el real, con la política vigente, es <strong>{real_label}</strong>)."
                + (" El cambio de umbral <strong>invierte el resultado</strong>." if changed else ""),
                kind="warning" if changed else "info",
            ),
            unsafe_allow_html=True,
        )
        for reason in blocking_reasons(sim_verdict, simulated):
            st.markdown(f"- {reason}")


def render_gate(baselines: list[BaselineSummary], runs: list[dict]) -> None:
    """Selection, verdict and metric table. Returns early on any unusable input
    so the baseline-creation section below still renders."""
    st.markdown(
        f'<span style="font-size:1.1rem; font-weight:700; color:{PALETTE["text"]};">1 · Qué comparar</span>',
        unsafe_allow_html=True,
    )
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        b_idx = st.selectbox(
            "📌 Baseline",
            range(len(baselines)),
            format_func=lambda i: f"{baselines[i].name} · {baselines[i].samples} muestras · {baselines[i].n_cases} casos",
            key="gate_baseline",
        )
    with sel_col2:
        r_idx = st.selectbox(
            "📁 Run a juzgar",
            range(len(runs)),
            format_func=lambda i: f"{runs[i].get('run_id', '?')} · {runs[i].get('chatbot_id', '?')} · {runs[i].get('timestamp', '')[:19]}",
            key="gate_run",
        )

    policy = _load_gate_policy()

    try:
        baseline_file = load_baseline(baselines[b_idx].path)
    except BaselineError as e:
        st.markdown(callout(f"No se pudo leer el baseline: {e}", kind="error"), unsafe_allow_html=True)
        return

    try:
        summary = RunSummary.model_validate(runs[r_idx])
        current = build_baseline([summary])
    except (BaselineError, ValueError) as e:
        st.markdown(callout(f"No se pudo leer el run seleccionado: {e}", kind="error"), unsafe_allow_html=True)
        return

    _render_drift(baseline_file, summary)

    try:
        verdict = evaluate_gate(baseline_file, current, policy)
    except CompatibilityError as e:
        st.markdown(
            callout(
                f"<strong>Estos dos no son comparables:</strong> {e}<br>"
                "El gate se niega a emitir un veredicto en vez de compararlos a medias — "
                "en CI esto es un error de ejecución (exit 2), no una regresión.",
                kind="error",
            ),
            unsafe_allow_html=True,
        )
        return

    st.divider()
    _render_verdict_card(verdict, blocking_reasons(verdict, policy))

    if verdict.samples == 1:
        st.markdown(
            callout(
                "Con <strong>1 muestra</strong> la potencia estadística es baja: casi nada llega a ser "
                "significativo. Usa 3 o más muestras en Run Evaluation para un veredicto fiable.",
                kind="info",
            ),
            unsafe_allow_html=True,
        )
    if verdict.new_case_ids:
        st.markdown(
            callout(f"Casos sin baseline (no se juzgan): <code>{', '.join(verdict.new_case_ids)}</code>", kind="info"),
            unsafe_allow_html=True,
        )
    if verdict.removed_case_ids:
        st.markdown(
            callout(f"Casos del baseline ausentes en el run: <code>{', '.join(verdict.removed_case_ids)}</code>", kind="info"),
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="font-size:1.1rem; font-weight:700; color:{PALETTE["text"]}; margin-bottom:0.25rem;">📐 Métricas</div>
        <div class="metric-explain" style="margin-bottom:1rem;">
            Cada fila compara una métrica entre baseline y run con un bootstrap pareado por caso.
            <strong>Solo las métricas gateadas pueden romper la build</strong>; el resto es informativo.
            Un intervalo de confianza que no cruza el cero indica una diferencia real, no ruido.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(verdict_rows(verdict), use_container_width=True, hide_index=True)

    _render_policy_panel(baseline_file, current, verdict, policy)

    st.page_link("pages/3_compare.py", label="🔄 Comparar dos runs entre sí (sin veredicto) →", use_container_width=False)


def render_baseline_creation(runs: list[dict], has_baselines: bool) -> None:
    """Build a baseline out of stored runs.

    Overwriting asks for explicit confirmation: these files are committed and
    decide whether builds pass.
    """
    with st.expander("📌 Crear un baseline a partir de runs", expanded=not has_baselines):
        st.markdown(
            """
            <div class="metric-explain" style="margin-bottom:1rem;">
                Un baseline congela el resultado de uno o varios runs del <strong>mismo dataset</strong>.
                Cuantos más runs incluyas, más fiable es: el gate necesita varias muestras para
                distinguir una regresión real del ruido normal de un LLM.
                El fichero se guarda en <code>baselines/</code> y está pensado para commitearse.
            </div>
            """,
            unsafe_allow_html=True,
        )

        run_labels = {i: f"{r.get('run_id', '?')} · {r.get('chatbot_id', '?')} · {r.get('timestamp', '')[:19]}" for i, r in enumerate(runs)}
        picked = st.multiselect(
            "Runs a incluir",
            options=list(run_labels),
            format_func=lambda i: run_labels[i],
            default=[0],
            key="gate_new_baseline_runs",
        )
        new_name = st.text_input("Nombre del baseline", value="main", key="gate_new_baseline_name")

        target_path = os.path.join(BASELINES_DIR, f"{new_name}.json")
        confirmed = True
        if new_name and os.path.exists(target_path):
            st.markdown(
                callout(f"Ya existe <code>baselines/{new_name}.json</code>. Marca la casilla para reemplazarlo.", kind="warning"),
                unsafe_allow_html=True,
            )
            confirmed = st.checkbox(
                f"Sobrescribir el baseline existente «{new_name}»",
                value=False,
                key="gate_overwrite",
                help="El fichero actual se reemplaza. Si estaba commiteado, el cambio aparecerá en tu próximo diff.",
            )

        if st.button("📌 Guardar baseline", type="primary", disabled=not picked or not new_name or not confirmed):
            try:
                summaries = [RunSummary.model_validate(runs[i]) for i in picked]
                saved_path = save_baseline(build_baseline(summaries), target_path)
            except BaselineError as e:
                st.markdown(
                    callout(
                        f"No se pudo construir el baseline: {e}<br>Los runs deben cubrir los mismos casos y el mismo modo de chatbot.",
                        kind="error",
                    ),
                    unsafe_allow_html=True,
                )
            except OSError as e:
                st.markdown(callout(f"No se pudo escribir el fichero: {e}", kind="error"), unsafe_allow_html=True)
            else:
                st.markdown(
                    callout(f"Baseline guardado en <code>{saved_path}</code> ({len(picked)} run(s)).", kind="success"),
                    unsafe_allow_html=True,
                )
                st.rerun()


# ── Page body ─────────────────────────────────────────────────────────────────
_runs = list_runs()
_baselines = list_baselines(BASELINES_DIR)

if not _runs:
    st.markdown(
        """
        <div class="empty-state">
            <span class="empty-icon">🎯</span>
            <div class="empty-title">No hay runs que juzgar</div>
            <div class="empty-desc">
                El gate compara un run ya ejecutado contra un baseline.<br>
                Lanza primero una evaluación.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_run.py", label="🚀 Ir a Run Evaluation", use_container_width=False)
    st.stop()

if not _baselines:
    st.markdown(
        f"""
        <div class="empty-state">
            <span class="empty-icon">📌</span>
            <div class="empty-title">Todavía no hay ningún baseline</div>
            <div class="empty-desc">
                Un baseline congela el resultado de uno o varios runs para poder detectar
                regresiones contra él. Se guarda en <code>{BASELINES_DIR}</code> y se commitea al repo.<br><br>
                Créalo abajo a partir de los runs que ya tienes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    render_gate(_baselines, _runs)

st.divider()
render_baseline_creation(_runs, has_baselines=bool(_baselines))
