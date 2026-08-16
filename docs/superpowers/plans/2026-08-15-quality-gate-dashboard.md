# Quality Gate Dashboard Page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dashboard page that emits the regression gate's verdict for an already-executed run against a chosen baseline, lists and creates baselines, and shows the policy that produced the verdict.

**Architecture:** One Streamlit page (`5_gate.py`) that only renders, plus one pure-function module (`components/gate_view.py`) holding every decision the page needs. The gate engine (`src/gate/`) is consumed as-is — no evaluation logic is reimplemented.

**Tech Stack:** Python 3.11+, Streamlit, Pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-08-15-quality-gate-dashboard-design.md`

## Global Constraints

- The page never executes an evaluation. Run Evaluation owns that.
- The page never writes `config/gate.yaml`. Policy simulation is in-memory only.
- Dataset drift only informs; it never changes the verdict.
- `src/dashboard/**` is excluded from coverage (`pyproject.toml`), but `gate_view.py` still gets full unit tests — exclusion is about the coverage gate, not about whether tests exist.
- UI copy is Spanish (matching the rest of the dashboard); code, docstrings and commit messages are English.
- Line length 140, `ruff` + `mypy src/ --ignore-missing-imports` must stay clean.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/dashboard/components/gate_view.py` | **Create.** All gate-page logic as pure functions: baseline listing, dataset drift, verdict rows, blocking reasons. No Streamlit import. |
| `src/dashboard/components/shared.py` | **Modify.** Add `BASELINES_DIR` next to the existing `RESULTS_DIR` / `DATASETS_DIR` constants. |
| `src/dashboard/pages/5_gate.py` | **Create.** Rendering only: selectors, verdict, metric table, policy panel, baseline creation. |
| `src/dashboard/pages/3_compare.py` | **Modify.** Cross-link to the new page. |
| `tests/test_dashboard_gate_view.py` | **Create.** Unit tests for every `gate_view` function. |
| `README.md`, `CHANGELOG.md` | **Modify.** Document the page. |

---

### Task 1: Baseline listing

**Files:**
- Create: `src/dashboard/components/gate_view.py`
- Modify: `src/dashboard/components/shared.py` (after line 57, next to `CONFIG_PATH`)
- Test: `tests/test_dashboard_gate_view.py`

**Interfaces:**
- Consumes: `src.gate.baseline.load_baseline`, `src.gate.baseline.BaselineError`, `src.gate.models.BaselineFile`
- Produces: `BaselineSummary` (Pydantic model with fields `name: str`, `path: str`, `timestamp: str`, `chatbot_id: str`, `chatbot_mode: str`, `samples: int`, `n_cases: int`, `run_ids: list[str]`) and `list_baselines(baselines_dir: str) -> list[BaselineSummary]`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the Quality Gate page logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard.components.gate_view import list_baselines
from src.gate.baseline import build_baseline, save_baseline
from tests.gate_helpers import make_summary


def _write_baseline(directory: Path, name: str, *, run_id: str = "run_a") -> None:
    summary = make_summary({"case_a": {"rule_based": 0.9}}, run_id=run_id)
    save_baseline(build_baseline([summary]), str(directory / f"{name}.json"))


class TestListBaselines:
    def test_returns_empty_when_directory_is_missing(self, tmp_path: Path) -> None:
        assert list_baselines(str(tmp_path / "nope")) == []

    def test_reads_name_and_metadata_from_each_file(self, tmp_path: Path) -> None:
        _write_baseline(tmp_path, "main", run_id="run_main")

        found = list_baselines(str(tmp_path))

        assert len(found) == 1
        assert found[0].name == "main"
        assert found[0].run_ids == ["run_main"]
        assert found[0].n_cases == 1
        assert found[0].samples == 1

    def test_sorts_by_name(self, tmp_path: Path) -> None:
        _write_baseline(tmp_path, "zeta")
        _write_baseline(tmp_path, "alpha")

        assert [b.name for b in list_baselines(str(tmp_path))] == ["alpha", "zeta"]

    def test_an_unreadable_file_does_not_hide_the_others(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_baseline(tmp_path, "good")
        (tmp_path / "broken.json").write_text("{not json")

        with caplog.at_level("WARNING"):
            found = list_baselines(str(tmp_path))

        assert [b.name for b in found] == ["good"]
        assert "broken.json" in caplog.text

    def test_ignores_non_json_files(self, tmp_path: Path) -> None:
        _write_baseline(tmp_path, "main")
        (tmp_path / "README.md").write_text("not a baseline")

        assert [b.name for b in list_baselines(str(tmp_path))] == ["main"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard_gate_view.py -p no:cacheprovider --no-cov -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.dashboard.components.gate_view'`

- [ ] **Step 3: Add the constant**

In `src/dashboard/components/shared.py`, directly after `CONFIG_PATH`:

```python
BASELINES_DIR = os.path.join(_ROOT_DIR, "baselines")
```

- [ ] **Step 4: Write the minimal implementation**

Create `src/dashboard/components/gate_view.py`:

```python
"""Logic behind the Quality Gate page.

Pure functions with no Streamlit import, so the page is left with rendering
only and every decision here is directly testable. Nothing in this module
re-implements the gate: verdicts come from ``src.gate`` unchanged.
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel

from src.gate.baseline import BaselineError, load_baseline

logger = logging.getLogger(__name__)


class BaselineSummary(BaseModel):
    """One baseline file, described for a picker."""

    name: str
    path: str
    timestamp: str
    chatbot_id: str
    chatbot_mode: str
    samples: int
    n_cases: int
    run_ids: list[str]


def list_baselines(baselines_dir: str) -> list[BaselineSummary]:
    """Describe every readable baseline in ``baselines_dir``, sorted by name.

    A corrupt file is logged and skipped rather than hiding the rest: the
    picker must keep working when one file goes bad.
    """
    if not os.path.isdir(baselines_dir):
        return []

    summaries: list[BaselineSummary] = []
    for filename in sorted(os.listdir(baselines_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(baselines_dir, filename)
        try:
            baseline = load_baseline(path)
        except BaselineError as e:
            logger.warning("Skipping unreadable baseline %s: %s", filename, e)
            continue
        summaries.append(
            BaselineSummary(
                name=filename[: -len(".json")],
                path=path,
                timestamp=baseline.timestamp,
                chatbot_id=baseline.chatbot_id,
                chatbot_mode=baseline.chatbot_mode,
                samples=baseline.samples,
                n_cases=len(baseline.cases),
                run_ids=baseline.run_ids,
            )
        )
    return summaries
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dashboard_gate_view.py -p no:cacheprovider --no-cov -q`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/components/gate_view.py src/dashboard/components/shared.py tests/test_dashboard_gate_view.py
git commit -m "feat: list gate baselines for the dashboard picker"
```

---

### Task 2: Dataset drift detection

**Files:**
- Modify: `src/dashboard/components/gate_view.py`
- Test: `tests/test_dashboard_gate_view.py`

**Interfaces:**
- Consumes: `src.gate.baseline.compute_dataset_hash`, `src.gate.models.BaselineFile`, `src.runner.models.TestCase`
- Produces: `DriftReport` (fields `comparable: bool`, `drifted: bool`, `missing_ids: list[str]`, `baseline_hash: str`, `current_hash: str | None`) and `dataset_drift(baseline: BaselineFile, run_cases: Sequence[TestCase]) -> DriftReport`

This is the task that gives `dataset_hash` its first reader. Read §5 of the spec before writing it — comparing against `datasets/` on disk is the wrong answer and the third test below is what pins that down.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard_gate_view.py`:

```python
from src.dashboard.components.gate_view import dataset_drift
from src.runner.models import TestCase


def _case(case_id: str, text: str = "q") -> TestCase:
    return TestCase(
        id=case_id,
        category="functional",
        input=text,
        expected_behavior="answers",
        evaluation_type=["rule_based"],
        severity="medium",
    )


def _baseline_over(case_ids: list[str], text: str = "q"):
    """Baseline built from a run covering exactly ``case_ids``."""
    summary = make_summary({cid: {"rule_based": 0.9} for cid in case_ids})
    for result in summary.results:
        result.test_case.input = text
    return build_baseline([summary])


class TestDatasetDrift:
    def test_identical_cases_show_no_drift(self) -> None:
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a"), _case("b")])

        assert report.comparable is True
        assert report.drifted is False

    def test_same_id_with_changed_text_is_drift(self) -> None:
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a", "a completely different question"), _case("b")])

        assert report.comparable is True
        assert report.drifted is True

    def test_a_run_covering_extra_cases_is_not_drift(self) -> None:
        """The hash is computed over the baseline's ids only. Hashing the whole
        current dataset would flag every baseline built from a subset — a normal
        usage — and an alarm that fires almost always gets ignored."""
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a"), _case("b"), _case("c"), _case("d")])

        assert report.comparable is True
        assert report.drifted is False

    def test_a_missing_id_makes_it_not_comparable(self) -> None:
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a")])

        assert report.comparable is False
        assert report.missing_ids == ["b"]
        assert report.current_hash is None

    def test_not_comparable_is_never_reported_as_drift(self) -> None:
        baseline = _baseline_over(["a", "b"])

        report = dataset_drift(baseline, [_case("a", "changed too")])

        assert report.drifted is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard_gate_view.py::TestDatasetDrift -p no:cacheprovider --no-cov -q`
Expected: FAIL — `ImportError: cannot import name 'dataset_drift'`

- [ ] **Step 3: Write the minimal implementation**

Add to `src/dashboard/components/gate_view.py` (extend the imports with `from collections.abc import Sequence`, `from src.gate.baseline import compute_dataset_hash`, `from src.gate.models import BaselineFile`, `from src.runner.models import TestCase`):

```python
class DriftReport(BaseModel):
    """Whether a baseline still describes the same test cases as a run."""

    comparable: bool
    drifted: bool
    missing_ids: list[str] = []
    baseline_hash: str
    current_hash: str | None = None


def dataset_drift(baseline: BaselineFile, run_cases: Sequence[TestCase]) -> DriftReport:
    """Detect test cases that changed content while keeping their id.

    Compares the baseline against the cases the *run* was executed with, not
    against ``datasets/`` on disk: the question is whether the two sides of the
    verdict describe the same tests. ``build_baseline`` hashes only the cases of
    its own run, so a run covering extra cases is not drift — the hash is
    recomputed over the baseline's ids alone.

    If the run is missing any of the baseline's ids the hashes cannot be
    compared at all (the stored one covers the full set), and the report says
    so instead of claiming drift.
    """
    baseline_ids = sorted({case.id for case in baseline.cases})
    by_id = {case.id: case for case in run_cases}

    missing = [case_id for case_id in baseline_ids if case_id not in by_id]
    if missing:
        return DriftReport(
            comparable=False,
            drifted=False,
            missing_ids=missing,
            baseline_hash=baseline.dataset_hash,
            current_hash=None,
        )

    current_hash = compute_dataset_hash([by_id[case_id] for case_id in baseline_ids])
    return DriftReport(
        comparable=True,
        drifted=current_hash != baseline.dataset_hash,
        missing_ids=[],
        baseline_hash=baseline.dataset_hash,
        current_hash=current_hash,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dashboard_gate_view.py -p no:cacheprovider --no-cov -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/components/gate_view.py tests/test_dashboard_gate_view.py
git commit -m "feat: detect baselines built from a different version of the cases"
```

---

### Task 3: Verdict presentation

**Files:**
- Modify: `src/dashboard/components/gate_view.py`
- Test: `tests/test_dashboard_gate_view.py`

**Interfaces:**
- Consumes: `src.gate.models.GateVerdict`, `src.gate.models.GatePolicy`
- Produces: `verdict_rows(verdict: GateVerdict) -> list[dict]` and `blocking_reasons(verdict: GateVerdict, policy: GatePolicy) -> list[str]`

`blocking_reasons` takes the policy because a useful message quotes the limit that was crossed, and `GateVerdict` does not carry it.

The three failure causes must stay distinguishable: a hard rule, a breaching metric, and a gated metric that could not be compared. The third is a configuration error (exit 2 in CI), not a regression — conflating them in the UI would repeat the mistake that produced the false PASS in PR #11.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard_gate_view.py`:

```python
from src.dashboard.components.gate_view import blocking_reasons, verdict_rows
from src.gate.models import GatePolicy, GateVerdict, MetricComparison, MetricPolicy


def _comparison(metric: str, *, regression: float = 0.0, gated: bool = False, breaches: bool = False) -> MetricComparison:
    return MetricComparison(
        metric=metric,
        baseline_mean=0.70,
        current_mean=0.70 - regression,
        regression=regression,
        ci_low=-0.11,
        ci_high=-0.04,
        p_value=0.01,
        n_cases=43,
        significant=breaches,
        gated=gated,
        breaches=breaches,
    )


def _verdict(**kwargs) -> GateVerdict:
    defaults = dict(
        passed=True,
        comparisons=[],
        hard_rule_violations=[],
        missing_gated_metrics=[],
        new_case_ids=[],
        removed_case_ids=[],
        mean_flakiness=0.0,
        samples=3,
    )
    return GateVerdict(**{**defaults, **kwargs})


class TestVerdictRows:
    def test_one_row_per_comparison(self) -> None:
        verdict = _verdict(comparisons=[_comparison("pass_rate"), _comparison("rule_based")])

        rows = verdict_rows(verdict)

        assert [row["Métrica"] for row in rows] == ["pass_rate", "rule_based"]

    def test_marks_which_metrics_are_gated(self) -> None:
        verdict = _verdict(comparisons=[_comparison("pass_rate", gated=True), _comparison("rule_based", gated=False)])

        rows = verdict_rows(verdict)

        assert rows[0]["Gateada"] == "sí"
        assert rows[1]["Gateada"] == "no"

    def test_regression_keeps_its_sign(self) -> None:
        verdict = _verdict(comparisons=[_comparison("pass_rate", regression=0.08)])

        assert verdict_rows(verdict)[0]["Regresión"].startswith("+")

    def test_no_comparisons_gives_no_rows(self) -> None:
        assert verdict_rows(_verdict()) == []


class TestBlockingReasons:
    def test_a_passing_verdict_has_no_reasons(self) -> None:
        assert blocking_reasons(_verdict(), GatePolicy()) == []

    def test_reports_hard_rule_violations_verbatim(self) -> None:
        verdict = _verdict(passed=False, hard_rule_violations=["New critical failures: safety_004"])

        reasons = blocking_reasons(verdict, GatePolicy())

        assert any("safety_004" in reason for reason in reasons)

    def test_a_breaching_metric_quotes_its_limit(self) -> None:
        verdict = _verdict(passed=False, comparisons=[_comparison("pass_rate", regression=0.08, gated=True, breaches=True)])
        policy = GatePolicy(metrics={"pass_rate": MetricPolicy(max_regression=0.05)})

        reasons = blocking_reasons(verdict, policy)

        assert len(reasons) == 1
        assert "pass_rate" in reasons[0]
        assert "0.05" in reasons[0]

    def test_a_non_breaching_metric_is_not_a_reason(self) -> None:
        verdict = _verdict(passed=False, comparisons=[_comparison("rule_based", regression=0.01)])

        assert blocking_reasons(verdict, GatePolicy()) == []

    def test_missing_gated_metric_is_not_worded_as_a_regression(self) -> None:
        """In CI this is exit 2, a configuration error — the exact case that
        produced a false PASS in PR #11 when an evaluator silently dropped out."""
        verdict = _verdict(passed=False, missing_gated_metrics=["pass_rate"])

        reasons = blocking_reasons(verdict, GatePolicy())

        assert len(reasons) == 1
        assert "regres" not in reasons[0].lower()
        assert "compar" in reasons[0].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard_gate_view.py -p no:cacheprovider --no-cov -q`
Expected: FAIL — `ImportError: cannot import name 'blocking_reasons'`

- [ ] **Step 3: Write the minimal implementation**

Add to `src/dashboard/components/gate_view.py` (extend imports with `from src.gate.models import GatePolicy, GateVerdict`):

```python
def verdict_rows(verdict: GateVerdict) -> list[dict[str, str]]:
    """Metric table rows, mirroring the columns of the CI console report so the
    two are recognisably the same table."""
    return [
        {
            "Métrica": c.metric,
            "Baseline": f"{c.baseline_mean:.4f}",
            "Actual": f"{c.current_mean:.4f}",
            "Regresión": f"{c.regression:+.4f}",
            "IC 95%": f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]",
            "p-valor": f"{c.p_value:.4f}",
            "Gateada": "sí" if c.gated else "no",
            "Veredicto": "❌ regresión" if c.breaches else "✅ ok",
        }
        for c in verdict.comparisons
    ]


def blocking_reasons(verdict: GateVerdict, policy: GatePolicy) -> list[str]:
    """Why the gate fails, in plain sentences.

    Keeps the three causes apart on purpose. A metric that cannot be compared is
    a configuration error — CI exits 2, not 1 — and reads nothing like a quality
    regression.
    """
    reasons: list[str] = list(verdict.hard_rule_violations)

    for c in verdict.comparisons:
        if not c.breaches:
            continue
        limit = policy.metrics[c.metric].max_regression if c.metric in policy.metrics else None
        limit_text = f", por encima del límite permitido ({limit:.2f})" if limit is not None else ""
        reasons.append(f"«{c.metric}» empeora {c.regression:+.4f}{limit_text}.")

    for metric in verdict.missing_gated_metrics:
        reasons.append(
            f"La métrica «{metric}» no se puede comparar: falta en alguno de los dos lados. "
            "Es un error de configuración — en CI provoca exit 2, no un fallo de calidad."
        )

    return reasons
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dashboard_gate_view.py -p no:cacheprovider --no-cov -q`
Expected: PASS (19 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/components/gate_view.py tests/test_dashboard_gate_view.py
git commit -m "feat: format the gate verdict for the dashboard"
```

---

### Task 4: The page — selection, verdict and metric table

**Files:**
- Create: `src/dashboard/pages/5_gate.py`
- Test: manual, by driving the running dashboard (see Step 4)

**Interfaces:**
- Consumes: `gate_view.list_baselines`, `gate_view.dataset_drift`, `gate_view.verdict_rows`, `gate_view.blocking_reasons`, `shared.list_runs`, `shared.BASELINES_DIR`, `src.gate.baseline.build_baseline`, `src.gate.policy.load_policy`, `src.gate.comparison.CompatibilityError`, `src.gate.policy.evaluate_gate`
- Produces: nothing consumed by later tasks; Tasks 5 and 6 append sections to this file.

Follow the structure of `3_compare.py`: `set_page_config` → `inject_css()` → `render_sidebar()` → `page_header(...)`, empty states via the `empty-state` CSS class, callouts via `styles.callout`.

- [ ] **Step 1: Write the page**

```python
"""Page 5: Quality Gate — verdict for a stored run against a committed baseline.

This page never runs an evaluation (Run Evaluation owns that) and never writes
config/gate.yaml. It reads what exists and applies the gate engine unchanged.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.dashboard.components.gate_view import blocking_reasons, dataset_drift, list_baselines, verdict_rows
from src.dashboard.components.shared import BASELINES_DIR, list_runs
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.styles import callout, inject_css, page_header
from src.gate.baseline import BaselineError, build_baseline, load_baseline
from src.gate.comparison import CompatibilityError
from src.gate.models import GatePolicy
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

# ── Preconditions ─────────────────────────────────────────────────────────────
runs = list_runs()
baselines = list_baselines(BASELINES_DIR)

if not runs:
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

if not baselines:
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

# ── Selection ─────────────────────────────────────────────────────────────────
if baselines:
    st.markdown(
        '<span style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">1 · Qué comparar</span>',
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
            format_func=lambda i: (
                f"{runs[i].get('run_id', '?')} · {runs[i].get('chatbot_id', '?')} · {runs[i].get('timestamp', '')[:19]}"
            ),
            key="gate_run",
        )

    chosen_baseline = baselines[b_idx]
    chosen_run = runs[r_idx]

    # ── Load and evaluate ─────────────────────────────────────────────────────
    verdict = None
    policy = GatePolicy()
    try:
        policy = load_policy(_POLICY_PATH) if os.path.exists(_POLICY_PATH) else GatePolicy()
    except PolicyError as e:
        st.markdown(
            callout(f"La política <code>gate.yaml</code> no es válida ({e}). Se usan los valores por defecto.", kind="warning"),
            unsafe_allow_html=True,
        )

    try:
        baseline_file = load_baseline(chosen_baseline.path)
        summary = RunSummary.model_validate(chosen_run)
        current = build_baseline([summary])
    except BaselineError as e:
        st.markdown(callout(f"No se pudo leer el baseline: {e}", kind="error"), unsafe_allow_html=True)
        st.stop()
    except Exception as e:
        st.markdown(callout(f"No se pudo leer el run seleccionado: {e}", kind="error"), unsafe_allow_html=True)
        st.stop()

    # Dataset drift — informational, never changes the verdict.
    run_cases = [r.test_case for r in summary.results]
    drift = dataset_drift(baseline_file, run_cases)
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
        st.stop()

    # ── Verdict ───────────────────────────────────────────────────────────────
    st.divider()
    reasons = blocking_reasons(verdict, policy)
    color = "#22c55e" if verdict.passed else "#ef4444"
    label = "✅ PASS" if verdict.passed else "❌ FAIL"
    reasons_html = (
        "".join(f'<li style="margin-bottom:0.25rem;">{r}</li>' for r in reasons)
        if reasons
        else '<li style="color:#94a3b8;">Ninguna métrica gateada empeoró de forma significativa.</li>'
    )
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1a1a2e,#22223d); border:1px solid #2d2d44;
             border-left:4px solid {color}; border-radius:12px; padding:1.25rem; margin:1rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:2rem;">
                <div style="flex:1;">
                    <div style="font-size:0.7rem; color:#6366f1; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Veredicto del gate</div>
                    <ul style="margin:0.5rem 0 0 1rem; padding:0; color:#e2e8f0; font-size:0.9rem;">{reasons_html}</ul>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:2.2rem; font-weight:900; color:{color};">{label}</div>
                    <div style="font-size:0.8rem; color:#94a3b8;">{verdict.samples} muestra(s) · flakiness {verdict.mean_flakiness:.2f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # ── Metric table ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.25rem;">📐 Métricas</div>
        <div class="metric-explain" style="margin-bottom:1rem;">
            Cada fila compara una métrica entre baseline y run con un bootstrap pareado por caso.
            <strong>Solo las métricas gateadas pueden romper la build</strong>; el resto es informativo.
            El intervalo de confianza que no cruza el cero indica una diferencia real, no ruido.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(verdict_rows(verdict), use_container_width=True, hide_index=True)

    st.page_link("pages/3_compare.py", label="🔄 Comparar dos runs entre sí (sin veredicto) →", use_container_width=False)
```

- [ ] **Step 2: Verify it compiles**

Run: `python3 -m py_compile src/dashboard/pages/5_gate.py`
Expected: no output

- [ ] **Step 3: Create a baseline to test against**

```bash
python3 -m src run --provider mock --samples 3 --evaluators rule_based,safety
# take the three run ids printed at the end
python3 -m src baseline save <run_id_1> <run_id_2> <run_id_3> --name main
```

- [ ] **Step 4: Drive the running dashboard**

```bash
python3 -m streamlit run src/dashboard/app.py --server.port 8511 --server.headless true
```

Check, on the Quality Gate page:
1. A PASS verdict appears for a run consistent with the baseline.
2. Deleting `baselines/main.json` shows the "no baselines yet" empty state.
3. The metric table lists `pass_rate` with `Gateada = sí`.

- [ ] **Step 5: Run the full suite and linters**

Run: `python3 -m pytest tests/ -q && python3 -m ruff check src/ tests/ && python3 -m ruff format --check src/ tests/ && python3 -m mypy src/ --ignore-missing-imports`
Expected: all green

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/pages/5_gate.py
git commit -m "feat: quality gate page with verdict and metric table"
```

---

### Task 5: Policy panel with simulation

**Files:**
- Modify: `src/dashboard/pages/5_gate.py` (append after the metric table)

**Interfaces:**
- Consumes: `policy` and `verdict` from Task 4, `evaluate_gate`, `GatePolicy`, `MetricPolicy`
- Produces: nothing

The sliders build a fresh `GatePolicy` and re-evaluate. Nothing is written to disk. Adding or removing gated metrics is not simulated — that changes the shape of the policy and belongs in the YAML.

- [ ] **Step 1: Append the section**

```python
    # ── Policy ────────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚖️ Política vigente · simular otros umbrales"):
        source = _POLICY_PATH if os.path.exists(_POLICY_PATH) else "valores por defecto integrados"
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

        pol_cols = st.columns(3)
        with pol_cols[0]:
            sim_alpha = st.slider(
                "Nivel de significancia (p)",
                0.01, 0.20, float(policy.significance_level), 0.01,
                help="Un p-valor por debajo de este umbral se considera una diferencia real, no ruido.",
            )
        with pol_cols[1]:
            sim_effect = st.slider(
                "Efecto mínimo",
                0.0, 0.30, float(policy.min_effect_size), 0.01,
                help="Regresiones más pequeñas que esto nunca rompen la build, aunque sean significativas.",
            )
        with pol_cols[2]:
            gated_metric = next(iter(policy.metrics), None)
            sim_max_regression = st.slider(
                f"Regresión máxima · {gated_metric or 'sin métricas gateadas'}",
                0.0, 0.50,
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
                    {**policy.metrics, gated_metric: MetricPolicy(max_regression=sim_max_regression)}
                    if gated_metric
                    else policy.metrics
                ),
            }
        )

        try:
            sim_verdict = evaluate_gate(baseline_file, current, simulated)
        except CompatibilityError as e:
            st.markdown(callout(f"No se puede simular: {e}", kind="error"), unsafe_allow_html=True)
        else:
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
```

Extend the imports at the top of the file with `MetricPolicy`:

```python
from src.gate.models import GatePolicy, MetricPolicy
```

- [ ] **Step 2: Verify it compiles**

Run: `python3 -m py_compile src/dashboard/pages/5_gate.py`
Expected: no output

- [ ] **Step 3: Drive the dashboard**

With the server running, open the expander and confirm:
1. Moving "Regresión máxima" to 0.0 flips a PASS verdict to FAIL and the callout says so.
2. `config/gate.yaml` is unchanged afterwards: `git diff --exit-code config/gate.yaml`

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/pages/5_gate.py
git commit -m "feat: show the gate policy and let it be simulated without writing it"
```

---

### Task 6: Baseline creation

**Files:**
- Modify: `src/dashboard/pages/5_gate.py` (append at the end, outside the `if baselines:` block so it also shows when none exist)

**Interfaces:**
- Consumes: `shared.list_runs`, `src.gate.baseline.build_baseline`, `src.gate.baseline.save_baseline`, `src.gate.baseline.BaselineError`
- Produces: nothing

Overwriting an existing baseline requires an explicit confirmation checkbox: these files are committed and decide whether builds pass.

- [ ] **Step 1: Append the section**

```python
# ── Create a baseline ─────────────────────────────────────────────────────────
st.divider()
with st.expander("📌 Crear un baseline a partir de runs", expanded=not baselines):
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

    run_labels = {
        i: f"{r.get('run_id', '?')} · {r.get('chatbot_id', '?')} · {r.get('timestamp', '')[:19]}" for i, r in enumerate(runs)
    }
    picked = st.multiselect(
        "Runs a incluir",
        options=list(run_labels),
        format_func=lambda i: run_labels[i],
        default=[0],
        key="gate_new_baseline_runs",
    )
    new_name = st.text_input("Nombre del baseline", value="main", key="gate_new_baseline_name")

    target_path = os.path.join(BASELINES_DIR, f"{new_name}.json")
    exists = os.path.exists(target_path)
    confirmed = True
    if exists:
        confirmed = st.checkbox(
            f"Sobrescribir el baseline existente «{new_name}»",
            value=False,
            key="gate_overwrite",
            help="El fichero actual se reemplaza. Si estaba commiteado, el cambio aparecerá en tu próximo diff.",
        )
        st.markdown(
            callout(f"Ya existe <code>baselines/{new_name}.json</code>. Marca la casilla para reemplazarlo.", kind="warning"),
            unsafe_allow_html=True,
        )

    if st.button("📌 Guardar baseline", type="primary", disabled=not picked or not new_name or not confirmed):
        try:
            summaries = [RunSummary.model_validate(runs[i]) for i in picked]
            saved_path = save_baseline(build_baseline(summaries), target_path)
        except BaselineError as e:
            st.markdown(
                callout(
                    f"No se pudo construir el baseline: {e}<br>"
                    "Los runs deben cubrir los mismos casos y el mismo modo de chatbot.",
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
```

- [ ] **Step 2: Verify it compiles**

Run: `python3 -m py_compile src/dashboard/pages/5_gate.py`
Expected: no output

- [ ] **Step 3: Drive the dashboard**

1. With no baselines, the section is open by default and creating one works.
2. Re-saving the same name shows the overwrite warning and the button stays disabled until the checkbox is ticked.
3. Selecting runs with different case sets shows the `BaselineError` message, not a traceback.

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/pages/5_gate.py
git commit -m "feat: create gate baselines from the dashboard"
```

---

### Task 7: Cross-link and documentation

**Files:**
- Modify: `src/dashboard/pages/3_compare.py` (after the statistical comparison section, around line 459)
- Modify: `README.md` (dashboard bullet list and project tree)
- Modify: `CHANGELOG.md` (`[Unreleased]` → `### Added`)

- [ ] **Step 1: Add the cross-link in Compare Runs**

At the end of `3_compare.py`, after the statistical section:

```python
st.caption(
    "Esta comparación no emite veredicto: enfrenta dos runs entre sí. "
    "Para juzgar un run contra un baseline commiteado, usa el Quality Gate."
)
st.page_link("pages/5_gate.py", label="🎯 Ir al Quality Gate →", use_container_width=False)
```

- [ ] **Step 2: Update the README**

In the dashboard bullet list, after "Compare runs side-by-side":

```markdown
- Judge a run against a committed baseline with the same gate CI uses, and see
  why it passes or fails
```

In the project tree, under `components/`:

```
│       │   ├── gate_view.py      # Quality Gate page logic (baselines, drift, verdict rows)
```

under `pages/`:

```
│           ├── 5_gate.py         # Quality Gate page (verdict, policy, baseline creation)
```

and under `tests/`:

```
│   ├── test_dashboard_gate_view.py # Quality Gate page logic tests
```

- [ ] **Step 3: Update the CHANGELOG**

Under `## [Unreleased]` → `### Added`:

```markdown
- **Quality Gate page in the dashboard.** The gate engine had been CLI-only:
  its only trace in the UI was an unlabelled statistics table at the bottom of
  Compare Runs. The page judges a stored run against a committed baseline and
  explains the verdict, keeping the three failure causes distinct — a hard rule,
  a breaching metric, and a gated metric that cannot be compared (a
  configuration error that exits 2 in CI, not a regression). Baselines can be
  listed and created from the UI; the policy is shown and can be simulated, but
  never written back to `config/gate.yaml`.
- **`dataset_hash` finally has a reader.** Written into every baseline since the
  gate landed but never read, so a test case whose text changed while keeping
  its id was silently compared against a stale entry. The page now warns. The
  hash is recomputed over the baseline's own case ids against the cases stored
  in the run's report — not against `datasets/` on disk, which would flag every
  baseline built from a subset of datasets.
```

- [ ] **Step 4: Update the test count**

Run the suite, read the totals, and update both occurrences in the README (`Runs N tests` and `# N tests with coverage report`) plus the coverage percentage in the Features list.

Run: `python3 -m pytest tests/ --cov=src --cov-report=term -q | tail -3`

- [ ] **Step 5: Full verification**

Run: `python3 -m pytest tests/ -q && python3 -m ruff check src/ tests/ && python3 -m ruff format --check src/ tests/ && python3 -m mypy src/ --ignore-missing-imports`
Expected: all green

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/pages/3_compare.py README.md CHANGELOG.md
git commit -m "docs: document the quality gate page"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §3 Page structure (5 sections) | 4 (1-3), 5 (4), 6 (5) |
| §4 `list_baselines` | 1 |
| §4 `dataset_drift` | 2 |
| §4 `verdict_rows`, `blocking_reasons` | 3 |
| §5 Dataset drift semantics | 2 |
| §6 Error states | 4 (compatibility, baseline, policy, empty states), 6 (BaselineError, OSError) |
| §7 Compare Runs cross-link | 7 |
| §8 Policy simulation | 5 |
| §9 Test strategy | 1, 2, 3 (unit), 4-6 (browser verification) |
| §10 Overwrite confirmation | 6 |

**Type consistency:** `BaselineSummary.path` is what Task 4 passes to `load_baseline`. `dataset_drift` takes `run_cases` (from `summary.results[].test_case`), matching Task 2's signature. `blocking_reasons(verdict, policy)` takes two arguments in Tasks 3, 4 and 5 alike.

**Note for the executor:** Tasks 4-6 all modify `5_gate.py` and must be done in order — Task 5 references `baseline_file`, `current`, `policy` and `verdict`, all bound in Task 4.
