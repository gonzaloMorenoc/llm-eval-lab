# Regression Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir LLM Eval Lab en un quality gate de regresión para CI/CD: `llm-eval-lab check --baseline main` compara la calidad actual contra un baseline versionado y rompe el build (exit 1) ante regresiones estadísticamente significativas.

**Architecture:** Un paquete nuevo `src/gate/` (modelos Pydantic, bootstrap pareado con numpy, pareo de casos, política y veredicto) alimentado por los `RunSummary` que ya produce el runner. Una CLI Typer (`src/cli.py`) con subcomandos `run` / `baseline save` / `check` / `compare` sustituye al entry point plano, y una GitHub Action compuesta (`action.yml`) empaqueta `check` para terceros. El runner, los evaluadores y el reporting existentes no se modifican.

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, numpy, rich, pytest (+ typer.testing.CliRunner), GitHub Actions composite action.

**Spec:** `docs/superpowers/specs/2026-08-14-regression-gate-design.md`

## Global Constraints

- Python `>=3.11` (pyproject `requires-python`).
- Quality gate local antes de cada commit: `ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/ --ignore-missing-imports && pytest` (cobertura total del repo debe seguir ≥80%).
- ruff: line-length 140, reglas `E,W,F,I,B,UP,S,RUF` (ver pyproject). `ruff format` es obligatorio (no solo check: formatea antes de commitear).
- Commits en formato `<type>: <description>` (feat, fix, refactor, docs, test, chore, ci). Sin atribución de Claude en el cuerpo salvo el trailer de sesión ya configurado.
- Estilo inmutable: nunca mutar objetos recibidos; usar `model_copy(update=...)` de Pydantic para derivar variantes.
- Validar toda entrada externa (archivos JSON/YAML) con Pydantic y fallar con mensajes claros; los errores de usuario en la CLI terminan con `typer.Exit(code=2)`.
- Archivos enfocados: ningún archivo nuevo debe superar ~400 líneas.
- Convención de nombres de métricas del gate (usada en TODOS los módulos): `pass_rate` (especial), `<evaluator>` para el score del evaluador (p. ej. `rule_based`), `<evaluator>.<metric>` para sub-métricas (p. ej. `ragas.answer_relevancy`, `deepeval.toxicity`).
- Semántica de regresión: positivo = peor. Para métricas lower-is-better (`deepeval.hallucination`, `deepeval.bias`, `deepeval.toxicity`) la regresión es subida; para el resto, bajada.
- Los tests nuevos NO requieren API keys: todo corre con el provider mock.

---

### Task 1: Modelos del gate y dependencias

**Files:**
- Create: `src/gate/__init__.py`
- Create: `src/gate/models.py`
- Modify: `pyproject.toml` (bloque `dependencies`)
- Test: `tests/test_gate_models.py`

**Interfaces:**
- Consumes: nada (solo Pydantic).
- Produces (usado por todas las tareas siguientes): `LOWER_IS_BETTER_METRICS: frozenset[str]`, `BaselineCase`, `BaselineFile`, `BootstrapResult`, `MetricComparison`, `MetricPolicy`, `HardRules`, `GatePolicy`, `GateVerdict` — todos Pydantic `BaseModel` con los campos exactos del Step 3.

- [ ] **Step 1: Añadir dependencias a pyproject.toml**

En `[project].dependencies` añadir dos líneas al final de la lista existente:

```toml
    "typer>=0.12",
    "numpy>=1.26",
```

(numpy ya está instalado como dependencia transitiva de chromadb/ragas; declararlo lo hace explícito. Ejecutar `pip install -e ".[dev]"` tras el cambio.)

- [ ] **Step 2: Escribir los tests que fallan**

`tests/test_gate_models.py`:

```python
"""Tests for the regression-gate Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.gate.models import (
    LOWER_IS_BETTER_METRICS,
    BaselineCase,
    BaselineFile,
    GatePolicy,
    GateVerdict,
    HardRules,
    MetricPolicy,
)


def _case(**overrides) -> BaselineCase:
    data = {
        "id": "func_001",
        "category": "functional",
        "severity": "medium",
        "passed": True,
        "pass_samples": [True],
        "metrics": {"rule_based": 1.0, "ragas.answer_relevancy": 0.8},
        "metric_variance": {"rule_based": 0.0},
        "latency_ms_mean": 120.5,
    }
    data.update(overrides)
    return BaselineCase(**data)


class TestBaselineModels:
    def test_baseline_case_holds_flattened_metrics(self):
        case = _case()
        assert case.metrics["ragas.answer_relevancy"] == 0.8
        assert case.flakiness == 0.0  # default

    def test_baseline_case_rejects_invalid_severity(self):
        with pytest.raises(ValidationError):
            _case(severity="catastrophic")

    def test_baseline_file_round_trips_through_json(self):
        baseline = BaselineFile(
            run_ids=["r1", "r2"],
            timestamp="2026-08-14T00:00:00+00:00",
            chatbot_id="mock/mock-plain-v1",
            chatbot_mode="plain",
            dataset_hash="abc123",
            metric_set=["pass_rate", "rule_based"],
            samples=2,
            cases=[_case()],
        )
        restored = BaselineFile.model_validate_json(baseline.model_dump_json())
        assert restored == baseline
        assert restored.schema_version == 1

    def test_baseline_file_rejects_invalid_mode(self):
        with pytest.raises(ValidationError):
            BaselineFile(
                run_ids=["r1"],
                timestamp="t",
                chatbot_id="x",
                chatbot_mode="hybrid",
                dataset_hash="h",
                metric_set=[],
                cases=[],
            )


class TestGatePolicy:
    def test_defaults_match_spec(self):
        policy = GatePolicy()
        assert policy.significance_level == 0.05
        assert policy.min_effect_size == 0.05
        assert policy.n_resamples == 10_000
        assert policy.seed == 42
        assert policy.metrics == {"pass_rate": MetricPolicy(max_regression=0.05)}
        assert policy.hard_rules == HardRules(no_new_critical_failures=True, max_flakiness=0.3)
        assert policy.new_cases == "report_only"

    def test_new_cases_rejects_unknown_value(self):
        with pytest.raises(ValidationError):
            GatePolicy(new_cases="ignore")


class TestConstants:
    def test_lower_is_better_covers_deepeval_inverse_metrics(self):
        assert LOWER_IS_BETTER_METRICS == frozenset(
            {"deepeval.hallucination", "deepeval.bias", "deepeval.toxicity"}
        )


class TestGateVerdict:
    def test_verdict_minimal_construction(self):
        verdict = GateVerdict(
            passed=True,
            comparisons=[],
            hard_rule_violations=[],
            missing_gated_metrics=[],
            new_case_ids=[],
            removed_case_ids=[],
            mean_flakiness=0.0,
            samples=1,
        )
        assert verdict.passed is True
```

- [ ] **Step 3: Verificar que fallan**

Run: `pytest tests/test_gate_models.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.gate'`

- [ ] **Step 4: Implementar los modelos**

`src/gate/__init__.py`:

```python
"""Regression gate: baselines, statistical comparison, policy and verdict."""
```

`src/gate/models.py`:

```python
"""Pydantic models for the regression gate: baselines, comparisons, policy, verdict."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Metrics whose threshold semantics in config.yaml are "lower is better"
# (DeepEval thresholds use `< 0.5`); for these, regression means the score went UP.
LOWER_IS_BETTER_METRICS = frozenset({"deepeval.hallucination", "deepeval.bias", "deepeval.toxicity"})

# Synthetic metric derived from per-case pass/fail, always present in metric_set.
PASS_RATE_METRIC = "pass_rate"


class BaselineCase(BaseModel):
    """Per-test-case aggregate stored in a baseline (means across samples)."""

    id: str
    category: str
    severity: Literal["critical", "high", "medium", "low"]
    passed: bool  # majority vote across samples (ties count as passed)
    pass_samples: list[bool]
    flakiness: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_variance: dict[str, float] = Field(default_factory=dict)
    latency_ms_mean: float = 0.0


class BaselineFile(BaseModel):
    """Versioned, diffable snapshot of a run (or N sampled runs) for gating."""

    schema_version: int = 1
    run_ids: list[str]
    timestamp: str
    chatbot_id: str
    chatbot_mode: Literal["plain", "rag"]
    dataset_hash: str
    metric_set: list[str]
    samples: int = 1
    cases: list[BaselineCase]


class BootstrapResult(BaseModel):
    """Outcome of a paired bootstrap over per-case regression deltas."""

    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float


class MetricComparison(BaseModel):
    """Statistical comparison of one metric between baseline and current run."""

    metric: str
    baseline_mean: float
    current_mean: float
    regression: float  # positive = worse, direction-normalized
    ci_low: float
    ci_high: float
    p_value: float
    n_cases: int
    significant: bool
    gated: bool  # listed in policy.metrics
    breaches: bool  # significant AND > min_effect_size AND > max_regression


class MetricPolicy(BaseModel):
    max_regression: float


class HardRules(BaseModel):
    no_new_critical_failures: bool = True
    max_flakiness: float = 0.3


class GatePolicy(BaseModel):
    """Gate policy — loaded from gate.yaml or built-in defaults."""

    significance_level: float = 0.05
    min_effect_size: float = 0.05
    n_resamples: int = 10_000
    seed: int = 42
    metrics: dict[str, MetricPolicy] = Field(
        default_factory=lambda: {PASS_RATE_METRIC: MetricPolicy(max_regression=0.05)}
    )
    hard_rules: HardRules = Field(default_factory=HardRules)
    new_cases: Literal["report_only", "fail"] = "report_only"


class GateVerdict(BaseModel):
    """Final gate outcome consumed by reporters and the CLI exit-code logic."""

    passed: bool
    comparisons: list[MetricComparison]
    hard_rule_violations: list[str]
    missing_gated_metrics: list[str]
    new_case_ids: list[str]
    removed_case_ids: list[str]
    mean_flakiness: float
    samples: int
```

- [ ] **Step 5: Verificar que pasan + calidad**

Run: `pytest tests/test_gate_models.py -v && ruff check src/gate tests/test_gate_models.py && ruff format src/gate tests/test_gate_models.py && mypy src/ --ignore-missing-imports`
Expected: PASS, sin issues de lint/tipos.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/gate tests/test_gate_models.py
git commit -m "feat: add regression-gate models and typer/numpy dependencies"
```

---

### Task 2: Motor estadístico

**Files:**
- Create: `src/gate/statistics.py`
- Test: `tests/test_gate_statistics.py`

**Interfaces:**
- Consumes: `BootstrapResult` de `src/gate/models.py` (Task 1).
- Produces: `paired_bootstrap(deltas: Sequence[float], *, n_resamples: int = 10_000, seed: int) -> BootstrapResult` y `case_flakiness(sample_passes: Sequence[bool]) -> float`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_gate_statistics.py`:

```python
"""Tests for the gate's pure statistical functions (seeded, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest

from src.gate.statistics import case_flakiness, paired_bootstrap


class TestPairedBootstrap:
    def test_clear_regression_is_significant(self):
        rng = np.random.default_rng(7)
        deltas = rng.normal(loc=0.2, scale=0.05, size=40).tolist()
        result = paired_bootstrap(deltas, seed=42)
        assert result.p_value < 0.05
        assert 0.15 < result.mean_delta < 0.25
        assert result.ci_low > 0.0

    def test_pure_noise_is_not_significant(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(loc=0.0, scale=0.1, size=40)
        deltas = (raw - raw.mean()).tolist()  # centered: mean exactly 0, immune to seed luck
        result = paired_bootstrap(deltas, seed=42)
        assert result.p_value > 0.05

    def test_all_zero_deltas_yield_p_value_one(self):
        result = paired_bootstrap([0.0] * 20, seed=42)
        assert result.mean_delta == 0.0
        assert result.p_value == 1.0

    def test_same_seed_is_reproducible(self):
        deltas = [0.1, -0.05, 0.2, 0.0, 0.07]
        a = paired_bootstrap(deltas, seed=123)
        b = paired_bootstrap(deltas, seed=123)
        assert a == b

    def test_different_seed_changes_resamples_not_mean(self):
        deltas = [0.1, -0.05, 0.2, 0.0, 0.07]
        a = paired_bootstrap(deltas, seed=1)
        b = paired_bootstrap(deltas, seed=2)
        assert a.mean_delta == b.mean_delta
        assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)

    def test_empty_deltas_raise(self):
        with pytest.raises(ValueError, match="at least one delta"):
            paired_bootstrap([], seed=42)

    def test_ci_bounds_are_ordered(self):
        result = paired_bootstrap([0.1, 0.2, -0.1, 0.05], seed=42)
        assert result.ci_low <= result.mean_delta <= result.ci_high


class TestCaseFlakiness:
    def test_all_samples_agree_pass(self):
        assert case_flakiness([True, True, True]) == 0.0

    def test_all_samples_agree_fail(self):
        assert case_flakiness([False, False]) == 0.0

    def test_one_dissenter_of_five(self):
        assert case_flakiness([True, True, True, True, False]) == pytest.approx(0.2)

    def test_even_split_is_half(self):
        assert case_flakiness([True, False, True, False]) == pytest.approx(0.5)

    def test_single_sample_is_zero(self):
        assert case_flakiness([True]) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one sample"):
            case_flakiness([])
```

- [ ] **Step 2: Verificar que fallan**

Run: `pytest tests/test_gate_statistics.py -v`
Expected: FAIL con `ModuleNotFoundError` / `ImportError` sobre `src.gate.statistics`.

- [ ] **Step 3: Implementar**

`src/gate/statistics.py`:

```python
"""Pure statistical functions for the regression gate: paired bootstrap and flakiness."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.gate.models import BootstrapResult


def paired_bootstrap(deltas: Sequence[float], *, n_resamples: int = 10_000, seed: int) -> BootstrapResult:
    """Bootstrap the mean of paired per-case regression deltas (positive = worse).

    Returns the observed mean, a 95% percentile CI and a one-sided p-value for
    H1 "mean regression > 0", computed as the fraction of resampled means <= 0.
    """
    if len(deltas) == 0:
        raise ValueError("paired_bootstrap requires at least one delta")
    arr = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    return BootstrapResult(
        mean_delta=float(arr.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        p_value=float(np.mean(means <= 0.0)),
    )


def case_flakiness(sample_passes: Sequence[bool]) -> float:
    """Fraction of samples whose pass/fail disagrees with the majority (an even split scores 0.5)."""
    if len(sample_passes) == 0:
        raise ValueError("case_flakiness requires at least one sample")
    n_pass = sum(1 for p in sample_passes if p)
    minority = min(n_pass, len(sample_passes) - n_pass)
    return minority / len(sample_passes)
```

- [ ] **Step 4: Verificar que pasan + calidad**

Run: `pytest tests/test_gate_statistics.py -v && ruff check src/gate tests/test_gate_statistics.py && ruff format src/gate tests/test_gate_statistics.py && mypy src/ --ignore-missing-imports`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gate/statistics.py tests/test_gate_statistics.py
git commit -m "feat: add paired bootstrap and flakiness statistics for the gate"
```

---

### Task 3: Construcción, persistencia y carga de baselines

**Files:**
- Create: `src/gate/baseline.py`
- Create: `tests/gate_helpers.py` (helper compartido por los tests del gate)
- Test: `tests/test_gate_baseline.py`

**Interfaces:**
- Consumes: `BaselineCase`, `BaselineFile` (Task 1); `case_flakiness` (Task 2); `RunSummary`, `TestCase`, `TestResult`, `EvaluationResult` de `src/runner/models.py`.
- Produces:
  - `BaselineError(Exception)`
  - `compute_dataset_hash(test_cases: Sequence[TestCase]) -> str`
  - `build_baseline(summaries: Sequence[RunSummary]) -> BaselineFile`
  - `save_baseline(baseline: BaselineFile, path: str) -> str`
  - `load_baseline(path: str) -> BaselineFile`
  - Helper de tests: `make_summary(case_scores: dict[str, dict[str, float]], *, passed: dict[str, bool] | None = None, severities: dict[str, str] | None = None, run_id: str = "run_test", chatbot_mode: str = "plain") -> RunSummary` en `tests/gate_helpers.py`.

- [ ] **Step 1: Escribir el helper de tests**

`tests/gate_helpers.py`:

```python
"""Helpers to build synthetic RunSummary objects for gate tests."""

from __future__ import annotations

from src.runner.models import EvaluationResult, RunSummary, TestCase, TestResult


def make_summary(
    case_scores: dict[str, dict[str, float]],
    *,
    passed: dict[str, bool] | None = None,
    severities: dict[str, str] | None = None,
    run_id: str = "run_test",
    chatbot_mode: str = "plain",
) -> RunSummary:
    """Build a minimal RunSummary. ``case_scores`` maps case_id -> {metric -> score}.

    Metric names may be plain evaluator names ("rule_based") or dotted
    sub-metrics ("ragas.answer_relevancy"), mirroring the gate's flattening.
    Cases default to passed=True and severity="medium" unless overridden.
    """
    passed = passed or {}
    severities = severities or {}
    results = []
    for case_id, metrics in case_scores.items():
        evaluations: dict[str, EvaluationResult] = {}
        for name, score in metrics.items():
            if "." in name:
                evaluator, sub = name.split(".", 1)
                ev = evaluations.setdefault(
                    evaluator,
                    EvaluationResult(evaluator=evaluator, passed=True, score=None, details={"metric_scores": {}}),
                )
                ev.details["metric_scores"][sub] = score
            else:
                evaluations[name] = EvaluationResult(evaluator=name, passed=True, score=score)
        test_case = TestCase(
            id=case_id,
            category="functional",
            input="q",
            expected_behavior="answers",
            evaluation_type=["rule_based"],
            severity=severities.get(case_id, "medium"),
        )
        results.append(
            TestResult(
                test_case=test_case,
                response="a",
                chatbot_mode=chatbot_mode,
                latency_ms=100.0,
                evaluations=list(evaluations.values()),
                overall_passed=passed.get(case_id, True),
                overall_score=None,
            )
        )
    n_passed = sum(1 for r in results if r.overall_passed)
    return RunSummary(
        run_id=run_id,
        timestamp="2026-08-14T00:00:00+00:00",
        chatbot_id="mock/mock-plain-v1",
        chatbot_mode=chatbot_mode,
        total=len(results),
        passed=n_passed,
        failed=len(results) - n_passed,
        results=results,
    )
```

- [ ] **Step 2: Escribir los tests que fallan**

`tests/test_gate_baseline.py`:

```python
"""Tests for baseline building, hashing, saving and loading."""

from __future__ import annotations

import pytest

from src.gate.baseline import (
    BaselineError,
    build_baseline,
    compute_dataset_hash,
    load_baseline,
    save_baseline,
)
from tests.gate_helpers import make_summary


class TestComputeDatasetHash:
    def test_hash_is_stable_and_order_independent(self):
        s1 = make_summary({"a": {"rule_based": 1.0}, "b": {"rule_based": 0.5}})
        s2 = make_summary({"b": {"rule_based": 0.5}, "a": {"rule_based": 1.0}})
        cases1 = [r.test_case for r in s1.results]
        cases2 = [r.test_case for r in reversed(s2.results)]
        assert compute_dataset_hash(cases1) == compute_dataset_hash(cases2)

    def test_hash_changes_when_a_case_changes(self):
        s1 = make_summary({"a": {"rule_based": 1.0}})
        s2 = make_summary({"a": {"rule_based": 1.0}}, severities={"a": "critical"})
        h1 = compute_dataset_hash([r.test_case for r in s1.results])
        h2 = compute_dataset_hash([r.test_case for r in s2.results])
        assert h1 != h2


class TestBuildBaseline:
    def test_single_summary_flattens_metrics(self):
        summary = make_summary({"a": {"rule_based": 1.0, "ragas.answer_relevancy": 0.8}})
        baseline = build_baseline([summary])
        assert baseline.samples == 1
        case = baseline.cases[0]
        assert case.metrics == {"rule_based": 1.0, "ragas.answer_relevancy": 0.8}
        assert case.flakiness == 0.0
        assert case.pass_samples == [True]
        assert "pass_rate" in baseline.metric_set
        assert "ragas.answer_relevancy" in baseline.metric_set

    def test_multi_sample_means_variance_and_flakiness(self):
        s1 = make_summary({"a": {"rule_based": 0.8}}, passed={"a": True}, run_id="r1")
        s2 = make_summary({"a": {"rule_based": 0.4}}, passed={"a": False}, run_id="r2")
        s3 = make_summary({"a": {"rule_based": 0.6}}, passed={"a": True}, run_id="r3")
        baseline = build_baseline([s1, s2, s3])
        case = baseline.cases[0]
        assert baseline.samples == 3
        assert baseline.run_ids == ["r1", "r2", "r3"]
        assert case.metrics["rule_based"] == pytest.approx(0.6)
        assert case.metric_variance["rule_based"] == pytest.approx(0.026667, abs=1e-4)
        assert case.passed is True  # majority 2/3
        assert case.flakiness == pytest.approx(1 / 3)

    def test_tie_counts_as_passed(self):
        s1 = make_summary({"a": {"rule_based": 1.0}}, passed={"a": True}, run_id="r1")
        s2 = make_summary({"a": {"rule_based": 1.0}}, passed={"a": False}, run_id="r2")
        baseline = build_baseline([s1, s2])
        assert baseline.cases[0].passed is True
        assert baseline.cases[0].flakiness == pytest.approx(0.5)

    def test_empty_summaries_raise(self):
        with pytest.raises(BaselineError, match="At least one"):
            build_baseline([])

    def test_mismatched_case_ids_raise(self):
        s1 = make_summary({"a": {"rule_based": 1.0}}, run_id="r1")
        s2 = make_summary({"b": {"rule_based": 1.0}}, run_id="r2")
        with pytest.raises(BaselineError, match="same test case ids"):
            build_baseline([s1, s2])

    def test_mismatched_modes_raise(self):
        s1 = make_summary({"a": {"rule_based": 1.0}}, run_id="r1", chatbot_mode="plain")
        s2 = make_summary({"a": {"rule_based": 1.0}}, run_id="r2", chatbot_mode="rag")
        with pytest.raises(BaselineError, match="chatbot_mode"):
            build_baseline([s1, s2])


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        baseline = build_baseline([make_summary({"a": {"rule_based": 1.0}})])
        path = save_baseline(baseline, str(tmp_path / "baselines" / "main.json"))
        assert load_baseline(path) == baseline

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(BaselineError, match="not found"):
            load_baseline(str(tmp_path / "nope.json"))

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(BaselineError, match="Invalid baseline"):
            load_baseline(str(path))
```

- [ ] **Step 3: Verificar que fallan**

Run: `pytest tests/test_gate_baseline.py -v`
Expected: FAIL con `ModuleNotFoundError` sobre `src.gate.baseline`.

- [ ] **Step 4: Implementar**

`src/gate/baseline.py`:

```python
"""Build, persist and load regression-gate baselines from run summaries."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence

from src.gate.models import PASS_RATE_METRIC, BaselineCase, BaselineFile
from src.gate.statistics import case_flakiness
from src.runner.models import RunSummary, TestCase, TestResult


class BaselineError(Exception):
    """Raised when a baseline cannot be built, loaded or validated."""


def compute_dataset_hash(test_cases: Sequence[TestCase]) -> str:
    """SHA-256 over the canonical JSON of the test cases, sorted by id."""
    canonical = json.dumps(
        [tc.model_dump() for tc in sorted(test_cases, key=lambda t: t.id)],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _case_metrics(result: TestResult) -> dict[str, float]:
    """Flatten evaluator scores into {metric_name: score} using the gate naming convention."""
    metrics: dict[str, float] = {}
    for ev in result.evaluations:
        if ev.score is not None:
            metrics[ev.evaluator] = ev.score
        metric_scores = ev.details.get("metric_scores") or {}
        for name, score in metric_scores.items():
            metrics[f"{ev.evaluator}.{name}"] = float(score)
    return metrics


def build_baseline(summaries: Sequence[RunSummary]) -> BaselineFile:
    """Aggregate N sampled runs over the same dataset into one BaselineFile."""
    if not summaries:
        raise BaselineError("At least one run summary is required")
    first = summaries[0]
    ids = {r.test_case.id for r in first.results}
    for s in summaries[1:]:
        if s.chatbot_mode != first.chatbot_mode:
            raise BaselineError("All samples must share the same chatbot_mode")
        if {r.test_case.id for r in s.results} != ids:
            raise BaselineError("All samples must cover the same test case ids")

    per_case: dict[str, list[TestResult]] = {}
    for s in summaries:
        for r in s.results:
            per_case.setdefault(r.test_case.id, []).append(r)

    cases: list[BaselineCase] = []
    metric_set: set[str] = {PASS_RATE_METRIC}
    for case_id in sorted(per_case):
        results = per_case[case_id]
        test_case = results[0].test_case
        sample_metrics = [_case_metrics(r) for r in results]
        metric_names = sorted({name for m in sample_metrics for name in m})
        means: dict[str, float] = {}
        variances: dict[str, float] = {}
        for name in metric_names:
            values = [m[name] for m in sample_metrics if name in m]
            mean = sum(values) / len(values)
            means[name] = round(mean, 6)
            variances[name] = round(sum((v - mean) ** 2 for v in values) / len(values), 6)
        pass_samples = [r.overall_passed for r in results]
        latencies = [r.latency_ms for r in results]
        metric_set.update(metric_names)
        cases.append(
            BaselineCase(
                id=case_id,
                category=test_case.category,
                severity=test_case.severity,
                passed=sum(pass_samples) * 2 >= len(pass_samples),
                pass_samples=pass_samples,
                flakiness=case_flakiness(pass_samples),
                metrics=means,
                metric_variance=variances,
                latency_ms_mean=round(sum(latencies) / len(latencies), 2),
            )
        )

    return BaselineFile(
        run_ids=[s.run_id for s in summaries],
        timestamp=first.timestamp,
        chatbot_id=first.chatbot_id,
        chatbot_mode=first.chatbot_mode,
        dataset_hash=compute_dataset_hash([r.test_case for r in first.results]),
        metric_set=sorted(metric_set),
        samples=len(summaries),
        cases=cases,
    )


def save_baseline(baseline: BaselineFile, path: str) -> str:
    """Write the baseline as indented JSON (small, diffable). Returns the path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(baseline.model_dump_json(indent=2))
    return path


def load_baseline(path: str) -> BaselineFile:
    """Load and validate a baseline file. Raises BaselineError on any problem."""
    if not os.path.exists(path):
        raise BaselineError(f"Baseline file not found: {path}")
    try:
        with open(path) as f:
            return BaselineFile.model_validate_json(f.read())
    except Exception as e:
        raise BaselineError(f"Invalid baseline file {path}: {e}") from e
```

- [ ] **Step 5: Verificar que pasan + calidad**

Run: `pytest tests/test_gate_baseline.py -v && ruff check src/gate tests/ && ruff format src/gate tests/gate_helpers.py tests/test_gate_baseline.py && mypy src/ --ignore-missing-imports`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gate/baseline.py tests/gate_helpers.py tests/test_gate_baseline.py
git commit -m "feat: build, save and load regression-gate baselines"
```

---

### Task 4: Pareo de casos y comparación estadística por métrica

**Files:**
- Create: `src/gate/comparison.py`
- Test: `tests/test_gate_comparison.py`

**Interfaces:**
- Consumes: modelos de Task 1; `paired_bootstrap` (Task 2); `build_baseline`/`make_summary` (Task 3) en tests.
- Produces:
  - `CompatibilityError(Exception)`
  - `CasePair = tuple[BaselineCase, BaselineCase]`
  - `validate_compatibility(baseline: BaselineFile, current: BaselineFile) -> None`
  - `pair_cases(baseline: BaselineFile, current: BaselineFile) -> tuple[list[CasePair], list[str], list[str]]` (pairs, new_ids, removed_ids)
  - `regression_deltas(pairs: list[CasePair], metric: str) -> list[float]`
  - `compare_metrics(baseline: BaselineFile, current: BaselineFile, policy: GatePolicy) -> list[MetricComparison]`

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_gate_comparison.py`:

```python
"""Tests for case pairing and per-metric statistical comparison."""

from __future__ import annotations

import pytest

from src.gate.baseline import build_baseline
from src.gate.comparison import (
    CompatibilityError,
    compare_metrics,
    pair_cases,
    regression_deltas,
    validate_compatibility,
)
from src.gate.models import GatePolicy, MetricPolicy
from tests.gate_helpers import make_summary


def _baseline(case_scores, **kwargs):
    return build_baseline([make_summary(case_scores, **kwargs)])


class TestValidateCompatibility:
    def test_same_mode_and_shared_metrics_ok(self):
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 0.9}})
        validate_compatibility(base, curr)  # no raise

    def test_mode_mismatch_raises(self):
        base = _baseline({"a": {"rule_based": 1.0}}, chatbot_mode="plain")
        curr = _baseline({"a": {"rule_based": 1.0}}, chatbot_mode="rag")
        with pytest.raises(CompatibilityError, match="chatbot_mode"):
            validate_compatibility(base, curr)

    def test_pass_rate_always_shared(self):
        # Even with disjoint evaluator metrics, pass_rate exists on both sides.
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"llm_judge": 0.8}})
        validate_compatibility(base, curr)  # no raise


class TestPairCases:
    def test_pairs_new_and_removed(self):
        base = _baseline({"a": {"rule_based": 1.0}, "b": {"rule_based": 0.9}})
        curr = _baseline({"b": {"rule_based": 0.8}, "c": {"rule_based": 0.7}})
        pairs, new_ids, removed_ids = pair_cases(base, curr)
        assert [(p[0].id, p[1].id) for p in pairs] == [("b", "b")]
        assert new_ids == ["c"]
        assert removed_ids == ["a"]


class TestRegressionDeltas:
    def test_higher_is_better_direction(self):
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 0.8}})
        pairs, _, _ = pair_cases(base, curr)
        assert regression_deltas(pairs, "rule_based") == [pytest.approx(0.2)]

    def test_lower_is_better_direction(self):
        base = _baseline({"a": {"deepeval.toxicity": 0.1}})
        curr = _baseline({"a": {"deepeval.toxicity": 0.4}})
        pairs, _, _ = pair_cases(base, curr)
        # toxicity went UP -> regression positive
        assert regression_deltas(pairs, "deepeval.toxicity") == [pytest.approx(0.3)]

    def test_pass_rate_deltas_from_passed_flags(self):
        base = _baseline({"a": {"rule_based": 1.0}, "b": {"rule_based": 1.0}}, passed={"a": True, "b": True})
        curr = _baseline({"a": {"rule_based": 1.0}, "b": {"rule_based": 1.0}}, passed={"a": True, "b": False})
        pairs, _, _ = pair_cases(base, curr)
        assert sorted(regression_deltas(pairs, "pass_rate")) == [0.0, 1.0]

    def test_cases_missing_the_metric_are_skipped(self):
        base = _baseline({"a": {"rule_based": 1.0}, "b": {"llm_judge": 0.9}})
        curr = _baseline({"a": {"rule_based": 0.9}, "b": {"llm_judge": 0.9}})
        pairs, _, _ = pair_cases(base, curr)
        assert len(regression_deltas(pairs, "rule_based")) == 1


class TestCompareMetrics:
    def test_identical_runs_show_no_regression(self):
        scores = {f"c{i}": {"rule_based": 0.9} for i in range(10)}
        base = _baseline(scores)
        curr = _baseline(scores)
        comparisons = compare_metrics(base, curr, GatePolicy())
        by_name = {c.metric: c for c in comparisons}
        assert by_name["rule_based"].regression == 0.0
        assert by_name["rule_based"].p_value == 1.0
        assert not by_name["rule_based"].breaches
        assert by_name["pass_rate"].gated is True
        assert by_name["rule_based"].gated is False

    def test_clear_gated_regression_breaches(self):
        base = _baseline({f"c{i}": {"rule_based": 0.9} for i in range(20)})
        curr = _baseline({f"c{i}": {"rule_based": 0.5} for i in range(20)})
        policy = GatePolicy(metrics={"rule_based": MetricPolicy(max_regression=0.1)})
        comparisons = compare_metrics(base, curr, policy)
        rule = next(c for c in comparisons if c.metric == "rule_based")
        assert rule.significant is True
        assert rule.breaches is True
        assert rule.n_cases == 20

    def test_ungated_regression_reported_but_never_breaches(self):
        base = _baseline({f"c{i}": {"rule_based": 0.9} for i in range(20)})
        curr = _baseline({f"c{i}": {"rule_based": 0.5} for i in range(20)})
        comparisons = compare_metrics(base, curr, GatePolicy())  # rule_based not in policy
        rule = next(c for c in comparisons if c.metric == "rule_based")
        assert rule.significant is True
        assert rule.breaches is False

    def test_small_effect_filtered_by_min_effect_size(self):
        base = _baseline({f"c{i}": {"rule_based": 0.90} for i in range(30)})
        curr = _baseline({f"c{i}": {"rule_based": 0.88} for i in range(30)})
        policy = GatePolicy(metrics={"rule_based": MetricPolicy(max_regression=0.01)})
        comparisons = compare_metrics(base, curr, policy)
        rule = next(c for c in comparisons if c.metric == "rule_based")
        # delta constante 0.02 es "significativo" pero < min_effect_size (0.05)
        assert rule.breaches is False
```

- [ ] **Step 2: Verificar que fallan**

Run: `pytest tests/test_gate_comparison.py -v`
Expected: FAIL con `ModuleNotFoundError` sobre `src.gate.comparison`.

- [ ] **Step 3: Implementar**

`src/gate/comparison.py`:

```python
"""Pair baseline vs current cases and compute per-metric statistical comparisons."""

from __future__ import annotations

from src.gate.models import (
    LOWER_IS_BETTER_METRICS,
    PASS_RATE_METRIC,
    BaselineCase,
    BaselineFile,
    GatePolicy,
    MetricComparison,
)
from src.gate.statistics import paired_bootstrap


class CompatibilityError(Exception):
    """Raised when two runs cannot be meaningfully compared."""


CasePair = tuple[BaselineCase, BaselineCase]


def validate_compatibility(baseline: BaselineFile, current: BaselineFile) -> None:
    """Fail fast when runs are not comparable (spec §4: exit 2 at the CLI layer)."""
    if baseline.chatbot_mode != current.chatbot_mode:
        raise CompatibilityError(
            f"chatbot_mode mismatch: baseline={baseline.chatbot_mode}, current={current.chatbot_mode}"
        )
    if not set(baseline.metric_set) & set(current.metric_set):
        raise CompatibilityError("No shared metrics between baseline and current run")


def pair_cases(baseline: BaselineFile, current: BaselineFile) -> tuple[list[CasePair], list[str], list[str]]:
    """Pair cases by id. Returns (pairs, new_case_ids, removed_case_ids)."""
    base_by_id = {c.id: c for c in baseline.cases}
    curr_by_id = {c.id: c for c in current.cases}
    pairs = [(base_by_id[i], curr_by_id[i]) for i in sorted(base_by_id.keys() & curr_by_id.keys())]
    new_ids = sorted(curr_by_id.keys() - base_by_id.keys())
    removed_ids = sorted(base_by_id.keys() - curr_by_id.keys())
    return pairs, new_ids, removed_ids


def regression_deltas(pairs: list[CasePair], metric: str) -> list[float]:
    """Per-case regression (positive = worse), skipping pairs missing the metric."""
    deltas: list[float] = []
    for base, curr in pairs:
        if metric == PASS_RATE_METRIC:
            deltas.append(float(base.passed) - float(curr.passed))
            continue
        if metric not in base.metrics or metric not in curr.metrics:
            continue
        if metric in LOWER_IS_BETTER_METRICS:
            deltas.append(curr.metrics[metric] - base.metrics[metric])
        else:
            deltas.append(base.metrics[metric] - curr.metrics[metric])
    return deltas


def _metric_mean(cases: list[BaselineCase], metric: str) -> float:
    if metric == PASS_RATE_METRIC:
        return sum(1.0 for c in cases if c.passed) / len(cases) if cases else 0.0
    values = [c.metrics[metric] for c in cases if metric in c.metrics]
    return sum(values) / len(values) if values else 0.0


def compare_metrics(baseline: BaselineFile, current: BaselineFile, policy: GatePolicy) -> list[MetricComparison]:
    """Compare every metric shared by both runs; only policy-listed metrics can breach."""
    pairs, _, _ = pair_cases(baseline, current)
    comparisons: list[MetricComparison] = []
    for metric in sorted(set(baseline.metric_set) & set(current.metric_set)):
        deltas = regression_deltas(pairs, metric)
        if not deltas:
            continue
        boot = paired_bootstrap(deltas, n_resamples=policy.n_resamples, seed=policy.seed)
        significant = boot.p_value < policy.significance_level
        metric_policy = policy.metrics.get(metric)
        breaches = False
        if metric_policy is not None and significant:
            breaches = boot.mean_delta > policy.min_effect_size and boot.mean_delta > metric_policy.max_regression
        comparisons.append(
            MetricComparison(
                metric=metric,
                baseline_mean=round(_metric_mean(baseline.cases, metric), 4),
                current_mean=round(_metric_mean(current.cases, metric), 4),
                regression=round(boot.mean_delta, 4),
                ci_low=round(boot.ci_low, 4),
                ci_high=round(boot.ci_high, 4),
                p_value=round(boot.p_value, 4),
                n_cases=len(deltas),
                significant=significant,
                gated=metric_policy is not None,
                breaches=breaches,
            )
        )
    return comparisons
```

- [ ] **Step 4: Verificar que pasan + calidad**

Run: `pytest tests/test_gate_comparison.py -v && ruff check src/gate tests/test_gate_comparison.py && ruff format src/gate tests/test_gate_comparison.py && mypy src/ --ignore-missing-imports`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gate/comparison.py tests/test_gate_comparison.py
git commit -m "feat: pair cases and compare metrics with direction-aware bootstrap"
```

---

### Task 5: Política de gate y veredicto

**Files:**
- Create: `src/gate/policy.py`
- Create: `config/gate.yaml` (ejemplo documentado de la política)
- Test: `tests/test_gate_policy.py`

**Interfaces:**
- Consumes: modelos (Task 1); `validate_compatibility`, `pair_cases`, `compare_metrics`, `CompatibilityError` (Task 4); `build_baseline`/`make_summary` (Task 3) en tests.
- Produces:
  - `PolicyError(Exception)`
  - `load_policy(path: str | None) -> GatePolicy` (None → defaults empaquetados; el YAML puede llevar clave raíz `gate:` o los campos directamente)
  - `evaluate_gate(baseline: BaselineFile, current: BaselineFile, policy: GatePolicy) -> GateVerdict` (propaga `CompatibilityError`)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_gate_policy.py`:

```python
"""Tests for policy loading and full gate evaluation."""

from __future__ import annotations

import pytest

from src.gate.baseline import build_baseline
from src.gate.comparison import CompatibilityError
from src.gate.models import GatePolicy, HardRules, MetricPolicy
from src.gate.policy import PolicyError, evaluate_gate, load_policy
from tests.gate_helpers import make_summary


def _baseline(case_scores, **kwargs):
    return build_baseline([make_summary(case_scores, **kwargs)])


class TestLoadPolicy:
    def test_none_returns_defaults(self):
        assert load_policy(None) == GatePolicy()

    def test_loads_yaml_with_gate_key(self, tmp_path):
        path = tmp_path / "gate.yaml"
        path.write_text(
            "gate:\n"
            "  significance_level: 0.01\n"
            "  metrics:\n"
            "    rule_based: {max_regression: 0.2}\n"
            "  new_cases: fail\n"
        )
        policy = load_policy(str(path))
        assert policy.significance_level == 0.01
        assert policy.metrics == {"rule_based": MetricPolicy(max_regression=0.2)}
        assert policy.new_cases == "fail"
        assert policy.min_effect_size == 0.05  # default preserved

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(PolicyError, match="Cannot read"):
            load_policy(str(tmp_path / "nope.yaml"))

    def test_invalid_content_raises(self, tmp_path):
        path = tmp_path / "gate.yaml"
        path.write_text("gate:\n  new_cases: whatever\n")
        with pytest.raises(PolicyError, match="Invalid policy"):
            load_policy(str(path))


class TestEvaluateGate:
    def test_identical_runs_pass(self):
        scores = {f"c{i}": {"rule_based": 0.9} for i in range(10)}
        verdict = evaluate_gate(_baseline(scores), _baseline(scores), GatePolicy())
        assert verdict.passed is True
        assert verdict.hard_rule_violations == []
        assert verdict.missing_gated_metrics == []
        assert verdict.samples == 1

    def test_new_critical_failure_violates_hard_rule(self):
        base = _baseline(
            {"crit": {"rule_based": 1.0}, "ok": {"rule_based": 1.0}},
            passed={"crit": True, "ok": True},
            severities={"crit": "critical"},
        )
        curr = _baseline(
            {"crit": {"rule_based": 1.0}, "ok": {"rule_based": 1.0}},
            passed={"crit": False, "ok": True},
            severities={"crit": "critical"},
        )
        verdict = evaluate_gate(base, curr, GatePolicy())
        assert verdict.passed is False
        assert any("crit" in v for v in verdict.hard_rule_violations)

    def test_already_failing_critical_case_is_not_a_new_failure(self):
        base = _baseline({"crit": {"rule_based": 0.1}}, passed={"crit": False}, severities={"crit": "critical"})
        curr = _baseline({"crit": {"rule_based": 0.1}}, passed={"crit": False}, severities={"crit": "critical"})
        verdict = evaluate_gate(base, curr, GatePolicy())
        assert verdict.hard_rule_violations == []

    def test_new_cases_report_only_vs_fail(self):
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 1.0}, "b": {"rule_based": 1.0}})
        report_only = evaluate_gate(base, curr, GatePolicy())
        assert report_only.passed is True
        assert report_only.new_case_ids == ["b"]
        failing = evaluate_gate(base, curr, GatePolicy(new_cases="fail"))
        assert failing.passed is False
        assert any("b" in v for v in failing.hard_rule_violations)

    def test_missing_gated_metric_fails_verdict(self):
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = _baseline({"a": {"rule_based": 1.0}})
        policy = GatePolicy(metrics={"ragas.answer_relevancy": MetricPolicy(max_regression=0.1)})
        verdict = evaluate_gate(base, curr, policy)
        assert verdict.missing_gated_metrics == ["ragas.answer_relevancy"]
        assert verdict.passed is False

    def test_flakiness_hard_rule(self):
        s1 = make_summary({"a": {"rule_based": 1.0}}, passed={"a": True}, run_id="r1")
        s2 = make_summary({"a": {"rule_based": 1.0}}, passed={"a": False}, run_id="r2")
        base = _baseline({"a": {"rule_based": 1.0}})
        curr = build_baseline([s1, s2])  # flakiness 0.5 > 0.3
        verdict = evaluate_gate(base, curr, GatePolicy())
        assert verdict.mean_flakiness == pytest.approx(0.5)
        assert any("flakiness" in v.lower() for v in verdict.hard_rule_violations)
        relaxed = evaluate_gate(base, curr, GatePolicy(hard_rules=HardRules(max_flakiness=0.6)))
        assert relaxed.hard_rule_violations == []

    def test_incompatible_runs_propagate(self):
        base = _baseline({"a": {"rule_based": 1.0}}, chatbot_mode="plain")
        curr = _baseline({"a": {"rule_based": 1.0}}, chatbot_mode="rag")
        with pytest.raises(CompatibilityError):
            evaluate_gate(base, curr, GatePolicy())
```

- [ ] **Step 2: Verificar que fallan**

Run: `pytest tests/test_gate_policy.py -v`
Expected: FAIL con `ModuleNotFoundError` sobre `src.gate.policy`.

- [ ] **Step 3: Implementar**

`src/gate/policy.py`:

```python
"""Load the gate policy and produce the final verdict."""

from __future__ import annotations

import yaml

from src.gate.comparison import compare_metrics, pair_cases, validate_compatibility
from src.gate.models import BaselineFile, GatePolicy, GateVerdict


class PolicyError(Exception):
    """Raised when the gate policy file is missing or invalid."""


def load_policy(path: str | None) -> GatePolicy:
    """Load a gate policy YAML (top-level ``gate:`` key optional). None -> built-in defaults."""
    if path is None:
        return GatePolicy()
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except OSError as e:
        raise PolicyError(f"Cannot read policy file {path}: {e}") from e
    try:
        return GatePolicy.model_validate(data.get("gate", data))
    except Exception as e:
        raise PolicyError(f"Invalid policy file {path}: {e}") from e


def evaluate_gate(baseline: BaselineFile, current: BaselineFile, policy: GatePolicy) -> GateVerdict:
    """Full gate evaluation. Raises CompatibilityError for non-comparable runs."""
    validate_compatibility(baseline, current)
    pairs, new_ids, removed_ids = pair_cases(baseline, current)
    comparisons = compare_metrics(baseline, current, policy)

    compared = {c.metric for c in comparisons}
    missing_gated = sorted(m for m in policy.metrics if m not in compared)

    violations: list[str] = []
    if policy.hard_rules.no_new_critical_failures:
        broken = [b.id for b, c in pairs if b.severity == "critical" and b.passed and not c.passed]
        if broken:
            violations.append(f"New critical failures: {', '.join(broken)}")
    mean_flakiness = sum(c.flakiness for c in current.cases) / len(current.cases) if current.cases else 0.0
    if mean_flakiness > policy.hard_rules.max_flakiness:
        violations.append(f"Mean flakiness {mean_flakiness:.2f} exceeds limit {policy.hard_rules.max_flakiness:.2f}")
    if new_ids and policy.new_cases == "fail":
        violations.append(f"Cases without baseline: {', '.join(new_ids)}")

    passed = not violations and not missing_gated and not any(c.breaches for c in comparisons)
    return GateVerdict(
        passed=passed,
        comparisons=comparisons,
        hard_rule_violations=violations,
        missing_gated_metrics=missing_gated,
        new_case_ids=new_ids,
        removed_case_ids=removed_ids,
        mean_flakiness=round(mean_flakiness, 4),
        samples=current.samples,
    )
```

`config/gate.yaml` (ejemplo con los defaults, para que los usuarios lo copien y ajusten):

```yaml
# Regression gate policy. Used by `llm-eval-lab check --policy config/gate.yaml`.
# Without --policy, these same built-in defaults apply.
gate:
  significance_level: 0.05   # p-value threshold for the paired bootstrap
  min_effect_size: 0.05      # regressions smaller than this never break the build
  n_resamples: 10000
  seed: 42                   # bootstrap seed, for reproducible CI verdicts

  # Only metrics listed here can break the build. Everything else is report-only.
  # Metric names: pass_rate, <evaluator> (e.g. rule_based), <evaluator>.<metric>
  # (e.g. ragas.answer_relevancy, deepeval.toxicity).
  metrics:
    pass_rate: {max_regression: 0.05}

  hard_rules:
    no_new_critical_failures: true   # a newly failing critical case always fails the gate
    max_flakiness: 0.3               # mean per-case flakiness across the run

  new_cases: report_only   # report_only | fail — cases present now but absent in the baseline
```

- [ ] **Step 4: Verificar que pasan + calidad**

Run: `pytest tests/test_gate_policy.py -v && ruff check src/gate tests/test_gate_policy.py && ruff format src/gate tests/test_gate_policy.py && mypy src/ --ignore-missing-imports`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gate/policy.py config/gate.yaml tests/test_gate_policy.py
git commit -m "feat: gate policy loading and verdict evaluation with hard rules"
```

---

### Task 6: Reporter del gate (consola rich + Markdown para PR)

**Files:**
- Create: `src/reporting/gate_reporter.py`
- Test: `tests/test_gate_reporter.py`

**Interfaces:**
- Consumes: `GateVerdict`, `MetricComparison` (Task 1).
- Produces:
  - `render_gate_console(verdict: GateVerdict, console: Console) -> None`
  - `generate_gate_markdown(verdict: GateVerdict, output_dir: str) -> str` (escribe `gate_report.md`, hace append a `$GITHUB_STEP_SUMMARY` si está definido, devuelve la ruta)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_gate_reporter.py`:

```python
"""Tests for the gate console and Markdown reporters."""

from __future__ import annotations

import os

from rich.console import Console

from src.gate.models import GateVerdict, MetricComparison
from src.reporting.gate_reporter import generate_gate_markdown, render_gate_console


def _verdict(*, passed: bool = True, samples: int = 1) -> GateVerdict:
    return GateVerdict(
        passed=passed,
        comparisons=[
            MetricComparison(
                metric="pass_rate",
                baseline_mean=0.9,
                current_mean=0.8,
                regression=0.1,
                ci_low=0.02,
                ci_high=0.18,
                p_value=0.01,
                n_cases=40,
                significant=True,
                gated=True,
                breaches=not passed,
            )
        ],
        hard_rule_violations=[] if passed else ["New critical failures: safety_001"],
        missing_gated_metrics=[],
        new_case_ids=["new_1"],
        removed_case_ids=[],
        mean_flakiness=0.05,
        samples=samples,
    )


class TestConsoleReporter:
    def test_renders_table_and_verdict(self):
        console = Console(record=True, width=200)
        render_gate_console(_verdict(passed=False), console)
        output = console.export_text()
        assert "pass_rate" in output
        assert "FAIL" in output
        assert "New critical failures" in output
        assert "new_1" in output

    def test_single_sample_warning(self):
        console = Console(record=True, width=200)
        render_gate_console(_verdict(samples=1), console)
        assert "low statistical power" in console.export_text()

    def test_no_warning_with_multiple_samples(self):
        console = Console(record=True, width=200)
        render_gate_console(_verdict(samples=3), console)
        assert "low statistical power" not in console.export_text()


class TestMarkdownReporter:
    def test_writes_markdown_table(self, tmp_path):
        path = generate_gate_markdown(_verdict(passed=False), str(tmp_path))
        assert os.path.basename(path) == "gate_report.md"
        content = open(path).read()
        assert "# Regression gate: ❌ FAIL" in content
        assert "| pass_rate |" in content
        assert "New critical failures: safety_001" in content
        assert "new_1" in content

    def test_appends_to_github_step_summary(self, tmp_path, monkeypatch):
        summary_file = tmp_path / "step_summary.md"
        summary_file.write_text("previous content\n")
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
        generate_gate_markdown(_verdict(), str(tmp_path / "out"))
        content = summary_file.read_text()
        assert content.startswith("previous content\n")
        assert "# Regression gate: ✅ PASS" in content

    def test_no_step_summary_env_is_fine(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        path = generate_gate_markdown(_verdict(), str(tmp_path))
        assert os.path.exists(path)
```

- [ ] **Step 2: Verificar que fallan**

Run: `pytest tests/test_gate_reporter.py -v`
Expected: FAIL con `ModuleNotFoundError` sobre `src.reporting.gate_reporter`.

- [ ] **Step 3: Implementar**

`src/reporting/gate_reporter.py`:

```python
"""Gate verdict reporters — rich console table and Markdown for PR comments."""

from __future__ import annotations

import os

from rich.console import Console
from rich.table import Table

from src.gate.models import GateVerdict

_STATUS = {True: "✅ PASS", False: "❌ FAIL"}
_LOW_POWER_NOTE = "Note: --samples 1 has low statistical power; use --samples 3 or more for reliable significance."


def render_gate_console(verdict: GateVerdict, console: Console) -> None:
    """Print the verdict as a rich table plus hard-rule and case-diff notes."""
    table = Table(title=f"Regression gate — {_STATUS[verdict.passed]} (samples: {verdict.samples})")
    for column in ("Metric", "Baseline", "Current", "Regression", "95% CI", "p-value", "Gated", "Verdict"):
        table.add_column(column)
    for c in verdict.comparisons:
        table.add_row(
            c.metric,
            f"{c.baseline_mean:.4f}",
            f"{c.current_mean:.4f}",
            f"{c.regression:+.4f}",
            f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]",
            f"{c.p_value:.4f}",
            "yes" if c.gated else "no",
            "❌ regression" if c.breaches else "✅ ok",
        )
    console.print(table)
    for violation in verdict.hard_rule_violations:
        console.print(f"[red]Hard rule violated:[/red] {violation}")
    for metric in verdict.missing_gated_metrics:
        console.print(f"[red]Gated metric not comparable:[/red] {metric}")
    if verdict.new_case_ids:
        console.print(f"[yellow]Cases without baseline:[/yellow] {', '.join(verdict.new_case_ids)}")
    if verdict.removed_case_ids:
        console.print(f"[yellow]Cases removed since baseline:[/yellow] {', '.join(verdict.removed_case_ids)}")
    if verdict.samples == 1:
        console.print(f"[yellow]{_LOW_POWER_NOTE}[/yellow]")


def generate_gate_markdown(verdict: GateVerdict, output_dir: str) -> str:
    """Write gate_report.md (and append to $GITHUB_STEP_SUMMARY when set). Returns the path."""
    lines = [
        f"# Regression gate: {_STATUS[verdict.passed]}",
        "",
        f"Samples per case: {verdict.samples} · Mean flakiness: {verdict.mean_flakiness:.2f}",
        "",
        "| Metric | Baseline | Current | Regression | 95% CI | p-value | Gated | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in verdict.comparisons:
        lines.append(
            f"| {c.metric} | {c.baseline_mean:.4f} | {c.current_mean:.4f} | {c.regression:+.4f} "
            f"| [{c.ci_low:+.4f}, {c.ci_high:+.4f}] | {c.p_value:.4f} "
            f"| {'yes' if c.gated else 'no'} | {'❌ regression' if c.breaches else '✅ ok'} |"
        )
    if verdict.hard_rule_violations:
        lines += ["", "## Hard rule violations", ""] + [f"- {v}" for v in verdict.hard_rule_violations]
    if verdict.missing_gated_metrics:
        lines += ["", "## Gated metrics not comparable", ""] + [f"- {m}" for m in verdict.missing_gated_metrics]
    if verdict.new_case_ids:
        lines += ["", f"Cases without baseline: {', '.join(verdict.new_case_ids)}"]
    if verdict.removed_case_ids:
        lines += ["", f"Cases removed since baseline: {', '.join(verdict.removed_case_ids)}"]
    if verdict.samples == 1:
        lines += ["", f"_{_LOW_POWER_NOTE}_"]
    content = "\n".join(lines) + "\n"

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "gate_report.md")
    with open(path, "w") as f:
        f.write(content)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(content)
    return path
```

- [ ] **Step 4: Verificar que pasan + calidad**

Run: `pytest tests/test_gate_reporter.py -v && ruff check src/reporting tests/test_gate_reporter.py && ruff format src/reporting tests/test_gate_reporter.py && mypy src/ --ignore-missing-imports`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reporting/gate_reporter.py tests/test_gate_reporter.py
git commit -m "feat: gate console and Markdown reporters with GITHUB_STEP_SUMMARY support"
```

---

### Task 7: Modo de variabilidad en los mocks

**Files:**
- Modify: `src/chatbots/mock_adapter.py` (constructores de `MockChatbot` y `MockRAGChatbot`)
- Test: `tests/test_chatbots.py` (añadir una clase de tests al final)

**Interfaces:**
- Consumes: nada nuevo.
- Produces: parámetro opcional `transform: Callable[[str], str] | None = None` en `MockChatbot.__init__` y `MockRAGChatbot.__init__`; cuando está definido se aplica al `content` de cada respuesta. Determinista: la variación la controla el test que pasa el callable.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir al final de `tests/test_chatbots.py`:

```python
class TestMockTransform:
    async def test_transform_applies_to_plain_mock(self):
        bot = MockChatbot(transform=lambda text: text[:5])
        response = await bot.complete([{"role": "user", "content": "What is machine learning?"}])
        assert len(response.content) == 5

    async def test_transform_applies_to_rag_mock(self):
        bot = MockRAGChatbot(transform=lambda text: text + " EXTRA")
        response = await bot.complete([{"role": "user", "content": "What is python?"}])
        assert response.content.endswith(" EXTRA")
        assert response.retrieved_contexts  # retrieval untouched

    async def test_no_transform_keeps_default_behavior(self):
        bot = MockChatbot()
        response = await bot.complete([{"role": "user", "content": "capital of france"}])
        assert response.content == "The capital of France is Paris."
```

(Si `tests/test_chatbots.py` no importa ya `MockChatbot`/`MockRAGChatbot`, añadir el import correspondiente de `src.chatbots.mock_adapter` en la cabecera.)

- [ ] **Step 2: Verificar que fallan**

Run: `pytest tests/test_chatbots.py -k Transform -v`
Expected: FAIL con `TypeError: __init__() got an unexpected keyword argument 'transform'`.

- [ ] **Step 3: Implementar**

En `src/chatbots/mock_adapter.py`, añadir el import y cambiar ambos constructores y `complete`:

```python
from collections.abc import Callable
```

`MockChatbot`:

```python
    def __init__(
        self,
        latency_range: tuple[float, float] = (50.0, 200.0),
        transform: Callable[[str], str] | None = None,
    ) -> None:
        self._latency_range = latency_range
        self._transform = transform
```

y en su `complete`, sustituir `content = _find_response(last_user_msg)` por:

```python
        content = _find_response(last_user_msg)
        if self._transform is not None:
            content = self._transform(content)
```

Aplicar exactamente el mismo patrón en `MockRAGChatbot` (`__init__` con `latency_range: tuple[float, float] = (80.0, 300.0)` y `transform`, y la misma sustitución en su `complete`; el `retrieve` no cambia).

- [ ] **Step 4: Verificar que pasan (y que nada existente se rompe)**

Run: `pytest tests/test_chatbots.py -v && ruff check src/chatbots tests/test_chatbots.py && ruff format src/chatbots tests/test_chatbots.py && mypy src/ --ignore-missing-imports`
Expected: PASS completo, incluidos los tests preexistentes del archivo.

- [ ] **Step 5: Commit**

```bash
git add src/chatbots/mock_adapter.py tests/test_chatbots.py
git commit -m "feat: optional response transform on mock chatbots for gate testing"
```

---

### Task 8: CLI Typer con comando `run` (refactor del entry point)

**Files:**
- Create: `src/cli.py`
- Modify: `src/__main__.py` (queda como shim de la CLI)
- Modify: `pyproject.toml` (añadir `[project.scripts]`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `EvalRunner`, `load_all_datasets`, `load_dataset` (`src/runner/runner.py`); reporters JSON/Markdown existentes; chatbots y evaluadores existentes.
- Produces (para Tasks 9-11):
  - `app: typer.Typer` y `baseline_app: typer.Typer` en `src/cli.py`
  - `_execute_runs(provider: str | None, mode: str, samples: int, evaluators_csv: str | None, datasets_csv: str | None, results_dir: str) -> list[RunSummary]` (async)
  - `_load_summary(results_dir: str, run_id: str) -> RunSummary`
  - `_DEFAULT_RESULTS_DIR: str` (`<repo>/results`)
  - `main()` en `src/__main__.py` (entry point de consola; sin argumentos equivale a `run`)

**Nota:** `_build_chatbot` y `_build_evaluators` se MUEVEN desde `src/__main__.py` a `src/cli.py` sin cambios de lógica (nadie más los importa; `pages/1_run.py` tiene sus propias copias). `src/__main__.py` sigue en el omit de cobertura; `src/cli.py` NO se añade al omit — lo cubren los tests de esta task. Aviso: `[project.scripts]` puede dar conflicto trivial al mergear `chore/repo-housekeeping` si esa rama también tocó `pyproject.toml`; resolver conservando ambos cambios.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_cli.py`:

```python
"""CLI tests using typer's CliRunner and the mock provider (no API keys)."""

from __future__ import annotations

import os

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


class TestRunCommand:
    def test_run_with_mock_provider_writes_reports(self, tmp_path):
        result = runner.invoke(
            app,
            ["run", "--provider", "mock", "--datasets", "functional", "--results-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        run_dirs = os.listdir(tmp_path)
        assert len(run_dirs) == 1
        assert os.path.exists(tmp_path / run_dirs[0] / "report.json")
        assert os.path.exists(tmp_path / run_dirs[0] / "report.md")
        assert "Estimated API calls" in result.output

    def test_run_with_samples_creates_one_run_per_sample(self, tmp_path):
        result = runner.invoke(
            app,
            ["run", "--provider", "mock", "--datasets", "functional", "--samples", "2", "--results-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert len(os.listdir(tmp_path)) == 2

    def test_unknown_dataset_exits_2(self, tmp_path):
        result = runner.invoke(
            app,
            ["run", "--provider", "mock", "--datasets", "nope", "--results-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        assert "Unknown dataset" in result.output

    def test_unknown_evaluator_exits_2(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "run",
                "--provider", "mock",
                "--datasets", "functional",
                "--evaluators", "ghost_eval",
                "--results-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        assert "Unknown or unavailable evaluators" in result.output

    def test_evaluator_filter_limits_evaluations(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "run",
                "--provider", "mock",
                "--datasets", "safety",
                "--evaluators", "rule_based",
                "--results-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
```

- [ ] **Step 2: Verificar que fallan**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.cli'`.

- [ ] **Step 3: Implementar la CLI**

`src/cli.py`:

```python
"""Typer CLI: run evaluations, manage baselines, and gate quality regressions."""

from __future__ import annotations

import asyncio
import os

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.reporting.json_reporter import generate_json_report
from src.reporting.markdown_reporter import generate_markdown_report
from src.runner.models import RunSummary, TestCase
from src.runner.runner import EvalRunner, load_all_datasets, load_dataset

console = Console()
app = typer.Typer(help="LLM Eval Lab — evaluate chatbots and gate quality regressions.")
baseline_app = typer.Typer(help="Manage regression-gate baselines.")
app.add_typer(baseline_app, name="baseline")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
_LLM_EVALUATORS = {"ragas", "deepeval", "llm_judge"}


def _build_chatbot(mode: str, provider: str | None):
    """Build the appropriate chatbot based on mode and provider (moved from src/__main__.py)."""
    from src.chatbots.mock_adapter import MockChatbot, MockRAGChatbot
    from src.chatbots.openai_compatible import OpenAICompatibleChatbot

    if provider == "mock":
        if mode == "rag":
            return MockRAGChatbot()
        return MockChatbot()

    if mode == "rag":
        from src.chatbots.rag_chatbot import DemoRAGChatbot

        return DemoRAGChatbot(provider_name=provider)

    return OpenAICompatibleChatbot(provider_name=provider)


def _build_evaluators(use_llm_judge: bool = False) -> dict:
    """Build the set of evaluators to use (moved from src/__main__.py, logic unchanged)."""
    from src.evaluators.consistency import ConsistencyEvaluator
    from src.evaluators.deepeval_evaluator import DeepEvalEvaluator
    from src.evaluators.llm_judge import LLMJudgeEvaluator
    from src.evaluators.ragas_evaluator import RagasEvaluator
    from src.evaluators.rule_based import RuleBasedEvaluator
    from src.evaluators.safety import SafetyEvaluator

    evaluators = {
        "rule_based": RuleBasedEvaluator(),
        "safety": SafetyEvaluator(),
    }

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            evaluators["ragas"] = RagasEvaluator()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize RAGAS evaluator: {e}[/yellow]")
    else:
        console.print("[yellow]Warning: OPENAI_API_KEY not set — RAGAS evaluator disabled.[/yellow]")

    use_deepeval = os.getenv("USE_DEEPEVAL", "false").lower() == "true"
    if use_deepeval and openai_key:
        try:
            evaluators["deepeval"] = DeepEvalEvaluator()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize DeepEval evaluator: {e}[/yellow]")

    use_consistency = os.getenv("USE_CONSISTENCY", "false").lower() == "true"
    if use_consistency:
        evaluators["consistency"] = ConsistencyEvaluator()

    if use_llm_judge:
        try:
            evaluators["llm_judge"] = LLMJudgeEvaluator()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize LLM judge: {e}[/yellow]")

    return evaluators


def _select_datasets(datasets_csv: str | None) -> list[TestCase]:
    """Load all datasets, or only the comma-separated named ones (functional,safety,...)."""
    if not datasets_csv:
        return load_all_datasets()
    datasets_dir = os.path.join(_PROJECT_ROOT, "datasets")
    cases: list[TestCase] = []
    for name in [n.strip() for n in datasets_csv.split(",") if n.strip()]:
        path = os.path.join(datasets_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            console.print(f"[red]Unknown dataset: {name}[/red]")
            raise typer.Exit(code=2)
        cases.extend(load_dataset(path))
    return cases


def _filter_evaluators(evaluators: dict, evaluators_csv: str | None) -> dict:
    """Restrict the registered evaluators to a comma-separated subset."""
    if not evaluators_csv:
        return evaluators
    wanted = {n.strip() for n in evaluators_csv.split(",") if n.strip()}
    unknown = wanted - evaluators.keys()
    if unknown:
        console.print(f"[red]Unknown or unavailable evaluators: {', '.join(sorted(unknown))}[/red]")
        raise typer.Exit(code=2)
    return {name: evaluator for name, evaluator in evaluators.items() if name in wanted}


def _print_cost_estimate(n_cases: int, samples: int, evaluator_names: set[str]) -> None:
    """Rough API-call estimate so CI users see the cost before the run starts."""
    chatbot_calls = n_cases * samples
    llm_eval_calls = n_cases * samples * len(evaluator_names & _LLM_EVALUATORS)
    console.print(f"Estimated API calls — chatbot: {chatbot_calls}, LLM evaluators: {llm_eval_calls}")


async def _execute_runs(
    provider: str | None,
    mode: str,
    samples: int,
    evaluators_csv: str | None,
    datasets_csv: str | None,
    results_dir: str,
) -> list[RunSummary]:
    """Run the evaluation ``samples`` times, persisting each run's reports."""
    load_dotenv(os.path.join(_PROJECT_ROOT, "config", ".env"))
    chatbot = _build_chatbot(mode, provider)
    use_llm_judge = os.getenv("USE_LLM_JUDGE", "false").lower() == "true"
    evaluators = _filter_evaluators(_build_evaluators(use_llm_judge), evaluators_csv)
    test_cases = _select_datasets(datasets_csv)
    _print_cost_estimate(len(test_cases), samples, set(evaluators.keys()))

    summaries: list[RunSummary] = []
    for i in range(samples):
        if samples > 1:
            console.print(f"\n[bold]Sample {i + 1}/{samples}[/bold]")
        eval_runner = EvalRunner(chatbot=chatbot, evaluators=evaluators)
        summary = await eval_runner.run(test_cases)
        output_dir = os.path.join(results_dir, summary.run_id)
        generate_json_report(summary, output_dir)
        generate_markdown_report(summary, output_dir)
        summaries.append(summary)
    return summaries


def _load_summary(results_dir: str, run_id: str) -> RunSummary:
    """Load a persisted run's report.json as a RunSummary (exit 2 if missing)."""
    path = os.path.join(results_dir, run_id, "report.json")
    if not os.path.exists(path):
        console.print(f"[red]Run not found: {path}[/red]")
        raise typer.Exit(code=2)
    with open(path) as f:
        return RunSummary.model_validate_json(f.read())


@app.command()
def run(
    provider: str | None = typer.Option(None, help="Provider name (overrides ACTIVE_PROVIDER)."),
    mode: str | None = typer.Option(None, help="plain or rag (overrides CHATBOT_MODE)."),
    samples: int = typer.Option(1, min=1, help="Times each test case is executed."),
    evaluators: str | None = typer.Option(None, help="Comma-separated evaluator subset (e.g. rule_based,safety)."),
    datasets: str | None = typer.Option(None, help="Comma-separated dataset names (e.g. functional,safety)."),
    results_dir: str = typer.Option(_DEFAULT_RESULTS_DIR, help="Directory where run reports are written."),
) -> None:
    """Run an evaluation (default command when no subcommand is given)."""
    resolved_provider = provider or os.getenv("ACTIVE_PROVIDER") or None
    resolved_mode = (mode or os.getenv("CHATBOT_MODE", "plain")).lower()
    summaries = asyncio.run(
        _execute_runs(resolved_provider, resolved_mode, samples, evaluators, datasets, results_dir)
    )
    console.print(f"\nRun ids: {', '.join(s.run_id for s in summaries)}")
```

- [ ] **Step 4: Convertir `src/__main__.py` en shim**

Reemplazar TODO el contenido de `src/__main__.py` por:

```python
"""Entry point: python -m src / llm-eval-lab console script."""

from __future__ import annotations

import sys

from src.cli import app


def main() -> None:
    """Invoke the CLI; with no arguments, default to the ``run`` command."""
    args = sys.argv[1:]
    app(args or ["run"])


def cli() -> None:
    """Backwards-compatible alias for the old console-script entry point."""
    main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Registrar el script de consola y silenciar B008 en la CLI**

En `pyproject.toml`, añadir tras el bloque `[project.optional-dependencies]`:

```toml
[project.scripts]
llm-eval-lab = "src.__main__:main"
```

y en `[tool.ruff.lint.per-file-ignores]` añadir la línea (el patrón idiomático de Typer usa `typer.Option(...)` como default de parámetro, que dispara bugbear B008):

```toml
"src/cli.py" = ["B008"]  # typer.Option/Argument as parameter defaults is the Typer idiom
```

Ejecutar `pip install -e ".[dev]"` para regenerar el script.

- [ ] **Step 6: Verificar que pasan + smoke manual**

Run: `pytest tests/test_cli.py -v && pytest && ruff check src/ tests/ && ruff format src/ tests/ && mypy src/ --ignore-missing-imports`
Expected: PASS completo (la suite entera, para confirmar que el refactor del entry point no rompe nada).

Smoke: `ACTIVE_PROVIDER=mock python -m src run --datasets functional` termina con exit 0 y crea un run en `results/` (nota: con argumentos hay que nombrar el subcomando; solo el invocación sin argumentos equivale a `run`). También verificar que el bare `python -m src` sigue arrancando la evaluación completa (interrumpir con Ctrl+C tras ver la barra de progreso) (borrarlo después: es un artefacto local, `results/` ya está en `.gitignore`; verificar con `git status`).

- [ ] **Step 7: Commit**

```bash
git add src/cli.py src/__main__.py pyproject.toml tests/test_cli.py
git commit -m "feat: typer CLI with run command, samples and dataset/evaluator filters"
```

---

### Task 9: Comandos `baseline save` y `compare`

**Files:**
- Modify: `src/cli.py` (añadir comandos al final del archivo)
- Test: `tests/test_cli.py` (añadir clases de tests)

**Interfaces:**
- Consumes: `build_baseline`, `save_baseline`, `BaselineError` (Task 3); `evaluate_gate`, `GatePolicy` (Task 5); `CompatibilityError` (Task 4); `render_gate_console` (Task 6); `_load_summary`, `app`, `baseline_app` (Task 8).
- Produces: comandos `llm-eval-lab baseline save <run_id>... [--name] [--results-dir] [--baselines-dir]` y `llm-eval-lab compare <run_a> <run_b> [--results-dir]`.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_cli.py`:

```python
def _do_run(tmp_path, datasets="functional", samples=1):
    """Run the mock provider and return the created run ids (oldest first)."""
    results_dir = tmp_path / "results"
    args = ["run", "--provider", "mock", "--datasets", datasets, "--results-dir", str(results_dir)]
    if samples > 1:
        args += ["--samples", str(samples)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return results_dir, sorted(os.listdir(results_dir))


class TestBaselineSave:
    def test_save_single_run(self, tmp_path):
        results_dir, run_ids = _do_run(tmp_path)
        baselines_dir = tmp_path / "baselines"
        result = runner.invoke(
            app,
            [
                "baseline", "save", run_ids[0],
                "--name", "main",
                "--results-dir", str(results_dir),
                "--baselines-dir", str(baselines_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (baselines_dir / "main.json").exists()

    def test_save_multiple_runs_records_samples(self, tmp_path):
        from src.gate.models import BaselineFile

        results_dir, run_ids = _do_run(tmp_path, samples=2)
        baselines_dir = tmp_path / "baselines"
        result = runner.invoke(
            app,
            ["baseline", "save", *run_ids, "--results-dir", str(results_dir), "--baselines-dir", str(baselines_dir)],
        )
        assert result.exit_code == 0, result.output
        baseline = BaselineFile.model_validate_json((baselines_dir / "main.json").read_text())
        assert baseline.samples == 2

    def test_save_unknown_run_exits_2(self, tmp_path):
        result = runner.invoke(
            app,
            ["baseline", "save", "no_such_run", "--results-dir", str(tmp_path), "--baselines-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        assert "Run not found" in result.output


class TestCompareCommand:
    def test_compare_two_runs_renders_table(self, tmp_path):
        results_dir, run_ids = _do_run(tmp_path, samples=2)
        result = runner.invoke(app, ["compare", run_ids[0], run_ids[1], "--results-dir", str(results_dir)])
        assert result.exit_code == 0, result.output
        assert "pass_rate" in result.output
        assert "Regression gate" in result.output

    def test_compare_missing_run_exits_2(self, tmp_path):
        results_dir, run_ids = _do_run(tmp_path)
        result = runner.invoke(app, ["compare", run_ids[0], "ghost", "--results-dir", str(results_dir)])
        assert result.exit_code == 2
```

- [ ] **Step 2: Verificar que fallan**

Run: `pytest tests/test_cli.py -k "BaselineSave or CompareCommand" -v`
Expected: FAIL — typer responde `No such command 'baseline'` / `'compare'` (exit code distinto del esperado).

- [ ] **Step 3: Implementar**

Añadir al final de `src/cli.py` (más los imports nuevos en la cabecera del archivo):

```python
from src.gate.baseline import BaselineError, build_baseline, save_baseline
from src.gate.comparison import CompatibilityError
from src.gate.models import GatePolicy
from src.gate.policy import evaluate_gate
from src.reporting.gate_reporter import render_gate_console
```

```python
@baseline_app.command("save")
def baseline_save(
    run_ids: list[str] = typer.Argument(..., help="One or more run ids under --results-dir."),
    name: str = typer.Option("main", help="Baseline name (file becomes <baselines-dir>/<name>.json)."),
    results_dir: str = typer.Option(_DEFAULT_RESULTS_DIR, help="Directory containing run reports."),
    baselines_dir: str = typer.Option("baselines", help="Directory for baseline files."),
) -> None:
    """Aggregate one or more runs into a committed, diffable baseline file."""
    summaries = [_load_summary(results_dir, run_id) for run_id in run_ids]
    try:
        baseline = build_baseline(summaries)
    except BaselineError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e
    path = save_baseline(baseline, os.path.join(baselines_dir, f"{name}.json"))
    console.print(f"Baseline saved: {path} (samples: {baseline.samples}, cases: {len(baseline.cases)})")


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="Run id used as baseline side."),
    run_b: str = typer.Argument(..., help="Run id used as current side."),
    results_dir: str = typer.Option(_DEFAULT_RESULTS_DIR, help="Directory containing run reports."),
) -> None:
    """Statistical comparison between two stored runs (no verdict, no special exit code)."""
    baseline = build_baseline([_load_summary(results_dir, run_a)])
    current = build_baseline([_load_summary(results_dir, run_b)])
    try:
        verdict = evaluate_gate(baseline, current, GatePolicy())
    except CompatibilityError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e
    render_gate_console(verdict, console)
```

- [ ] **Step 4: Verificar que pasan + calidad**

Run: `pytest tests/test_cli.py -v && ruff check src/cli.py tests/test_cli.py && ruff format src/cli.py tests/test_cli.py && mypy src/ --ignore-missing-imports`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cli.py tests/test_cli.py
git commit -m "feat: baseline save and compare CLI commands"
```

---

### Task 10: Comando `check` (el gate)

**Files:**
- Modify: `src/cli.py` (añadir `_resolve_baseline_path` y el comando `check`)
- Test: `tests/test_cli.py` (añadir clase de tests)

**Interfaces:**
- Consumes: `load_baseline`/`BaselineError` (Task 3), `load_policy`/`PolicyError`/`evaluate_gate` (Task 5), `CompatibilityError` (Task 4), `render_gate_console`/`generate_gate_markdown` (Task 6), `_execute_runs`/`_load_summary` (Task 8).
- Produces: comando `llm-eval-lab check --baseline <name|path> [--provider] [--mode] [--samples] [--evaluators] [--datasets] [--policy] [--results-dir] [--baselines-dir]` con contrato de exit codes: `0` pasa, `1` regresión, `2` error de ejecución/config/incompatibilidad.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_cli.py`:

```python
class TestCheckCommand:
    def _run_and_save_baseline(self, tmp_path, datasets="safety"):
        results_dir, run_ids = _do_run(tmp_path, datasets=datasets)
        baselines_dir = tmp_path / "baselines"
        result = runner.invoke(
            app,
            ["baseline", "save", run_ids[0], "--results-dir", str(results_dir), "--baselines-dir", str(baselines_dir)],
        )
        assert result.exit_code == 0, result.output
        return results_dir, baselines_dir / "main.json"

    def test_check_against_own_baseline_passes(self, tmp_path):
        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline", str(baseline_path),
                "--provider", "mock",
                "--datasets", "safety",
                "--results-dir", str(results_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output
        run_ids = sorted(os.listdir(results_dir))
        assert os.path.exists(results_dir / run_ids[-1] / "gate_report.md")

    def test_check_detects_new_critical_failure(self, tmp_path):
        from src.gate.models import BaselineFile

        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        baseline = BaselineFile.model_validate_json(baseline_path.read_text())
        critical_failed = [c for c in baseline.cases if c.severity == "critical" and not c.passed]
        # El mock responde de forma insegura a los triggers de safety.jsonl a propósito;
        # si esta precondición falla, ampliar _UNSAFE_TRIGGERS en mock_adapter.py
        # para cubrir al menos un caso critical del dataset safety.
        assert critical_failed, "expected the mock to fail at least one critical safety case"
        flipped = [
            c.model_copy(update={"passed": True, "pass_samples": [True]})
            if (c.severity == "critical" and not c.passed)
            else c
            for c in baseline.cases
        ]
        baseline_path.write_text(baseline.model_copy(update={"cases": flipped}).model_dump_json(indent=2))
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline", str(baseline_path),
                "--provider", "mock",
                "--datasets", "safety",
                "--results-dir", str(results_dir),
            ],
        )
        assert result.exit_code == 1, result.output
        assert "New critical failures" in result.output

    def test_check_missing_baseline_exits_2(self, tmp_path):
        result = runner.invoke(
            app,
            ["check", "--baseline", str(tmp_path / "ghost.json"), "--provider", "mock", "--results-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        assert "not found" in result.output

    def test_check_mode_mismatch_exits_2(self, tmp_path):
        results_dir = tmp_path / "results"
        r = runner.invoke(
            app,
            ["run", "--provider", "mock", "--mode", "rag", "--datasets", "functional", "--results-dir", str(results_dir)],
        )
        assert r.exit_code == 0, r.output
        run_id = os.listdir(results_dir)[0]
        baselines_dir = tmp_path / "baselines"
        r = runner.invoke(
            app,
            ["baseline", "save", run_id, "--results-dir", str(results_dir), "--baselines-dir", str(baselines_dir)],
        )
        assert r.exit_code == 0, r.output
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline", str(baselines_dir / "main.json"),
                "--provider", "mock",
                "--mode", "plain",
                "--datasets", "functional",
                "--results-dir", str(results_dir),
            ],
        )
        assert result.exit_code == 2
        assert "chatbot_mode" in result.output

    def test_check_resolves_baseline_by_name(self, tmp_path):
        results_dir, baseline_path = self._run_and_save_baseline(tmp_path)
        result = runner.invoke(
            app,
            [
                "check",
                "--baseline", "main",
                "--baselines-dir", str(baseline_path.parent),
                "--provider", "mock",
                "--datasets", "safety",
                "--results-dir", str(results_dir),
            ],
        )
        assert result.exit_code == 0, result.output
```

- [ ] **Step 2: Verificar que fallan**

Run: `pytest tests/test_cli.py -k CheckCommand -v`
Expected: FAIL — `No such command 'check'`.

- [ ] **Step 3: Implementar**

Añadir a los imports de gate en `src/cli.py`: `load_baseline` (de `src.gate.baseline`) y `PolicyError, load_policy` (de `src.gate.policy`), y `generate_gate_markdown` (de `src.reporting.gate_reporter`). Después añadir al final del archivo:

```python
def _resolve_baseline_path(baseline: str, baselines_dir: str) -> str:
    """A value with a path separator or .json suffix is a path; otherwise a name in baselines_dir."""
    if baseline.endswith(".json") or os.path.sep in baseline:
        return baseline
    return os.path.join(baselines_dir, f"{baseline}.json")


@app.command()
def check(
    baseline: str = typer.Option("main", help="Baseline name or path to a baseline JSON file."),
    provider: str | None = typer.Option(None, help="Provider name (overrides ACTIVE_PROVIDER)."),
    mode: str | None = typer.Option(None, help="plain or rag (overrides CHATBOT_MODE)."),
    samples: int = typer.Option(1, min=1, help="Times each test case is executed."),
    evaluators: str | None = typer.Option(None, help="Comma-separated evaluator subset."),
    datasets: str | None = typer.Option(None, help="Comma-separated dataset names."),
    policy: str | None = typer.Option(None, help="Path to a gate policy YAML (default: built-in policy)."),
    results_dir: str = typer.Option(_DEFAULT_RESULTS_DIR, help="Directory where run reports are written."),
    baselines_dir: str = typer.Option("baselines", help="Directory containing baseline files."),
) -> None:
    """Run an evaluation and fail (exit 1) on significant quality regressions vs the baseline."""
    try:
        gate_policy = load_policy(policy)
        baseline_file = load_baseline(_resolve_baseline_path(baseline, baselines_dir))
    except (PolicyError, BaselineError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e

    resolved_provider = provider or os.getenv("ACTIVE_PROVIDER") or None
    resolved_mode = (mode or os.getenv("CHATBOT_MODE", "plain")).lower()
    try:
        summaries = asyncio.run(
            _execute_runs(resolved_provider, resolved_mode, samples, evaluators, datasets, results_dir)
        )
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise typer.Exit(code=2) from e

    current = build_baseline(summaries)
    try:
        verdict = evaluate_gate(baseline_file, current, gate_policy)
    except CompatibilityError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e

    render_gate_console(verdict, console)
    output_dir = os.path.join(results_dir, summaries[-1].run_id)
    md_path = generate_gate_markdown(verdict, output_dir)
    console.print(f"Gate report: {md_path}")

    if verdict.missing_gated_metrics:
        raise typer.Exit(code=2)
    if not verdict.passed:
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Verificar que pasan + suite completa**

Run: `pytest tests/test_cli.py -v && pytest && ruff check src/ tests/ && ruff format src/ tests/ && mypy src/ --ignore-missing-imports`
Expected: PASS completo con cobertura ≥80%.

- [ ] **Step 5: Commit**

```bash
git add src/cli.py tests/test_cli.py
git commit -m "feat: check command gating regressions with CI exit codes"
```

---

### Task 11: Fixture de baseline, GitHub Action compuesta y job de dogfooding en CI

**Files:**
- Create: `tests/fixtures/baseline_mock.json` (generado con la CLI, luego commiteado)
- Create: `action.yml` (raíz del repo)
- Modify: `.github/workflows/ci.yml` (job nuevo `gate-dogfood`)

**Interfaces:**
- Consumes: CLI completa (Tasks 8-10).
- Produces: action invocable como `uses: gonzaloMorenoc/llm-eval-lab@<ref>` (o `uses: ./` desde el propio repo) con inputs `provider`, `mode`, `baseline`, `samples`, `evaluators`, `datasets`, `policy`, `python-version`.

- [ ] **Step 1: Generar el fixture de baseline con el provider mock**

```bash
mkdir -p tests/fixtures
ACTIVE_PROVIDER=mock llm-eval-lab run --samples 3 --results-dir /tmp/gate-fixture-results
# Copiar los tres run ids que imprime la línea final "Run ids: ..." y usarlos aquí:
llm-eval-lab baseline save <run_id_1> <run_id_2> <run_id_3> \
  --name baseline_mock \
  --results-dir /tmp/gate-fixture-results \
  --baselines-dir tests/fixtures
rm -rf /tmp/gate-fixture-results
```

Verificar: `tests/fixtures/baseline_mock.json` existe, tiene `"samples": 3` y 43 casos.

- [ ] **Step 2: Verificar el gate localmente contra el fixture (el "test que falla" de esta task es este comando antes de crear la action)**

```bash
ACTIVE_PROVIDER=mock llm-eval-lab check --baseline tests/fixtures/baseline_mock.json --results-dir /tmp/gate-check-results
echo "exit: $?"
rm -rf /tmp/gate-check-results
```

Expected: tabla del gate con veredicto PASS y `exit: 0` (el mock es determinista en scores; solo varía la latencia, que no está gateada).

- [ ] **Step 3: Crear `action.yml`**

```yaml
name: "LLM Eval Lab regression gate"
description: "Run LLM evaluations and fail the build on statistically significant quality regressions."
branding:
  icon: "check-circle"
  color: "green"
inputs:
  provider:
    description: "Chatbot provider name (set API keys as env/secrets on the calling job)"
    default: "mock"
  mode:
    description: "plain or rag"
    default: "plain"
  baseline:
    description: "Baseline name or path to a baseline JSON in the calling repo"
    default: "baselines/main.json"
  samples:
    description: "Times each test case is executed"
    default: "1"
  evaluators:
    description: "Comma-separated evaluator subset (empty = all registered)"
    default: ""
  datasets:
    description: "Comma-separated dataset names (empty = all)"
    default: ""
  policy:
    description: "Path to a gate policy YAML (empty = built-in defaults)"
    default: ""
  python-version:
    description: "Python version to install"
    default: "3.11"
runs:
  using: "composite"
  steps:
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ inputs.python-version }}
    - name: Install llm-eval-lab
      shell: bash
      # Editable install: the CLI resolves datasets/ and config/ relative to the
      # source tree, so the package must stay in place (a regular install would
      # point _PROJECT_ROOT at site-packages, where datasets/ does not exist).
      run: pip install -e "${{ github.action_path }}"
    - name: Run regression gate
      shell: bash
      env:
        ACTIVE_PROVIDER: ${{ inputs.provider }}
        CHATBOT_MODE: ${{ inputs.mode }}
      run: |
        ARGS=(check --baseline "${{ inputs.baseline }}" --samples "${{ inputs.samples }}")
        if [ -n "${{ inputs.evaluators }}" ]; then ARGS+=(--evaluators "${{ inputs.evaluators }}"); fi
        if [ -n "${{ inputs.datasets }}" ]; then ARGS+=(--datasets "${{ inputs.datasets }}"); fi
        if [ -n "${{ inputs.policy }}" ]; then ARGS+=(--policy "${{ inputs.policy }}"); fi
        llm-eval-lab "${ARGS[@]}"
```

- [ ] **Step 4: Añadir el job de dogfooding a `.github/workflows/ci.yml`**

Añadir al final del archivo, al mismo nivel que `lint` y `test`:

```yaml
  gate-dogfood:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Run regression gate against the mock baseline
        uses: ./
        with:
          provider: mock
          baseline: tests/fixtures/baseline_mock.json
```

- [ ] **Step 5: Verificación local final**

Run: `pytest && ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: PASS (el fixture no afecta a la suite; el job de CI se verificará al abrir el PR — revisar que `gate-dogfood` sale verde).

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/baseline_mock.json action.yml .github/workflows/ci.yml
git commit -m "ci: composite gate action with mock-baseline dogfooding job"
```

---

### Task 12: Capa estadística en la página Compare Runs

**Files:**
- Modify: `src/dashboard/pages/3_compare.py` (imports en cabecera + sección nueva al final del archivo)

**Interfaces:**
- Consumes: `build_baseline` (Task 3), `evaluate_gate`/`GatePolicy` (Task 5), `RunSummary`, `RESULTS_DIR` y `list_runs()` de `src/dashboard/components/shared.py` (los dicts `run_a`/`run_b` ya en scope en la página son los `report.json` cargados).
- Produces: sección "📐 Comparación estadística" en la página; sin página nueva.

**Nota:** el dashboard está excluido de cobertura; la verificación es lint + mypy + smoke manual. Antes de codificar, leer `src/dashboard/components/shared.py:79-95` para confirmar si `list_runs()` devuelve el report completo (con `results`) o un resumen; el helper de abajo cubre ambos casos.

- [ ] **Step 1: Añadir imports en la cabecera de `3_compare.py`** (junto a los imports existentes de `src.dashboard.components`):

```python
from src.dashboard.components.shared import RESULTS_DIR, list_runs
from src.gate.baseline import build_baseline
from src.gate.models import GatePolicy
from src.gate.policy import evaluate_gate
from src.runner.models import RunSummary
```

(La línea existente `from src.dashboard.components.shared import list_runs` se sustituye por la de arriba.)

- [ ] **Step 2: Añadir la sección al final del archivo**

```python
# ── Statistical comparison (gate engine) ──────────────────────────────────────
st.divider()
st.subheader("📐 Comparación estadística")


def _full_summary(run: dict) -> RunSummary:
    """Load the complete RunSummary for a run dict coming from list_runs()."""
    if run.get("results"):
        return RunSummary.model_validate(run)
    path = os.path.join(RESULTS_DIR, run.get("run_id", ""), "report.json")
    with open(path) as f:
        return RunSummary.model_validate_json(f.read())


try:
    _verdict = evaluate_gate(
        build_baseline([_full_summary(run_a)]),
        build_baseline([_full_summary(run_b)]),
        GatePolicy(),
    )
    st.dataframe(
        [
            {
                "Métrica": c.metric,
                "A (baseline)": round(c.baseline_mean, 4),
                "B (actual)": round(c.current_mean, 4),
                "Regresión": round(c.regression, 4),
                "IC 95%": f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]",
                "p-valor": round(c.p_value, 4),
                "Significativa": "sí" if c.significant else "no",
            }
            for c in _verdict.comparisons
        ],
        use_container_width=True,
    )
    st.caption(
        "Bootstrap pareado por caso: B como run actual frente a A como baseline. "
        "Con 1 muestra por caso la potencia estadística es baja; usa `--samples 3` o más en la CLI."
    )
except Exception as e:
    st.info(f"Comparación estadística no disponible para estos runs: {e}")
```

(El `except Exception` amplio es deliberado: la página no debe romperse por runs incomparables — p. ej. plain vs rag — ni por reports antiguos; el mensaje informa del motivo.)

- [ ] **Step 3: Verificar calidad + smoke manual**

Run: `ruff check src/dashboard && ruff format src/dashboard && mypy src/ --ignore-missing-imports && pytest`
Expected: PASS.

Smoke: con al menos 2 runs en `results/` (generarlos con `ACTIVE_PROVIDER=mock llm-eval-lab run --datasets functional` dos veces si hace falta), lanzar `streamlit run src/dashboard/app.py`, abrir Compare Runs y comprobar que la sección nueva muestra la tabla con p-valores. Borrar los runs generados después si se crearon solo para el smoke.

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/pages/3_compare.py
git commit -m "feat: statistical significance section in the Compare Runs page"
```

---

### Task 13: Documentación (README + CHANGELOG)

**Files:**
- Modify: `README.md` (features, sección nueva "Regression Gate", CLI en Quickstart, project structure)
- Modify: `CHANGELOG.md` (entradas en `[Unreleased]` → `### Added`)

**Interfaces:** solo documentación de lo construido en Tasks 1-12; usar los nombres exactos de comandos y archivos de esas tareas.

- [ ] **Step 1: Actualizar README.md**

1. En la lista **Features**, añadir tras la línea del dashboard:

```markdown
- **Regression gate for CI/CD**: `llm-eval-lab check` compares a run against a committed baseline with paired bootstrap statistics and breaks the build on significant regressions; reusable GitHub Action included
```

2. Nueva sección tras "## Dashboard":

````markdown
## Regression Gate (CI/CD)

Turn evaluations into a quality gate: compare the current state against a committed
baseline and fail the build (exit code 1) on statistically significant regressions.

```bash
# 1. Create a baseline (3 samples per case for statistical power) and commit it
llm-eval-lab run --samples 3
llm-eval-lab baseline save <run_id_1> <run_id_2> <run_id_3> --name main
git add baselines/main.json && git commit -m "chore: update eval baseline"

# 2. In a PR, gate against it
llm-eval-lab check --baseline main --samples 3
echo $?   # 0 = pass, 1 = regression, 2 = execution/config error
```

The statistical engine pairs test cases between runs and bootstraps the per-case
regression deltas (95% CI + one-sided p-value). A metric only breaks the build when
it is listed in the gate policy AND the regression is statistically significant AND
larger than both `min_effect_size` and its `max_regression`. Hard rules bypass
statistics entirely: a newly failing `critical` case always fails the gate.
Policy reference: [`config/gate.yaml`](config/gate.yaml).

### GitHub Action

```yaml
- uses: gonzaloMorenoc/llm-eval-lab@main
  with:
    provider: groq          # default: mock (free, no keys)
    baseline: baselines/main.json
    samples: "3"
  env:
    GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

The default CI preset uses only free evaluators (rule-based, safety); LLM-based
metrics opt in via the `evaluators` input. `check` prints an API-call estimate
before running, and writes `gate_report.md` (also to `$GITHUB_STEP_SUMMARY`)
ready to paste as a PR comment.

### CLI reference

| Command | Purpose |
|---------|---------|
| `llm-eval-lab run [--samples N] [--datasets a,b] [--evaluators x,y]` | Run an evaluation (N runs with `--samples`) |
| `llm-eval-lab baseline save <run_id>... [--name main]` | Aggregate runs into `baselines/<name>.json` |
| `llm-eval-lab check --baseline main [--policy config/gate.yaml]` | Gate: exit 0/1/2 |
| `llm-eval-lab compare <run_a> <run_b>` | Statistical comparison, no verdict |

`python -m src` (sin subcomando) sigue equivaliendo a `run`.
````

3. En **Project Structure**, añadir las entradas nuevas en sus lugares: `action.yml`, `config/gate.yaml`, `src/cli.py`, `src/gate/` (con sus 5 módulos y una línea de descripción cada uno), `src/reporting/gate_reporter.py`, `tests/fixtures/baseline_mock.json` y los 6 archivos de test nuevos.

- [ ] **Step 2: Actualizar CHANGELOG.md**

Añadir al principio de `## [Unreleased]` → `### Added`:

```markdown
- Regression quality gate for CI/CD: `src/gate/` (paired bootstrap statistics,
  baseline build/load, case pairing, policy + verdict), `llm-eval-lab` CLI with
  `run` / `baseline save` / `check` / `compare` subcommands (exit codes 0/1/2),
  `gate_report.md` + `$GITHUB_STEP_SUMMARY` output, composite GitHub Action
  (`action.yml`) with a mock-baseline dogfooding job in CI, statistical section
  in the Compare Runs dashboard page, and `config/gate.yaml` policy reference.
- `MockChatbot`/`MockRAGChatbot` accept an optional `transform` callable to
  simulate response drift in gate tests.
```

- [ ] **Step 3: Verificación final del proyecto completo**

Run: `pytest && ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/ --ignore-missing-imports`
Expected: PASS, cobertura ≥80%.

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: document the regression gate, CLI and GitHub Action"
```

---

## Verificación final del plan completo

Tras la última task, ejecutar de una vez:

```bash
pytest && ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/ --ignore-missing-imports
ACTIVE_PROVIDER=mock llm-eval-lab check --baseline tests/fixtures/baseline_mock.json --results-dir /tmp/final-check && echo "GATE OK"
rm -rf /tmp/final-check
```

Expected: suite verde con cobertura ≥80% y `GATE OK`. Abrir PR contra `main` y comprobar que los tres jobs de CI (lint, test, gate-dogfood) salen verdes.





