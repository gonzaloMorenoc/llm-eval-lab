"""Multi-sample stability view — which test cases contradict themselves.

Running the suite N times is only worth the API calls if the extra runs are
allowed to disagree. These helpers turn a baseline built from N samples into
what the Run page shows: how many cases are unstable, and which ones.

Pure functions with no Streamlit imports, so they can be tested directly.
"""

from __future__ import annotations

from typing import Any

from src.gate.models import BaselineCase, BaselineFile


def sample_pattern(case: BaselineCase) -> str:
    """Render a case's per-sample outcomes in run order, e.g. ``✅❌✅``."""
    return "".join("✅" if passed else "❌" for passed in case.pass_samples)


def unstable_case_rows(baseline: BaselineFile) -> list[dict[str, Any]]:
    """Table rows for the cases whose samples disagree, most unstable first.

    A case that fails in every sample is *stable* — it is a plain failure and
    already shown as such. Only self-contradiction lands here.
    """
    flaky = [case for case in baseline.cases if case.flakiness > 0]
    flaky.sort(key=lambda case: (-case.flakiness, case.id))
    return [
        {
            "Caso": case.id,
            "Categoría": case.category,
            "Severidad": case.severity,
            "Muestras": sample_pattern(case),
            "Inestabilidad": round(case.flakiness, 3),
        }
        for case in flaky
    ]


def stability_headline(baseline: BaselineFile) -> dict[str, Any]:
    """Summary counters for the stability callout."""
    cases = baseline.cases
    unstable = sum(1 for case in cases if case.flakiness > 0)
    mean_flakiness = sum(case.flakiness for case in cases) / len(cases) if cases else 0.0
    return {
        "unstable": unstable,
        "total": len(cases),
        "samples": baseline.samples,
        "mean_flakiness": round(mean_flakiness, 4),
    }
