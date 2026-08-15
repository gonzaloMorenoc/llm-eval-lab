"""Logic behind the Quality Gate page.

Pure functions with no Streamlit import, so the page is left with rendering
only and every decision here is directly testable. Nothing in this module
re-implements the gate: verdicts come from ``src.gate`` unchanged.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from pydantic import BaseModel

from src.gate.baseline import BaselineError, compute_dataset_hash, load_baseline
from src.gate.models import BaselineFile
from src.runner.models import TestCase

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
