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
