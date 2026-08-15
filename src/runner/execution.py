"""Run an evaluation suite N times and persist every report.

Both the CLI (``--samples``) and the dashboard need "run the suite N times,
write each report to disk, hand back the summaries". Keeping one implementation
means the two entry points cannot disagree about what a sample is — the same
reason the dashboard's statistical comparison calls the gate engine instead of
recomputing it.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from src.chatbots.base import BaseChatbot
from src.evaluators.base import BaseEvaluator
from src.reporting.json_reporter import generate_json_report
from src.reporting.markdown_reporter import generate_markdown_report
from src.runner.models import RunSummary, TestCase
from src.runner.runner import EvalRunner


async def run_samples(
    chatbot: BaseChatbot,
    evaluators: dict[str, BaseEvaluator],
    test_cases: list[TestCase],
    results_dir: str,
    samples: int = 1,
    config: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_sample_start: Callable[[int, int], None] | None = None,
) -> list[RunSummary]:
    """Evaluate ``test_cases`` ``samples`` times, persisting each run's reports.

    ``on_progress(completed, total)`` counts cases across *all* samples, so a
    caller's progress bar advances once from 0 to ``samples * len(test_cases)``
    instead of restarting at every sample boundary.

    ``on_sample_start(index, samples)`` fires before each sample (1-indexed), for
    callers that label them in their output.
    """
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")

    total_units = samples * len(test_cases)
    completed_before = 0
    summaries: list[RunSummary] = []

    for index in range(1, samples + 1):
        if on_sample_start is not None:
            on_sample_start(index, samples)

        sample_progress = _offset_progress(on_progress, completed_before, total_units)
        runner = EvalRunner(chatbot=chatbot, evaluators=evaluators, config=config)
        summary = await runner.run(test_cases, on_progress=sample_progress)

        output_dir = os.path.join(results_dir, summary.run_id)
        generate_json_report(summary, output_dir)
        generate_markdown_report(summary, output_dir)

        summaries.append(summary)
        completed_before += len(test_cases)

    return summaries


def _offset_progress(
    on_progress: Callable[[int, int], None] | None,
    offset: int,
    total: int,
) -> Callable[[int, int], None] | None:
    """Shift one run's per-case count into the whole multi-sample sequence."""
    if on_progress is None:
        return None

    def report(done: int, _run_total: int) -> None:
        on_progress(offset + done, total)

    return report
