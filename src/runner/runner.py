"""Async orchestrator — loads datasets, dispatches evaluators, collects results."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.chatbots.base import BaseChatbot
from src.evaluators.base import BaseEvaluator
from src.runner.models import (
    CategoryStats,
    EvaluationResult,
    RunSummary,
    TestCase,
    TestResult,
)

console = Console()


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(os.path.abspath(config_path)) as f:
        return yaml.safe_load(f)


def load_dataset(path: str) -> list[TestCase]:
    """Load test cases from a JSONL file."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(TestCase(**data))
    return cases


def load_all_datasets(datasets_dir: str | None = None) -> list[TestCase]:
    """Load all JSONL datasets from the datasets/ directory (excluding rag_knowledge_base)."""
    if datasets_dir is None:
        datasets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
    datasets_dir = os.path.abspath(datasets_dir)

    all_cases: list[TestCase] = []
    for filename in sorted(os.listdir(datasets_dir)):
        if not filename.endswith(".jsonl"):
            continue
        if filename == "rag_knowledge_base.jsonl":
            continue
        all_cases.extend(load_dataset(os.path.join(datasets_dir, filename)))
    return all_cases


class EvalRunner:
    """Main orchestrator for running evaluations."""

    def __init__(
        self,
        chatbot: BaseChatbot,
        evaluators: dict[str, BaseEvaluator],
        config: dict | None = None,
    ) -> None:
        self._chatbot = chatbot
        self._evaluators = evaluators
        self._config = config or _load_config()

        runner_cfg = self._config.get("runner", {})
        self._max_concurrent = runner_cfg.get("max_concurrent", 5)
        self._retry_attempts = runner_cfg.get("retry_attempts", 3)
        self._retry_backoff_base = runner_cfg.get("retry_backoff_base", 2)

    async def _call_chatbot(self, messages: list[dict]) -> tuple[str, list[str] | None, float, str | None]:
        """Call the chatbot with retry logic for network errors. Returns (content, contexts, latency, error)."""
        last_error = None
        for attempt in range(self._retry_attempts):
            try:
                response = await self._chatbot.complete(messages)
                return response.content, response.retrieved_contexts, response.latency_ms, None
            except Exception as e:
                last_error = str(e)
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                is_network = "timeout" in str(e).lower() or "connect" in str(e).lower()
                if is_rate_limit or is_network:
                    wait = self._retry_backoff_base ** attempt
                    await asyncio.sleep(wait)
                else:
                    break
        return "", None, 0.0, last_error

    async def _run_single(self, test_case: TestCase, semaphore: asyncio.Semaphore) -> TestResult:
        """Execute a single test case."""
        async with semaphore:
            # Build messages
            if isinstance(test_case.input, list):
                messages = test_case.input
            else:
                messages = [{"role": "user", "content": test_case.input}]

            # Call chatbot
            content, contexts, latency, error = await self._call_chatbot(messages)
            mode = "rag" if self._chatbot.is_rag else "plain"

            result = TestResult(
                test_case=test_case,
                response=content,
                retrieved_contexts=contexts,
                chatbot_mode=mode,
                latency_ms=latency,
                error=error,
                evaluations=[],
                overall_passed=False,
                overall_score=None,
            )

            if error:
                return result

            # Run evaluators
            evaluations: list[EvaluationResult] = []
            for eval_type in test_case.evaluation_type:
                evaluator = self._evaluators.get(eval_type)
                if evaluator is None:
                    continue
                try:
                    eval_result = await evaluator.evaluate(
                        test_case=test_case,
                        response=content,
                        retrieved_contexts=contexts,
                        latency_ms=latency,
                    )
                    evaluations.append(eval_result)
                except Exception as e:
                    evaluations.append(EvaluationResult(
                        evaluator=eval_type,
                        passed=False,
                        score=None,
                        reason=f"Evaluator error: {e}",
                        details={"error": str(e)},
                    ))

            result.evaluations = evaluations
            result.overall_passed = all(e.passed for e in evaluations) if evaluations else False

            scored = [e.score for e in evaluations if e.score is not None]
            result.overall_score = round(sum(scored) / len(scored), 4) if scored else None

            return result

    async def run(self, test_cases: list[TestCase]) -> RunSummary:
        """Run all test cases and return a summary."""
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat()
        mode = "rag" if self._chatbot.is_rag else "plain"

        console.print(
            f"\n[bold]Starting evaluation run[/bold] — "
            f"chatbot: [cyan]{self._chatbot.get_id()}[/cyan] | "
            f"mode: [yellow]{mode}[/yellow] | "
            f"cases: [green]{len(test_cases)}[/green]\n"
        )

        semaphore = asyncio.Semaphore(self._max_concurrent)
        results: list[TestResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(test_cases))

            async def _run_and_track(tc: TestCase) -> TestResult:
                r = await self._run_single(tc, semaphore)
                progress.advance(task)
                return r

            tasks = [_run_and_track(tc) for tc in test_cases]
            results = await asyncio.gather(*tasks)

        # Build summary
        results_list = list(results)
        passed = sum(1 for r in results_list if r.overall_passed)
        failed = sum(1 for r in results_list if not r.overall_passed and r.error is None)
        errors = sum(1 for r in results_list if r.error is not None)
        total = len(results_list)

        scored = [r.overall_score for r in results_list if r.overall_score is not None]
        avg_score = round(sum(scored) / len(scored), 4) if scored else 0.0

        latencies = [r.latency_ms for r in results_list if r.latency_ms > 0]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        critical_failures = sum(
            1 for r in results_list
            if not r.overall_passed and r.test_case.severity == "critical"
        )

        # By category
        by_category: dict[str, CategoryStats] = {}
        for r in results_list:
            cat = r.test_case.category
            if cat not in by_category:
                by_category[cat] = CategoryStats()
            stats = by_category[cat]
            stats.total += 1
            if r.overall_passed:
                stats.passed += 1
            else:
                stats.failed += 1

        for stats in by_category.values():
            stats.pass_rate = round(stats.passed / stats.total, 4) if stats.total > 0 else 0.0

        # RAGAS aggregate scores
        ragas_scores: dict[str, list[float]] = {}
        for r in results_list:
            for ev in r.evaluations:
                if ev.evaluator == "ragas" and ev.details.get("metric_scores"):
                    for metric_name, score in ev.details["metric_scores"].items():
                        ragas_scores.setdefault(metric_name, []).append(score)

        ragas_aggregate = {
            m: round(sum(scores) / len(scores), 4)
            for m, scores in ragas_scores.items()
        }

        summary = RunSummary(
            run_id=run_id,
            timestamp=timestamp,
            chatbot_id=self._chatbot.get_id(),
            chatbot_mode=mode,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=round(passed / total, 4) if total > 0 else 0.0,
            avg_score=avg_score,
            avg_latency_ms=avg_latency,
            critical_failures=critical_failures,
            by_category=by_category,
            ragas_aggregate=ragas_aggregate,
            results=results_list,
        )

        # Print summary
        console.print(f"\n[bold green]Run complete:[/bold green] {run_id}")
        console.print(f"  Total: {total} | Passed: {passed} | Failed: {failed} | Errors: {errors}")
        console.print(f"  Pass rate: {summary.pass_rate:.1%} | Avg score: {avg_score}")
        console.print(f"  Avg latency: {avg_latency:.0f}ms | Critical failures: {critical_failures}")
        if ragas_aggregate:
            console.print("  RAGAS scores:", ragas_aggregate)

        return summary
