"""Typer CLI: run evaluations, manage baselines, and gate quality regressions."""

from __future__ import annotations

import asyncio
import os

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.gate.baseline import BaselineError, build_baseline, load_baseline, save_baseline
from src.gate.comparison import CompatibilityError
from src.gate.models import GatePolicy
from src.gate.policy import PolicyError, evaluate_gate, load_policy
from src.reporting.gate_reporter import generate_gate_markdown, render_gate_console
from src.runner.execution import run_samples
from src.runner.models import RunSummary, TestCase
from src.runner.runner import load_all_datasets, load_dataset

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

    def _announce(index: int, total: int) -> None:
        if total > 1:
            console.print(f"\n[bold]Sample {index}/{total}[/bold]")

    return await run_samples(
        chatbot=chatbot,
        evaluators=evaluators,
        test_cases=test_cases,
        results_dir=results_dir,
        samples=samples,
        on_sample_start=_announce,
    )


def _fail_on_execution_errors(summaries: list[RunSummary]) -> None:
    """Spec §3: an API failure is an execution error (exit 2), never a regression (exit 1).

    The runner does not raise on chatbot failure — it records ``TestResult.error`` and
    counts it in ``RunSummary.errors``, and ``build_baseline`` drops both fields. Without
    this check a provider outage reaches the gate as mass case failures, collapses
    ``pass_rate`` and blames the author's change in the PR report.
    """
    total = sum(s.errors for s in summaries)
    if total == 0:
        return
    first_error = next((r.error for s in summaries for r in s.results if r.error), "unknown error")
    console.print(f"[red]Execution errors during evaluation: {total} case(s) got no response.[/red]")
    console.print(f"[red]First error: {first_error}[/red]")
    raise typer.Exit(code=2)


def _load_summary(results_dir: str, run_id: str) -> RunSummary:
    """Load a persisted run's report.json as a RunSummary (exit 2 if missing or invalid)."""
    path = os.path.join(results_dir, run_id, "report.json")
    if not os.path.exists(path):
        console.print(f"[red]Run not found: {path}[/red]")
        raise typer.Exit(code=2)
    try:
        with open(path) as f:
            return RunSummary.model_validate_json(f.read())
    except Exception as e:
        console.print(f"[red]Failed to parse {path}: {e}[/red]")
        raise typer.Exit(code=2) from e


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
    default_mode: str = os.getenv("CHATBOT_MODE", "plain")
    resolved_mode = (mode or default_mode).lower()
    summaries = asyncio.run(_execute_runs(resolved_provider, resolved_mode, samples, evaluators, datasets, results_dir))
    console.print(f"\nRun ids: {', '.join(s.run_id for s in summaries)}")


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
        path = save_baseline(baseline, os.path.join(baselines_dir, f"{name}.json"))
    except BaselineError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e
    except OSError as e:
        # A non-writable baselines directory is a config error, not a crash.
        console.print(f"[red]Cannot write baseline file: {e}[/red]")
        raise typer.Exit(code=2) from e
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
    default_mode: str = os.getenv("CHATBOT_MODE", "plain")
    resolved_mode = (mode or default_mode).lower()
    try:
        summaries = asyncio.run(_execute_runs(resolved_provider, resolved_mode, samples, evaluators, datasets, results_dir))
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise typer.Exit(code=2) from e

    _fail_on_execution_errors(summaries)

    # Everything from here on is inside the exit-code contract: an unexpected failure
    # (unwritable results dir, unwritable $GITHUB_STEP_SUMMARY, ...) must exit 2, not
    # leak a traceback with Python's exit code 1, which CI would read as "regression".
    try:
        current = build_baseline(summaries)
        verdict = evaluate_gate(baseline_file, current, gate_policy)
        render_gate_console(verdict, console)
        output_dir = os.path.join(results_dir, summaries[-1].run_id)
        md_path = generate_gate_markdown(verdict, output_dir)
        console.print(f"Gate report: {md_path}")

        if verdict.missing_gated_metrics:
            raise typer.Exit(code=2)
        if not verdict.passed:
            raise typer.Exit(code=1)
    except typer.Exit:
        # typer.Exit subclasses RuntimeError, so the broad handler below would swallow
        # the two deliberate exits above and turn them into exit 2.
        raise
    except CompatibilityError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e
    except Exception as e:
        console.print(f"[red]Gate evaluation failed: {e}[/red]")
        raise typer.Exit(code=2) from e
