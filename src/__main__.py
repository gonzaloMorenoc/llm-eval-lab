"""Entry point: python -m src"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from rich.console import Console

from src.chatbots.mock_adapter import MockChatbot, MockRAGChatbot
from src.chatbots.openai_compatible import OpenAICompatibleChatbot
from src.evaluators.consistency import ConsistencyEvaluator
from src.evaluators.deepeval_evaluator import DeepEvalEvaluator
from src.evaluators.ragas_evaluator import RagasEvaluator
from src.evaluators.rule_based import RuleBasedEvaluator
from src.evaluators.safety import SafetyEvaluator
from src.evaluators.llm_judge import LLMJudgeEvaluator
from src.reporting.json_reporter import generate_json_report
from src.reporting.markdown_reporter import generate_markdown_report
from src.runner.runner import EvalRunner, load_all_datasets

console = Console()


def _build_chatbot(mode: str, provider: str | None):
    """Build the appropriate chatbot based on mode and provider."""
    if provider == "mock":
        if mode == "rag":
            return MockRAGChatbot()
        return MockChatbot()

    if mode == "rag":
        from src.chatbots.rag_chatbot import DemoRAGChatbot
        return DemoRAGChatbot(provider_name=provider)

    return OpenAICompatibleChatbot(provider_name=provider)


def _build_evaluators(mode: str, use_llm_judge: bool = False) -> dict:
    """Build the set of evaluators to use."""
    evaluators = {
        "rule_based": RuleBasedEvaluator(),
        "safety": SafetyEvaluator(),
    }

    # RAGAS evaluator (requires OPENAI_API_KEY for LLM-based metrics)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            evaluators["ragas"] = RagasEvaluator()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize RAGAS evaluator: {e}[/yellow]")
    else:
        console.print("[yellow]Warning: OPENAI_API_KEY not set — RAGAS evaluator disabled.[/yellow]")

    # DeepEval evaluator (requires OPENAI_API_KEY)
    use_deepeval = os.getenv("USE_DEEPEVAL", "false").lower() == "true"
    if use_deepeval and openai_key:
        try:
            evaluators["deepeval"] = DeepEvalEvaluator()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize DeepEval evaluator: {e}[/yellow]")

    # Consistency evaluator
    use_consistency = os.getenv("USE_CONSISTENCY", "false").lower() == "true"
    if use_consistency:
        evaluators["consistency"] = ConsistencyEvaluator()

    if use_llm_judge:
        try:
            evaluators["llm_judge"] = LLMJudgeEvaluator()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize LLM judge: {e}[/yellow]")

    return evaluators


async def main() -> None:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

    mode = os.getenv("CHATBOT_MODE", "plain").lower()
    provider = os.getenv("ACTIVE_PROVIDER") or None
    use_llm_judge = os.getenv("USE_LLM_JUDGE", "false").lower() == "true"

    console.print(f"\n[bold]LLM Eval Lab[/bold]")
    console.print(f"  Mode: [yellow]{mode}[/yellow]")
    console.print(f"  Provider: [cyan]{provider or 'from config.yaml'}[/cyan]")

    chatbot = _build_chatbot(mode, provider)
    evaluators = _build_evaluators(mode, use_llm_judge)

    console.print(f"  Chatbot: [cyan]{chatbot.get_id()}[/cyan]")
    console.print(f"  Evaluators: [green]{', '.join(evaluators.keys())}[/green]")

    test_cases = load_all_datasets()
    console.print(f"  Test cases loaded: [green]{len(test_cases)}[/green]")

    runner = EvalRunner(chatbot=chatbot, evaluators=evaluators)
    summary = await runner.run(test_cases)

    # Generate reports
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", summary.run_id
    )
    json_path = generate_json_report(summary, output_dir)
    md_path = generate_markdown_report(summary, output_dir)

    console.print(f"\n[bold green]Reports generated:[/bold green]")
    console.print(f"  JSON: {json_path}")
    console.print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
