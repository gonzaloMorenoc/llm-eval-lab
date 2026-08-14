# Contributing to LLM Eval Lab

Thanks for your interest. This project exists to make LLM quality assurance
legible — so contributions that clarify *why* an evaluation works are as
welcome as ones that add features.

## Getting set up

```bash
git clone https://github.com/gonzaloMorenoc/llm-eval-lab.git
cd llm-eval-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,dev]"
```

No API key is needed to develop or run the test suite — the mock adapters cover
everything. You only need provider keys to evaluate a real model.

## The quality gate

Run this before pushing. CI runs exactly the same three commands:

```bash
ruff check src/ tests/ && ruff format --check src/ tests/
mypy src/ --ignore-missing-imports
pytest tests/ --cov=src --cov-fail-under=80
```

## Conventions

- **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`, `ci:`.
- **Tests come first.** Coverage must stay at or above 80%; bug fixes should
  include a test that fails before the fix.
- **English** for code, comments, docstrings, and documentation.
- **Explain the reasoning.** Module docstrings in this repo describe what a
  component measures, why that approach was chosen, and where it falls short.
  New modules should do the same — the limitations section matters most.

## Adding an evaluator

1. Implement `BaseEvaluator` in `src/evaluators/`.
2. Add its name to the `evaluation_type` Literal in `src/runner/models.py`.
3. Register it in `src/__main__.py` → `_build_evaluators()` and in
   `src/dashboard/pages/1_run.py` → `_build_evaluators()`.
4. Add thresholds to `config/config.yaml` — no hardcoded magic numbers.
5. Add test cases to `datasets/` that exercise it.

## Adding a provider

Add a block to `config/config.yaml` under `providers:`. Any OpenAI-compatible
API works without code changes.

## Pull requests

- Branch from `main`, one logical change per PR.
- Describe what breaks without the change, not only what the change does.
- Update `CHANGELOG.md` under `## [Unreleased]`.
- If you change evaluation behaviour, say how it shifts scores — thresholds
  are load-bearing for anyone tracking runs over time.

## Reporting bugs

Open an issue with the reproduction, the expected versus actual result, and
your Python version. If it involves a provider, include which one — behaviour
varies a lot between them.
