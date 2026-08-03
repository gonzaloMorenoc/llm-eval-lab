## What changes

<!-- What breaks without this change? Describe the problem, then the fix. -->

## Type

- [ ] `fix` — bug fix
- [ ] `feat` — new functionality
- [ ] `refactor` — no behaviour change
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `chore` / `ci` — tooling, dependencies, pipeline

## Effect on evaluation results

<!-- Does this shift scores, thresholds, or pass/fail verdicts for existing
     runs? Say so explicitly — people track these over time. Write "none" if
     the change is inert. -->

## Quality gate

- [ ] `ruff check src/ tests/` and `ruff format --check src/ tests/` pass
- [ ] `mypy src/ --ignore-missing-imports` passes
- [ ] `pytest tests/ --cov=src --cov-fail-under=80` passes
- [ ] Bug fixes include a test that fails without the fix
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`

## Notes for the reviewer

<!-- Anything you are unsure about, or deliberately left out of scope. -->
